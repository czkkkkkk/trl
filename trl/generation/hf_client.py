# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HFClient: HTTP + ZMQ client for the HF serve-mode rollout server.

Mirrors the NkipyClient interface but targets a plain HF transformers server
(``rl_examples/deepmath/hf_serve/hf_serve.py``) that runs
``AutoModelForCausalLM.generate()`` under ``torch.compile(backend="neuron")``
in a separate process.  Weight updates ride over ZMQ REQ/REP with msgpack
metadata and raw torch-tensor bytes (no numpy hop, so bf16 travels natively).
"""

from __future__ import annotations

import logging
import socket
import time
from urllib.parse import urlparse

import msgpack
import requests
import zmq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from torch import nn


logger = logging.getLogger(__name__)


class HFClient:
    """Client for the HF serve-mode rollout server.

    Weight sync lifecycle (per training step):
        1. ``init_communicator()`` — POST ``/init_communicator/`` + connect ZMQ
        2. ``update_named_param()`` x N — send each parameter via ZMQ
        3. ``close_communicator()`` — send ZMQ "done" + close socket

    Args:
        base_url (`str`, *optional*):
            Base URL for the HF server (e.g., ``"http://localhost:30000"``).
            If provided, ``host`` and ``server_port`` are ignored.
        host (`str`, *optional*, defaults to ``"0.0.0.0"``):
            IP address of the HF server.
        server_port (`int`, *optional*, defaults to ``30000``):
            Port number of the HF server.
        zmq_port (`int`, *optional*, defaults to ``5558``):
            Port for ZMQ weight sync communication.
        connection_timeout (`float`, *optional*, defaults to ``0.0``):
            Timeout in seconds to wait for the server.
    """

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 30000,
        zmq_port: int = 5558,
        connection_timeout: float = 0.0,
    ):
        self.session = requests.Session()

        retry_strategy = Retry(
            total=5,
            connect=5,
            read=5,
            status=3,
            status_forcelist=[500, 502, 503],
            backoff_factor=2,
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if base_url is not None:
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"

        self.zmq_port = zmq_port
        self.zmq_timeout_ms = 300_000  # 5 min per param; surfaces hangs instead of wedging
        self._zmq_ctx: zmq.Context | None = None
        self._zmq_sock: zmq.Socket | None = None

        self.check_server(connection_timeout)

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """Check server availability, retrying until *total_timeout* elapses."""
        url = f"{self.base_url}/health"
        start_time = time.time()

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                if time.time() - start_time >= total_timeout:
                    raise ConnectionError(
                        f"The HF server can't be reached at {self.base_url} after "
                        f"{total_timeout} seconds. Make sure the server is running."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            logger.info(f"Server not up yet. Retrying in {retry_interval}s...")
            time.sleep(retry_interval)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_tokens: int = 16,
        logprobs: int | None = 0,
        generation_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Generate completions for *prompts*.

        Returns a dict matching the VLLMClient/NkipyClient interface::

            {
                "prompt_ids":        list[list[int]],
                "completion_ids":    list[list[int]],
                "logprobs":          list[list[list[float]]] | None,
                "logprob_token_ids": list[list[list[int]]] | None,
            }
        """
        url = f"{self.base_url}/generate"

        # HF disables top_k with 0 (nkipy used -1). Pass through unchanged.
        logger.info(f"[HFClient] POST {url}: {len(prompts)} prompts, n={n}")
        t0 = time.time()
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_new_tokens": max_tokens,
                "return_logprob": logprobs is not None,
                "generation_kwargs": generation_kwargs or {},
            },
        )
        gen_elapsed = time.time() - t0
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        json_response = response.json()
        results = json_response.get("results", [json_response])

        prompt_ids_map: dict[int, list[int]] = {}
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[list[float]] | None] = []
        all_logprob_token_ids: list[list[list[int]] | None] = []

        for r in results:
            idx = r.get("index", 0)
            if idx not in prompt_ids_map:
                prompt_ids_map[idx] = r["prompt_ids"]
            all_completion_ids.append(r["completion_ids"])

            token_logprobs = r.get("token_logprobs")
            if token_logprobs is not None:
                all_logprobs.append([[entry[0]] for entry in token_logprobs])
                all_logprob_token_ids.append([[entry[1]] for entry in token_logprobs])
            else:
                all_logprobs.append(None)
                all_logprob_token_ids.append(None)

        prompt_ids = [prompt_ids_map[i] for i in sorted(prompt_ids_map)]

        num_completions = len(all_completion_ids)
        logger.info(
            f"[HFClient.generate] Received {num_completions} completions "
            f"(expected {len(prompts) * n}) in {gen_elapsed:.2f}s"
        )

        has_logprobs = any(lp is not None for lp in all_logprobs)
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs if has_logprobs else None,
            "logprob_token_ids": all_logprob_token_ids if has_logprobs else None,
        }

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def init_communicator(self):
        """Start a weight sync session: tell the server, then connect ZMQ."""
        url = f"{self.base_url}/init_communicator/"
        response = self.session.post(
            url,
            json={"host": "0.0.0.0", "port": self.zmq_port},
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay for the server to bind the ZMQ REP socket.
        time.sleep(1)

        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
        self._zmq_sock.setsockopt(zmq.RCVTIMEO, self.zmq_timeout_ms)
        self._zmq_sock.setsockopt(zmq.SNDTIMEO, self.zmq_timeout_ms)
        self._zmq_sock.setsockopt(zmq.LINGER, 0)
        self._zmq_sock.connect(f"tcp://{self.host}:{self.zmq_port}")

        logger.info(f"ZMQ connected to tcp://{self.host}:{self.zmq_port}")

    def update_named_param(
        self, name: str, weights: torch.Tensor, verbose: bool = False
    ) -> dict:
        """Send one named parameter to the server via ZMQ.

        Ships the torch tensor's raw bytes directly — no numpy hop, so bf16
        travels without needing ``ml_dtypes``.
        """
        if self._zmq_sock is None:
            raise RuntimeError(
                "Communicator not initialized. Call init_communicator() first."
            )

        start_time = time.time()
        tensor = weights.detach().cpu().contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        data_size_mb = nbytes / 1e6

        # Zero-copy ZMQ send: pass the tensor's memoryview directly. This skips
        # the 600 MB+ bytes() materialization that makes ``embed_tokens.weight``
        # look like a hang.
        view = memoryview(tensor.numpy()).cast("b")
        meta = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        # copy=False: zmq sends directly from the memoryview; the tensor must
        # stay alive until send completes (it does — we hold ``tensor``).
        self._zmq_sock.send(msgpack.packb(meta), zmq.SNDMORE)
        self._zmq_sock.send(view, copy=False)
        ack = msgpack.unpackb(self._zmq_sock.recv())

        if ack.get("status") != "ok":
            raise RuntimeError(f"Server error for {name}: {ack}")

        elapsed = time.time() - start_time

        if verbose:
            logger.info(f"Updated {name}: {data_size_mb:.2f} MB in {elapsed:.3f}s")

        return {
            "param_name": name,
            "data_size_mb": data_size_mb,
            "transfer_time": elapsed,
        }

    def update_model_params(
        self, model: nn.Module, verbose: bool = False, log_every: int = 50
    ) -> dict:
        """Update all model parameters via ZMQ."""
        all_stats = []
        total_size_gb = 0.0
        total_time = 0.0

        sync_start = time.time()
        params = list(model.named_parameters())
        for i, (name, param) in enumerate(params, start=1):
            stats = self.update_named_param(name, param.data, verbose=verbose)
            all_stats.append(stats)
            total_size_gb += stats["data_size_mb"] / 1e3
            total_time += stats["transfer_time"]
            if log_every and (i == 1 or i % log_every == 0 or i == len(params)):
                logger.info(
                    f"[HFClient.update_model_params] {i}/{len(params)} "
                    f"{name} ({stats['data_size_mb']:.1f} MB in {stats['transfer_time']:.2f}s)"
                )
        wall_elapsed = time.time() - sync_start
        logger.info(
            f"[HFClient.update_model_params] synced {len(params)} params "
            f"({total_size_gb:.2f} GB) in {wall_elapsed:.2f}s "
            f"(sum per-param {total_time:.2f}s)"
        )

        return {
            "num_params": len(all_stats),
            "total_size_gb": total_size_gb,
            "total_time": total_time,
            "wall_time": wall_elapsed,
            "per_param_stats": all_stats,
        }

    def close_communicator(self):
        """End the weight sync session (send ZMQ ``"done"`` + close socket)."""
        if self._zmq_sock is not None:
            self._zmq_sock.send_multipart([msgpack.packb({"cmd": "done"}), b""])
            ack = msgpack.unpackb(self._zmq_sock.recv())
            logger.info(f"Weight sync session ended: {ack}")
            self._zmq_sock.close()
            self._zmq_sock = None

        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None

        try:
            url = f"{self.base_url}/close_communicator/"
            self.session.post(url)
        except requests.ConnectionError:
            pass  # server may already be down
