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

"""HF serve-mode generation backend for TRL trainers.

Server-mode only: sends generation requests via HTTP and weight updates via
ZMQ to a separate HF transformers server that runs ``generate()`` under
``torch.compile(backend="neuron")``.  Mirrors the NkipyGeneration interface
so GRPOTrainer can swap in this backend transparently.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING

from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from ..data_utils import is_conversational
from ..extras.profiling import ProfilingContext
from .hf_client import HFClient


if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import PeftModel


logger = logging.getLogger(__name__)


class HFGeneration:
    """Handles HF serve-mode generation for trainers.

    Extracts all HF-server-specific logic (init, generation, weight sync) into
    a separate class that matches the ``NkipyGeneration`` / ``VLLMGeneration``
    interface.

    Args:
        model: Training model whose weights will be synced.
        accelerator: Accelerator for distributed training.
        is_fsdp_enabled: Whether FSDP is enabled.
        processing_class: Tokenizer or processor.
        server_base_url: Base URL (e.g. ``"http://localhost:30000"``).
        server_host: Server host.  Ignored if *server_base_url* is set.
        server_port: Server port.  Ignored if *server_base_url* is set.
        server_timeout: Seconds to wait for the server to come up.
        zmq_port: Port for ZMQ weight sync channel.
        repetition_penalty / temperature / top_p / top_k / min_p: Sampling params.
        max_completion_length: Max tokens to generate.
        logprobs: ``0`` to return sampled-token logprob only, ``None`` to skip.
        generation_kwargs: Extra generation params (e.g. ``cache_implementation``,
            ``prefill_chunk_size``).
        chat_template / chat_template_kwargs / tools: Chat configuration.
    """

    def __init__(
        self,
        model: "PreTrainedModel | PeftModel",
        accelerator: "Accelerator",
        is_fsdp_enabled: bool,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        # Server configuration
        server_base_url: str | None = None,
        server_host: str = "0.0.0.0",
        server_port: int = 30000,
        server_timeout: float = 240.0,
        zmq_port: int = 5558,
        # Generation configuration
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_completion_length: int = 16,
        logprobs: int | None = 0,
        generation_kwargs: dict | None = None,
        # Chat / tool configuration
        chat_template: str | None = None,
        chat_template_kwargs: dict | None = None,
        tools: list | None = None,
    ):
        self.model = model
        self.accelerator = accelerator
        self.is_fsdp_enabled = is_fsdp_enabled
        self.processing_class = processing_class

        self.server_base_url = server_base_url
        self.server_host = server_host
        self.server_port = server_port
        self.server_timeout = server_timeout
        self.zmq_port = zmq_port

        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_completion_length = max_completion_length
        self.logprobs = logprobs
        self.generation_kwargs = generation_kwargs or {}

        self.chat_template = chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tools = tools

        self._init_hf()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_hf(self):
        accelerator = self.accelerator

        if accelerator.is_main_process:
            if self.server_base_url is not None:
                base_url = self.server_base_url
            else:
                base_url = f"http://{self.server_host}:{self.server_port}"

            self.hf_client = HFClient(
                base_url=base_url,
                zmq_port=self.zmq_port,
                connection_timeout=self.server_timeout,
            )

        accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # Weight synchronization
    # ------------------------------------------------------------------

    def _fix_param_name(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        """Strip FSDP / checkpoint-wrapping prefixes."""
        extra_prefixes = extra_prefixes or []
        for prefix in ["_checkpoint_wrapped_module."] + extra_prefixes:
            name = name.replace(prefix, "")
        return name

    @staticmethod
    def _is_lora_internal(name: str) -> bool:
        return any(
            k in name
            for k in [".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."]
        )

    def _sync_fsdp2_params(self, module: nn.Module):
        """FSDP2-specific sync: ``state_dict()`` already returns full tensors."""
        accelerator = self.accelerator
        state_dict = module.state_dict()

        for orig_name, param in state_dict.items():
            name = orig_name.removeprefix("base_model.model.").replace(".base_layer", "")
            if self._is_lora_internal(name):
                continue
            if "original_module" in name:
                continue
            name = self._fix_param_name(name, extra_prefixes=["modules_to_save.default."])

            param = param.full_tensor()

            if accelerator.is_main_process:
                self.hf_client.update_named_param(name, param)

    def sync_weights(self):
        """Synchronize training model weights to the HF server.

        Opens a ZMQ weight-sync session, sends all parameters, then closes it.
        Handles FSDP2 and PEFT.
        """
        model = self.model
        accelerator = self.accelerator
        is_fsdp_enabled = self.is_fsdp_enabled

        if accelerator.is_main_process:
            self.hf_client.init_communicator()
        accelerator.wait_for_everyone()

        if is_peft_model(model):
            model.merge_adapter()

            if is_fsdp_enabled:
                self._sync_fsdp2_params(model.get_base_model())
            else:
                base_model = model.get_base_model()
                for name, param in base_model.named_parameters():
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self._is_lora_internal(name):
                        continue
                    if model.prefix in name:
                        continue
                    if "original_module" in name:
                        continue
                    name = self._fix_param_name(
                        name, extra_prefixes=["modules_to_save.default."]
                    )
                    if accelerator.is_main_process:
                        self.hf_client.update_named_param(name, param.data)

            model.unmerge_adapter()
        else:
            if is_fsdp_enabled:
                self._sync_fsdp2_params(model)
            else:
                for name, param in model.named_parameters():
                    name = self._fix_param_name(name)
                    if accelerator.is_main_process:
                        self.hf_client.update_named_param(name, param.data)

        if accelerator.is_main_process:
            self.hf_client.close_communicator()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list,
        num_generations: int,
        profiler: ProfilingContext | None = None,
    ) -> tuple:
        """Generate completions using the HF serve-mode server.

        Returns:
            ``(prompt_ids, completion_ids, logprobs, logprob_token_ids, extra_fields)``
        """
        profiler = profiler or nullcontext()
        accelerator = self.accelerator

        all_prompts = gather_object(prompts)

        if accelerator.is_main_process:
            # Deduplicate: prompts already contain num_generations copies.
            ordered_set_of_prompts = all_prompts[::num_generations]

            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                text_prompts = [
                    self.processing_class.apply_chat_template(
                        conversation=msgs,
                        tools=self.tools,
                        chat_template=self.chat_template,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.chat_template_kwargs,
                    )
                    for msgs in ordered_set_of_prompts
                ]
            else:
                text_prompts = ordered_set_of_prompts

            with profiler:
                output = self.hf_client.generate(
                    prompts=text_prompts,
                    n=num_generations,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    logprobs=self.logprobs,
                    generation_kwargs=self.generation_kwargs,
                )
                required_keys = {"prompt_ids", "completion_ids", "logprobs", "logprob_token_ids"}
                extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                payload = (
                    output["prompt_ids"],
                    output["completion_ids"],
                    output["logprobs"],
                    output.get("logprob_token_ids"),
                    extra_fields,
                )
        else:
            payload = None

        obj_list = [payload]
        broadcast_object_list(obj_list, from_process=0)
        (
            all_prompt_ids,
            all_completion_ids,
            all_logprobs,
            all_logprob_token_ids,
            all_extra_fields,
        ) = obj_list[0]

        all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

        process_slice = slice(
            accelerator.process_index * len(prompts),
            (accelerator.process_index + 1) * len(prompts),
        )
        prompt_ids = all_prompt_ids[process_slice]
        completion_ids = all_completion_ids[process_slice]
        logprobs = all_logprobs[process_slice] if all_logprobs is not None else None
        logprob_token_ids = (
            all_logprob_token_ids[process_slice]
            if all_logprob_token_ids is not None
            else None
        )

        extra_fields = {}
        for key, values in all_extra_fields.items():
            if isinstance(values, list):
                extra_fields[key] = values[process_slice]
            else:
                extra_fields[key] = values

        return prompt_ids, completion_ids, logprobs, logprob_token_ids, extra_fields
