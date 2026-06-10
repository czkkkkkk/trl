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

from argparse import Namespace

from .base import Command, CommandContext


class HfServeCommand(Command):
    """CLI command for serving TRL models with the HF data-parallel rollout server."""

    def __init__(self):
        super().__init__(name="hf-serve", help_text="Serve a model with the HF rollout server")

    def register(self, subparsers) -> None:
        subparsers.add_parser(self.name, help=self.help_text, add_help=False)

    def run(self, args: Namespace, context: CommandContext) -> int:
        from ...scripts.hf_serve import main as hf_serve_main
        from ...scripts.hf_serve import make_parser as make_hf_serve_parser

        parser = make_hf_serve_parser()
        (script_args,) = parser.parse_args_and_config(args=context.argv_after(self.name))
        hf_serve_main(script_args)
        return 0
