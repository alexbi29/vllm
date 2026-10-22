# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig


class Gemma4MTPSpeculatorConfig(MTPSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.gemma4.speculator.Gemma4Speculator"
    )
    # Inherit MTP head quantization: Gemma quantizes its assistant's own head.
    supported_options = MTPSpeculatorConfig.supported_options - {
        "draft_lm_head_token_ranges"
    }
