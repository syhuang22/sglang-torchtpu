"""Cosmos3-Nano generation and forward dynamics on one TPU device (nightly).

Same harness as the Wan2.2 TPU smoke test. Quality is checked offline against the
official cosmos-framework outputs; this guards that both Cosmos3 request kinds
reach TPU and return a decodable video of the requested length.
"""

from __future__ import annotations

import io
import json
import logging

import pytest
import torch
from huggingface_hub import hf_hub_download

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerContext,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

logger = logging.getLogger(__name__)

register_tpu_ci(
    est_time=2700,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "cosmos3-nano", "diffusion"],
)

REPO = "nvidia/Cosmos3-Nano"
SERVER = DiffusionServerArgs(
    model_path=resolve_tpu_model_path(REPO), modality="video", num_gpus=1
)
# Workaround: seeding a TPU torch.Generator can deadlock on the GIL in torch_tpu's
# blocking host-to-device copy; drop once TorchTPU fixes it.
COMMON = {"num_inference_steps": 4, "seed": 0, "generator_device": "cpu"}


def _forward_dynamics_case() -> DiffusionTestCase:
    example = json.load(
        open(hf_hub_download(REPO, "assets/example_action_fd_agibotworld_action_chunks.json"))
    )
    return DiffusionTestCase(
        "cosmos3_nano_forward_dynamics",
        SERVER,
        DiffusionSamplingParams(
            prompt=example["prompt"],
            image_path=hf_hub_download(
                REPO, "assets/example_action_fd_agibotworld_first_frame.png"
            ),
            output_size="640x640",
            num_frames=example["action_chunk_size"] + 1,
            extras={
                **COMMON,
                "action_mode": "forward_dynamics",
                "domain_name": example["domain_name"],
                "action_view_point": example["view_point"],
                "action_fps": example["fps"],
                "action": example["action_chunks"][0],
            },
        ),
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
    )


CASES = [
    DiffusionTestCase(
        "cosmos3_nano_t2v",
        SERVER,
        DiffusionSamplingParams(
            prompt="A robotic arm wipes a plate with a green sponge in a bright kitchen",
            output_size="832x480",
            num_frames=9,
            extras=COMMON,
        ),
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
    _forward_dynamics_case(),
]


def _decoded_frame_count(video_bytes: bytes) -> int:
    import av

    with av.open(io.BytesIO(video_bytes)) as container:
        return sum(1 for _ in container.decode(video=0))


@pytest.mark.skipif(not hasattr(torch, "tpu"), reason="torch_tpu found no TPU")
class TestCosmos3NanoTPU(DiffusionServerBase):
    """Nightly single-device TPU Cosmos3 video and forward-dynamics generation."""

    @pytest.fixture(params=CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        return request.param

    def test_diffusion_generation(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,
    ):
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )
        _, content = self.run_and_collect(
            diffusion_server, case.id, generate_fn, collect_perf=False
        )
        frames = _decoded_frame_count(content)
        logger.info("decoded %d frames, %d bytes", frames, len(content))
        want = case.sampling_params.num_frames
        assert frames == want, f"expected {want} frames, decoded {frames}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
