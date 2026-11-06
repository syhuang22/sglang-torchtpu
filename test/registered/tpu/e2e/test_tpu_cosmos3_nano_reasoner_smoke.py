"""Cosmos3-Nano reasoner (its understanding tower as a VLM) on one TPU device.

Serves the checkpoint through srt and asks the checkpoint's own reasoning example.
Greedy decoding on TPU reproduces the official transformers-cosmos3 fp32 plan word
for word, so the test pins its key objects rather than only a non-empty reply.
"""

import base64
import json
import os
import unittest

import requests
from huggingface_hub import hf_hub_download

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=1800,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "cosmos3-nano", "multimodal", "smoke"],
)

REPO = "nvidia/Cosmos3-Nano"


def _skip_if_no_tpu() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch_tpu") is None


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuCosmos3NanoReasonerSmoke(CustomTestCase):
    """Launch the reasoner on TPU and check its plan for the checkpoint example."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            resolve_tpu_model_path(REPO),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dtype",
                "bfloat16",
                "--device",
                "tpu",
                "--attention-backend",
                "pallas_tpu_rpa_v3",
                "--tp-size",
                "1",
                "--grammar-backend",
                "none",
                "--page-size",
                "128",
                "--context-length",
                "8192",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_reasoning_example_plan(self):
        example = json.load(open(hf_hub_download(REPO, "assets/example_reasoning_prompt.json")))
        with open(hf_hub_download(REPO, "assets/example_reasoning_input.png"), "rb") as f:
            image = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "default",
                "temperature": 0,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image}},
                            {"type": "text", "text": example["prompt"]},
                        ],
                    }
                ],
            },
            timeout=600,
        )
        self.assertEqual(resp.status_code, 200)
        text = resp.json()["choices"][0]["message"]["content"]
        # The official fp32 greedy plan: arm to flower, grasp, arm to bottle, place.
        for words in ("flower", "red bottle", "Grasp"):
            self.assertIn(words, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
