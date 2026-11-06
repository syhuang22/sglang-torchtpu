# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)


class SoundTokenizerLoader(PlainStateDictComponentLoader):
    component_names = ["sound_tokenizer"]
    default_precision_attr = "vae_precision"

    def build_model_config(self, config, component_name):
        return {"config": config}
