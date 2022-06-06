# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/expert.py ]
#   Synopsis     [ the HuBERT wrapper ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import fairseq
from ..interfaces import UpstreamBase


############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt]
        )
        self.model = model[0]
        # if len(self.model.encoder.layers) >= 24 and hasattr(self.model.encoder.layers[23].self_attn, "fp32_attention"):
        #     self.model.encoder.layers[23].self_attn.fp32_attention = False
        # if len(self.model.encoder.layers) >= 12 and hasattr(self.model.encoder.layers[11].self_attn, "fp32_attention"):
        #     self.model.encoder.layers[11].self_attn.fp32_attention = False
        # for i in range(len(self.model.encoder.layers)):
        #     if hasattr(self.model.encoder.layers[i].self_attn, "attention_relaxation"):
        #         self.model.encoder.layers[i].self_attn.attention_relaxation = False
        self.task = task

        if len(self.hooks) == 0:
            if hasattr(self.model, "speech_encoder"):
                module_name = "self.model.speech_encoder.encoder.layers"
            else:
                module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            if hasattr(self.model, "speech_encoder"):
                self.add_hook("self.model.speech_encoder.encoder", lambda input, output: output[0])
            else:
                self.add_hook("self.model.encoder", lambda input, output: output[0])

            if hasattr(self.model, "shared_encoder"):
                module_name = "self.model.shared_encoder"
                for module_id in range(len(eval(module_name))):
                    self.add_hook(
                        f"{module_name}[{module_id}]",
                        lambda input, output: output[0].transpose(0, 1),
                    )

            if hasattr(self.model, "decoder"):
                module_name = "self.model.decoder"
                if hasattr(self.model.decoder, "layers"):
                    module_name = "self.model.decoder.layers"
                for module_id in range(len(eval(module_name))):
                    self.add_hook(
                        f"{module_name}[{module_id}]",
                        lambda input, output: output[0].transpose(0, 1),
                    )

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))
            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        if hasattr(self.model, "feature_stride"):
            print(f"The downsample rate of model is {self.model.feature_stride}")
            return self.model.feature_stride
        return 320

    def forward(self, wavs):
        if self.task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        res = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
