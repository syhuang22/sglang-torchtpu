# SPDX-License-Identifier: Apache-2.0
"""Audio tokenizer (spectrogram ConvNeXt encoder, Oobleck decoder) for Cosmos3 sound."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)


class Snake1d(nn.Module):
    def __init__(self, hidden_dim: int, logscale: bool = True) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        alpha = torch.exp(self.alpha) if self.logscale else self.alpha
        beta = torch.exp(self.beta) if self.logscale else self.beta
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)


class OobleckResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = weight_norm(
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        )
        self.snake2 = Snake1d(dim)
        self.conv2 = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.snake1(x))
        y = self.conv2(self.snake2(y))
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class OobleckDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        output_padding: int,
    ) -> None:
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=output_padding,
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        return self.res_unit3(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list[int],
        channel_multiples: list[int],
    ) -> None:
        super().__init__()
        strides = upsampling_ratios
        mults = [1] + list(channel_multiples)

        self.conv1 = weight_norm(
            nn.Conv1d(input_channels, channels * mults[-1], kernel_size=7, padding=3)
        )

        blocks = []
        for i, stride in enumerate(strides):
            blocks.append(
                OobleckDecoderBlock(
                    input_dim=channels * mults[len(strides) - i],
                    output_dim=channels * mults[len(strides) - i - 1],
                    stride=stride,
                    output_padding=stride % 2,
                )
            )
        self.block = nn.ModuleList(blocks)
        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(
            nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.block:
            x = layer(x)
        x = self.snake1(x)
        return self.conv2(x)


class _FP32LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32 whatever the module dtype, as the checkpoint was trained."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps
        ).to(x.dtype)


class Cosmos3AudioConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, use_snake: bool) -> None:
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.ConstantPad1d((3, 3), 0),
            nn.Conv1d(dim, dim, kernel_size=7, groups=dim),
        )
        self.norm = _FP32LayerNorm(dim, eps=1e-5, bias=False)
        self.pwconv1 = nn.Conv1d(dim, dim * 4, kernel_size=1)
        self.act = Snake1d(dim * 4) if use_snake else nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * 4, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.pwconv2(self.act(self.pwconv1(y)))
        return x + y


class Cosmos3AudioSpectrogramEncoder(nn.Module):
    """Waveform -> VAE moments: STFT features through strided ConvNeXt stages."""

    def __init__(
        self,
        audio_channels: int,
        channels: int,
        latent_dim: int,
        channel_multiples: list[int],
        strides: list[int],
        num_blocks: int,
        n_fft: int,
        hop_length: int,
        use_snake: bool,
        padding_mode: str,
    ) -> None:
        super().__init__()
        self.audio_channels = audio_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        mults = list(channel_multiples)
        # Real and imaginary parts of n_fft // 2 + 1 bins per channel.
        layers: list[nn.Module] = [
            weight_norm(
                nn.Conv1d(
                    (n_fft + 2) * audio_channels,
                    mults[0] * channels,
                    kernel_size=1,
                    bias=False,
                )
            )
        ]
        for i, stride in enumerate(strides):
            in_dim = mults[i] * channels
            out_dim = mults[min(i + 1, len(mults) - 1)] * channels
            layers += [Cosmos3AudioConvNeXtBlock(in_dim, use_snake) for _ in range(num_blocks)]
            layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_dim,
                        out_dim,
                        kernel_size=2 * stride,
                        stride=stride,
                        padding=math.ceil(stride / 2),
                        padding_mode=padding_mode,
                    )
                )
            )
        layers.append(
            weight_norm(
                nn.Conv1d(mults[-1] * channels, latent_dim, kernel_size=1, bias=False)
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch, num_channels, num_samples = audio.shape
        pad = self.n_fft - self.hop_length
        waveform = F.pad(
            audio.reshape(batch * num_channels, num_samples).float(),
            (pad // 2, pad - pad // 2),
        )
        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        # [B*C, F, T] complex -> [B, C * 2F, T]: per channel, real bins then imaginary.
        spec = torch.cat([spec.real, spec.imag], dim=1).to(audio.dtype)
        spec = spec.reshape(batch, num_channels * spec.shape[1], spec.shape[2])
        return self.layers(spec)


def _cfg(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        v = config.get(k)
        if v is not None:
            return v
    return default


class Cosmos3AVAEAudioTokenizer(nn.Module, LayerwiseOffloadableModuleMixin):
    """Cosmos3 audio tokenizer: latents → waveform via an Oobleck decoder stack."""

    layerwise_offload_dit_group_enabled = False
    layer_names = ["decoder.block"]

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.sample_rate = int(
            _cfg(config, "sampling_rate", "sample_rate", default=48000)
        )
        self.audio_channels = int(
            _cfg(
                config,
                "dec_out_channels",
                "audio_channels",
                default=2 if bool(config.get("stereo", True)) else 1,
            )
        )
        self.latent_channels = int(
            _cfg(config, "vocoder_input_dim", "io_channels", "latent_ch", default=64)
        )
        dec_strides = [
            int(s) for s in _cfg(config, "dec_strides", default=[2, 4, 5, 6, 8])
        ]
        self.hop_size = int(
            _cfg(
                config,
                "hop_size",
                default=math.prod(dec_strides) if dec_strides else 1920,
            )
        )
        stride_product = math.prod(dec_strides)
        if stride_product != self.hop_size:
            raise ValueError(
                "Cosmos3 AVAE dec_strides product must equal hop_size: "
                f"product={stride_product}, hop_size={self.hop_size}."
            )

        norm = str(_cfg(config, "normalization_type", default="none"))
        if bool(_cfg(config, "normalize_latents", default=False)) and norm == "none":
            norm = "tanh"
        self.normalization_type = norm
        self.tanh_input_scale = float(_cfg(config, "tanh_input_scale", default=1.5))
        self.tanh_output_scale = float(_cfg(config, "tanh_output_scale", default=3.5))
        self.tanh_clamp = float(_cfg(config, "tanh_clamp", default=0.995))

        self.normalize_volume = bool(_cfg(config, "normalize_volume", default=True))
        stereo = bool(config.get("stereo", True))
        self.encoder = Cosmos3AudioSpectrogramEncoder(
            audio_channels=int(_cfg(config, "input_channels", default=1)) * (2 if stereo else 1),
            channels=int(_cfg(config, "enc_dim", default=192)),
            latent_dim=int(_cfg(config, "enc_latent_dim", default=128)),
            channel_multiples=list(_cfg(config, "enc_c_mults", default=[1, 2, 4])),
            strides=list(_cfg(config, "enc_strides", default=[4, 5, 6])),
            num_blocks=int(_cfg(config, "enc_num_blocks", default=2)),
            n_fft=int(_cfg(config, "enc_n_fft", default=64)),
            hop_length=int(_cfg(config, "enc_hop_length", default=16)),
            use_snake=bool(_cfg(config, "enc_use_snake", default=True)),
            padding_mode=str(_cfg(config, "padding_mode", default="zeros")),
        )

        self.decoder = OobleckDecoder(
            channels=int(_cfg(config, "dec_dim", default=320)),
            input_channels=self.latent_channels,
            audio_channels=self.audio_channels,
            upsampling_ratios=list(reversed(dec_strides)),
            channel_multiples=list(
                _cfg(config, "dec_c_mults", default=[1, 2, 4, 8, 16])
            ),
        )

    @property
    def temporal_compression_factor(self) -> int:
        return self.hop_size

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(num_audio_samples) // self.hop_size

    def get_encoded_latent_num_samples(self, num_audio_samples: int) -> int:
        """Latent length of an encoded clip; encoding pads it up to a whole hop."""
        return -(-int(num_audio_samples) // self.hop_size)

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(num_latent_samples) * self.hop_size

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.normalization_type == "tanh":
            in_dtype = latent.dtype
            x = torch.clamp(
                latent.float() / self.tanh_output_scale,
                -self.tanh_clamp,
                self.tanh_clamp,
            )
            return (torch.atanh(x) * self.tanh_input_scale).to(in_dtype)
        if self.normalization_type != "none":
            raise ValueError(
                f"Unsupported AVAE normalization_type={self.normalization_type!r}."
            )
        return latent

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Waveform ``[B, C, N]`` -> posterior-mean latents ``[B, latent_channels, ceil(N / hop)]``."""
        if self.normalization_type != "none":
            raise NotImplementedError(
                f"encoding with AVAE normalization_type={self.normalization_type!r}"
            )
        if self.normalize_volume:
            waveform = waveform / (waveform.abs().max() + 1e-5) * 0.95
        pad = -waveform.shape[-1] % self.hop_size
        waveform = F.pad(waveform, (0, pad))
        encoder = next(self.encoder.parameters())
        moments = self.encoder(waveform.to(device=encoder.device, dtype=encoder.dtype))
        # The VAE bottleneck stores [mean, scale] on the channel axis.
        return moments[:, : self.latent_channels]

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        squeeze = latent.ndim == 2
        if squeeze:
            latent = latent.unsqueeze(0)
        decoder_dtype = next(self.decoder.parameters()).dtype
        decoder_device = next(self.decoder.parameters()).device
        z = self._denormalize_latent(latent.to(decoder_device)).to(decoder_dtype)
        audio = self.decoder(z).clamp(-1.0, 1.0).to(latent.dtype)
        return audio.squeeze(0) if squeeze else audio


EntryClass = Cosmos3AVAEAudioTokenizer
