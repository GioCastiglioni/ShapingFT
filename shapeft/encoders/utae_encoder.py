from collections import OrderedDict
from logging import Logger
from typing import Sequence

import torch
import torch.nn as nn

from shapeft.encoders.base import Encoder


class UTAE_Encoder(Encoder):
    """
    Multi Temporal UTAE Encoder for Supervised Baseline, to be trained from scratch.
    It supports single time frame inputs with optical bands

    Args:
        input_bands (dict[str, list[str]]): Band names, specifically expecting the 'optical' key with a list of bands.
        input_size (int): Size of the input images (height and width).
        topology (Sequence[int]): The number of feature channels at each stage of the U-Net encoder.

    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        multi_temporal: int,
        topology: Sequence[int],
        output_dim: int | list[int],
        download_url: str,
        encoder_weights: str | None = None,
    ):
        super().__init__(
            model_name="utae_encoder",
            encoder_weights=encoder_weights,  # no pre-trained weights, train from scratch
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=0,
            output_dim=output_dim,
            output_layers=None,
            multi_temporal=multi_temporal,
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
        )

        self.in_channels = len(input_bands["optical"])  # number of optical bands
        self.topology = topology

        self.in_conv = ConvBlock(
            nkernels=[self.in_channels] + [self.topology[0], self.topology[0]],
            pad_value=0,
            norm="group",
            padding_mode="reflect",
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=self.topology[i],
                d_out=self.topology[i + 1],
                k=4,
                s=2,
                p=1,
                pad_value=0,
                norm="group",
                padding_mode="reflect",
            )
            for i in range(len(self.topology) - 1)
        )

    def forward(self, image):
        pass

    def load_encoder_weights(self, logger: Logger, from_scratch: bool = True) -> None:
        pass


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.reshape(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.reshape(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out
