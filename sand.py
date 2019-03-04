from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_sand_ckpt(model, ckpt_file):
    map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
    ckpt = torch.load(ckpt_file, map_location=map_location)

    # Single branch, so remove 'branch.' prefix from variables
    state = {}
    for k, v in ckpt['model'].items():
        state[k.replace('branch.', '')] = v
    model.load_state_dict(state)


def convbn(in_channels, out_channels, kernel_size, stride, padding, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=dilation if dilation > 1 else padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample, pad, dilation):
        super().__init__()

        self.conv1 = nn.Sequential(
            convbn(in_channels, out_channels, 3, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )

        self.conv2 = convbn(out_channels, out_channels, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Sand(nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
        self.in_channels = 32

        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True)
        )

        self.lastconv = nn.Sequential(
            convbn(320, 128, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False)
        )

        self.final = nn.Conv2d(32, self.n_dims, kernel_size=1, padding=0, stride=1, bias=False)

    def _make_layer(self, block, out_channels, blocks, stride, pad, dilation):
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        else:
            downsample = None

        layers = [block(self.in_channels, out_channels, stride, downsample, pad, dilation)]
        self.in_channels = out_channels*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):

        pad = 50
        x = F.pad(x, (pad,) * 4)

        output = self.firstconv(x)
        skip = output
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, output_skip.size()[-2:], mode='bilinear', align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, output_skip.size()[-2:], mode='bilinear', align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, output_skip.size()[-2:], mode='bilinear', align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, output_skip.size()[-2:], mode='bilinear', align_corners=True)

        output_feature = torch.cat((output_raw, output_skip, output_branch4,
                                    output_branch3, output_branch2, output_branch1), 1)

        output_feature = F.interpolate(output_feature, skip.size()[-2:], mode='bilinear', align_corners=True)
        output_feature = self.lastconv(output_feature) + skip
        output_feature = F.interpolate(output_feature, x.size()[-2:], mode='bilinear', align_corners=True)
        output_feature = self.final(output_feature)
        output_feature = output_feature[..., pad:-pad, pad:-pad]

        return output_feature
