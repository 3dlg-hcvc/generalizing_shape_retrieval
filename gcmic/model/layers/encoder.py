import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

import timm


class QueryEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=128, backbone="resnet50"):
        super(QueryEncoder, self).__init__()
        self.dim = out_dim
        self.resnet = resnet50(pretrained=True) if backbone=="resnet50" else resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        fc_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                nn.BatchNorm1d(fc_features*1),
                nn.Linear(fc_features*1, self.dim),
            )

    def forward(self, input):
        embeddings = self.resnet(input)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class RenderingEncoder(nn.Module):
    def __init__(self, in_dim=1, out_dim=128):
        super(RenderingEncoder, self).__init__()
        self.dim = out_dim
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        fc_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                nn.BatchNorm1d(fc_features*1),
                nn.Linear(fc_features*1, self.dim),
            )

    def forward(self, inputs):
        inputs = inputs.float()
        embeddings = self.resnet(inputs)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ViTEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super(ViTEncoder, self).__init__()
        self.dim = out_dim
        self.vit = timm.create_model('vit_small_patch32_224_in21k', pretrained=True)
        self.vit.head = None
        self.proj = nn.Conv2d(in_dim, 3, kernel_size=1, stride=1, padding=0, bias=False)
        embed_dim = self.vit.embed_dim
        self.fc = nn.Sequential(
                # nn.BatchNorm1d(embed_dim),
                nn.LayerNorm(embed_dim, eps=1e-6),
                nn.Linear(embed_dim, self.dim),
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.vit.forward_features(x)
        # out = self.fc(x[:, 1:].mean(dim=1))
        out = self.fc(x[:, 0])
        out = F.normalize(out, p=2, dim=1)
        return out