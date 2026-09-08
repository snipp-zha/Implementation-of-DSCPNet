import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.act(self.block(x) + self.shortcut(x))
        return self.pool(x)


class ResNet12(nn.Module):
    def __init__(self, channels=(64, 160, 320, 640)):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.append(BasicBlock(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x):
        return self.encoder(x)


class DSFA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_select = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.channel_fc = nn.Linear(channels * 2, channels * 2)
        self.recalibration = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        f1 = self.local_branch(z)
        f2 = self.context_branch(z)
        u = torch.cat([f1, f2], dim=1)

        avg_map = torch.mean(u, dim=1, keepdim=True)
        max_map = torch.max(u, dim=1, keepdim=True)[0]
        spatial = self.spatial_select(torch.cat([avg_map, max_map], dim=1))
        s1, s2 = spatial[:, 0:1], spatial[:, 1:2]

        channel_context = F.adaptive_avg_pool2d(u, 1).flatten(1)
        channel = self.channel_fc(channel_context).view(z.size(0), 2, z.size(1), 1, 1)
        channel = F.softmax(channel, dim=1)
        c1, c2 = channel[:, 0], channel[:, 1]

        fused = s1 * c1 * f1 + s2 * c2 * f2
        return z + self.gamma * self.recalibration(fused)


class CFSI(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fore_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.scene_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, a):
        b, c, h, w = a.shape
        fe = self.fore_proj(a)
        fa = self.scene_proj(a)

        fe_m = fe.flatten(2)
        fa_m = fa.flatten(2)
        interaction_ea = torch.tanh(torch.bmm(fe_m.transpose(1, 2), fa_m) / (c ** 0.5))
        fa_refined = torch.bmm(fa_m, interaction_ea.transpose(1, 2)).view(b, c, h, w)

        interaction_ae = torch.tanh(torch.bmm(fa_m.transpose(1, 2), fe_m) / (c ** 0.5))
        fe_refined = torch.bmm(fe_m, interaction_ae.transpose(1, 2)).view(b, c, h, w)

        return a + self.out_proj(torch.cat([fe_refined, fa_refined], dim=1))


class CPFR(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.query_align = nn.Conv2d(channels, channels, 1, bias=False)
        self.proto_fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
        self.query_fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
        self.proto_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.query_out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, support_feat, support_labels, query_feat, way):
        prototypes = []
        for cls in range(way):
            prototypes.append(support_feat[support_labels == cls].mean(dim=0))
        proto = torch.stack(prototypes, dim=0)
        query = self.query_align(query_feat)

        p_vec = F.adaptive_avg_pool2d(proto, 1).flatten(1)
        q_vec = F.adaptive_avg_pool2d(query, 1).flatten(1)
        p_gate = torch.sigmoid(self.proto_fc(p_vec)).view(way, -1, 1, 1)
        q_gate = torch.sigmoid(self.query_fc(q_vec)).view(query.size(0), -1, 1, 1)

        proto = self.proto_out(proto * p_gate) + proto
        query = self.query_out(query * q_gate) + query
        proto_emb = F.adaptive_avg_pool2d(proto, 1).flatten(1)
        query_emb = F.adaptive_avg_pool2d(query, 1).flatten(1)
        return F.normalize(proto_emb, dim=1), F.normalize(query_emb, dim=1)


class DSCPNet(nn.Module):
    def __init__(self, num_base_classes=5, temperature=10.0):
        super().__init__()
        self.backbone = ResNet12()
        channels = self.backbone.out_channels
        self.dsfa = DSFA(channels)
        self.cfsi = CFSI(channels)
        self.cpfr = CPFR(channels)
        self.classifier = nn.Linear(channels, num_base_classes)
        self.temperature = temperature

    def extract_feature_map(self, x):
        z = self.backbone(x)
        a = self.dsfa(z)
        return self.cfsi(a)

    def forward_classification(self, x):
        feat = self.extract_feature_map(x)
        emb = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.classifier(emb)

    def forward_episode(self, support_x, support_y, query_x, way):
        support_feat = self.extract_feature_map(support_x)
        query_feat = self.extract_feature_map(query_x)
        proto_emb, query_emb = self.cpfr(support_feat, support_y, query_feat, way)
        logits = self.temperature * torch.mm(query_emb, proto_emb.t())
        return logits
