import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentMLPGRU(nn.Module):
    """Standard (vanilla) QMIX agent net: an MLP over a fixed-K nearest-neighbor feature
    vector -> GRU -> Q-values. No CNN spatial grid, and the remote/connected side-channel
    is IGNORED (pure-local observation baseline). Same forward signature as AgentCNNGRU so
    it is a drop-in in build_agent / DRLCAVAgent.

    Expects `local_grid` to carry the flattened kNN features shaped (K*F, 1, 1) (see
    ObservationBuilder.build_knn_obs), i.e. local_channels == K*F and L==W==1.
    """

    def __init__(
        self,
        local_channels: int,
        ego_dim: int,
        remote_dim: int,
        n_actions: int = 3,
        cnn_out_dim: int = 128,
        fusion_dim: int = 128,
        gru_hidden_dim: int = 128,
    ):
        super().__init__()
        # MLP encoder over the flat neighbor features (local_channels = K*F, L=W=1).
        self.enc = nn.Sequential(
            nn.Linear(local_channels, cnn_out_dim),
            nn.ReLU(),
            nn.Linear(cnn_out_dim, cnn_out_dim),
            nn.ReLU(),
        )
        # By default pure-local: fuse encoder features with ego only (remote ignored).
        # MLP_REMOTE=1 additionally fuses the remote (V2V) channel -> tests whether giving
        # the standard-QMIX baseline remote CAV information changes its behavior (2x2 ablation).
        self._use_remote = os.environ.get("MLP_REMOTE", "0") == "1"
        fuse_in = cnn_out_dim + ego_dim + (remote_dim if self._use_remote else 0)
        self.fc_fusion = nn.Linear(fuse_in, fusion_dim)
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        self.q_head = nn.Linear(gru_hidden_dim, n_actions)

        self.gru_hidden_dim = gru_hidden_dim
        self.n_actions = n_actions

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)

    def forward(self, local_grid, ego_vec, remote_vec, hidden=None):
        """local_grid: [B, T, C, L, W] (C*L*W = K*F); remote_vec accepted but IGNORED."""
        B, T, C, L, W = local_grid.shape
        x = local_grid.reshape(B * T, C * L * W)   # flatten neighbor features
        feat = self.enc(x).view(B, T, -1)          # [B, T, cnn_out_dim]

        if self._use_remote:
            fused = torch.cat([feat, ego_vec, remote_vec], dim=-1)
        else:
            fused = torch.cat([feat, ego_vec], dim=-1)  # pure-local: no remote
        fused = F.relu(self.fc_fusion(fused))       # [B, T, fusion_dim]

        if hidden is None:
            hidden = self.init_hidden(B, fused.device)

        gru_out, hidden = self.gru(fused, hidden)   # [B, T, H]
        q_values = self.q_head(gru_out)             # [B, T, A]
        return q_values, hidden


class AgentCNNGRU(nn.Module):
    """
    Shared per-agent network:
      local_grid -> CNN
      [CNN feat + ego + remote] -> FC -> GRU -> Q-values
    """

    def __init__(
        self,
        local_channels: int,
        ego_dim: int,
        remote_dim: int,
        n_actions: int = 3,
        cnn_out_dim: int = 128,
        fusion_dim: int = 128,
        gru_hidden_dim: int = 128,
    ):
        super().__init__()

        # CNN over [C, L, W]
        self.cnn = nn.Sequential(
            nn.Conv2d(local_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, cnn_out_dim),
            nn.ReLU(),
        )

        self.fc_fusion = nn.Linear(cnn_out_dim + ego_dim + remote_dim, fusion_dim)
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )
        self.q_head = nn.Linear(gru_hidden_dim, n_actions)

        self.gru_hidden_dim = gru_hidden_dim
        self.n_actions = n_actions

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)

    def forward(self, local_grid, ego_vec, remote_vec, hidden=None):
        """
        local_grid: [B, T, C, L, W]
        ego_vec:    [B, T, D_ego]
        remote_vec: [B, T, D_remote]
        hidden:     [1, B, H] or None

        returns:
          q_values: [B, T, A]
          hidden:   [1, B, H]
        """
        B, T, C, L, W = local_grid.shape

        x = local_grid.view(B * T, C, L, W)
        cnn_feat = self.cnn(x)                     # [B*T, cnn_out_dim]
        cnn_feat = cnn_feat.view(B, T, -1)        # [B, T, cnn_out_dim]

        fused = torch.cat([cnn_feat, ego_vec, remote_vec], dim=-1)
        fused = F.relu(self.fc_fusion(fused))     # [B, T, fusion_dim]

        if hidden is None:
            hidden = self.init_hidden(B, fused.device)

        gru_out, hidden = self.gru(fused, hidden) # [B, T, H]
        q_values = self.q_head(gru_out)           # [B, T, A]

        return q_values, hidden


class QMixer(nn.Module):
    """
    Standard QMIX mixer with hypernetworks.
    Mixes per-agent chosen Q values into Q_total.
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: [B, T, N]
        states:   [B, T, S]

        returns:
          q_total: [B, T, 1]
        """
        B, T, N = agent_qs.shape
        assert N == self.n_agents

        agent_qs = agent_qs.view(B * T, 1, N)             # [B*T, 1, N]
        states = states.view(B * T, self.state_dim)       # [B*T, S]

        w1 = torch.abs(self.hyper_w1(states))             # monotonicity
        w1 = w1.view(B * T, self.n_agents, self.embed_dim)

        b1 = self.hyper_b1(states).view(B * T, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)      # [B*T, 1, E]

        w2 = torch.abs(self.hyper_w2(states)).view(B * T, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(B * T, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2              # [B*T, 1, 1]
        q_total = q_total.view(B, T, 1)

        return q_total


class MultiAgentQMIX(nn.Module):
    """
    Wrapper:
      - shared per-agent network
      - QMIX mixer
    """

    def __init__(
        self,
        n_agents: int,
        local_channels: int,
        ego_dim: int,
        remote_dim: int,
        state_dim: int,
        n_actions: int = 3,
        cnn_out_dim: int = 128,
        fusion_dim: int = 128,
        gru_hidden_dim: int = 128,
        mixer_embed_dim: int = 64,
    ):
        super().__init__()

        self.agent_net = AgentCNNGRU(
            local_channels=local_channels,
            ego_dim=ego_dim,
            remote_dim=remote_dim,
            n_actions=n_actions,
            cnn_out_dim=cnn_out_dim,
            fusion_dim=fusion_dim,
            gru_hidden_dim=gru_hidden_dim,
        )

        self.mixer = QMixer(
            n_agents=n_agents,
            state_dim=state_dim,
            embed_dim=mixer_embed_dim,
        )

        self.n_agents = n_agents
        self.n_actions = n_actions

    def forward_agent(self, local_grid, ego_vec, remote_vec, hidden=None):
        return self.agent_net(local_grid, ego_vec, remote_vec, hidden)

    def forward_mixer(self, agent_qs, states):
        return self.mixer(agent_qs, states)

class AgentGATGRU(nn.Module):
    """Drop-in alternative to AgentCNNGRU with an IDENTICAL input/output signature.
    The convolutional encoder over the local grid is replaced by a self-attention
    (graph-attention/transformer) encoder over the grid cells: each of the L*W cells is a
    node carrying its C channel features plus a normalized (row, col) position; a self-attention
    block lets cells attend to one another, and a learned-query attention pool summarizes them to
    a fixed vector. The input (same ego-centric grid) is held identical to the CNN, so a CNN-vs-GAT
    comparison isolates the encoder architecture (attention vs convolution)."""

    def __init__(self, local_channels, ego_dim, remote_dim, n_actions=3,
                 cnn_out_dim=128, fusion_dim=128, gru_hidden_dim=128,
                 embed_dim=64, n_heads=4):
        super().__init__()
        self.node_in = nn.Linear(local_channels + 2, embed_dim)  # +2 = normalized (row, col)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.0)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                nn.Linear(embed_dim, embed_dim))
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.0)
        self.enc_out = nn.Linear(embed_dim, cnn_out_dim)

        self.fc_fusion = nn.Linear(cnn_out_dim + ego_dim + remote_dim, fusion_dim)
        self.gru = nn.GRU(input_size=fusion_dim, hidden_size=gru_hidden_dim, batch_first=True)
        self.q_head = nn.Linear(gru_hidden_dim, n_actions)
        self.gru_hidden_dim = gru_hidden_dim
        self.n_actions = n_actions

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)

    def forward(self, local_grid, ego_vec, remote_vec, hidden=None):
        # local_grid: [B, T, C, L, W]
        B, T, C, L, W = local_grid.shape
        x = local_grid.permute(0, 1, 3, 4, 2).reshape(B * T, L * W, C)   # [BT, N, C]
        rr = torch.linspace(0, 1, L, device=x.device).view(L, 1).expand(L, W).reshape(-1)
        cc = torch.linspace(0, 1, W, device=x.device).view(1, W).expand(L, W).reshape(-1)
        pos = torch.stack([rr, cc], dim=-1).unsqueeze(0).expand(B * T, L * W, 2)  # [BT, N, 2]
        h = self.node_in(torch.cat([x, pos], dim=-1))                    # [BT, N, E]
        a, _ = self.self_attn(h, h, h)                                   # cells attend to cells
        h = self.ln1(h + a)
        h = self.ln2(h + self.ff(h))
        q = self.query.expand(B * T, 1, h.shape[-1])
        pooled, _ = self.pool_attn(q, h, h)                             # [BT, 1, E]
        feat = F.relu(self.enc_out(pooled.squeeze(1)))                  # [BT, cnn_out_dim]
        feat = feat.view(B, T, -1)

        fused = torch.cat([feat, ego_vec, remote_vec], dim=-1)
        fused = F.relu(self.fc_fusion(fused))
        if hidden is None:
            hidden = self.init_hidden(B, fused.device)
        gru_out, hidden = self.gru(fused, hidden)
        return self.q_head(gru_out), hidden


class AgentCNNLNGRU(nn.Module):
    """Stability-test variant: identical to AgentCNNGRU but with LayerNorm on the CNN feature and on the
    fused feature. Isolates the transformer block's NORMALIZATION from its ATTENTION -- if this stabilizes
    CNN-QMIX (which collapses late without it), the GAT's stability benefit is attributable to LayerNorm,
    not to graph-attention. Drop-in: same (local_grid, ego, remote, hidden) -> (q, hidden) signature."""

    def __init__(self, local_channels, ego_dim, remote_dim, n_actions=3,
                 cnn_out_dim=128, fusion_dim=128, gru_hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(local_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, cnn_out_dim), nn.ReLU(),
        )
        self.ln_cnn = nn.LayerNorm(cnn_out_dim)
        self.fc_fusion = nn.Linear(cnn_out_dim + ego_dim + remote_dim, fusion_dim)
        self.ln_fusion = nn.LayerNorm(fusion_dim)
        self.gru = nn.GRU(fusion_dim, gru_hidden_dim, batch_first=True)
        self.q_head = nn.Linear(gru_hidden_dim, n_actions)
        self.gru_hidden_dim = gru_hidden_dim
        self.n_actions = n_actions

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)

    def forward(self, local_grid, ego_vec, remote_vec, hidden=None):
        B, T, C, L, W = local_grid.shape
        x = self.cnn(local_grid.reshape(B * T, C, L, W)).view(B, T, -1)
        x = self.ln_cnn(x)
        fused = F.relu(self.fc_fusion(torch.cat([x, ego_vec, remote_vec], dim=-1)))
        fused = self.ln_fusion(fused)
        if hidden is None:
            hidden = self.init_hidden(B, fused.device)
        gru_out, hidden = self.gru(fused, hidden)
        return self.q_head(gru_out), hidden
