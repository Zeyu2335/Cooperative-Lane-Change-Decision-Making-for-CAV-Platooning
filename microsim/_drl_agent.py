import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class EpisodeReplayBuffer:
    """
    Stores full episodes and samples fixed-length sequences.

    Each transition:
        (
            obs_batch,         # list length MAX_AGENTS
            agent_mask,        # [MAX_AGENTS]
            state,             # [state_dim]
            actions,           # [MAX_AGENTS]
            reward,            # scalar
            next_obs_batch,    # list length MAX_AGENTS
            next_agent_mask,   # [MAX_AGENTS]
            next_state,        # [state_dim]
            done               # scalar
        )
    """

    def __init__(self, capacity: int, seq_len: int):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push_episode(self, episode):
        if len(episode) > 0:
            self.buffer.append(episode)

    def sample(self, batch_size: int):
        # Return WHOLE episodes (variable length). The recurrent learner unrolls
        # the GRU from hidden=0 over each full episode, matching the act-time
        # hidden trajectory. Padding/masking to a common length is done in train_step.
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)


class DRLCAVAgent:
    """
    Shared RL brain for variable DRLCAV count using:
    - shared per-agent CNN+GRU
    - fixed MAX_AGENTS
    - padding + agent_mask
    """

    def __init__(
        self,
        agent_net,
        mixer,
        target_agent_net,
        target_mixer,
        n_agents: int,   # this is MAX_AGENTS
        device: str = "cpu",
        gamma: float = 0.99,
        lr: float = 5e-4,
        buffer_capacity: int = 5000,
        batch_size: int = 16,
        grad_clip: float = 10.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        target_update_interval: int = 500,
        seq_len: int = 10,
        tau: float = 0.005,
        mixing_mode: str = "qmix",   # "qmix" | "vdn" | "iql"
    ):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.target_update_interval = target_update_interval
        self.seq_len = seq_len
        self.tau = tau   # Polyak soft-update rate for target networks
        self.mixing_mode = mixing_mode   # credit assignment: qmix (mixer) | vdn (sum) | iql (per-agent)

        self.agent_net = agent_net.to(self.device)
        self.mixer = mixer.to(self.device)
        self.target_agent_net = target_agent_net.to(self.device)
        self.target_mixer = target_mixer.to(self.device)

        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=lr
        )

        self.replay_buffer = EpisodeReplayBuffer(buffer_capacity, seq_len=seq_len)
        self.current_episode = []

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learn_step_counter = 0

    # =====================================================
    # Episode storage
    # =====================================================
    def start_episode(self):
        self.current_episode = []

    def store_transition(
        self,
        obs_batch,
        agent_mask,
        state,
        actions,
        reward,
        next_obs_batch,
        next_agent_mask,
        next_state,
        done
    ):
        self.current_episode.append((
            obs_batch,
            agent_mask,
            state,
            actions,
            reward,
            next_obs_batch,
            next_agent_mask,
            next_state,
            done
        ))

    def end_episode(self):
        self.replay_buffer.push_episode(self.current_episode)
        self.current_episode = []
        # decay exploration once per EPISODE (not per train-step)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load_agent_only(self, path, epsilon=0.3):
        ckpt = torch.load(path, map_location=self.device)

        self.agent_net.load_state_dict(ckpt["agent_net"])
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())

        # Do NOT load mixer because state_dim changed
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Reset optimizer because mixer is new
        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=5e-6
        )

        # Reset exploration
        self.epsilon = epsilon

        # Reset replay
        self.replay_buffer.buffer.clear()
        self.current_episode = []

        print("Loaded agent_net only. Mixer reinitialized due to changed state_dim.")

    # =====================================================
    # Online action selection from 1-second sequence
    # =====================================================
    def select_action_from_seq_obs(
        self,
        obs_seq,
        hidden: Optional[torch.Tensor] = None,
        explore: bool = True,
    ):
        """
        obs_seq:
            {
                "local_grid": [T, C, L, W],
                "ego": [T, D_ego],
                "remote": [T, D_remote],
                "action_mask": [A]
            }
        """
        local_grid = torch.tensor(
            obs_seq["local_grid"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1,T,C,L,W]

        ego_vec = torch.tensor(
            obs_seq["ego"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1,T,D]

        remote_vec = torch.tensor(
            obs_seq["remote"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1,T,D]

        action_mask = torch.tensor(
            obs_seq["action_mask"], dtype=torch.bool, device=self.device
        )  # [A]

        with torch.no_grad():
            q_values, hidden = self.agent_net(local_grid, ego_vec, remote_vec, hidden)
            q_values = q_values[0, -1]  # [A]
            q_values = q_values.masked_fill(~action_mask, -1e9)

            if explore and np.random.rand() < self.epsilon:
                valid_actions = torch.where(action_mask)[0]
                if len(valid_actions) == 0:
                    action = 0
                else:
                    idx = torch.randint(len(valid_actions), (1,), device=self.device)
                    action = valid_actions[idx].item()
            else:
                action = torch.argmax(q_values).item()

        return int(action), hidden

    def select_action(
        self,
        obs,
        hidden: Optional[torch.Tensor] = None,
        explore: bool = True,
    ):
        """
        Single-step recurrent action selection.
        obs: {"local_grid":[C,L,W], "ego":[D], "remote":[D], "action_mask":[A]}
        The GRU advances ONE step from `hidden`; returns (action, new_hidden).
        """
        local_grid = torch.tensor(
            obs["local_grid"], dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)   # [1,1,C,L,W]
        ego_vec = torch.tensor(
            obs["ego"], dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)   # [1,1,D]
        remote_vec = torch.tensor(
            obs["remote"], dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)   # [1,1,D]
        action_mask = torch.tensor(
            obs["action_mask"], dtype=torch.bool, device=self.device
        )  # [A]

        with torch.no_grad():
            q_values, hidden = self.agent_net(local_grid, ego_vec, remote_vec, hidden)
            q_values = q_values[0, -1]                       # [A]
            q_values = q_values.masked_fill(~action_mask, -1e9)

            if explore and np.random.rand() < self.epsilon:
                valid_actions = torch.where(action_mask)[0]
                if len(valid_actions) == 0:
                    action = 0
                else:
                    idx = torch.randint(len(valid_actions), (1,), device=self.device)
                    action = valid_actions[idx].item()
            else:
                action = torch.argmax(q_values).item()

        return int(action), hidden

    # =====================================================
    # Target update
    # =====================================================
    def update_targets(self):
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_update_targets(self):
        # Polyak averaging: target <- tau*online + (1-tau)*target, every step.
        # Smoother than a hard copy every N steps -> avoids target-jump loss spikes.
        with torch.no_grad():
            for tp, p in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # =====================================================
    # Sequence QMIX training with agent padding + mask
    # =====================================================
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        B = len(batch)
        N = self.n_agents
        T = max(len(ep) for ep in batch)   # full-episode unroll; pad to batch max

        # shapes from the first transition
        t0 = batch[0][0]
        state_dim = np.asarray(t0[2]).shape[0]
        sample_obs = t0[0][0]
        C, L, W = np.asarray(sample_obs["local_grid"]).shape
        ego_dim = np.asarray(sample_obs["ego"]).shape[0]
        remote_dim = np.asarray(sample_obs["remote"]).shape[0]
        A = np.asarray(sample_obs["action_mask"]).shape[0]

        # zero-padded buffers; time_mask marks valid (real) timesteps
        time_mask = np.zeros((B, T), dtype=np.float32)
        reward_seq = np.zeros((B, T), dtype=np.float32)
        done_seq = np.zeros((B, T), dtype=np.float32)
        state_seq = np.zeros((B, T, state_dim), dtype=np.float32)
        next_state_seq = np.zeros((B, T, state_dim), dtype=np.float32)
        agent_mask_seq = np.zeros((B, T, N), dtype=np.float32)
        next_agent_mask_seq = np.zeros((B, T, N), dtype=np.float32)
        actions_seq = np.zeros((B, T, N), dtype=np.int64)
        local = np.zeros((B, T, N, C, L, W), dtype=np.float32)
        ego = np.zeros((B, T, N, ego_dim), dtype=np.float32)
        remote = np.zeros((B, T, N, remote_dim), dtype=np.float32)
        next_local = np.zeros((B, T, N, C, L, W), dtype=np.float32)
        next_ego = np.zeros((B, T, N, ego_dim), dtype=np.float32)
        next_remote = np.zeros((B, T, N, remote_dim), dtype=np.float32)
        next_amask = np.zeros((B, T, N, A), dtype=np.float32)

        for b, ep in enumerate(batch):
            for t, trans in enumerate(ep):
                (obs_batch, agent_mask, state, actions, reward,
                 next_obs_batch, next_agent_mask, next_state, done) = trans
                time_mask[b, t] = 1.0
                reward_seq[b, t] = reward
                done_seq[b, t] = done
                state_seq[b, t] = state
                next_state_seq[b, t] = next_state
                agent_mask_seq[b, t] = agent_mask
                next_agent_mask_seq[b, t] = next_agent_mask
                actions_seq[b, t] = actions
                for n in range(N):
                    local[b, t, n] = obs_batch[n]["local_grid"]
                    ego[b, t, n] = obs_batch[n]["ego"]
                    remote[b, t, n] = obs_batch[n]["remote"]
                    next_local[b, t, n] = next_obs_batch[n]["local_grid"]
                    next_ego[b, t, n] = next_obs_batch[n]["ego"]
                    next_remote[b, t, n] = next_obs_batch[n]["remote"]
                    next_amask[b, t, n] = next_obs_batch[n]["action_mask"]

        dev = self.device

        def to(a, dt=torch.float32):
            return torch.tensor(a, dtype=dt, device=dev)

        time_mask_t = to(time_mask)
        reward_seq = to(reward_seq)
        done_seq = to(done_seq)
        state_seq = to(state_seq)
        next_state_seq = to(next_state_seq)
        agent_mask_seq = to(agent_mask_seq)
        next_agent_mask_seq = to(next_agent_mask_seq)
        actions_t = to(actions_seq, torch.long)
        local_t = to(local); ego_t = to(ego); remote_t = to(remote)
        next_local_t = to(next_local); next_ego_t = to(next_ego); next_remote_t = to(next_remote)
        next_amask_t = to(next_amask).bool()

        chosen_q_list = []
        target_q_list = []

        for n in range(N):
            # current Q (GRU unrolls over the FULL episode from hidden=0)
            q_values, _ = self.agent_net(local_t[:, :, n], ego_t[:, :, n], remote_t[:, :, n])  # [B,T,A]
            chosen_q = q_values.gather(2, actions_t[:, :, n].unsqueeze(-1)).squeeze(-1)         # [B,T]
            chosen_q_list.append(chosen_q)

            # Double-Q target
            with torch.no_grad():
                online_next_q, _ = self.agent_net(
                    next_local_t[:, :, n], next_ego_t[:, :, n], next_remote_t[:, :, n]
                )
                online_next_q = online_next_q.masked_fill(~next_amask_t[:, :, n], -1e9)
                next_actions = online_next_q.argmax(dim=2, keepdim=True)

                target_q_values, _ = self.target_agent_net(
                    next_local_t[:, :, n], next_ego_t[:, :, n], next_remote_t[:, :, n]
                )
                target_q_values = target_q_values.masked_fill(~next_amask_t[:, :, n], -1e9)
                max_target_q = target_q_values.gather(2, next_actions).squeeze(-1)             # [B,T]
                target_q_list.append(max_target_q)

        chosen_qs = torch.stack(chosen_q_list, dim=2)   # [B,T,N]
        target_qs = torch.stack(target_q_list, dim=2)   # [B,T,N]

        # mask padded agents so they contribute zero
        chosen_qs = chosen_qs * agent_mask_seq
        target_qs = target_qs * next_agent_mask_seq

        # ---- credit assignment: QMIX (mixer) | VDN (sum) | IQL (per-agent) ----
        # Ablation baselines. chosen_qs/target_qs are already agent-masked above, so
        # padded agents contribute 0 to the VDN sum automatically. The mixer is only
        # used in qmix mode; in vdn/iql it gets no gradient and stays frozen at init.
        if self.mixing_mode == "iql":
            # Independent Q-learning: per-agent TD on the shared team reward, no joint value.
            with torch.no_grad():
                y = reward_seq.unsqueeze(-1) + self.gamma * (1.0 - done_seq).unsqueeze(-1) * target_qs  # [B,T,N]
            per_step = F.smooth_l1_loss(chosen_qs, y, reduction="none")          # [B,T,N]
            w = time_mask_t.unsqueeze(-1) * agent_mask_seq                       # [B,T,N]
            loss = (per_step * w).sum() / w.sum().clamp(min=1.0)
        else:
            if self.mixing_mode == "vdn":
                q_total = chosen_qs.sum(dim=2)                                   # [B,T]
                with torch.no_grad():
                    target_q_total = target_qs.sum(dim=2)                        # [B,T]
            else:  # qmix
                q_total = self.mixer(chosen_qs, state_seq).squeeze(-1)          # [B,T]
                with torch.no_grad():
                    target_q_total = self.target_mixer(target_qs, next_state_seq).squeeze(-1)  # [B,T]
            with torch.no_grad():
                y = reward_seq + self.gamma * (1.0 - done_seq) * target_q_total

            # masked TD loss over valid (non-padded) timesteps only
            per_step = F.smooth_l1_loss(q_total, y, reduction="none")   # [B,T]
            loss = (per_step * time_mask_t).sum() / time_mask_t.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            self.grad_clip
        )
        self.optimizer.step()

        self.learn_step_counter += 1
        self.soft_update_targets()

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            "agent_net": self.agent_net.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent_net": self.target_agent_net.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
            "seq_len": self.seq_len,
            "n_agents": self.n_agents,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(ckpt["agent_net"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_agent_net.load_state_dict(ckpt["target_agent_net"])
        self.target_mixer.load_state_dict(ckpt["target_mixer"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.learn_step_counter = ckpt.get("learn_step_counter", 0)