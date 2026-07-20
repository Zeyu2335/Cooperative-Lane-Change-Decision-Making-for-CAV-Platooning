import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random

from _microsim import simulation
from _constants import VEHTYPE, RANDSEED
from _qmix_network import AgentCNNGRU, AgentGATGRU, AgentCNNLNGRU, AgentMLPGRU, QMixer
from _drl_agent import DRLCAVAgent


MAX_AGENTS = 12


# ---------------------------------------------------------------------------------------
# Heterogeneous human drivers (car-following IDM + lane-change MOBIL, jointly per driver).
# Each human is independently assigned one of three literature-based profiles
# (Treiber et al. 2000 IDM; Kesting et al. 2007 MOBIL), so a single episode contains a
# realistic MIX of driving styles rather than one homogeneous human model.
#   IDM  : a [m/s^2] accel, b [m/s^2] comfortable decel, T [s] headway, s0 [m] jam gap
#   MOBIL: pol politeness, athr lane-change threshold [m/s^2], bsafe max safe decel [m/s^2]
# maps to vehicle attrs ad, bd, T, d (IDM) and pol, athr, bsafe (MOBIL).
DRIVER_PROFILES = {
    "conservative": dict(a=0.8, b=1.5, T=2.0, s0=2.5, pol=1.0, athr=2.0, bsafe=2.0),
    "normal":       dict(a=1.0, b=2.0, T=1.5, s0=2.0, pol=0.8, athr=1.0, bsafe=3.0),
    "aggressive":   dict(a=2.0, b=3.0, T=1.0, s0=1.5, pol=0.3, athr=0.3, bsafe=5.0),
}
_PROFILE_NAMES = ("conservative", "normal", "aggressive")


# Env-gated aggressive human model: draw each human IDM param from an independent normal
# N(mean, (AGG_CV*mean)^2). Defaults: a=3, b=3, s0=2, T=1, 20% variation. MOBIL params
# use the aggressive profile. Enable with HET_MODE=normal (default "discrete" = 3 profiles).
_HET_MODE = os.environ.get("HET_MODE", "discrete")
_AGG_CV   = float(os.environ.get("AGG_CV", "0.20"))
_AGG_MEAN = dict(a=float(os.environ.get("AGG_A", "3.0")),
                 b=float(os.environ.get("AGG_B", "3.0")),
                 s0=float(os.environ.get("AGG_S0", "2.0")),
                 T=float(os.environ.get("AGG_T", "1.0")))


def apply_heterogeneous_drivers(sim, rng):
    """Assign each human (MOBIL type) vehicle an independent driver profile (both IDM
    car-following and MOBIL lane-change params). `rng` is a random.Random-like source
    (pass the seeded module for reproducible validation). Returns the profile counts."""
    counts = {k: 0 for k in _PROFILE_NAMES}
    counts["normal_agg"] = 0
    for v in sim.svs:
        if v.type != VEHTYPE.MOBIL:
            continue
        if _HET_MODE == "normal":                # per-driver aggressiveness ~ N(mean, (CV*mean)^2)
            def _n(key):
                m = _AGG_MEAN[key]
                return max(0.1, rng.gauss(m, _AGG_CV * m))
            v.ad, v.bd, v.d, v.T = _n("a"), _n("b"), _n("s0"), _n("T")
            ap = DRIVER_PROFILES["aggressive"]   # aggressive MOBIL lane-change params
            v.pol, v.athr, v.bsafe = ap["pol"], ap["athr"], ap["bsafe"]
            counts["normal_agg"] += 1
        else:                                    # default: 3 discrete profiles
            prof = _PROFILE_NAMES[rng.randrange(3)]
            p = DRIVER_PROFILES[prof]
            v.ad, v.bd, v.T, v.d = p["a"], p["b"], p["T"], p["s0"]
            v.pol, v.athr, v.bsafe = p["pol"], p["athr"], p["bsafe"]
            counts[prof] += 1
    return counts


def validate_heterogeneous(rl_agent, XRTK, YRTK, decision_interval_steps,
                           n_episodes=20, seed_base=100000):
    """Held-out validation on a FIXED set of heterogeneous-driver scenarios: the same
    `n_episodes` seeds every call, so the logged curve is directly comparable across
    checkpoints. Greedy (no exploration). Global RNG state and the agent's epsilon are
    saved/restored so ongoing training stays bit-for-bit reproducible."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    saved_eps = rl_agent.epsilon
    rl_agent.epsilon = 0.0
    P, L, S, C = [], [], [], []
    for k in range(n_episodes):
        vseed = seed_base + k
        random.seed(vseed)
        np.random.seed(vseed)
        args = build_args()
        sim = simulation(args, XRTK, YRTK, CAVtype=3,
                         filename="/tmp/val_ep.csv", rl_agent=rl_agent, explore=False)
        sim.explore = False
        apply_heterogeneous_drivers(sim, random)
        speed_sum, niv, done = 0.0, 0, False
        while not done:
            _, done = run_one_decision_interval(sim, decision_interval_steps=decision_interval_steps)
            vs = [sv.v for sv in sim.svs]
            if vs:
                speed_sum += float(np.mean(vs)); niv += 1
            if getattr(sim, "done", False):
                done = True
        lc, plat, col = compute_episode_metrics(sim)
        P.append(plat); L.append(lc); S.append(speed_sum / max(niv, 1)); C.append(col)
    rl_agent.epsilon = saved_eps
    random.setstate(py_state)
    np.random.set_state(np_state)
    return dict(platoon=float(np.mean(P)), lc=float(np.mean(L)),
                speed=float(np.mean(S)), collision=float(np.mean(C)))


def build_args():
    class Args:
        HZ = 10
        numSvs = 24
        numEgos = 0
        route = 5
        geometry = 0   # 3-lane road only (ego-centric obs still allows 2/4-lane transfer TESTING)
        traffic = 2
        useActive = 1
        useVerbose = 0
        useVisual = 0
        recordVisual = 0
        followId = 1
        useRealtime = 0
        cavRate = random.choice([0.17, 0.25, 0.33])
        logging = False
    return Args()



def evaluate_policy(rl_agent, args, XRTK, YRTK, decision_interval_steps):
    sim = simulation(
        args,
        XRTK,
        YRTK,
        CAVtype=3,
        filename="eval.csv",
        rl_agent=rl_agent,
        explore=False
    )

    # make sure evaluation does not use exploration
    sim.explore = False

    total_reward = 0.0
    total_decision_intervals = 0
    platoon_rate_sum = 0.0

    collision_count = 0
    had_collision = False

    done = False

    while not done:
        interval_reward, done = run_one_decision_interval(
            sim,
            decision_interval_steps=decision_interval_steps
        )

        total_reward += (interval_reward / decision_interval_steps)
        total_decision_intervals += 1

        # platoon/parallel rate at this decision interval
        platoon_rate_sum += sim.compute_platoon_reward()

        # collision
        if getattr(sim, "collison", 0) > 0:
            had_collision = True
            collision_count += getattr(sim, "collison", 0)

        # also stop if your simulator sets done=True
        if getattr(sim, "done", False):
            done = True

    avg_platoon_rate = platoon_rate_sum / max(total_decision_intervals, 1)

    # action_count: 0=right, 1=keep, 2=left
    right_lc = sim.action_count.get(0, 0)
    keep = sim.action_count.get(1, 0)
    left_lc = sim.action_count.get(2, 0)

    lane_change_count = right_lc + left_lc

    return {
        "reward": total_reward,
        "platoon_rate": avg_platoon_rate,
        "lane_change_count": lane_change_count,
        "right_lane_changes": right_lc,
        "left_lane_changes": left_lc,
        "keep_count": keep,
        "collision_count": collision_count,
        "had_collision": had_collision,
    }


def make_dummy_obs(example_obs):
    return {
        "ego": np.zeros_like(example_obs["ego"], dtype=np.float32),
        "local_grid": np.zeros_like(example_obs["local_grid"], dtype=np.float32),
        "remote": np.zeros_like(example_obs["remote"], dtype=np.float32),
        "action_mask": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }


def collect_drlcav_obs(sim, max_agents=MAX_AGENTS):
    real_obs_batch = []

    for j in range(sim.numSvs):
        ego = sim.svs[j]
        if ego.type != VEHTYPE.DRLCAV:
            continue

        svs = [sim.svs[i] for i in range(sim.numSvs) if i != j]

        obs = sim.obs_builder.build_observation(
            ego=ego,
            svs=svs,
            tls=sim.tls,
            obs=sim.obs,
            divs=sim.divs
        )
        real_obs_batch.append(obs)

    n_real = len(real_obs_batch)
    if n_real == 0:
        raise ValueError("No DRLCAV found in this episode.")
    if n_real > max_agents:
        raise ValueError(f"n_real={n_real} exceeds MAX_AGENTS={max_agents}")

    example_obs = real_obs_batch[0]
    obs_batch = list(real_obs_batch)

    while len(obs_batch) < max_agents:
        obs_batch.append(make_dummy_obs(example_obs))

    agent_mask = np.array(
        [1.0] * n_real + [0.0] * (max_agents - n_real),
        dtype=np.float32
    )

    return obs_batch, agent_mask, n_real


def infer_dims(sim, max_agents=MAX_AGENTS):
    obs_batch, agent_mask, n_real = collect_drlcav_obs(sim, max_agents=max_agents)

    sample_obs = obs_batch[0]
    local_channels = sample_obs["local_grid"].shape[0]
    ego_dim = sample_obs["ego"].shape[0]
    remote_dim = sample_obs["remote"].shape[0]
    state_dim = sim.build_global_state().shape[0]
    n_agents = max_agents

    return local_channels, ego_dim, remote_dim, state_dim, n_agents


def build_agent(sim, device, mixing_mode="qmix"):
    local_channels, ego_dim, remote_dim, state_dim, n_agents = infer_dims(sim, max_agents=MAX_AGENTS)

    if os.environ.get("ALGO", "") == "ippo":
        from _ippo import AgentACGRU, IPPOAgent
        _ipnet = AgentACGRU(local_channels, ego_dim, remote_dim, n_actions=3,
                            cnn_out_dim=128, fusion_dim=128, gru_hidden_dim=128)
        return IPPOAgent(_ipnet, n_agents=n_agents, device=device)

    _enc = os.environ.get("ENCODER", "cnn")
    _EncCls = (AgentMLPGRU if _enc == "mlp" else
               AgentGATGRU if _enc == "gat" else
               AgentCNNLNGRU if _enc == "cnnln" else AgentCNNGRU)
    agent_net = _EncCls(
        local_channels=local_channels,
        ego_dim=ego_dim,
        remote_dim=remote_dim,
        n_actions=3,
        cnn_out_dim=128,
        fusion_dim=128,
        gru_hidden_dim=128,
    )

    mixer = QMixer(
        n_agents=n_agents,
        state_dim=state_dim,
        embed_dim=64,
    )

    target_agent_net = copy.deepcopy(agent_net)
    target_mixer = copy.deepcopy(mixer)

    rl_agent = DRLCAVAgent(
        agent_net=agent_net,
        mixer=mixer,
        target_agent_net=target_agent_net,
        target_mixer=target_mixer,
        n_agents=n_agents,
        device=device,
        gamma=0.99,
        lr=1e-4,                # lowered (was 3e-4) to smooth loss spikes
        buffer_capacity=2000,
        batch_size=8,           # full-episode BPTT (recurrent) -> smaller batch
        grad_clip=5.0,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,    # now applied per EPISODE -> ~0.05 over ~1000 eps
        target_update_interval=1000,
        seq_len=10,
        mixing_mode=mixing_mode,   # qmix (default) | vdn | iql ablation baselines
    )

    return rl_agent




def run_one_decision_interval(sim, decision_interval_steps=10,debug=False):
    """
    Hold current action policy over one 1-second interval.
    Accumulate reward over those 10 sim steps.
    Store only one transition for this whole interval.
    """
    total_reward = 0.0
    done = False

    for _ in range(decision_interval_steps):
        reward = sim.step(debug=debug)
        total_reward += reward

        if sim.t >= 80.0:
            done = True
            break

    return total_reward, done

def compute_episode_metrics(sim):
    drlcavs = [sv for sv in sim.svs if sv.type == VEHTYPE.DRLCAV]

    # lane-change number
    lane_change_num = sum(getattr(sv, "LC", 0) for sv in drlcavs)

    # platoon rate among DRLCAVs
    platoon_count = sum(getattr(sv, "in_platoon", 0.0) for sv in drlcavs)
    platoon_rate = platoon_count / max(len(drlcavs), 1)

    # collision rate
    collision_rate = 1.0 if getattr(sim, "collison", 0) > 0 else 0.0

    return lane_change_num, platoon_rate, collision_rate


def plot_training_curves(
    reward_history,
    loss_history,
    lane_change_history,
    platoon_rate_history,
    collision_rate_history,
    save_path="training_curves.png",
    ma_window=100
):


    def moving_avg(x, window):
        x = np.array(x, dtype=np.float32)
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    episodes = np.arange(len(reward_history))

    plt.figure(figsize=(10, 18))

    # -------------------------------------------------
    # 1 Reward
    # -------------------------------------------------
    plt.subplot(5, 1, 1)
    plt.plot(episodes, reward_history, alpha=0.35, label="Raw")

    if len(reward_history) >= ma_window:
        ma = moving_avg(reward_history, ma_window)
        plt.plot(
            np.arange(ma_window - 1, len(reward_history)),
            ma,
            linewidth=2.5,
            label=f"MA({ma_window})"
        )

    plt.title("1. Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # -------------------------------------------------
    # 2 Loss
    # -------------------------------------------------
    plt.subplot(5, 1, 2)
    if len(loss_history) > 0:
        plt.plot(loss_history, alpha=0.35, label="Raw")

        if len(loss_history) >= ma_window:
            ma = moving_avg(loss_history, ma_window)
            plt.plot(
                np.arange(ma_window - 1, len(loss_history)),
                ma,
                linewidth=2.5,
                label=f"MA({ma_window})"
            )

    plt.title("2. Loss Curve")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # -------------------------------------------------
    # 3 Lane Change
    # -------------------------------------------------
    plt.subplot(5, 1, 3)
    plt.plot(episodes, lane_change_history, alpha=0.35, label="Raw")

    if len(lane_change_history) >= ma_window:
        ma = moving_avg(lane_change_history, ma_window)
        plt.plot(
            np.arange(ma_window - 1, len(lane_change_history)),
            ma,
            linewidth=2.5,
            label=f"MA({ma_window})"
        )

    plt.title("3. Lane-Change Number per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Lane Changes")
    plt.legend()
    plt.grid(True)

    # -------------------------------------------------
    # 4 Platoon Rate
    # -------------------------------------------------
    plt.subplot(5, 1, 4)
    plt.plot(episodes, platoon_rate_history, alpha=0.35, label="Raw")

    if len(platoon_rate_history) >= ma_window:
        ma = moving_avg(platoon_rate_history, ma_window)
        plt.plot(
            np.arange(ma_window - 1, len(platoon_rate_history)),
            ma,
            linewidth=2.5,
            label=f"MA({ma_window})"
        )

    plt.title("4. Platoon Rate per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Platoon Rate")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    # -------------------------------------------------
    # 5 Collision Rate
    # -------------------------------------------------
    plt.subplot(5, 1, 5)
    plt.plot(episodes, collision_rate_history, alpha=0.35, label="Raw")

    if len(collision_rate_history) >= ma_window:
        ma = moving_avg(collision_rate_history, ma_window)
        plt.plot(
            np.arange(ma_window - 1, len(collision_rate_history)),
            ma,
            linewidth=2.5,
            label=f"MA({ma_window})"
        )

    plt.title("5. Collision Rate per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Collision")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train(num_episodes=500, save_every=50, save_dir="checkpoints", fig_dir="figures", resume_path=None, mixing_mode="qmix"):
    _seed = int(os.environ.get("SEED", RANDSEED))
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    random.seed(_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    args = build_args()
    save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    fig_dir = fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    if resume_path is not None:
        resume_path = resume_path

    XRTK = np.linspace(0, 3000, 2)
    YRTK = np.linspace(0, 0, 2)

    # warmup only to infer feature dimensions
    warmup_sim = simulation(
        args,
        XRTK,
        YRTK,
        CAVtype=3,
        filename="results.csv",
        rl_agent=None
    )

    rl_agent = build_agent(warmup_sim, device, mixing_mode=mixing_mode)
    if resume_path is not None:
        # Proper warm-start fine-tune: load the FULL checkpoint (agent + mixer +
        # targets + optimizer at lr=1e-4), then raise epsilon for re-exploration
        # under the new reward. (Not load_agent_only, which reinits the mixer and
        # drops lr to 5e-6.)
        rl_agent.load(resume_path)
        rl_agent.epsilon = 0.3
        rl_agent.epsilon_min = 0.05


    reward_history = []
    loss_history = []
    lane_change_history = []
    platoon_rate_history = []
    collision_rate_history = []
    avg_reward_history = []

    decision_interval_steps = int(round(1.0 / warmup_sim.dt))  # 10 when HZ=10
    best_eval_score = -float("inf")
    best_episode = -1

    # Per-episode training log for paper convergence plots (one row/episode).
    log_path = os.path.join(save_dir, "train_log.csv")
    with open(log_path, "w") as _lf:
        _lf.write("episode,return,avg_reward,loss,platoon_rate,lane_changes,collision,epsilon\n")

    # Held-out heterogeneous-driver validation (fixed seeds), logged for paper convergence.
    VAL_EVERY = int(os.environ.get("VAL_EVERY", "100"))
    VAL_EPISODES = int(os.environ.get("VAL_EPISODES", "20"))
    val_log_path = os.path.join(save_dir, "validation.csv")
    with open(val_log_path, "w") as _vf:
        _vf.write("episode,het_platoon,het_lc,het_speed,het_collision\n")

    for ep in range(num_episodes):
        args = build_args()
        sim = simulation(
            args,
            XRTK,
            YRTK,
            CAVtype=3,
            filename=f"train_ep_{ep}.csv",
            rl_agent=rl_agent
        )

        # Domain randomization of human lane-change frequency: per-episode base LC
        # threshold (log-uniform, aggressive->calm) + per-driver jitter. Trains the
        # policy across the human-LC-frequency axis that drives the SUMO transfer gap.
        if os.environ.get("DR_ATHR"):
            import math
            _lo = float(os.environ.get("DR_ATHR_LO", "0.3"))
            _hi = float(os.environ.get("DR_ATHR_HI", "8.0"))
            _base = math.exp(random.uniform(math.log(_lo), math.log(_hi)))
            for _v in sim.svs:
                if _v.type == VEHTYPE.MOBIL:
                    _v.athr = _base * math.exp(random.gauss(0.0, 0.2))

        # Domain randomization of human gap acceptance: per-episode base safe-deceleration
        # (b_safe; higher = accepts tighter gaps) + per-driver jitter. Tests whether varying the
        # MOBIL gap-acceptance dynamics (not just frequency) helps cross-simulator transfer.
        if os.environ.get("DR_BSAFE"):
            import math
            _blo = float(os.environ.get("DR_BSAFE_LO", "1.0"))
            _bhi = float(os.environ.get("DR_BSAFE_HI", "9.0"))
            _bbase = math.exp(random.uniform(math.log(_blo), math.log(_bhi)))
            for _v in sim.svs:
                if _v.type == VEHTYPE.MOBIL:
                    _v.bsafe = _bbase * math.exp(random.gauss(0.0, 0.2))

        # Heterogeneous human drivers: per-vehicle mix of conservative/normal/aggressive
        # IDM + MOBIL profiles (default ON in this folder). Uses the training RNG, so the
        # mix varies episode-to-episode. Set HET_DRIVERS=0 to fall back to homogeneous.
        if os.environ.get("HET_DRIVERS", "1") == "1":
            apply_heterogeneous_drivers(sim, random)

        rl_agent.start_episode()

        episode_reward = 0.0
        episode_intervals = 0
        n_loss0 = len(loss_history)   # to average this episode's TD losses
        done = False
        

        if ep % 500 == 0:
            debug = True
        else:
            debug = False

        while not done:
            # Observation/state at DECISION time
            state = sim.build_global_state()
            obs_batch, agent_mask, n_real = collect_drlcav_obs(sim, max_agents=MAX_AGENTS)

            # Run exactly one 1-second high-level decision interval
            interval_reward, done = run_one_decision_interval(
                sim,
                decision_interval_steps=decision_interval_steps,
                debug=debug
            )
            transition_reward = interval_reward / decision_interval_steps
            transition_reward = np.clip(transition_reward, -5.0, 5.0)

            # Next observation/state AFTER consequence of that decision
            next_state = sim.build_global_state()
            next_obs_batch, next_agent_mask, next_n_real = collect_drlcav_obs(sim, max_agents=MAX_AGENTS)

            # Actions actually used by real DRLCAVs, then pad
            # Use simulator-held last decision action, not sv.marl_action,
            # because sv.marl_action may be reset to 0 after lane change completion.
            real_actions = []
            action_index_map = {-1: 0, 0: 1, 1: 2}
            for sv in sim.svs:
                if sv.type == VEHTYPE.DRLCAV:
                    # Train on the action the policy actually issued at s_t
                    # (last_decision_actions), not a relabeled "keep".
                    sim_action = int(sim.last_decision_actions[sv.id])  # -1, 0, 1
                    real_actions.append(action_index_map[sim_action])

            if len(real_actions) > MAX_AGENTS:
                raise ValueError(
                    f"len(real_actions)={len(real_actions)} exceeds MAX_AGENTS={MAX_AGENTS}"
                )

            actions = np.array(
                real_actions + [1] * (MAX_AGENTS - len(real_actions)),
                dtype=np.int64
            )

            # Collision is the true terminal: cut the bootstrap (done=1) and end the
            # episode. This anchors the value function (prevents value inflation) and
            # makes the agent forfeit the future positive-reward stream on collision,
            # which is a strong avoidance signal. The t>=80 limit stays a truncation.
            collided = getattr(sim, "collison", 0) > 0
            terminal = 1.0 if collided else 0.0

            rl_agent.store_transition(
                obs_batch=obs_batch,
                agent_mask=agent_mask,
                state=state,
                actions=actions,
                reward=transition_reward,          # summed over 10 sim steps
                next_obs_batch=next_obs_batch,
                next_agent_mask=next_agent_mask,
                next_state=next_state,
                done=terminal
            )

            loss = rl_agent.train_step()
            if loss is not None:
                loss_history.append(loss)

            episode_reward += transition_reward
            episode_intervals += 1

            if collided:
                done = True

        rl_agent.end_episode()
        reward_history.append(episode_reward)
        lane_change_num, platoon_rate, collision_rate = compute_episode_metrics(sim)
        lane_change_history.append(lane_change_num)
        platoon_rate_history.append(platoon_rate)
        collision_rate_history.append(collision_rate)

        episode_avg_reward = episode_reward / max(episode_intervals, 1)
        avg_reward_history.append(episode_avg_reward)

        # append this episode's row to the training log
        ep_loss = float(np.mean(loss_history[n_loss0:])) if len(loss_history) > n_loss0 else float("nan")
        with open(log_path, "a") as _lf:
            _lf.write("%d,%.4f,%.4f,%.4f,%.4f,%d,%.1f,%.4f\n" % (
                ep, episode_reward, episode_avg_reward, ep_loss,
                platoon_rate, lane_change_num, collision_rate, rl_agent.epsilon))

        # Held-out heterogeneous validation for the paper's convergence curve.
        if (ep + 1) % VAL_EVERY == 0 or ep == 0:
            vres = validate_heterogeneous(
                rl_agent, XRTK, YRTK, decision_interval_steps, n_episodes=VAL_EPISODES)
            with open(val_log_path, "a") as _vf:
                _vf.write("%d,%.4f,%.4f,%.4f,%.4f\n" % (
                    ep + 1, vres["platoon"], vres["lc"], vres["speed"], vres["collision"]))
            print(f"[VAL ep{ep+1}] het platoon={vres['platoon']:.3f} "
                  f"lc={vres['lc']:.2f} speed={vres['speed']:.2f} coll={vres['collision']:.3f}")

        right_n = sim.action_count[0]
        keep_n  = sim.action_count[1]
        left_n  = sim.action_count[2]

        total_n = max(right_n + keep_n + left_n, 1)
        if ep % 100 == 0:
            print(
                f"Actions | "
                f"R {right_n/total_n:.2f} "
                f"K {keep_n/total_n:.2f} "
                f"L {left_n/total_n:.2f}"
            )

        # if (ep + 1) % 1000 == 0:

        #     num_eval_episodes = 1
        #     eval_results = []

        #     for eval_ep in range(num_eval_episodes):
        #         eval_result = evaluate_policy(
        #             rl_agent,
        #             args,
        #             XRTK,
        #             YRTK,
        #             decision_interval_steps
        #         )
        #         eval_results.append(eval_result)

        #     avg_reward = np.mean([r["reward"] for r in eval_results])
        #     avg_platoon_rate = np.mean([r["platoon_rate"] for r in eval_results])
        #     avg_lane_changes = np.mean([r["lane_change_count"] for r in eval_results])
        #     avg_collision_count = np.mean([r["collision_count"] for r in eval_results])
        #     collision_episode_rate = np.mean([1.0 if r["had_collision"] else 0.0 for r in eval_results])

        #     print(
        #         f"Eval({num_eval_episodes} eps) | "
        #         f"reward={avg_reward:.2f}, "
        #         f"platoon_rate={avg_platoon_rate:.3f}, "
        #         f"lane_changes={avg_lane_changes:.2f}, "
        #         f"collision_count={avg_collision_count:.2f}, "
        #         f"collision_ep_rate={collision_episode_rate:.2f}"
        #     )

        #     # collision-dominant score
        #     eval_score = (
        #         avg_reward
        #         + 100.0 * avg_platoon_rate
        #         - 1000.0 * avg_collision_count
        #         - 500.0 * collision_episode_rate
        #         - 2.0 * avg_lane_changes
        #     )

        #     print(f"Eval Score = {eval_score:.2f}")

        #     # save best model only if collision-free across all eval episodes
        #     if collision_episode_rate == 0.0 and eval_score > best_eval_score:
        #         best_eval_score = eval_score
        #         best_episode = ep + 1

        #         best_path = os.path.join(save_dir, "qmix_best.pt")
        #         rl_agent.save(best_path)

        #         print(
        #             f">>> New BEST collision-free model saved at episode {best_episode} | "
        #             f"score={best_eval_score:.2f}"
        #         )

        if (ep + 1) % save_every == 0:
            rl_agent.save(
                os.path.join(save_dir, f"qmix_checkpoint_ep{ep+1}.pt")
            )
        if (ep + 1) % (save_every*10) == 0:
            fig_path = os.path.join(
                fig_dir,
                f"training_curve_ep{ep+1}.png"
            )

            plot_training_curves(
                avg_reward_history,
                loss_history,
                lane_change_history,
                platoon_rate_history,
                collision_rate_history,
                save_path=fig_path
            )

            print(f"Saved figure: {fig_path}")

    rl_agent.save(
        os.path.join(save_dir, "qmix_final.pt")
    )

    return rl_agent, reward_history, loss_history


if __name__ == "__main__":
    # MIX_MODE env selects the credit-assignment variant: qmix (default) | vdn | iql.
    # Each writes its own checkpoint/figure dir so the QMIX paper model is never overwritten.
    mode = os.environ.get("MIX_MODE", "qmix")
    n_ep = int(os.environ.get("NUM_EPISODES", "5000"))
    save_dir = os.environ.get("SAVE_DIR", f"checkpoints_{mode}")
    fig_dir = os.environ.get("FIG_DIR", f"figures_{mode}")
    train(num_episodes=n_ep, save_every=100,
          save_dir=save_dir, fig_dir=fig_dir,
          mixing_mode=mode)#, resume_path="trained_model/qmix_checkpoint_2cav6v.pt")