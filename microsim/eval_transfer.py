#!/usr/bin/env python3
"""
Unified transferability evaluation harness (ego-centric DRLCAV).

One tool for all OOD axes. Every knob defaults to the in-distribution (ID) reference
(3 lanes, baseline humans, numSvs=24, training cavRate), so any single deviation is a
clean one-factor-at-a-time transfer test; combine several for a stress/factorial cell.

Knobs:
  --controller {drlcav, cav}   drlcav = trained RL policy (needs --checkpoint);
                               cav    = rule-based MOBIL-CAV baseline (type 1)
  --driver {baseline, aggressive_idm, heterogeneous, aggressive_mobil}   human behavior
  --lanes {2,3,4,5}            road lane count
  --num-svs N                  total vehicles (traffic density)         [ID 24]
  --penetration P              CAV fraction; omit -> ID random {0.17,0.25,0.33}
  --episodes N                 episodes (50 typical; 100+ for collision rate)
  --checkpoint PATH            required for drlcav
  --out CSV                    append one summary row to this CSV

Metrics: LC/veh, platoon rate, collision rate over the controlled CAVs (type CAV or DRLCAV);
avg speed over ALL vehicles (network-level, CAVs + humans = system throughput proxy).
Full 80 s episodes; traffic seeded (paired across conditions).

Driver params from literature: IDM (Treiber et al. 2000; Treiber & Kesting 2013),
MOBIL (Kesting et al. 2007).
"""
import argparse
import csv
import os
import numpy as np
import random
import torch

from _constants import VEHTYPE, RANDSEED, VEHLENGTH, VD, GEOMETRY
from _microsim import simulation
from train import build_args, build_agent, run_one_decision_interval

CONTROLLED_TYPES = (VEHTYPE.CAV, VEHTYPE.DRLCAV)
GEOM = {2: GEOMETRY.TWOLANE, 3: GEOMETRY.NONE, 4: GEOMETRY.FOURLANE, 5: GEOMETRY.FIVELANE}

# --- literature driver profiles ---
IDM = {  # a [m/s^2], b [m/s^2], T [s], s0 [m]
    "conservative": dict(a=0.8, b=1.5, T=2.0, s0=2.5),
    "normal":       dict(a=1.0, b=2.0, T=1.5, s0=2.0),
    "aggressive":   dict(a=2.0, b=3.0, T=1.0, s0=1.5),
}
MOBIL_AGGR = dict(p=0.0, a_th=0.1, b_safe=5.0)   # selfish, eager, forces harder braking

# --- energy model (instantaneous tractive power, passenger car; illustrative literature-range
# coefficients) with platoon drafting drag reduction. P = (F_roll + F_aero + F_inertia)*v,
# positive power only (no regen) + idle. Drag reduction eta(gap) = ETA_MAX*exp(-gap/LAMBDA)
# applied to the follower's Cd when a leader is ahead -> close platoons draft -> less energy. ---
VEH_MASS = 1500.0      # kg (compact car)
C_ROLL = 0.013         # rolling-resistance coefficient
RHO_AIR = 1.225        # kg/m^3
CD0 = 0.30             # baseline drag coefficient
FRONT_AREA = 2.5       # m^2 frontal area
G_ACC = 9.81
P_IDLE = 1500.0        # W accessory/idle baseline
DRAG_ETA_MAX = float(os.environ.get("DRAG_ETA_MAX", "0.30"))    # max drafting drag reduction (gap -> 0)
DRAG_LAMBDA  = float(os.environ.get("DRAG_LAMBDA", "12.0"))     # m, decay length of the drafting benefit

def _front_gap(veh, svs):
    """Bumper gap (m) to the nearest same-lane vehicle ahead, or None if none ahead."""
    ahead = [o for o in svs if o.id != veh.id
             and int(round(o.lane)) == int(round(veh.lane)) and o.s > veh.s]
    if not ahead:
        return None
    f = min(ahead, key=lambda o: o.s - veh.s)
    return max(0.1, f.s - veh.s - VEHLENGTH)

def _veh_power(v, a, gap):
    """Instantaneous tractive power [W] (positive-only + idle) with drafting drag reduction."""
    eta = DRAG_ETA_MAX * np.exp(-gap / DRAG_LAMBDA) if gap is not None else 0.0
    cd = CD0 * (1.0 - eta)
    f_roll = VEH_MASS * G_ACC * C_ROLL
    f_aero = 0.5 * RHO_AIR * cd * FRONT_AREA * v * v
    f_inertia = VEH_MASS * a
    p_wheel = (f_roll + f_aero + f_inertia) * v
    return max(0.0, p_wheel) + P_IDLE


def configure_humans(sim, driver):
    """Shift the human (MOBIL type-0) drivers for the chosen scenario; CAVs untouched."""
    humans = [sv for sv in sim.svs if sv.type == VEHTYPE.MOBIL]
    if driver == "baseline":
        return "training defaults"
    if driver == "aggressive_idm":
        for v in humans:
            v.ad, v.bd, v.T, v.d = IDM["aggressive"]["a"], IDM["aggressive"]["b"], IDM["aggressive"]["T"], IDM["aggressive"]["s0"]
        return f"{len(humans)}x aggressive IDM"
    if driver == "heterogeneous":
        counts = {"conservative": 0, "normal": 0, "aggressive": 0}
        for v in humans:
            prof = random.choice(["conservative", "normal", "aggressive"])
            p = IDM[prof]; v.ad, v.bd, v.T, v.d = p["a"], p["b"], p["T"], p["s0"]
            counts[prof] += 1
        return f"heterogeneous IDM {counts}"
    if driver == "aggressive_mobil":
        for v in humans:
            v.pol, v.athr, v.bsafe = MOBIL_AGGR["p"], MOBIL_AGGR["a_th"], MOBIL_AGGR["b_safe"]
        return f"{len(humans)}x aggressive MOBIL"
    raise ValueError(driver)


def cav_collision(sim, cav_ids):
    svs = sim.svs
    for i in range(len(svs)):
        vi = svs[i]
        for j in range(i + 1, len(svs)):
            vj = svs[j]
            if (vi.id not in cav_ids) and (vj.id not in cav_ids):
                continue
            if abs(vi.s - vj.s) < VEHLENGTH and abs(vi.l - vj.l) < 0.5:
                return True
    return False


def run(controller, driver, lanes, num_svs, penetration, num_episodes, checkpoint, device):
    np.random.seed(RANDSEED); random.seed(RANDSEED); torch.manual_seed(RANDSEED)
    XRTK = np.linspace(0, 3000, 2); YRTK = np.linspace(0, 0, 2)
    cav_type = 3 if controller == "drlcav" else 1

    def mk_args():
        a = build_args()
        a.geometry = GEOM[lanes]
        a.numSvs = num_svs
        if penetration is not None:
            a.cavRate = penetration
        return a

    rl_agent = None
    if controller == "drlcav":
        # Build the agent from a STANDARD warmup (24 veh, 3 lanes). The ego-centric obs
        # dims are fixed regardless of numSvs/lanes, so this avoids the MAX_AGENTS=12
        # dim-inference cap when a cell has >12 CAVs (numSvs*penetration > 12). The
        # rollout itself (per-agent select_action) has no such cap.
        warm_args = build_args(); warm_args.geometry = GEOM[3]; warm_args.numSvs = 24
        warm = simulation(warm_args, XRTK, YRTK, CAVtype=3, filename="/tmp/ev_warm.csv", rl_agent=None)
        rl_agent = build_agent(warm, device)
        # Eval uses only the per-agent policy (select_action); the QMIX mixer is unused
        # at test time and its state_dim = numSvs*7, so load ONLY agent_net (lets us vary
        # numSvs/penetration without a mixer-shape mismatch).
        ckpt = torch.load(checkpoint, map_location=device)
        rl_agent.agent_net.load_state_dict(ckpt["agent_net"])
        rl_agent.epsilon = 0.0

    np.random.seed(RANDSEED); random.seed(RANDSEED)   # paired traffic across conditions

    ROAD_LEN = float(XRTK[-1])   # road length (m), for density/flow
    lc, spd, plat, col, thr, eng, jrk = [], [], [], [], [], [], []
    info = ""
    for ep in range(num_episodes):
        sim = simulation(mk_args(), XRTK, YRTK, CAVtype=cav_type,
                         filename=f"/tmp/ev_ep_{ep}.csv", rl_agent=rl_agent, explore=False)
        sim.explore = False
        info = configure_humans(sim, driver)
        dis = sim.decision_interval_steps
        dt_int = dis * sim.dt                         # decision-interval duration (s), ~1.0

        cavs = [sv for sv in sim.svs if sv.type in CONTROLLED_TYPES]
        cav_ids = {sv.id for sv in cavs}
        n = max(1, len(cavs))
        had_col = 0; psum = ssum = fsum = 0.0; iv = 0
        energy_J = 0.0; dist_m = 0.0                  # network energy + distance -> energy/km
        jerk_sum = 0.0; jerk_cnt = 0; a_prev = {}     # network |jerk| (1 s resolution)
        done = False
        while not done:
            _, done = run_one_decision_interval(sim, decision_interval_steps=dis)
            if cav_collision(sim, cav_ids):
                had_col = 1
            svs = sim.svs
            vbar = float(np.mean([sv.v for sv in svs]))
            psum += sum(sim.obs_builder.is_in_platoon(sv, [o for o in svs if o.id != sv.id]) for sv in cavs) / n
            ssum += vbar                                              # NETWORK speed (all vehicles)
            fsum += (len(svs) / (lanes * ROAD_LEN)) * vbar * 3600.0   # per-lane flow (veh/hr)
            for sv in svs:                                            # network energy + jerk
                energy_J += _veh_power(sv.v, sv.a, _front_gap(sv, svs)) * dt_int
                dist_m += sv.v * dt_int
                if sv.id in a_prev:
                    jerk_sum += abs(sv.a - a_prev[sv.id]) / dt_int; jerk_cnt += 1
                a_prev[sv.id] = sv.a
            iv += 1
            if getattr(sim, "done", False):
                done = True
        lc.append(sum(getattr(sv, "LC", 0) for sv in cavs) / n)
        spd.append(ssum / max(iv, 1)); plat.append(psum / max(iv, 1)); col.append(had_col)
        thr.append(fsum / max(iv, 1))                                 # veh/hr/lane
        eng.append(energy_J / max(dist_m, 1e-6))                      # J/m == kJ/km
        jrk.append(jerk_sum / max(jerk_cnt, 1))                       # m/s^3 (1 s resolution)

    def ms(x):
        return float(np.mean(x)), float(np.std(x))
    res = {
        "controller": controller, "driver": driver, "lanes": lanes,
        "num_svs": num_svs, "penetration": penetration if penetration is not None else "ID",
        "episodes": num_episodes,
        "lc_per_veh_mean": ms(lc)[0], "lc_per_veh_std": ms(lc)[1],
        "speed_mean": ms(spd)[0], "speed_std": ms(spd)[1],
        "platoon_mean": ms(plat)[0], "platoon_std": ms(plat)[1],
        "collision_mean": ms(col)[0], "collision_std": ms(col)[1],
        "throughput_mean": ms(thr)[0], "throughput_std": ms(thr)[1],   # veh/hr/lane
        "energy_mean": ms(eng)[0], "energy_std": ms(eng)[1],           # kJ/km (lower=better)
        "jerk_mean": ms(jrk)[0], "jerk_std": ms(jrk)[1],               # m/s^3 (lower=smoother)
    }
    print(f"[{controller}|{driver}|L{lanes}|N{num_svs}|p{res['penetration']}] "
          f"humans={info} | LC/veh {res['lc_per_veh_mean']:.2f} | speed {res['speed_mean']:.2f} | "
          f"platoon {res['platoon_mean']:.3f} | collision {res['collision_mean']:.3f} | "
          f"flow {res['throughput_mean']:.0f} | energy {res['energy_mean']:.1f} | jerk {res['jerk_mean']:.3f}")
    return res


def append_csv(path, res):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.keys()))
        if new:
            w.writeheader()
        w.writerow(res)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified transferability eval for the ego-centric DRLCAV.")
    p.add_argument("--controller", choices=["drlcav", "cav"], default="drlcav")
    p.add_argument("--driver", choices=["baseline", "aggressive_idm", "heterogeneous", "aggressive_mobil"], default="baseline")
    p.add_argument("--lanes", type=int, choices=[2, 3, 4, 5], default=3)
    p.add_argument("--num-svs", type=int, default=24)
    p.add_argument("--penetration", type=float, default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="append summary row to this CSV")
    a = p.parse_args()
    if a.controller == "drlcav" and not a.checkpoint:
        p.error("--controller drlcav requires --checkpoint")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    res = run(a.controller, a.driver, a.lanes, a.num_svs, a.penetration, a.episodes, a.checkpoint, dev)
    if a.out:
        append_csv(a.out, res); print(f"appended -> {a.out}")
