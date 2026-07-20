#!/usr/bin/env python3
"""
eval_paper.py - in-distribution test evaluation for the Multi_agent_Platooning paper.

Rolls out episodes with HETEROGENEOUS human drivers (matching training) and computes the
paper's test-table metrics for one method at one penetration:
  platoon rate, max platoon length, platoon formation time, lane changes per CAV,
  network speed, and collision rate.

Method is selected by --controller plus environment variables (set by the launcher):
  drlcav  : ENCODER in {cnnln, cnn, mlp} (+ OBS_MODE=knn KNN=6 for mlp), --checkpoint
  cav     : MOBIL rule baseline
  cav + GREEDY_FOLLOW_CAV=1 GREEDY_MOBIL=1 : C-MOBIL cooperative baseline
"""
import os, argparse, csv
import numpy as np, torch, random
from collections import defaultdict
from _microsim import simulation
from _constants import VEHTYPE, RANDSEED, VEHLENGTH
from train import build_args, build_agent, run_one_decision_interval, apply_heterogeneous_drivers
from eval_transfer import GEOM, cav_collision, _veh_power, _front_gap

CONNECTED = (VEHTYPE.CAV, VEHTYPE.DRLCAV)
TTC_TH = [0.5, 1.0, 2.0, 3.0, 5.0]
NUMSVS = int(os.environ.get("NUMSVS", "24"))     # thresholds for the TTC cumulative distribution


def cav_closing_ttcs(sim):
    """Bumper-to-bumper time-to-collision (s) for each connected vehicle that is closing on
    its same-lane leader (rel. speed > 0); safe/opening pairs contribute no TTC sample."""
    out = []
    svs = sim.svs
    for veh in svs:
        if veh.type not in CONNECTED:
            continue
        lane = int(round(veh.lane))
        front = [o for o in svs if o.id != veh.id
                 and int(round(o.lane)) == lane and o.s > veh.s]
        if not front:
            continue
        f = min(front, key=lambda x: x.s - veh.s)
        rel_v = veh.v - f.v
        if rel_v > 1e-3:
            gap = max(0.1, f.s - veh.s - VEHLENGTH)
            out.append(gap / rel_v)
    return out


def max_platoon_len(sim):
    """Longest run of consecutive connected vehicles in a lane (no HV between adjacent
    members), matching the immediate-neighbor platoon definition of is_in_platoon()."""
    lanes = defaultdict(list)
    for sv in sim.svs:
        lanes[int(round(sv.lane))].append(sv)
    best = 0
    for vs in lanes.values():
        vs.sort(key=lambda x: x.s)
        run = 0
        for sv in vs:
            if sv.type in CONNECTED:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return best


def _shift_humans(sim, s):
    """OOD human-model shift: scale each human's fixed profile parameters toward more
    aggressive (s>0: shorter headway/gap, higher accel, less polite, lower LC threshold,
    tighter gap acceptance) or calmer (s<0). Deterministic per driver (no jitter) -> each
    driver keeps fixed parameters; only the whole heterogeneous mix is shifted off training."""
    for v in sim.svs:
        if v.type == VEHTYPE.MOBIL:
            v.ad *= (1 + s); v.bd *= (1 + s); v.bsafe *= (1 + s)
            v.T *= (1 - s); v.d *= (1 - s); v.pol *= (1 - s); v.athr *= (1 - s)


def run(controller, penetration, num_episodes, checkpoint, device):
    XRTK = np.linspace(0, 3000, 2); YRTK = np.linspace(0, 0, 2)
    cav_type = 3 if controller == "drlcav" else 1
    human_shift = float(os.environ.get("HUMAN_SHIFT", "0"))   # OOD: 0=in-dist, +aggr / -calm

    def mk_args():
        a = build_args(); a.geometry = 0; a.numSvs = NUMSVS; a.cavRate = penetration
        return a

    rl_agent = None
    if controller == "drlcav":
        wa = build_args(); wa.geometry = GEOM[3]; wa.numSvs = NUMSVS; wa.cavRate = min(0.2, 8.0/NUMSVS)
        warm = simulation(wa, XRTK, YRTK, CAVtype=3, filename="/tmp/ep_warm.csv", rl_agent=None)
        rl_agent = build_agent(warm, device)
        ck = torch.load(checkpoint, map_location=device)
        rl_agent.agent_net.load_state_dict(ck["agent_net"])
        rl_agent.epsilon = 0.0

    plat, mlen, ftime, lcv, spd, col, eng, minttc = [], [], [], [], [], [], [], []
    cap = []
    ttc_below = [0] * len(TTC_TH); ttc_total = 0; ttc_sum = 0.0   # TTC distribution accumulators
    for ep in range(num_episodes):
        random.seed(RANDSEED + ep); np.random.seed(RANDSEED + ep)
        sim = simulation(mk_args(), XRTK, YRTK, CAVtype=cav_type,
                         filename=f"/tmp/ep_{ep}.csv", rl_agent=rl_agent, explore=False)
        sim.explore = False
        apply_heterogeneous_drivers(sim, random)      # in-distribution: same het mix as training
        if human_shift != 0.0:
            _shift_humans(sim, human_shift)            # OOD: shift the whole mix off training
        cavs = [sv for sv in sim.svs if sv.type in CONNECTED]
        cav_ids = {sv.id for sv in cavs}
        n = max(1, len(cavs))
        dis = sim.decision_interval_steps; dt_int = dis * sim.dt
        psum = ssum = 0.0; iv = 0; had_col = 0; emax = 0; form_t = None; t = 0.0
        energy_J = dist_m = 0.0                        # network energy (J/m over all vehicles)
        cap_vsum = cap_ssum = 0.0                       # road capacity: q = 3600*sum(v)/sum(space headway)
        ep_min_ttc = float("inf")
        done = False
        while not done:
            _, done = run_one_decision_interval(sim, decision_interval_steps=dis)
            t += dt_int
            if cav_collision(sim, cav_ids):
                had_col = 1
            svs = sim.svs
            psum += sum(sim.obs_builder.is_in_platoon(sv, [o for o in svs if o.id != sv.id])
                        for sv in cavs) / n
            ssum += float(np.mean([sv.v for sv in svs]))
            for sv in svs:                             # network energy, mirrors eval_transfer
                _g = _front_gap(sv, svs)
                energy_J += _veh_power(sv.v, sv.a, _g) * dt_int
                dist_m += sv.v * dt_int
                if _g is not None and _g < 60.0:       # capacity from constrained (following) vehicles
                    cap_vsum += sv.v
                    cap_ssum += (_g + VEHLENGTH)
            for ttc in cav_closing_ttcs(sim):          # safety: TTC distribution + min TTC
                ttc_total += 1; ttc_sum += ttc; ep_min_ttc = min(ep_min_ttc, ttc)
                for i, th in enumerate(TTC_TH):
                    if ttc < th:
                        ttc_below[i] += 1
            ml = max_platoon_len(sim); emax = max(emax, ml)
            if form_t is None and ml >= 2:
                form_t = t
            iv += 1
        plat.append(psum / max(1, iv)); spd.append(ssum / max(1, iv))
        mlen.append(emax); col.append(had_col)
        eng.append(energy_J / max(dist_m, 1e-6))       # J/m == kJ/km
        cap.append(3600.0 * cap_vsum / max(cap_ssum, 1e-6))   # veh/h/lane
        if ep_min_ttc < float("inf"):
            minttc.append(ep_min_ttc)
        if form_t is not None:
            ftime.append(form_t)
        lcv.append(sum(getattr(sv, "LC", 0) for sv in cavs) / n)

    def m(x): return float(np.mean(x)) if x else float("nan")
    def s(x): return float(np.std(x)) if x else float("nan")
    res = dict(tag=os.environ.get("TAG", controller), controller=controller,
               penetration=penetration, human_shift=human_shift, episodes=num_episodes,
               platoon=m(plat), platoon_std=s(plat), max_len=m(mlen),
               form_time=m(ftime), lc_per_veh=m(lcv), speed=m(spd), energy=m(eng), capacity=m(cap),
               collision=m(col), min_ttc=m(minttc),
               ttc_mean=(ttc_sum / ttc_total if ttc_total else float("nan")))
    for i, th in enumerate(TTC_TH):                    # cumulative TTC distribution (fraction below)
        res[f"ttc_lt{th}"] = (ttc_below[i] / ttc_total if ttc_total else float("nan"))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["drlcav", "cav"], default="drlcav")
    p.add_argument("--penetration", type=float, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = run(a.controller, a.penetration, a.episodes, a.checkpoint, dev)
    print(f"[{r['tag']}|p{a.penetration}] platoon={r['platoon']:.3f} maxlen={r['max_len']:.2f} "
          f"LC={r['lc_per_veh']:.2f} speed={r['speed']:.2f} energy={r['energy']:.1f} "
          f"coll={r['collision']:.3f} minTTC={r['min_ttc']:.2f} "
          f"TTC<1s={r['ttc_lt1.0']:.3f} TTC<2s={r['ttc_lt2.0']:.3f}")
    if a.out:
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys()))
            if new:
                w.writeheader()
            w.writerow(r)
