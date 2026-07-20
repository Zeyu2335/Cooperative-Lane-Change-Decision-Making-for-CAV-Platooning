# CNN-QMIX: Cooperative Lane-Change Decision Making for CAV Platooning

Reference implementation for the paper *"Multi-Agent Reinforcement Learning for
Cooperative Lane-Change Decision Making in Mixed Traffic."* A CNN-QMIX multi-agent
reinforcement-learning policy decides cooperative lane changes for connected automated
vehicles (CAVs) in mixed traffic, on a custom Python microsimulator with ACC/CACC
longitudinal control, MOBIL/IDM human drivers, and an MPC lane-change executor.

## Repository layout

```
microsim/
  _constants.py      constants (vehicle length, types, seeds, ...)
  _sensor.py         neighbor/leader queries on the road
  _dynamics.py       kinematic bicycle model + Stanley steering
  _mpc.py            lateral MPC lane-change controller (condensed QP)
  _agents.py         vehicle controllers: MOBIL (human), CAV/ACC/CACC, DRLCAV
  _input.py          observation builder: ego-centric CNN grid + remote V2V vector + kNN
  _qmix_network.py   AgentCNNLNGRU / AgentMLPGRU / QMixer / MultiAgentQMIX
  _microsim.py       the simulation environment (step, spawn, collisions, safety layers)
  _drl_agent.py      DRL agent wrapper (action selection, replay, learning)
  _animation.py      optional visualization
  _logging.py        optional trajectory logging
  train.py           training entry point (heterogeneous drivers + in-loop validation)
  eval_paper.py      evaluation: platoon rate, LC, speed, capacity, TTC, ...
  eval_transfer.py   shared eval helpers (geometry, capacity/power model, collisions)
checkpoints/
  cnn_qmix_ln.pt     proposed model (CNN encoder + LayerNorm + QMIX mixer)
  vdn.pt             VDN baseline (CNN encoder, additive mixing)
  std_qmix.pt        standard-QMIX baseline (kNN-MLP encoder, QMIX mixer)
```

## Install

```bash
pip install -r requirements.txt          # numpy, torch, scipy, matplotlib
```

## Evaluate a trained model

Run from inside `microsim/` (the modules use flat imports):

```bash
cd microsim
# CNN-QMIX (proposed) at 50% CAV market penetration, worth gate + merge arbiter on
USE_WORTH_GATE=1 USE_LC_ARBITER=1 ENCODER=cnnln \
python eval_paper.py --controller drlcav --checkpoint ../checkpoints/cnn_qmix_ln.pt \
    --penetration 0.50 --episodes 100 --out results_cnnqmix.csv

# VDN baseline
USE_WORTH_GATE=1 USE_LC_ARBITER=1 ENCODER=cnn \
python eval_paper.py --controller drlcav --checkpoint ../checkpoints/vdn.pt --penetration 0.50 --episodes 100

# standard-QMIX baseline (kNN-MLP)
USE_WORTH_GATE=1 USE_LC_ARBITER=1 ENCODER=mlp OBS_MODE=knn KNN=6 \
python eval_paper.py --controller drlcav --checkpoint ../checkpoints/std_qmix.pt --penetration 0.50 --episodes 100

# rule-based baselines
python eval_paper.py --controller cav --penetration 0.50 --episodes 100                       # MOBIL
GREEDY_FOLLOW_CAV=1 GREEDY_MOBIL=1 ENCODER=cnnln \
python eval_paper.py --controller drlcav --checkpoint ../checkpoints/cnn_qmix_ln.pt --penetration 0.50 --episodes 100  # Greedy
```

The evaluation prints and (with `--out`) logs platoon rate, max platoon length, mean
speed, lane changes per CAV, **road capacity** (veh/h/lane), and minimum TTC.

## Train from scratch

```bash
cd microsim
SAVE_DIR=./checkpoints_new FIG_DIR=./figs NUM_EPISODES=5000 \
ENCODER=cnnln MIX_MODE=qmix CONT_PERCEPTION=1 LANE_TOL=0.9 \
python train.py
```

Checkpoints are saved every 100 episodes; a held-out validation set (fixed seeds,
independently sampled heterogeneous drivers) is logged to `SAVE_DIR/validation.csv`.

## Method knobs (environment variables)

| Variable | Meaning | Values |
|---|---|---|
| `ENCODER` | per-agent encoder | `cnnln` (CNN+LayerNorm), `cnn`, `mlp` (kNN) |
| `MIX_MODE` | value-mixing | `qmix` (monotonic), `vdn` (additive) |
| `OBS_MODE`, `KNN` | kNN observation for the MLP encoder | `knn`, `6` |
| `USE_WORTH_GATE` | suppress non-beneficial lane changes | `0`/`1` |
| `USE_LC_ARBITER` | concurrent-merge arbiter (safety) | `0`/`1` |
| `USE_MPC_LC` | MPC lateral executor (else Stanley) | `0`/`1` (eval-time) |
| `T_ACC`, `T_CACC` | ACC / CACC time headway [s] | `1.5`, `1.0` |
| `HET_MODE`, `AGG_A/B/T/S0`, `AGG_CV` | human driver model | `discrete`/`normal`, ... |
| `HUMAN_DELAY` | human reaction delay [s] | e.g. `0.8` |
| `W_SPEED`,`W_PLATOON`,`W_SUSTAIN`,`W_COLLISION`,`W_TTC`,`W_LANE_CHANGE` | reward weights | see paper |

Defaults reproduce the paper configuration. The learned policies are trained with the
worth gate off (to isolate reward effects) and evaluated with `USE_WORTH_GATE=1
USE_LC_ARBITER=1`.

## Citation

```
@article{mu_cnnqmix,
  title  = {Multi-Agent Reinforcement Learning for Cooperative Lane-Change Decision Making in Mixed Traffic},
  author = {Mu, Zeyu and others},
  year   = {2026}
}
```
