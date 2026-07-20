#! /usr/bin/env python3
"""Lateral Model Predictive Controller for CAV lane-change execution.

Replaces the Stanley steering controller (eval-time only, env USE_MPC_LC=1). At each step it
solves a short-horizon condensed QP over a linearized kinematic-bicycle lateral model:

    state z = [e_y, e_psi]   (lateral offset from the target-lane centre [m], heading error [rad])
    input  d = front-wheel steering [rad]
    e_y_dot   = v*e_psi + 0.5*v*d      (small-angle slip beta ~ 0.5 d, Lf=Lr)
    e_psi_dot = (v/VEHLENGTH)*d

    min_D  sum_k  q_y e_y[k]^2 + q_psi e_psi[k]^2 + r d[k]^2 + r_d (d[k]-d[k-1])^2
    s.t.   |d[k]| <= d_max,   |lateral accel| <= a_lat_max  (soft, via r on d given a_lat ~ 0.5 v^2 d /L)

Solved by condensing z over the horizon (z = Sx z0 + Su D) into an unconstrained QP, taking the
analytic minimiser and projecting the applied steer onto the box -- fast (pure numpy, no solver),
receding-horizon. Only the first input is applied.
"""
import numpy as np
from _constants import VEHLENGTH

import os
_D_MAX = 35.0 * np.pi / 180.0        # steering limit [rad]
_N = int(os.environ.get("MPC_N", "40"))          # prediction horizon (~2 s at dt=0.05)
# effort penalty is scaled by v^4 so the *lateral acceleration* (~ v^2 * d) is what is penalized,
# giving speed-appropriate gentleness (steer less at higher speed). Tuned for a ~2.4 s lane change
# at ~2.8 m/s^2 peak lateral acceleration.
_Q_Y  = float(os.environ.get("MPC_QY",  "1.0"))
_Q_PSI = float(os.environ.get("MPC_QPSI", "20.0"))
_R    = float(os.environ.get("MPC_R",   "0.03"))     # multiplies (v^2 d)^2
_R_D  = float(os.environ.get("MPC_RD",  "0.05"))


def _condense(v, dt, N):
    """Build z_{1..N} = Sx z0 + Su D for the linearized lateral model at speed v."""
    A = np.array([[1.0, v * dt], [0.0, 1.0]])
    B = np.array([[0.5 * v * dt], [v * dt / VEHLENGTH]])
    nx = 2
    Sx = np.zeros((nx * N, nx))
    Su = np.zeros((nx * N, N))
    Apow = np.eye(nx)
    # precompute A^k
    Apows = [np.eye(nx)]
    for _ in range(N):
        Apows.append(A @ Apows[-1])
    for i in range(N):
        Sx[nx * i:nx * (i + 1), :] = Apows[i + 1]
        for j in range(i + 1):
            Su[nx * i:nx * (i + 1), j:j + 1] = Apows[i - j] @ B
    return Sx, Su


def mpc_steer(v, e_y, e_psi, dt, prev_d=0.0):
    """Return the MPC steering command [rad] to drive (e_y, e_psi) -> 0 at speed v."""
    v = max(float(v), 0.5)                 # avoid degenerate B at v~0
    N = _N
    Sx, Su = _condense(v, dt, N)
    z0 = np.array([e_y, e_psi])
    # state weights (block-diagonal over horizon)
    qvec = np.tile(np.array([_Q_Y, _Q_PSI]), N)
    Q = np.diag(qvec)
    # input effort + rate weight; scale by v^4 so we penalize lateral acceleration (~v^2 d), not
    # raw steering -> gentler, speed-appropriate maneuvers.
    reff = _R * (v * v) ** 2
    rdff = _R_D * (v * v) ** 2
    R = reff * np.eye(N)
    D2 = np.eye(N) - np.eye(N, k=-1)       # first difference (d[k]-d[k-1])
    Rd = rdff * (D2.T @ D2)
    # condensed cost: 1/2 D' H D + f' D  (dropping const)
    H = Su.T @ Q @ Su + R + Rd
    f = Su.T @ Q @ (Sx @ z0)
    # rate term also couples to prev_d through d[0]-prev_d
    f[0] += rdff * (-prev_d)
    # unconstrained minimiser, then saturate the applied steer
    try:
        D = np.linalg.solve(H, -f)
    except np.linalg.LinAlgError:
        D = -np.linalg.pinv(H) @ f
    d0 = float(np.clip(D[0], -_D_MAX, _D_MAX))
    return d0


if __name__ == "__main__":
    # self-test: 3.6 m lane change at several speeds. Report duration, peak steer, peak lateral accel.
    LANEW = 3.6
    dt = 0.05
    for v in (15.0, 20.0, 25.0):
        e_y, e_psi, d_prev = LANEW, 0.0, 0.0
        peak_d = 0.0; peak_alat = 0.0; t_done = None
        for k in range(200):
            d = mpc_steer(v, e_y, e_psi, dt, d_prev)
            beta = 0.5 * d
            alat = (v * v / VEHLENGTH) * beta      # lateral acceleration [m/s^2]
            e_y += dt * (v * e_psi + v * beta)
            e_psi += dt * (v / VEHLENGTH) * beta
            d_prev = d
            peak_d = max(peak_d, abs(d)); peak_alat = max(peak_alat, abs(alat))
            if t_done is None and abs(e_y) < 0.15:
                t_done = k * dt
        print(f"v={v:4.1f}  LC_time={t_done}  peak_steer={peak_d*180/np.pi:5.2f} deg  "
              f"peak_lat_acc={peak_alat:4.2f} m/s^2  final_e_y={e_y:.3f}")
