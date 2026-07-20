import numpy as np
import os
from _constants import *
import numpy as np
from _sensor import getOpenLanes, getSideVeh, getRearVeh, getFrontVeh, getFrontObs, getFrontTL

class ObservationBuilder:
    def __init__(
        self,
        num_lanes,
        local_back=40.0,
        local_front=80.0,
        cell_size=5.0,
        remote_front=300.0,
        v_max=35.0,
        a_max=8.0,
        ldot_max=2.0,
        s_max=300.0,
    ):
        self.num_lanes = num_lanes
        # EGO-CENTRIC observation: the local grid and remote vector use a window of
        # lanes CENTERED on the ego (current +/- half), so the observation dimension
        # is FIXED (3 lanes) regardless of the road's lane count -> lane-agnostic.
        self.grid_lanes = 3      # current +/- 1
        self.remote_lanes = 3    # current +/- 1 (far field)
        self.local_back = local_back
        self.local_front = local_front
        self.cell_size = cell_size
        self.remote_front = remote_front

        self.num_cells = int((local_back + local_front) / cell_size)
        self.v_max = v_max
        self.a_max = a_max
        self.ldot_max = ldot_max
        self.s_max = s_max
        # ABSOLUTE-ENCODING ablation (counterfactual to ego-centric): index lanes by
        # absolute lane number into a fixed 5-wide grid, and expose the ego's absolute
        # lane as a feature. Ties behavior to absolute lane position, so a policy trained
        # at one lane count should not transfer. Enabled by ABS_OBS=1.
        self._abs_obs = os.environ.get("ABS_OBS", "0") == "1"
        if self._abs_obs:
            self.grid_lanes = 5
            self.remote_lanes = 5

        # channels:
        # 0 occupancy
        # 1 rel speed
        # 2 rel accel
        # 3 type code
        # 4 lane-change intent (binary: is the occupant changing lanes)
        # 5 normalized rel distance
        # 6 (INTENT_OBS only) SIGNED lane-change intent of CONNECTED vehicles (V2V broadcast):
        #   -1 right / +1 left / 0 none; 0 for humans (they don't broadcast intent). Lets a CAV
        #   see which way a nearby connected vehicle is committed -> learn to yield on a shared gap.
        self._intent_obs = os.environ.get("INTENT_OBS", "0") == "1"
        self.num_channels = 7 if self._intent_obs else 6

        # Fixed-K nearest-neighbor observation (standard/vanilla QMIX baseline): replace the
        # ego-centric CNN grid with the K nearest vehicles' features (pure local). Enabled by
        # OBS_MODE=knn; K set by KNN (default 6). F=6 features/neighbor -> local_grid (K*F,1,1).
        self._knn = os.environ.get("OBS_MODE", "") == "knn"
        self._K = int(os.environ.get("KNN", "6"))
        self._knn_F = 6

        # LOCAL_ONLY=1 zeros the remote (V2V) channel, restricting perception to the local
        # window -> tests whether CNN-QMIX's extra lane changes come from the remote CAV
        # information rather than the CNN representation (2x2 ablation).
        self._local_only = os.environ.get("LOCAL_ONLY", "0") == "1"

        # PLATOON-LOCK (USE_PLATOON_LOCK=1): once a CAV is in a platoon, the worth gate no
        # longer grants a lane change merely because another connected vehicle is nearby in
        # the target lane (platoon-hopping); an already-platooned CAV may only change lanes
        # for a significant speed/gap benefit. Reduces platoon-breaking lane changes.
        self._platoon_lock = os.environ.get("USE_PLATOON_LOCK", "0") == "1"

    def _clip_norm(self, x, scale):
        return np.clip(x / scale, -1.0, 1.0)

    def _veh_type_code(self, veh_type):
        # VEHTYPE.NONE=-1, MOBIL=0, CAV or DRLCAV=1
        if veh_type == VEHTYPE.MOBIL:
            return 0.0
        elif veh_type == VEHTYPE.CAV or veh_type == VEHTYPE.DRLCAV:
            return 1.0
        else:
            return -1.0

    def _tl_code(self, tl_status):
        if tl_status == LIGHTSTATUS.RED:
            return -1.0
        elif tl_status == LIGHTSTATUS.AMBER:
            return 0.0
        return 1.0

    def build_local_grid(self, ego, svs):
        # Ego-centric grid: rows are lanes RELATIVE to the ego (0=right, 1=current,
        # 2=left for grid_lanes=3), so the grid is fixed-size on any road.
        grid = np.zeros((self.num_channels, self.grid_lanes, self.num_cells), dtype=np.float32)
        half = self.grid_lanes // 2
        ego_lane = int(round(ego.lane))

        for sv in svs:
            if sv.id == ego.id:
                continue

            ds = sv.s - ego.s
            if ds < -self.local_back or ds >= self.local_front:
                continue

            if self._abs_obs:
                row = int(round(sv.lane)) - 1          # ABSOLUTE lane index (lane 1 -> row 0)
                if row < 0 or row >= self.grid_lanes:
                    continue
            else:
                rel = int(round(sv.lane)) - ego_lane  # ego-relative lane offset
                if rel < -half or rel > half:
                    continue
                row = rel + half                      # 0..grid_lanes-1

            col = int((ds + self.local_back) / self.cell_size)
            col = np.clip(col, 0, self.num_cells - 1)

            # if multiple vehicles fall in same cell, keep nearest in |ds|
            prev_occ = grid[0, row, col]
            prev_abs_dist = abs(grid[5, row, col] * self.local_front) if prev_occ > 0 else 1e9
            if prev_occ > 0 and abs(ds) >= prev_abs_dist:
                continue

            grid[0, row, col] = 1.0
            grid[1, row, col] = self._clip_norm(sv.v - ego.v, self.v_max)
            grid[2, row, col] = self._clip_norm(sv.a - ego.a, self.a_max)
            grid[3, row, col] = self._veh_type_code(sv.type)
            grid[4, row, col] = float(abs(getattr(sv, "ul", sv.lane) - sv.lane) > 1e-3)
            grid[5, row, col] = self._clip_norm(ds, self.local_front)
            if self._intent_obs:
                # signed committed intent, connected vehicles only (V2V intent sharing)
                if sv.type in (VEHTYPE.CAV, VEHTYPE.DRLCAV):
                    grid[6, row, col] = float(np.clip(getattr(sv, "ul", sv.lane) - sv.lane, -1.0, 1.0))

        return grid

    def is_in_platoon(self, ego, svs):
        connected_types = [VEHTYPE.CAV, VEHTYPE.DRLCAV]

        same_lane = [
            sv for sv in svs
            if sv.id != ego.id
            and sv.lane == ego.lane
        ]

        if len(same_lane) == 0:
            return 0.0

        front_vehs = [sv for sv in same_lane if sv.s > ego.s]
        rear_vehs = [sv for sv in same_lane if sv.s < ego.s]

        front = min(front_vehs, key=lambda x: x.s - ego.s) if len(front_vehs) > 0 else None
        rear = max(rear_vehs, key=lambda x: x.s) if len(rear_vehs) > 0 else None

        if front is not None and front.type in connected_types:
            return 1.0

        if rear is not None and rear.type in connected_types:
            return 1.0

        return 0.0

    def build_remote_vector(self, ego, svs):
        # Ego-relative per-lane summary farther ahead than local front. Lanes are
        # relative to the ego (current +/- half) so the vector is fixed-size on any road.
        feat = []
        half = self.remote_lanes // 2
        ego_lane = int(round(ego.lane))

        lane_iter = (range(1, self.remote_lanes + 1) if self._abs_obs
                     else (ego_lane + rel for rel in range(-half, half + 1)))
        for lane in lane_iter:
            lane_vehs = [
                sv for sv in svs
                if sv.id != ego.id
                and int(round(sv.lane)) == lane
                and (sv.s - ego.s) >= self.local_front
                and (sv.s - ego.s) <= self.remote_front
            ]

            if len(lane_vehs) == 0:
                feat.extend([0.0, 0.0, 0.0, 0.0])
                continue

            ds_list = np.array([sv.s - ego.s for sv in lane_vehs], dtype=np.float32)
            v_list = np.array([sv.v for sv in lane_vehs], dtype=np.float32)
            a_list = np.array([sv.a for sv in lane_vehs], dtype=np.float32)
            cav_count = np.sum([1 if sv.type in [VEHTYPE.CAV, VEHTYPE.DRLCAV] else 0 for sv in lane_vehs])

            feat.extend([
                self._clip_norm(np.mean(v_list) - ego.v, self.v_max),
                self._clip_norm(np.mean(a_list), self.a_max),
                np.clip(cav_count / 10.0, 0.0, 1.0),
                self._clip_norm(np.min(ds_list), self.s_max),
            ])

        return np.array(feat, dtype=np.float32)

    

    def build_ego_vector(self, ego, svs, tls, obs, divs):
        dist_to_tl, tl_status, _ = getFrontTL(ego.s, ego.len, tls)
        dist_to_obs, _ = getFrontObs(ego.s, ego.len, ego.lane, obs)
        open_lanes = getOpenLanes(ego.s, ego.numLanes, obs=obs, l=ego.lane, divs=divs)

        left_available = 1.0 if ego.lane < max(open_lanes) else 0.0
        right_available = 1.0 if ego.lane > min(open_lanes) else 0.0

        in_platoon = self.is_in_platoon(ego, svs)

        # save status into vehicle object for logging
        if hasattr(ego, "type") and ego.type == VEHTYPE.DRLCAV:
            ego.in_platoon = in_platoon

        abs_lane_feat = [((ego.lane - 1) / 4.0)] if self._abs_obs else []
        ego_vec = np.array(abs_lane_feat + [
            self._clip_norm(ego.v, self.v_max),
            self._clip_norm(ego.a, self.a_max),
            # absolute lane position dropped (lane-agnostic); left/right_available
            # below already encode whether a lane exists on each side.
            np.clip(ego.l - ego.lane, -1.0, 1.0),
            self._clip_norm(ego.ldot, self.ldot_max),
            self._clip_norm(dist_to_obs, self.s_max),
            self._clip_norm(dist_to_tl, self.s_max),
            self._tl_code(tl_status),
            left_available,
            right_available,
            in_platoon,
            # platoon-hold duration as a fraction of the 3 s sustain threshold
            # (30 steps @10Hz), clipped to [0,1]. Makes the sustain reward observable
            # so the agent knows how close it is to (and whether it is past) threshold.
            min(getattr(ego, "platoon_hold_steps", 0) / 30.0, 1.0),
        ], dtype=np.float32)

        return ego_vec
        
    def _lane_change_safe(self, ego, svs, target_lane, ttc_thr=2.5):
        """
        Safety shield for a lane change into target_lane. Safe only if:
          - the side is clear (getSideVeh), AND
          - target-lane front gap >= max(10, v*1.0) and ego is not closing on it
            within ttc_thr seconds, AND
          - target-lane rear gap >= max(10, v*0.6) and the rear car is not closing
            on ego within ttc_thr seconds.
        Gap covers the low-relative-speed (tailgating) case; TTC covers the
        high-closing-speed case. Both must pass.
        """
        if not getSideVeh(svs, ego.x, ego.y, ego.theta, target_lane):
            return False

        # front vehicle in target lane
        fds, fv, fi = getFrontVeh(svs, ego.x, ego.y, ego.len, ego.theta, target_lane)
        if fi >= 0:
            if fds < max(10.0, ego.v * 1.0):
                return False
            closing = ego.v - fv
            if closing > 1e-3 and (fds / closing) < ttc_thr:
                return False

        # rear vehicle in target lane
        rds, rv, ri = getRearVeh(svs, ego.x, ego.y, ego.theta, target_lane)
        if ri >= 0:
            if abs(rds) < max(10.0, ego.v * 0.6):
                return False
            closing = rv - ego.v
            if closing > 1e-3 and (abs(rds) / closing) < ttc_thr:
                return False

        return True

    def _lane_change_worth(self, ego, svs, target_lane,
                           platoon_range=50.0, gain_spd=1.0, gain_gap=10.0):
        """
        Incentive gate (MOBIL-style): allow a change into target_lane only if it is likely
        to PAY OFF -- either
          (a) a connected vehicle (CAV/DRLCAV) is reachable ahead OR behind in the target
              lane within platoon_range  -> platoon-forming opportunity, OR
          (b) the target lane offers a speed/gap advantage over the current lane (faster
              front vehicle, or markedly larger front gap = room to accelerate).
        If the ego is already platooned the speed/gap bar is raised (platoon retention: do
        not leave a platoon for a marginal gain). Blocks the "no-benefit" lane changes the
        LC audit flagged (~44% baseline / ~59% low-penetration). Safety is checked separately.
        """
        connected = (VEHTYPE.CAV, VEHTYPE.DRLCAV)
        in_platoon = self.is_in_platoon(ego, svs) > 0.5
        # PLATOON-LOCK: an already-platooned CAV does NOT get the platoon-opportunity
        # allowance (a) -- it may only leave for a significant speed/gap benefit (b).
        locked = self._platoon_lock and in_platoon
        tfds, tfv, tfi = getFrontVeh(svs, ego.x, ego.y, ego.len, ego.theta, target_lane)
        # (a) platoon opportunity: connected vehicle close ahead/behind in the target lane
        if not locked:
            if tfi >= 0 and svs[tfi].type in connected and tfds <= platoon_range:
                return True
            trds, trv, tri = getRearVeh(svs, ego.x, ego.y, ego.theta, target_lane)
            if tri >= 0 and svs[tri].type in connected and abs(trds) <= platoon_range:
                return True
        # (b) speed/gap incentive vs the current lane
        cfds, cfv, cfi = getFrontVeh(svs, ego.x, ego.y, ego.len, ego.theta, ego.lane)
        cur_gap = cfds if cfi >= 0 else 1e9
        cur_spd = cfv if cfi >= 0 else VD
        tgt_gap = tfds if tfi >= 0 else 1e9
        tgt_spd = tfv if tfi >= 0 else VD
        spd_bar = 2.0 if in_platoon else gain_spd          # harder to leave a platoon
        gap_bar = 20.0 if in_platoon else gain_gap
        if (tgt_spd - cur_spd) >= spd_bar or (tgt_gap - cur_gap) >= gap_bar:
            return True
        return False

    def _partner_gate_allows(self, ego, svs, target_lane, fwd=300.0, rear=40.0):
        """
        Eval-time PARTNER GATE (USE_PARTNER_GATE). Targets the irrational low-MPR lane changes:
        a change is allowed ONLY if either (a) a connected vehicle (CAV/DRLCAV) is observable
        anywhere within the policy's horizon ([-rear, +fwd] m, +-1 lane) -> partner-seeking is
        justified, OR (b) the target lane offers a real speed/gap gain (legitimate overtake).
        Otherwise (no partner in view AND no incentive) the change is purposeless -> block.
        Fires only in the no-partner-visible state, so it never touches ID / high-MPR behavior
        where a partner is almost always visible.
        """
        for o in svs:
            if o.type in (VEHTYPE.CAV, VEHTYPE.DRLCAV):
                dlon = o.s - ego.s
                if -rear <= dlon <= fwd and abs(round(o.lane) - round(ego.lane)) <= 1:
                    return True
        cfds, cfv, cfi = getFrontVeh(svs, ego.x, ego.y, ego.len, ego.theta, ego.lane)
        tfds, tfv, tfi = getFrontVeh(svs, ego.x, ego.y, ego.len, ego.theta, target_lane)
        cur_gap = cfds if cfi >= 0 else 1e9
        cur_spd = cfv if cfi >= 0 else VD
        tgt_gap = tfds if tfi >= 0 else 1e9
        tgt_spd = tfv if tfi >= 0 else VD
        return (tgt_spd - cur_spd) >= 1.0 or (tgt_gap - cur_gap) >= 10.0

    def build_action_mask(self, ego, svs, obs, divs):
        """
        Action mask:
            0 -> change right
            1 -> keep lane
            2 -> change left

        1.0 means allowed
        0.0 means masked out
        """
        mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Incentive gate (_lane_change_worth) is toggleable: USE_WORTH_GATE=0 disables it
        # (V2 / committed-HEAD config = open-lane + safety shield only). Default "0" = V2 (gate off);
        # current behavior. Safety shield (_lane_change_safe) is ALWAYS enforced.
        use_worth = os.environ.get("USE_WORTH_GATE", "0") == "1"
        # Partner gate: blocks purposeless LCs when no connected veh is visible AND no
        # speed/gap gain. Eval-time only (default off); targets low-MPR wasteful LCs.
        use_partner = os.environ.get("USE_PARTNER_GATE", "0") == "1"

        lc = getOpenLanes(ego.s, ego.numLanes, obs=obs, l=ego.lane, divs=divs)

        # right lane action -> index 0 (must be open, safe, AND -- if gated -- worth it)
        right_lane = ego.lane - 1
        if (right_lane not in lc) or (not self._lane_change_safe(ego, svs, right_lane)) \
           or (use_worth and not self._lane_change_worth(ego, svs, right_lane)) \
           or (use_partner and not self._partner_gate_allows(ego, svs, right_lane)):
            mask[0] = 0.0

        # keep lane -> index 1 (always allowed -> never deadlocks)
        mask[1] = 1.0

        # left lane action -> index 2 (must be open, safe, AND -- if gated -- worth it)
        left_lane = ego.lane + 1
        if (left_lane not in lc) or (not self._lane_change_safe(ego, svs, left_lane)) \
           or (use_worth and not self._lane_change_worth(ego, svs, left_lane)) \
           or (use_partner and not self._partner_gate_allows(ego, svs, left_lane)):
            mask[2] = 0.0

        return mask
        
    def build_knn_obs(self, ego, svs):
        """Fixed-K nearest-neighbor feature block for the standard-QMIX baseline. Picks the
        K nearest vehicles by Euclidean distance (longitudinal + lateral) and encodes each as
        [valid, Δs, Δlane, Δv, Δa, type_code] with the SAME normalizations as the CNN grid.
        Zero-padded to K. Shaped (K*F, 1, 1) so it drops into the [B,T,C,L,W] agent forward."""
        ego_lane = int(round(ego.lane))
        cand = []
        for sv in svs:
            if sv.id == ego.id:
                continue
            ds = sv.s - ego.s
            dlane = int(round(sv.lane)) - ego_lane
            dist = (ds * ds + (dlane * LANEWIDTH) ** 2) ** 0.5
            cand.append((dist, ds, dlane, sv))
        cand.sort(key=lambda z: z[0])

        feat = np.zeros((self._K, self._knn_F), dtype=np.float32)
        for i, (dist, ds, dlane, sv) in enumerate(cand[:self._K]):
            feat[i, 0] = 1.0                                        # valid slot
            feat[i, 1] = self._clip_norm(ds, self.local_front)     # Δs
            feat[i, 2] = np.clip(dlane / 2.0, -1.0, 1.0)           # Δlane (±2 lanes)
            feat[i, 3] = self._clip_norm(sv.v - ego.v, self.v_max) # Δv
            feat[i, 4] = self._clip_norm(sv.a - ego.a, self.a_max) # Δa
            feat[i, 5] = self._veh_type_code(sv.type)              # 0 human / 1 connected
        return feat.reshape(self._K * self._knn_F, 1, 1)

    def build_observation(self, ego, svs, tls, obs, divs):
        ego_vec = self.build_ego_vector(ego, svs, tls, obs, divs)
        local_grid = self.build_knn_obs(ego, svs) if self._knn else self.build_local_grid(ego, svs)
        remote_vec = self.build_remote_vector(ego, svs)
        if self._local_only:
            remote_vec = np.zeros_like(remote_vec)
        action_mask = self.build_action_mask(ego, svs, obs, divs)

        return {
            "ego": ego_vec,
            "local_grid": local_grid,
            "remote": remote_vec,
            "action_mask": action_mask
        }

