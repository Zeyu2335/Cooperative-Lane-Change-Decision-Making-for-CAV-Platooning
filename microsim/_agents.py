#! /usr/bin/env python3

from math import sin, cos, sqrt, tan, atan, atan2, fmod, pi, inf
from scipy.integrate import odeint
import numpy as np
import os

from _constants import *
from _sensor import *
_K_FF   = float(os.environ.get("CACC_KFF", "1.0"))   # CACC feedforward gain (1=full replication, 0=off)
_FF_TAU = float(os.environ.get("CACC_TAU", "0.2"))   # CACC feedforward filter time constant [s]
_CACC_SMOOTH = float(os.environ.get("CACC_SMOOTH", "0.0"))  # attenuating CACC: command low-pass time const [s]
_PLOEG = os.environ.get("USE_PLOEG_CACC","0")=="1"   # string-stable CACC (Ploeg 2011): Gamma(s)=1/(h s+1)
from collections import deque

# --- Human reaction delay (env-gated) --------------------------------------
# Humans perceive the leader gap/speed with a lag (stale info -> overshoot -> more
# acceleration churn -> more energy). CAVs are unaffected.
_HUMAN_DELAY = float(os.environ.get("HUMAN_DELAY", "0"))   # seconds

# --- Cooperative gap-opening (env-gated) -----------------------------------
# CAVs open a gap proactively for CONNECTED mergers (react to a neighbor V2V lane
# command `ul`); humans yield reactively once a neighbor physically STARTS a merge
# (its lateral coordinate `l` has moved toward the ego lane).
_COOP_GAP    = os.environ.get("USE_COOP_GAP", "0") == "1"
_COOP_WINDOW = float(os.environ.get("COOP_GAP_WINDOW", "30.0"))   # longitudinal look-ahead [m]
_COOP_LAT    = float(os.environ.get("COOP_GAP_LATERAL", "0.15"))  # lateral start threshold [lane]


class _vehicle:
    '''Virtual vehicle super class'''

    def __init__(self, x, y, s, v, a, theta, id, lane, numLanes, obs, divs):
        # Vehicle states
        self.x = x # Rear axis x gps coordinate
        self.y = y # Rear axis y gps coordinate
        self.v = v # Forward velocity
        self.a = a # Forward acceleration
        self.theta = theta # Heading angle in gps frame
        self.s = s # Rear axis longitudinal position from origin of road
        self.id = id # vehicle ID

        self.lane = lane # Current most occupied lane - 1 is rightmost
        self.l = (float)(lane) # Lateral position from right shoulder - 1.0 is centerline of right lane, 2.0 is centerline of second lane, ...
        self.ldot = 0. # Rate of change of l
        self.LC = 0 # Number of lane change

        # Vehicle characteristics
        self.len = VEHLENGTH # Length of vehicle from axle to axle
        self.wid = VEHWIDTH # Width of vehicle
        self.in_platoon = 0
        

        self.type = VEHTYPE.NONE

        # Road characteristics
        self.numLanes = numLanes # Num of lanes on road
        self.obs = obs
        self.divs = divs



class MOBIL(_vehicle):
    '''MOBIL driver for virtual vehicles'''
    def __init__(self, x=0., y=0., s=0., v=0., a=0., theta=0., id=1, lane=1, numLanes=1, obs=[], divs=[], 
                 vd = 10., ad = 2., bd = 3., delta = 4., T = 1.2, d = 2.8, active=False):
        super().__init__(x, y, s, v, a, theta, id, lane, numLanes, obs, divs)

        self.type = VEHTYPE.MOBIL

        # IDM parameters
        self.vd = vd # Desired velocity
        self.ad = ad # IDM acceleration 
        self.bd = bd # IDM deceleration
        self.delta = delta # IDM acceleration exponent
        self.T = T # Desired time headway
        self.d = d # Desired standstill gap
        self.t = 0
        self.safe = 0
        
        self.ac = 1.5 # Comfortable deceleration to stop at intersections

        # MOBIL parameters
        self.pol = 0.8 # Politeness factor
        self.athr = 1 # Lane change threshold
        self.abias = 0.2 # Bias to prefer left lane - negative bias to prefer right lane
        self.bsafe = 3 # [m/s2] Maximal safe deceleration allowed for rear vehicle in target lane

        self.dtlc = 0.0 # Time before another lane change can be initiated - must be <0 before another lane change is allowed
        self.tlc = 7.0 # Time allowed between lane changes
        self.dtlcnear = 0.0 # Time before another lane change can be initiated - triggered from a surrounding vehicle lane changing - must be <0 before another lane change is allowed
        self.tlcnear = 2.5 # Time allowed between lane changes due to nearby surrounding vehicles

        self.ld = self.lane # Desired lane
        self.active = active # True/false to allow lane changes to activate
        self.vminlc = 5.5 # [m/s] Minimum velocity before non-emergency lane change decisions are allowed

        # Controls
        self.ua = 0. # Acceleration command
        self.ul = (float)(self.lane) # Lane command

        # Debug
        self.verbose = False # Verbose printing to the cl of the internal decision logic

    def getNearLC(self, svs, r=25.):
        '''Check if nearby vehicles have initiated a lane change'''
        nearLC = False

        for k in range(0, len(svs)):
            if abs(svs[k].s-self.s <= r) and (svs[k].ul - svs[k].lane) > 0:
                nearLC = True
                
        return nearLC
        
    def getAccel(self, ds, v, vf, vd, lightstatus = LIGHTSTATUS.GREEN, lightds = 2000.0, obsds = 2000.0):
        '''Get acceleration command considering neighboring vehicle and traffic light interactions'''
        # Light interaction
        ulight = self.ad
        if lightds > -1.0 and lightstatus is not LIGHTSTATUS.GREEN:
            dt = v/self.ac
            delta = (0.5*(vd+vd))**2./(2.*self.ac)

            if lightds <= delta + self.d and (lightstatus is LIGHTSTATUS.RED or dt > 4.):
                ulight = 0.1833*self.ac

                if lightds < ds:
                    ds = lightds-2.
                    vf = 0.

        # Obstacle interaction
        uobs = self.ad
        if obsds < 1:
            aa = 1
        if obsds > -1.0 and obsds < ds:
            uobs = 0.1833*self.ac
            ds = obsds
            vf = 0.

        # Vehicle interaction
        dv = v-vf # Positive if approaching front vehicle
        sstar = ( self.d+max(0., v*self.T+v*dv/sqrt(4.*self.ad*self.bd)) )

        uveh = self.ad * (1 - (v/vd)**self.delta - ( sstar/(max(ds, 0.01)) )**2.)
        
        return min(uveh, ulight, uobs)

    def _delayed_perception(self, ds, vr, dt):
        """Return (ds, vr) as perceived _HUMAN_DELAY seconds ago (per-vehicle ring buffer).
        During warm-up (buffer not yet full) returns the current values."""
        nd = max(1, int(round(_HUMAN_DELAY / max(dt, 1e-6))))
        buf = getattr(self, "_perc_buf", None)
        if buf is None or buf.maxlen != nd:
            buf = deque(maxlen=nd)
            self._perc_buf = buf
        old = buf[0] if len(buf) == nd else (ds, vr)   # oldest retained = nd steps ago
        buf.append((ds, vr))
        return old

    def _coop_gap_accel(self, svs, mode):
        """Cooperative gap-opening: decelerate to open room for a merging neighbor.
        mode="cav"   proactive -- react to a CONNECTED neighbor whose V2V lane command
                     `ul` targets the ego lane (acts on intent, before/at merge start).
        mode="human" reactive  -- react only once a neighbor has physically STARTED the
                     merge (lateral coord `l` moved >_COOP_LAT toward the ego lane).
        Treats the closest qualifying merger ahead as a virtual leader and returns the
        IDM following accel (a deceleration that opens the gap), or None if nobody merges.
        """
        if not _COOP_GAP:
            return None
        ego_lane = int(round(self.lane))
        best_ds = None
        best_v = 0.0
        for o in svs:
            if o.id == self.id:
                continue
            origin = int(round(o.lane))
            if abs(origin - ego_lane) != 1:          # merger must come from an adjacent lane
                continue
            direction = ego_lane - origin            # +1/-1 toward the ego lane
            if mode == "cav":
                if o.type not in (VEHTYPE.CAV, VEHTYPE.DRLCAV):
                    continue                         # only open gaps for connected mergers
                if int(round(o.ul)) != ego_lane:     # V2V command must target the ego lane
                    continue
            else:                                    # human: only after the merge visibly starts
                if (o.l - origin) * direction < _COOP_LAT:
                    continue
            dsig = o.s - self.s                      # merger becomes the ego leader when ahead
            if dsig < -2.0 or dsig > _COOP_WINDOW:
                continue
            ds_v = max(0.1, dsig - self.len)
            if best_ds is None or ds_v < best_ds:
                best_ds = ds_v
                best_v = o.v
        if best_ds is None:
            return None
        return self.getAccel(best_ds, self.v, best_v, self.vd)

    def is_target_lane_safe(self, svs, target_lane):
        min_front_gap = max(12.0, self.v * 1.2)
        min_rear_gap = max(10.0, self.v * 0.8)

        for veh in svs:
            if veh.lane != target_lane:
                continue

            gap = veh.s - self.s

            # front vehicle in target lane
            if gap > 0 and gap < min_front_gap:
                return False

            # rear vehicle in target lane
            if gap < 0 and abs(gap) < min_rear_gap:
                return False

        return True

    def setCommand(self, svs, tls, t, dt):
        '''Set acceleration command ua and lane command ul given interactions with svs and tls'''
        # Get nearest light in front, obstacle in front, and nearest vehicle in front - including check against ego vehicle that should have been appended to svs
        tlds, tlstatus, _ = getFrontTL(self.s, self.len, tls)
        obsds, obsk = getFrontObs(self.s, self.len, self.lane, self.obs)
        ds, vr, _ = getFrontVeh(svs, self.x, self.y, self.len, self.theta, self.lane)
        self.t = self.t + 1
        # Take action if desired speed is non-zero
        if self.vd > 0:
            ### Acceleration component
            if _HUMAN_DELAY > 0 and self.type == VEHTYPE.MOBIL:   # human reaction lag
                ds_d, vr_d = self._delayed_perception(ds, vr, dt)
                self.ua = self.getAccel(ds_d, self.v, vr_d, self.vd, tlstatus, tlds, obsds)
            else:
                self.ua = self.getAccel(ds, self.v, vr, self.vd, tlstatus, tlds, obsds)
            if self.type == VEHTYPE.MOBIL:                 # humans yield when a merge starts
                _cg = self._coop_gap_accel(svs, "human")
                if _cg is not None:
                    self.ua = min(self.ua, _cg)

            ### Lane change component
            # Valid lanes
            lc = getOpenLanes(self.s, self.numLanes, obs=self.obs, l=self.lane, divs=self.divs)

            # acc' acceleration after target lane decision
            # B' back vehicle at target lane decision
            self.dtlc -= dt

            if self.getNearLC(svs): # If another vehicle making a lane change then reset timer
                self.dtlcnear = self.tlcnear

            self.dtlcnear -= dt

            if self.active and (self.v > self.vminlc or obsk > -1) and self.dtlc < 0 and self.dtlcnear < 0:
                lutil = -inf
                rutil = -inf
                lcrit = False
                rcrit = False

                # Evaluate current lane
                dlane = max( min(0 + self.lane, self.numLanes), 1 )

                apr, ar = 0., 0.
                rds, rv, ri = getRearVeh(svs, self.x, self.y, self.theta, dlane)
                if ri >= 0: # Evaluate rear vehicle utility in lane
                    rtlds, rtlstatus, _ = getFrontTL(svs[ri].s, svs[ri].len, tls)
                    if rtlds > 0:
                        fds, fv, _ = getFrontVeh(svs, self.x, self.y, self.len, self.theta, dlane)

                        apr = self.getAccel(fds-rds, rv, fv, max(self.vd, rv), rtlstatus, rtlds) # acc'(B)
                        ar = self.getAccel(-rds, rv, self.v, max(self.vd, rv), rtlstatus, rtlds) # acc(B)
                
                # Evaluate left lane
                if self.lane < max(lc):
                    dlane = max(min(1 + self.lane, self.numLanes), 1)

                    aprp = 0.
                    rds, rv, ri = getRearVeh(svs, self.x, self.y, self.theta, dlane)
                    if ri >= 0: # Evaluate rear vehicle utility in lane
                        rtlds, rtlstatus, _ = getFrontTL(svs[ri].s, svs[ri].len, tls)
                        if rtlds > 0:
                            aprp = self.getAccel(-rds, rv, self.v, max(self.vd, rv), rtlstatus, rtlds) # acc'(B')

                    lsafe = (
                                (aprp > -self.bsafe)
                                and getSideVeh(svs, self.x, self.y, self.theta, dlane)
                                and self.is_target_lane_safe(svs, dlane)
                            )

                    lcrit = lsafe
                    
                    if lsafe:
                        fds, fv, _ = getFrontVeh(svs, self.x, self.y, self.len, self.theta, dlane)
                        ap = self.getAccel(fds, self.v, fv, self.vd, tlstatus, tlds) # acc'

                        arp = 0.
                        if ri >= 0:
                            rtlds, rtlstatus, _ = getFrontTL(svs[ri].s, svs[ri].len, tls)
                            if rtlds > 0:
                                fds, fv, _ = getFrontVeh(svs, self.x, self.y, self.len, self.theta, dlane)

                                arp = self.getAccel(fds-rds, rv, fv, max(self.vd, rv), rtlstatus, rtlds) # acc'(B)

                        # Incentive criterion
                        lcrit = ( ap - self.ua > self.pol*( (self.lane>min(lc))*(ar-apr) + arp - aprp ) + self.athr)
                        lutil = ( ap - self.ua + self.pol*( (self.lane>min(lc))*(ar-apr) + arp - aprp ) + self.abias)
                        
                # Evaluate right lane
                if self.lane > min(lc):
                    dlane = max(min(-1 + self.lane, self.numLanes), 1)

                    aprp = 0.
                    rds, rv, ri = getRearVeh(svs, self.x, self.y, self.theta, dlane)
                    if ri >= 0: # Evaluate rear vehicle utility in lane
                        rtlds, rtlstatus, _ = getFrontTL(svs[ri].s, svs[ri].len, tls)
                        if rtlds > 0:
                            aprp = self.getAccel(-rds, rv, self.v, max(self.vd, rv), rtlstatus, rtlds) # acc'(B')

                    rsafe = ((aprp > -self.bsafe) 
                            and getSideVeh(svs, self.x, self.y, self.theta, dlane)
                            and self.is_target_lane_safe(svs, dlane)
                            )

                    rcrit = rsafe
                    
                    if rsafe:
                        fds, fv, _ = getFrontVeh(svs, self.x, self.y, self.len, self.theta, dlane)
                        ap = self.getAccel(fds, self.v, fv, self.vd, tlstatus, tlds) # acc'

                        arp = 0.
                        if ri >= 0:
                            rtlds, rtlstatus, _ = getFrontTL(svs[ri].s, svs[ri].len, tls)
                            if rtlds > 0:
                                fds, fv, _ = getFrontVeh(svs, self.x, self.y, self.len, self.theta, dlane)

                                arp = self.getAccel(fds-rds, rv, fv, max(self.vd, rv), rtlstatus, rtlds) # acc'(B)

                        # Incentive criterion
                        rcrit = ( ap - self.ua > self.pol*( (self.lane>min(lc))*(ar-apr) + arp - aprp ) + self.athr )
                        rutil = ( ap - self.ua + self.pol*( (self.lane>min(lc))*(ar-apr) + arp - aprp ) )
                # -------------------------------------------------
                # No lane change near traffic light/intersection
                # -------------------------------------------------
                near_intersection = 0.0 <= tlds <= 80.0
                if near_intersection:
                    lcrit = False
                    rcrit = False    
                ### Get lane command
                le = 0
                if lcrit:
                    le = 1
                elif rcrit:
                    le = -1
                    
                if lcrit and rcrit: # both lanes are valid
                    if lutil - rutil > self.abias:
                        rcrit = False
                        le = 1
                    elif rutil - lutil > self.abias:
                        lcrit = False
                        le = -1
                    elif self.dtlc + 8.*self.tlc < 0: # Prefer to go back towards desired lane after an extended period of time
                        le = self.ld - self.lane
                    

                self.ul += le

                if le: # If ego deciding to make a lane change then reset timer
                    self.dtlc = self.tlc
                    self.LC += 1 

                if self.verbose:
                    print('\tlc: {:}, le: {:}, ar: {:0.2f}, apr: {:0.2f},\n\tlcrit: {:}, rcrit: {:}, lutil: {:0.2f}, rutil: {:0.2f}'.format(lc, le, ar, apr, lcrit, rcrit, lutil, rutil))

            # Saturate commands
            self.ua = min( max( -6.0, self.ua ), 2.0 ) # Prevent harsh acceleration or braking
            
            self.ul = min( max(self.ul, self.lane-1), self.lane+1 ) # Prevent multiple lane cross-over request
            self.ul = min( max(self.ul, min(lc)), max(lc) ) # Prevent out of bounds request
            
class CAV(MOBIL):
    """
    CAV driver:
    - ACC when preceding vehicle is HV/MOBIL
    - CACC when preceding vehicle is CAV/DRLCAV
    """

    def __init__(
        self, x=0., y=0., s=0., v=0., a=0., theta=0.,
        id=1, lane=1, numLanes=1, obs=[], divs=[],
        vd=10., active=False
    ):
        super().__init__(x, y, s, v, a, theta, id, lane, numLanes, obs, divs)

        self.type = VEHTYPE.CAV
        self.T = 1.2
        self.vd = vd
        self.active = active

        # control command
        self.ua = 0.0
        self.ul = float(self.lane)

        # CACC feedforward state
        self.fw = [0.0]
        self.prea = 0.0

        self.X = np.empty((5, 50))
        self.U = np.empty((2, 50))

        # controller gains (retuned for string stability: 2*k_rel*T_acc + k_gap*T_acc^2 >= 2)
        self.k_gap = 0.4
        self.k_rel = 0.6
        self.s0 = 4.2
        self.T_acc = float(os.environ.get("T_ACC", "1.5"))     # ACC headway (following a human) -- string-stable
        self.T_cacc = float(os.environ.get("T_CACC", "1.0"))    # CACC headway (following a connected vehicle) -- shorter, feedforward keeps it stable

        # acceleration limits
        self.acc_max = 2.0
        self.dec_max = -8.0

    def setCommand(self, svs, tls, t, dt):
        """
        Longitudinal controller:
        ACC/CACC with safe braking.
        """

        # keep MOBIL lane-change logic if active
        super().setCommand(svs, tls, t, dt)

        ds, vf, i = getFrontVeh(
            svs, self.x, self.y, self.len, self.theta, self.lane
        )

        # --------------------------------------------------
        # 1. If there is a preceding vehicle
        # --------------------------------------------------
        if i >= 0:
            front = svs[i]

            relative_speed = vf - self.v

            # ACC: preceding vehicle is human/MOBIL (headway T_acc)
            if front.type == VEHTYPE.MOBIL:
                spacing_error = ds - self.v * self.T_acc - self.s0
                acc_des = (
                    self.k_gap * spacing_error
                    + self.k_rel * relative_speed
                )

            # CACC: preceding vehicle is connected (shorter headway T_cacc + feedforward)
            else:
                spacing_error = ds - self.v * self.T_cacc - self.s0
                state = odeint(
                    self.derivatives_filter,
                    self.fw,
                    [0, dt],
                    args=(front.a,)
                ).T

                self.fw = state[-1]

                acc_des = (
                    self.k_gap * spacing_error
                    + self.k_rel * relative_speed
                    + _K_FF * self.fw[0]
                )
                if _PLOEG:              # Ploeg (2011) string-stable CACC: h*u_dot = -u + kp*e + kd*edot + u_lead
                    h = max(self.T_cacc, 1e-3)
                    e = ds - self.s0 - h * self.v                 # spacing error: gap - (r + h*v)
                    edot = relative_speed - h * self.a            # d/dt spacing error
                    u_prev = getattr(self, "u_cacc", acc_des)
                    acc_des = u_prev + (dt / h) * (-u_prev + self.k_gap * e + self.k_rel * edot + front.a)
                    self.u_cacc = acc_des
                if _CACC_SMOOTH > 0.0:              # attenuating (string-stable) CACC: low-pass the command
                    a_prev = getattr(self, "_a_cmd_prev", acc_des)
                    beta = dt / (_CACC_SMOOTH + dt)
                    acc_des = beta * acc_des + (1.0 - beta) * a_prev
                self._a_cmd_prev = acc_des

                self.prea = front.a

            # emergency braking if gap is very small
            min_safe_gap = max(2.0, 0.6 * self.v)

            if ds < min_safe_gap:
                # smooth emergency braking: ramp from the linear command to dec_max as the
                # gap collapses (continuous; reaches -8 only as ds -> 0, no bang-bang switch).
                frac = min(1.0, max(0.0, 1.0 - ds / max(min_safe_gap, 1e-3)))
                acc_des = (1.0 - frac) * acc_des + frac * self.dec_max

        # --------------------------------------------------
        # 2. If there is no preceding vehicle
        # --------------------------------------------------
        else:
            acc_des = 0.5 * (self.vd - self.v)

        # --------------------------------------------------
        # 3. Never overwrite braking just because v >= vd
        # --------------------------------------------------
        self.ua = acc_des

        _cg = self._coop_gap_accel(svs, "cav")             # open a gap for a connected merger
        if _cg is not None:
            self.ua = min(self.ua, _cg)

        # acceleration saturation
        self.ua = min(max(self.ua, self.dec_max), self.acc_max)

    def derivatives_filter(self, x, t, a_pre):
        """
        Low-pass filter for preceding vehicle acceleration.
        x: filtered acceleration state
        a_pre: raw preceding vehicle acceleration
        """
        tau = _FF_TAU
        return (a_pre - x) / tau

class DRLCAV(CAV):
    """
    DRLCAV:
    - ACC when preceding vehicle is HV/MOBIL
    - CACC when preceding vehicle is CAV/DRLCAV
    - lane-change decision comes from MARL
    - once a lane change starts, keep executing until finished
    """

    def __init__(
        self,
        x=0., y=0., s=0., v=0., a=0., theta=0.,
        id=1, lane=1, numLanes=1, obs=[], divs=[],
        vd=10., active=False
    ):
        super().__init__(
            x=x, y=y, s=s, v=v, a=a, theta=theta,
            id=id, lane=lane, numLanes=numLanes,
            obs=obs, divs=divs, vd=vd, active=active
        )

        self.type = VEHTYPE.DRLCAV
        self.in_platoon = 0.0

        # MARL action:
        # -1 = right, 0 = keep, +1 = left
        self.marl_action = 0

        # lane-change execution state
        self.is_lane_changing = False
        self.target_lane = self.lane
        self.did_lane_change = False

        # during-maneuver lane-change abort (eval-time safety layer). USE_LC_ABORT=1 enables
        # re-checking the target lane each step while crossing and aborting (steer back to the
        # origin lane) if a collision is IMMINENT -- fixes the "commit-and-forget" collisions.
        self._lc_abort = os.environ.get("USE_LC_ABORT", "0") == "1"
        self.lc_aborts = 0

        # cooldown
        self.dtlc = 0.0
        self.tlc = 7.0
        self.platoon_hold_steps = 0

        # repeated lane-change bookkeeping
        self.time_since_last_lc = 999.0
        self.repeated_lane_change = False
        self.repeat_lc_window = 8.0

        # controller parameters (retuned for string stability)
        self.k_gap = 0.4
        self.k_rel = 0.6
        self.s0 = 4.2
        self.T_acc = float(os.environ.get("T_ACC", "1.5"))     # ACC headway (following a human) -- string-stable
        self.T_cacc = float(os.environ.get("T_CACC", "1.0"))    # CACC headway (following a connected vehicle) -- shorter + feedforward
        self.acc_max = 2.0
        self.dec_max = -8.0

        # CACC feedforward state
        self.fw = [0.0]
        self.prea = 0.0

    def set_marl_action(self, action: int):
        """
        Set MARL lane action:
        -1 = right
         0 = keep
        +1 = left
        """
        self.marl_action = int(action)

    def _lc_collision_imminent(self, svs, tlane, ttc=1.2, hard_gap=None):
        """
        Collision-imminent check on the target lane, for the during-maneuver ABORT only.
        Uses TIGHT thresholds (about one vehicle length, or TTC < ~1.2 s) so it fires only when
        a crash is actually developing -- NOT the large gaps a normal platoon merge settles into
        (those would over-abort and cost platooning). Returns True if a target-lane vehicle is
        within hard_gap, or closing within ttc, ahead or behind.
        """
        if hard_gap is None:
            hard_gap = VEHLENGTH + 2.0
        fds, fv, fi = getFrontVeh(svs, self.x, self.y, self.len, self.theta, tlane)
        if fi >= 0:
            if fds < hard_gap:
                return True
            closing = self.v - fv
            if closing > 1e-3 and (fds / closing) < ttc:
                return True
        rds, rv, ri = getRearVeh(svs, self.x, self.y, self.theta, tlane)
        if ri >= 0:
            if abs(rds) < hard_gap:
                return True
            closing = rv - self.v
            if closing > 1e-3 and (abs(rds) / closing) < ttc:
                return True
        return False

    def lane_change_finished(self, tol=0.15):
        """
        Lane change is finished when lateral position is close to target lane center.
        """
        return abs(self.l - self.target_lane) <= tol

    def update_lane_change_status(self, tol=0.15):
        """
        Update lane-change execution status.
        """
        if self.is_lane_changing and self.lane_change_finished(tol=tol):
            self.is_lane_changing = False
            self.lane = int(round(self.target_lane))
            self.target_lane = self.lane

    def setCommand(self, svs, tls, t, dt):
        """
        Longitudinal:
        - ACC/CACC with safe braking

        Lateral:
        - MARL lane-change command
        - keep lane-change execution until finished
        """

        self.did_lane_change = False
        self.repeated_lane_change = False
        self.time_since_last_lc += dt
        self.update_lane_change_status()

        # ==================================================
        # 1. Longitudinal ACC/CACC control
        # ==================================================
        ds, vf, i = getFrontVeh(
            svs, self.x, self.y, self.len, self.theta, self.lane
        )

        if i >= 0:
            front = svs[i]

            relative_speed = vf - self.v

            # ACC: front vehicle is human/MOBIL (headway T_acc)
            if front.type == VEHTYPE.MOBIL:
                spacing_error = ds - self.v * self.T_acc - self.s0
                acc_des = (
                    self.k_gap * spacing_error
                    + self.k_rel * relative_speed
                )

            # CACC: front vehicle is connected vehicle (shorter headway T_cacc + feedforward)
            else:
                spacing_error = ds - self.v * self.T_cacc - self.s0
                state = odeint(
                    self.derivatives_filter,
                    self.fw,
                    [0, dt],
                    args=(front.a,)
                ).T

                self.fw = state[-1]

                acc_des = (
                    self.k_gap * spacing_error
                    + self.k_rel * relative_speed
                    + _K_FF * self.fw[0]
                )
                if _PLOEG:              # Ploeg (2011) string-stable CACC: h*u_dot = -u + kp*e + kd*edot + u_lead
                    h = max(self.T_cacc, 1e-3)
                    e = ds - self.s0 - h * self.v                 # spacing error: gap - (r + h*v)
                    edot = relative_speed - h * self.a            # d/dt spacing error
                    u_prev = getattr(self, "u_cacc", acc_des)
                    acc_des = u_prev + (dt / h) * (-u_prev + self.k_gap * e + self.k_rel * edot + front.a)
                    self.u_cacc = acc_des
                if _CACC_SMOOTH > 0.0:              # attenuating (string-stable) CACC: low-pass the command
                    a_prev = getattr(self, "_a_cmd_prev", acc_des)
                    beta = dt / (_CACC_SMOOTH + dt)
                    acc_des = beta * acc_des + (1.0 - beta) * a_prev
                self._a_cmd_prev = acc_des

                self.prea = front.a

            # emergency braking if gap is too small
            min_safe_gap = max(2.0, 0.6 * self.v)

            if ds < min_safe_gap:
                # smooth emergency braking: ramp from the linear command to dec_max as the
                # gap collapses (continuous; reaches -8 only as ds -> 0, no bang-bang switch).
                frac = min(1.0, max(0.0, 1.0 - ds / max(min_safe_gap, 1e-3)))
                acc_des = (1.0 - frac) * acc_des + frac * self.dec_max

        else:
            # no front vehicle: free-speed tracking
            acc_des = 0.5 * (self.vd - self.v)

        # Important:
        # do NOT set ua = 0 only because self.v >= self.vd
        self.ua = acc_des

        _cg = self._coop_gap_accel(svs, "cav")             # open a gap for a connected merger
        if _cg is not None:
            self.ua = min(self.ua, _cg)

        self.ua = min(max(self.ua, self.dec_max), self.acc_max)

        # ==================================================
        # 2. Lateral MARL lane-change control
        # ==================================================
        self.dtlc -= dt

        lc = getOpenLanes(
            self.s,
            self.numLanes,
            obs=self.obs,
            l=self.lane,
            divs=self.divs
        )

        # If already changing lane, continue to target lane -- unless a collision is now
        # imminent in the target lane, in which case ABORT: steer back to the origin lane.
        # (self.lane stays the origin lane until the maneuver finishes, so it is the abort target.)
        if self.is_lane_changing:
            if self._lc_abort and self.target_lane != self.lane \
               and self._lc_collision_imminent(svs, self.target_lane):
                self.target_lane = self.lane     # abort: revert destination to origin lane
                self.lc_aborts += 1
            self.ul = float(self.target_lane)

        else:
            requested_delta = int(self.marl_action)
            proposed_target_lane = self.lane + requested_delta

            if (
                self.active
                and self.dtlc <= 0.0
                and requested_delta != 0
                and proposed_target_lane in lc
                and abs(requested_delta) <= 1
            ):
                # start lane change
                self.is_lane_changing = True
                self.target_lane = proposed_target_lane
                self.ul = float(self.target_lane)

                self.did_lane_change = True

                if self.time_since_last_lc < self.repeat_lc_window:
                    self.repeated_lane_change = True

                self.time_since_last_lc = 0.0
                self.LC += 1
                self.dtlc = self.tlc

            else:
                # keep current lane
                self.ul = float(self.lane)

        # saturate lane command
        self.ul = min(max(self.ul, min(lc)), max(lc))

    def derivatives_filter(self, x, t, a_pre):
        """
        Low-pass filter for preceding vehicle acceleration.
        """
        tau = _FF_TAU
        return (a_pre - x) / tau

class TrafficLight:
    '''Traffic light fixed signal phase and timing for virtual intersections'''

    def __init__(self, s, x, y, theta, type, cycle, green, amber, phase, id = 1):
        # Light position
        self.s = s 
        self.x = x
        self.y = y
        self.theta = theta
        self.id = id

        # Light timings and current status
        self.status = LIGHTSTATUS.GREEN
        self.type = type
        self.cycle = cycle
        self.green = green
        self.amber = amber
        self.phase = phase

        # Unique light identifier
        # self.id = TrafficLight.id
        # TrafficLight.id += 1


    def setCommand(self, t):
        '''Get traffic intersection status'''
        if self.type == LIGHTTYPE.LIGHT:
            lt = fmod(t + self.phase, self.cycle)

            if lt < self.green:
                self.status = LIGHTSTATUS.GREEN
            elif lt < self.green + self.amber:
                self.status = LIGHTSTATUS.AMBER
            else:
                self.status = LIGHTSTATUS.RED

        elif self.type == LIGHTTYPE.STOP:
            self.status = LIGHTSTATUS.RED

        else:
            raise RuntimeError('Unknown light type')

