#! /usr/bin/env python3
import random
import time
import os
from math import sin, cos, sqrt, tan, atan, atan2, fmod, pi, inf

import numpy as np
from scipy.interpolate import interp1d

from _constants import *
from _sensor import *
from _mpc import mpc_steer
from _dynamics import *
from _animation import animation
from _logging import logger
from _agents import TrafficLight, MOBIL, CAV, DRLCAV
from _input import ObservationBuilder
from _drl_agent import DRLCAVAgent
from collections import deque

# np.random.seed(18)

def clipped_normal(mean, std, lower, upper, size):
    samples = np.random.normal(mean, std, size)
    return np.clip(samples, lower, upper)

def uniform_random_list(low, high, size, seed=None):
    if seed is None:
        # use the global (seeded) np.random stream so traffic is reproducible
        return np.random.uniform(low, high, size)
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size)

def generate_unique_numbers(start, end, count, min_diff):
    """
    Generate unique numbers within a given range ensuring a minimum difference.
    """
    if (end - start) < (count - 1) * min_diff:
        raise ValueError("Range too small to generate the required numbers with the given minimum difference.")
    numbers = []
    while len(numbers) < count:
        num = random.randint(start, end)

        # Check if it's valid (maintains min_diff)
        if all(abs(num - existing) >= min_diff for existing in numbers):
            numbers.append(num)
    return numbers  # Optional: Sort the numbers

# Class definitions
_USE_MPC_LC = os.environ.get("USE_MPC_LC", "0") == "1"  # MPC lateral control for CAV lane change (eval-time)
_ACT_LAG = os.environ.get("USE_ACT_LAG", "0") == "1"  # Ploeg drivetrain lag: a_dot=(u-a)/tau
_ACT_TAU = float(os.environ.get("ACT_TAU", "0.1"))

class simulation:
    '''Traffic simulation class that handles microsimulation'''
    def getEgoTrajXY(self, ego):
        '''Convert predicted state trajectory X into predicted x,y coordinate trajectory'''
        Xnp, Ynp = self.interpx(ego.X[0, :]), self.interpy(ego.X[0, :])

        for k in range(0, len(ego.X[1, :])):
            lnp = (ego.X[1, k]-1)*LANEWIDTH
            Xnp[k] -= lnp*sin(self.THRTK)
            Ynp[k] += lnp*cos(self.THRTK)

        Xnp[0], Ynp[0] = ego.x, ego.y # X is future predicted states x_k+1 - computation and refresh delay causes the signal to chatter so smooth signal visualization by replacing first element

        return Xnp, Ynp

    def setRoadDimensions(self):
        '''Set up road network assigning dimensions X, Y'''
        s0 = 0.
        self.SRTK = [s0]*len(self.XRTK)
        for i in range(1, len(self.XRTK)):
            s0 += sqrt((self.XRTK[i]-self.XRTK[i-1])**2 + (self.YRTK[i]-self.YRTK[i-1])**2)
            self.SRTK[i] = s0
        
        self.THRTK = atan2(self.YRTK[-1]-self.YRTK[0], self.XRTK[-1]-self.XRTK[0])

        self.interpx, self.interpy = interp1d(self.SRTK, self.XRTK, fill_value='extrapolate'), interp1d(self.SRTK, self.YRTK, fill_value='extrapolate')

    def setRoadGeometry(self):
        '''Set up road network assigning ramp geometry with Divider and Obstacle structures'''
        self.divs = [ [], [], [], [] ] # Default have no merging scenario
        self.obs = [ [], [], [], [] ] # Default have no merging scenario

        if self.geometry == GEOMETRY.MERGE1:
            # self.divs = [ 
            #     [-inf], # s_start
            #     [350.], # s_end
            #     [1.], # l_lower
            #     [2.] # l_upper
            # ] # Divider is between lanes 1 and 2

            # self.obs = [ 
            #     [1250.], # s_start
            #     [inf], # s_end
            #     [1.], # l_start
            #     [1.] # l_end
            # ] # Obstacle blocks lane 1 starting at 100 m

            self.numLanes = 3 # 1 shoulder and 2 freeway lanes

            print('Merging-1 geometry used.')

        elif self.geometry == GEOMETRY.SINGLELANE:
            self.numLanes = 1 # 1 freeway lane

            print('Single lane geometry used.')

        elif self.geometry == GEOMETRY.NONE:
            self.numLanes = 3 # 3 freeway lanes

            print('No obstacle geometry used.')

        elif self.geometry == GEOMETRY.TWOLANE:
            self.numLanes = 2 # 2 freeway lanes (no obstacle) - for lane-count transfer test

        elif self.geometry == GEOMETRY.FOURLANE:
            self.numLanes = 4 # 4 freeway lanes (no obstacle) - for lane-count transfer test

        elif self.geometry == GEOMETRY.FIVELANE:
            self.numLanes = 5 # 5 freeway lanes (no obstacle) - deeper lane-count OOD test

        else:
            raise NameError(f'Geometry scenario {self.geometry} not recognized!')

        self.numDivs, self.numObs = len(self.divs[0]), len(self.obs[0])
        
    def setRoadLights(self):
        '''Set up road network assigning traffic lights with RD structure'''
        self.RD = [ [self.SRTK[-1]-10], [LIGHTTYPE.STOP], [0], [0], [0], [0] ] # Default place a stop sign 10m before the end of road

        if self.route == ROUTE.SIM1:
            self.RD = [
                [400.00],
                [10.00],
                [95.20],
                [95.20],
                [0],
                [0]
            ]

        elif self.route == ROUTE.NONE:
            print('No route used.')

        else:
            raise NameError(f'Route scenario {self.route} not recognized!')

        self.numTLs = len(self.RD[0])
        self.tls = [TrafficLight(self.RD[0][i], self.interpx(self.RD[0][i]), self.interpy(self.RD[0][i]), self.THRTK, self.RD[1][i], self.RD[2][i], self.RD[3][i], self.RD[4][i], self.RD[5][i]) for i in range(0, self.numTLs) ]

    def setRoadTraffic(self, CAVtype=1):
        '''Set traffic on road network'''
        # Initialize surrounding vehicles
        n_cavs = 0
        self.svs = [None]*self.numSvs

        # Default values
        s0 = 0. # Agent Initial position
        v0 = 0. # Agent Initial velocity
        a0 = 0. # Initial accel
        vd = 0. # Desired speed
        lane = 1 # Initial lane
        type = VEHTYPE.MOBIL # Agent vehicle type used to decide constructor

        numcav = int(self.numSvs * self.descavrate)
        numvehlane = int(self.numSvs / self.numLanes)
        rand_id = random.sample(range(0, self.numSvs), self.numSvs)
        cav_id = rand_id[:numcav]
        # cav_id[0] = 0
        # cav_id = [3, 4, 5, 10, 11]
        # cav_id = [3, 4, 5]
        

        # intial speed
        numHV = self.numSvs - numcav
        desired_speed = np.round(clipped_normal(mean=VD,std=2,lower=VD*0.9,upper=VD*1.1,size=self.numSvs), 1)
        desired_speed[cav_id] = VD
        # initial position
        initial_gap = np.round(uniform_random_list(low=VD*2, high=VD*3, size=self.numSvs), 1)
        # True density sweep: override the randomized in-lane spacing with a fixed value (m),
        # so local density = 1000/FIXED_GAP veh/km/lane while the vehicle count is held constant.
        if os.environ.get("FIXED_GAP"):
            initial_gap = np.round(np.full(self.numSvs, float(os.environ["FIXED_GAP"])), 1)
        # Typed spawn state: CAVs at CACC time-gap (desired speed), humans at a larger gap.
        typed_init = os.environ.get("TYPED_INIT") == "1"
        is_cav_pre = np.zeros(self.numSvs, dtype=bool)
        is_cav_pre[cav_id] = True                     # CAV membership in pre-sort index space
        if typed_init:
            _tcacc = float(os.environ.get("T_CACC", "1.0"))
            _cav_gap = round(VEHLENGTH + 4.2 + VD * _tcacc, 1)          # s0 + VD*T_cacc (+ length)
            _human_gap = float(os.environ.get("HUMAN_INIT_GAP", str(VD * 4)))
            initial_gap = np.where(is_cav_pre, _cav_gap, _human_gap).round(1)
        # Distribute ALL numSvs vehicles across lanes, handling numSvs not divisible by
        # numLanes (first `rem` lanes get one extra) so no vehicle is left with lane=0.
        # Each lane block starts a new position group. Backward-compatible when divisible.
        base = self.numSvs // self.numLanes
        rem = self.numSvs % self.numLanes
        lane_counts = [base + (1 if ln < rem else 0) for ln in range(self.numLanes)]
        lane_id = []
        group_start = set()
        idx = 0
        for ln, cnt in enumerate(lane_counts):
            group_start.add(idx)
            lane_id += [ln + 1] * cnt
            idx += cnt
        initial_pos = []
        for i in range(self.numSvs):
            if i in group_start:
                initial_pos.append(random.randint(0, VD*2))
            else:
                initial_pos.append(initial_pos[-1] - initial_gap[i])
        initial_pos = np.round(initial_pos, 1)
        sorted_indices = sorted(range(len(initial_pos)), key=lambda i: initial_pos[i], reverse=True)
        initial_pos_sorted  = [initial_pos[i] for i in sorted_indices]
        lane_id_sorted  = [lane_id[i] for i in sorted_indices]
        is_cav_sorted = np.array([is_cav_pre[i] for i in sorted_indices])
        desired_speed_final = (np.array([desired_speed[i] for i in sorted_indices])
                               if typed_init else desired_speed)

        SIM = [] # SIM Matrix used for specifically assigning agent values in some cases - generated from veh2sim.m file
        # SIM Matrix is - position, velocity, acceleration, lane, vehtype, id, v_des
        # Random Intial parameters

        SIM = np.zeros([self.numSvs, 7])
        # random inital position
        SIM[:, 0] = initial_pos_sorted
        # inital speed
        SIM[:, 1] = desired_speed_final
        # acceleration
        SIM[:, 2] = 0
        # lane id
        SIM[:, 3] = lane_id_sorted
        
        # vehicle type
        if typed_init:
            SIM[np.where(is_cav_sorted)[0], 4] = CAVtype
        else:
            SIM[cav_id, 4] = CAVtype
        # vehicle id
        SIM[:, 5] = np.arange(1, self.numSvs+1)
        # random desired speed
        SIM[:, 6] = desired_speed_final
        
        # print(SIM[:,0])
        # print(cav_id)

        ### Loop to create each intended vehicle object
        for i in range(0, self.numSvs):
            # Assign specific traffic positions 
            if self.traffic == TRAFFIC.SIM1:
                # SIM Matrix is - position, velocity, acceleration, lane, vehtype, id, v_des

                s0 = SIM[i][0]
                v0 = SIM[i][1]
                a0 = SIM[i][2]
                vd = SIM[i][6]
                lane = SIM[i][3]
                type = SIM[i][4]
                id = SIM[i][5]

            else:
                raise NameError(f'Traffic scenario {self.traffic} not recognized!')

            ### Get class constructor based on vehicle type selected
            constructor = None

            if type == VEHTYPE.MOBIL:
                constructor = MOBIL
            elif type == VEHTYPE.CAV:
                constructor = CAV
                n_cavs += 1
            elif type== VEHTYPE.DRLCAV:
                constructor = DRLCAV
                n_cavs += 1

            else:
                raise NameError(f'Type of vehicle {type} not recognized!')

            # Create surrounding vehicle
            self.svs[i] = constructor(
                    x=self.interpx(s0)-(lane-1)*LANEWIDTH*sin(self.THRTK), # x
                    y=self.interpy(s0)+(lane-1)*LANEWIDTH*cos(self.THRTK), # y
                    s=s0, 
                    v=v0,
                    a=a0,
                    theta=self.THRTK, # theta0
                    id=id,
                    lane=lane, 
                    vd=vd,
                    numLanes=self.numLanes,
                    obs=self.obs,
                    divs=self.divs,
                    active=self.useActive
                )

            ### Error checks
            roadLength = self.SRTK[-1]
            assert s0 < roadLength, 'Vehicle initialized beyond desired road length.'
            
            # assert len(SIM)==0 or len(SIM) == self.numSvs, f'Input numSvs {self.numSvs} does not match the SIM numSvs {len(SIM)}!'

            lc = getOpenLanes(s0+8, self.numLanes, obs=self.obs)
            assert lane in lc, f''.format('Assigned lane for Vehicle outside valid lanes! Double check -numLanes and -geometry!')

        
        # Calculate number of CAVs in network
        self.cavrate = n_cavs/self.numSvs
        

    def __init__(self, args, XRTK = np.linspace(0, 1000, 2), YRTK = np.linspace(0, 0, 2), CAVtype=1,filename='data.csv', rl_agent=None, explore=True):
        '''Set up traffic simulation'''
        # random.seed(a=RANDSEED) # Fix random seed

        self.t = 0. # Total elapsed simulation time [s]

        ### Simulation Parameters
        self.numEgos = args.numEgos
        self.numSvs = args.numSvs
        self.route = args.route
        self.geometry = args.geometry
        self.traffic = args.traffic
        self.enable_logging = args.logging

        self.HZ = args.HZ
        self.dt = 1./self.HZ
        self.XRTK = XRTK # XRTK is the global coordinates for the road - vector of x points from start to end
        self.YRTK = YRTK # YRTK is the global coordinates for the road - vector of y points from start to end
        self.filename = filename
        self.cavtype = CAVtype
        self.collison = 0
        self.done = False

        self.useVerbose = args.useVerbose
        self.useActive = args.useActive
        self.useRealtime = False

        self.descavrate = args.cavRate # The desired CAV penetration rate
        self.cavrate = 0 # The realized CAV penetration rate
        self.explore = explore
        self.action_count = {
            0: 0,   # RIGHT
            1: 0,   # KEEP
            2: 0    # LEFT
        }

        self.rt_tic = time.time()
        self.timer = time.time()


        ### Setup road network
        # Road geometry from XRTK, YRTK - Assumes straight road with sorted coordinates
        # Rightmost lane or shoulder is lane 1, centerline is l = 1
        self.setRoadDimensions()

        # Traffic lights        
        self.setRoadLights()

        # Road geometry - [s_start], [s_end], [l_start], [l_end]
        self.setRoadGeometry()

        # Surrounding vehicles
        self.setRoadTraffic(CAVtype)
        
        self.obs_builder = ObservationBuilder(
            num_lanes=self.numLanes,
            local_back=40.0,
            local_front=80.0,
            cell_size=5.0,
            remote_front=300.0,
        )

        self.rl_agent = rl_agent
        self.init_rl_buffers()

        # RL decision interval: 1 second
        self.decision_interval_steps = max(1, int(round(1.0 / self.dt)))
        self.sim_step_count = 0

        # Per-DRLCAV last action and GRU hidden state
        self.last_rl_actions = {}
        self.rl_hidden_states = {}

        # Per-DRLCAV observation history for 1-second input to GRU
        self.obs_history = {}
        for sv in self.svs:
            if sv.type == VEHTYPE.DRLCAV:
                self.last_rl_actions[sv.id] = 0  # default keep lane
                self.rl_hidden_states[sv.id] = None
                self.obs_history[sv.id] = {
                    "local_grid": deque(maxlen=10),
                    "ego": deque(maxlen=10),
                    "remote": deque(maxlen=10),
                }

        ### Visualize
        self.anim = animation(XRTK=XRTK, YRTK=YRTK, laneWidths=[LANEWIDTH]*self.numLanes, laneOrient=self.THRTK, numTLs=self.numTLs, numEgos=self.numEgos, numSvs=self.numSvs, numObs=self.numObs, numDivs=self.numDivs, useVisual=args.useVisual, recordVisual=args.recordVisual, HZ=self.HZ, followId=args.followId)
        
        ### Logging
        self.logger = logger(self.dt, TEND, self.numTLs, self.numSvs, self.filename)

        ### Error checking
        assert self.numEgos == 0 or self.numEgos == 1, 'Multiple egos not yet supported.'
        assert self.descavrate >= 0 and self.descavrate <= 1, 'Desired CAV penetration rate should be within the range [0, 1].'
    

    def init_rl_buffers(self):
        self.decision_interval_steps = max(1, int(round(1.0 / self.dt)))
        self.sim_step_count = 0

        self.last_rl_actions = {}
        self.last_decision_actions = {}
        self.rl_hidden_states = {}
        self.obs_history = {}

        # Centralized same-tick lane-change arbiter (eval-time safety shield). USE_LC_ARBITER=1
        # enables it; default off so training / V2 reproduction is unchanged. Prevents two
        # vehicles from merging into the same gap on the same decision tick (the concurrent-
        # merge collisions the per-agent _lane_change_safe shield cannot see).
        self._lc_arbiter = os.environ.get("USE_LC_ARBITER", "0") == "1"
        self._lc_claims = []   # (target_lane, s) gaps reserved this decision tick

        # Greedy "follow-nearest-connected-vehicle" rule baseline (GREEDY_FOLLOW_CAV=1, eval-time).
        # Replaces the DRLCAV's LEARNED lateral decision with a hand-coded heuristic (steer toward
        # the nearest CAV/DRLCAV's lane, safety-gated); everything else (CACC, execution) identical.
        # Tests whether the learned policy beats a simple co-lane heuristic (= is cooperation learned).
        self._greedy = os.environ.get("GREEDY_FOLLOW_CAV", "0") == "1"

        seq_len = 10
        if self.rl_agent is not None and hasattr(self.rl_agent, "seq_len"):
            seq_len = self.rl_agent.seq_len

        for sv in self.svs:
            if sv.type == VEHTYPE.DRLCAV:
                self.last_rl_actions[sv.id] = 0
                self.last_decision_actions[sv.id] = 0
                self.rl_hidden_states[sv.id] = None
                self.obs_history[sv.id] = {
                    "local_grid": deque(maxlen=seq_len),
                    "ego": deque(maxlen=seq_len),
                    "remote": deque(maxlen=seq_len),
                }


    def update_obs_history(self, ego, obs):
        hist = self.obs_history[ego.id]
        hist["local_grid"].append(obs["local_grid"])
        hist["ego"].append(obs["ego"])
        hist["remote"].append(obs["remote"])


    def get_seq_obs_for_agent(self, ego, current_obs):
        hist = self.obs_history[ego.id]

        seq_len = 10
        if self.rl_agent is not None and hasattr(self.rl_agent, "seq_len"):
            seq_len = self.rl_agent.seq_len

        local_list = list(hist["local_grid"])
        ego_list = list(hist["ego"])
        remote_list = list(hist["remote"])

        if len(local_list) == 0:
            local_list.append(current_obs["local_grid"])
            ego_list.append(current_obs["ego"])
            remote_list.append(current_obs["remote"])
        elif not np.array_equal(local_list[-1], current_obs["local_grid"]):
            local_list.append(current_obs["local_grid"])
            ego_list.append(current_obs["ego"])
            remote_list.append(current_obs["remote"])

        while len(local_list) < seq_len:
            local_list.insert(0, local_list[0])
            ego_list.insert(0, ego_list[0])
            remote_list.insert(0, remote_list[0])

        return {
            "local_grid": np.stack(local_list[-seq_len:], axis=0),
            "ego": np.stack(ego_list[-seq_len:], axis=0),
            "remote": np.stack(remote_list[-seq_len:], axis=0),
            "action_mask": current_obs["action_mask"],
        }

    def compute_platoon_reward(self):
        drlcavs = [sv for sv in self.svs if sv.type == VEHTYPE.DRLCAV]

        if len(drlcavs) == 0:
            return 0.0

        return sum(veh.in_platoon for veh in drlcavs) / len(drlcavs)

    def compute_team_reward(self, debug=False):
        # Reward weights are env-configurable (defaults = paper baseline) for the reward
        # sensitivity analysis; unset envs fall back to the baseline values.
        w_speed = float(os.environ.get("W_SPEED", "0.2"))
        w_platoon = float(os.environ.get("W_PLATOON", "0.3"))   # instantaneous platoon ~constant -> small
        w_collision = float(os.environ.get("W_COLLISION", "5.0"))  # REAL collisions, not a gap proxy
        # lane-change penalty (per-event, Markov). Applied UNDILUTED (see total_reward):
        # scaled by decision_interval_steps so it isn't washed out by the /interval averaging.
        w_lane_change = float(os.environ.get("W_LANE_CHANGE", "0.5"))
        # stronger penalty for repeated / excessive LC
        w_repeated_lc = float(os.environ.get("W_REPEATED_LC", "1.0"))
        w_ttc = float(os.environ.get("W_TTC", "1.0"))
        w_sustain = float(os.environ.get("W_SUSTAIN", "1.0"))  # reward HOLDING a platoon >=3s
        

        reward_speed = 0.0
        reward_collision = 0.0
        reward_lane_change = 0.0
        reward_repeated_lc = 0.0
        reward_ttc = 0.0
        reward_sustain = 0.0
        total_lc = 0

        sustain_threshold_steps = int(3.0 / self.dt)

        drlcavs = [sv for sv in self.svs if sv.type == VEHTYPE.DRLCAV]
        if len(drlcavs) == 0:
            return 0.0

        reward_platoon = self.compute_platoon_reward()

        for veh in drlcavs:
            reward_speed += min(veh.v / VD, 1.0)

            # update sustained platoon counter
            if getattr(veh, "in_platoon", 0.0) > 0.5:
                veh.platoon_hold_steps += 1
            else:
                veh.platoon_hold_steps = 0

            # reward only if platoon is maintained long enough
            if veh.platoon_hold_steps >= sustain_threshold_steps:
                reward_sustain += 1.0

            # lane-change penalty only when a lane change actually starts
            if getattr(veh, "did_lane_change", False):
                if getattr(veh, "in_platoon", 0.0) > 0.5:
                    reward_lane_change -= 1.5
                else:
                    reward_lane_change -= 1.0

            # stronger penalty if lane change is repeated within short time window
            if getattr(veh, "repeated_lane_change", False):
                reward_repeated_lc -= 1.0

            # total episode lane changes
            total_lc += getattr(veh, "LC", 0)
                

            same_lane_front = [
                other for other in self.svs
                if other.id != veh.id
                and int(round(other.lane)) == int(round(veh.lane))
                and other.s > veh.s
            ]

            if len(same_lane_front) > 0:
                front = min(same_lane_front, key=lambda x: x.s - veh.s)
                gap = front.s - veh.s

                rel_v = veh.v - front.v
                if rel_v > 1e-3:
                    ttc = gap / rel_v
                    if ttc < 1.0:
                        reward_ttc -= 1.0
                    elif ttc < 2.0:
                        reward_ttc -= 0.5
        
        # real geometric collisions this step (set by check_collisions() in step())
        reward_collision = -float(self.collison)
        reward_speed /= len(drlcavs)
        reward_lane_change /= len(drlcavs)
        reward_ttc /= len(drlcavs)
        reward_sustain /= len(drlcavs)
        reward_repeated_lc /= len(drlcavs)

        # total_reward = w_platoon * reward_platoon + w_lane_change * reward_lane_change + w_collision * reward_collision
        total_reward = (
            w_speed * reward_speed
            + w_platoon * reward_platoon
            + w_sustain * reward_sustain
            + w_collision * reward_collision
            # LC penalties are one-time events per decision -> multiply by the interval
            # length so they survive the /decision_interval_steps transition averaging.
            + w_lane_change * reward_lane_change * self.decision_interval_steps
            + w_repeated_lc * reward_repeated_lc * self.decision_interval_steps
            + w_ttc * reward_ttc
        )
        # total_reward = (
        #     w_speed * reward_speed
        #     + w_platoon * reward_platoon
        #     + w_collision * reward_collision
        #     + w_lane_change * reward_lane_change
        #     + w_ttc * reward_ttc
        # )

        if debug and (self.sim_step_count % self.decision_interval_steps == 0):
            print(
                f"t={self.t:5.1f} | "
                f"speed={w_speed * reward_speed:7.3f} "
                f"platoon={w_platoon * reward_platoon:7.3f} "
                f"sustain={w_sustain * reward_sustain:7.3f} "
                f"collision={w_collision * reward_collision:7.3f} "
                f"LC={w_lane_change * reward_lane_change:7.3f} "
                f"total={total_reward:7.3f}"
            )

        return total_reward
    
    def build_global_state(self):
        state = []

        for sv in sorted(self.svs, key=lambda x: x.id):
            state.extend([
                sv.s / 3000.0,
                sv.v / 35.0,
                sv.a / 8.0,
                (sv.lane - 1) / max(1, self.numLanes - 1),
                1.0 if sv.type == VEHTYPE.DRLCAV else 0.0,
                1.0 if sv.type == VEHTYPE.CAV else 0.0,
                1.0 if sv.type == VEHTYPE.MOBIL else 0.0,
            ])

        return np.array(state, dtype=np.float32)

    def _gap_claim_conflict(self, ego, tlane, svs, margin=2.0):
        """
        Same-tick lane-change arbitration. Return True if ego must NOT initiate a change into
        tlane this tick because the target gap is already taken by either:
          (1) a vehicle currently mid lane-change INTO tlane within the conflict span, or
          (2) a vehicle that already RESERVED this gap earlier in this tick's loop order.
        Geometric/TTC safety is still handled by _lane_change_safe (the action mask); this
        only adds the multi-agent concurrent-merge guard.
        """
        span = VEHLENGTH + margin
        for o in svs:
            if getattr(o, "is_lane_changing", False) \
               and int(round(getattr(o, "target_lane", o.lane))) == tlane \
               and abs(o.s - ego.s) < span:
                return True
        for (cl_lane, cl_s) in self._lc_claims:
            if cl_lane == tlane and abs(cl_s - ego.s) < span:
                return True
        return False

    def _greedy_action(self, ego, others, rng=200.0):
        """
        Greedy rule (0=right/-1, 1=keep, 2=left/+1). Pure platoon-seek by default; with GREEDY_MOBIL=1
        it is the CONJUNCTION of MOBIL and platoon-seek: change lanes only if BOTH the MOBIL incentive
        rule AND the platoon-seek rule independently choose the same direction (both say yes), and the
        move is safe. Otherwise keep lane. All moves safety-gated by is_target_lane_safe.
        """
        # platoon-seek decision: direction toward the nearest connected vehicle (CAV/DRLCAV) within rng
        best, bd = None, rng
        for o in others:
            if o.type in (VEHTYPE.CAV, VEHTYPE.DRLCAV):
                d = abs(o.s - ego.s)
                if d <= bd:
                    bd, best = d, o
        plat_delta = 0
        if best is not None:
            dl = int(round(best.lane)) - int(round(ego.lane))
            if dl != 0:
                plat_delta = 1 if dl > 0 else -1

        if os.environ.get("GREEDY_MOBIL", "0") != "1":
            # pure platoon-seek
            if plat_delta != 0 and ego.is_target_lane_safe(others, ego.lane + plat_delta):
                return 2 if plat_delta > 0 else 0
            return 1

        # MOBIL decision: direction of the largest safe speed/gap incentive
        cfds, cfv, cfi = getFrontVeh(others, ego.x, ego.y, ego.len, ego.theta, ego.lane)
        cur_spd = cfv if cfi >= 0 else VD
        cur_gap = cfds if cfi >= 0 else 1e9
        mobil_delta, best_gain = 0, 0.0
        for delta in (-1, 1):
            tl = ego.lane + delta
            if not ego.is_target_lane_safe(others, tl):
                continue
            tfds, tfv, tfi = getFrontVeh(others, ego.x, ego.y, ego.len, ego.theta, tl)
            tgt_spd = tfv if tfi >= 0 else VD
            tgt_gap = tfds if tfi >= 0 else 1e9
            if (tgt_spd - cur_spd) >= 1.0 or (tgt_gap - cur_gap) >= 10.0:  # MOBIL incentive satisfied
                if (tgt_spd - cur_spd) >= best_gain:
                    best_gain, mobil_delta = (tgt_spd - cur_spd), delta

        # conjunction: act only if MOBIL and platoon-seek agree on the same (safe) direction
        if mobil_delta != 0 and mobil_delta == plat_delta \
           and ego.is_target_lane_safe(others, ego.lane + mobil_delta):
            return 2 if mobil_delta > 0 else 0
        return 1

    def check_collisions(self):
        collision_count = 0

        for i in range(self.numSvs):
            vi = self.svs[i]
            for j in range(i + 1, self.numSvs):
                vj = self.svs[j]

                # Only count collisions the policy is responsible for: at least one
                # vehicle in the pair must be a DRLCAV. Human-human (MOBIL-MOBIL)
                # crashes are outside the agents' control and must not penalize them.
                if vi.type != VEHTYPE.DRLCAV and vj.type != VEHTYPE.DRLCAV:
                    continue

                longitudinal_overlap = abs(vi.s - vj.s) < VEHLENGTH
                lateral_overlap = abs(vi.l - vj.l) < 0.5

                if longitudinal_overlap and lateral_overlap:
                    collision_count += 1

        self.collison = collision_count
        return collision_count

    def rate(self):
        '''Sleep program until next time to execute another frame if running with real-time option'''
        if self.useRealtime:
            while time.time()-self.rt_tic < self.dt:
                pass

            self.rt_tic = time.time()

    def sim(self):
        '''Run traffic microsimulation from t to TEND'''
        ### Simulation
        # Main loop
        while self.t < TEND:
            self.step()
        # ID = np.array([])
        # # Set surrounding vehicle control
        # for j in range(0, self.numSvs):
        #     ID = np.append(ID, self.svs[j].id)
        # print(ID) 
        # Cleanup
        self.logger.save()
        self.anim.save()

    def step_dynamics_only(self, ext_cav=None):
        """
        Advance simulator one step after all commands are already set.
        This is the propagation/update part only.
        """
        # Step forward traffic dynamics
        for sv in self.svs:
            if _USE_MPC_LC and sv.type in (VEHTYPE.CAV, VEHTYPE.DRLCAV):
                e_y = (sv.l - sv.ul) * LANEWIDTH                       # lateral offset from target lane [m]
                steer = mpc_steer(sv.v, e_y, sv.theta - self.THRTK, self.dt,
                                  getattr(sv, "_mpc_prev_d", 0.0))
                sv._mpc_prev_d = steer
            else:
                steer = stanley(sv.v, -(sv.theta - self.THRTK), (sv.ul - sv.l))
            if _ACT_LAG:                                          # first-order actuator/drivetrain lag
                _alpha = 1.0 - np.exp(-self.dt / _ACT_TAU)
                a_in = sv.a + _alpha * (sv.ua - sv.a)             # actual accel lags the command sv.ua
            else:
                a_in = sv.ua
            z = RK4(
                dyn,
                self.dt,
                self.t,
                np.array([sv.x, sv.y, sv.v, sv.theta]),
                np.array([a_in, steer])
            )

            # Post process
            if z[2] >= 0:  # Prevent reversing
                sv.x, sv.y, sv.v, sv.theta = z[0], z[1], z[2], z[3]
                sv.a = a_in
            else:
                sv.v = 0.0
                sv.a = 0.0

            sv.s, sv.l, sv.ldot = getRoadPos(
                sv.x, sv.y, sv.v, sv.theta,
                self.THRTK, self.XRTK[0], self.YRTK[0], LANEWIDTH
            )
            sv.lane = round(sv.l)

        self.check_collisions()
        self.t += self.dt

        # Logging
        if self.enable_logging:
            self.logger.step(self.t, self.tls, self.svs)

        # Visualize
        xp, yp, tp = [], [], []
        Xnp, Ynp = [], []

        if ext_cav is not None:
            xp, yp, tp = ext_cav.x, ext_cav.y, ext_cav.theta
            Xnp, Ynp = self.getEgoTrajXY(ext_cav)

        self.anim.draw(
            xego=xp, yego=yp, tego=tp,
            svs=self.svs, tls=self.tls,
            obs=self.obs, divs=self.divs,
            Xego=Xnp, Yego=Ynp
        )

        sim_timer = time.time() - self.timer
        self.timer = time.time()

        if self.useVerbose:
            if ext_cav is not None:
                print('Ego s: {:0.2f}, l: {:0.1f}, v: {:0.1f}, a: {:0.1f}, x: {:0.1f}, y: {:0.1f}'.format(
                    ext_cav.s, ext_cav.l, ext_cav.v, ext_cav.a, ext_cav.x, ext_cav.y))

            for i in range(0, self.numSvs):
                print('  N{:d} type: {:d}, s: {:0.2f}, l: {:0.1f}, v: {:0.1f}, vd: {:0.1f}, ua: {:0.1f}, ul: {:0.0f}, x: {:0.1f}, y: {:0.1f}'.format(
                    i, self.svs[i].type, self.svs[i].s, self.svs[i].l, self.svs[i].v,
                    self.svs[i].vd, self.svs[i].ua, self.svs[i].ul,
                    self.svs[i].x, self.svs[i].y))

        self.rate()


    def step(self, le=np.zeros(2), ext_cav=None, debug= False):
        """
        Perform one simulator step.

        - simulator updates every dt
        - RL makes a new decision every 1 second
        - if a DRLCAV is already changing lanes, keep executing until finished
        - store the issued high-level decision separately from the currently held action
        """
        # traffic light control
        for tl in self.tls:
            tl.setCommand(self.t)

        # new RL decision only once per second
        make_new_decision = (self.sim_step_count % self.decision_interval_steps == 0)
        explore = self.explore

        # reset this tick's gap reservations for the concurrent-merge arbiter
        if make_new_decision and self._lc_arbiter:
            self._lc_claims = []
        # set vehicle commands
        for j in range(0, self.numSvs):
            ego = self.svs[j]
            svs = [self.svs[i] for i in range(0, self.numSvs) if i != j]

            if ego.type == VEHTYPE.DRLCAV and self.rl_agent is not None:
                # current single-step observation
                obs = self.obs_builder.build_observation(
                    ego=ego,
                    svs=svs,
                    tls=self.tls,
                    obs=self.obs,
                    divs=self.divs
                )

                # update lane-change execution status
                ego.update_lane_change_status()

                # if lane change has finished, execution action returns to keep lane
                if not ego.is_lane_changing and ego.lane_change_finished():
                    self.last_rl_actions[ego.id] = 0

                # Recurrent act path: step the GRU ONCE per decision interval on the
                # single current obs + carried hidden. The hidden state is the memory,
                # so it advances at the same 1 Hz cadence the training loop unrolls.
                # Step it EVERY interval (even mid lane-change) so the act-time hidden
                # trajectory matches training exactly.
                if make_new_decision:
                    if self._greedy:
                        # rule baseline: lateral action from the greedy follow-CAV heuristic
                        action_idx = self._greedy_action(ego, svs)
                    else:
                        action_idx, next_hidden = self.rl_agent.select_action(
                            obs,
                            hidden=self.rl_hidden_states[ego.id],
                            explore=explore
                        )
                        self.rl_hidden_states[ego.id] = next_hidden
                    self.action_count[action_idx] += 1

                    # network action index -> simulator lane delta (0=right,1=keep,2=left)
                    action_map = {0: -1, 1: 0, 2: 1}
                    sim_action = action_map[action_idx]

                    # record the decision every interval (for replay), so the stored
                    # action matches the GRU step that produced this hidden state
                    self.last_decision_actions[ego.id] = sim_action

                    # only (re)issue the execution command when not mid lane-change
                    if not ego.is_lane_changing:
                        # concurrent-merge arbiter: defer a NEW lane change if another vehicle
                        # has claimed the same target gap this tick (or is already merging into
                        # it). The recorded decision (last_decision_actions) is left as the raw
                        # policy action; only execution is shielded.
                        if self._lc_arbiter and sim_action != 0:
                            tlane = ego.lane + sim_action
                            if self._gap_claim_conflict(ego, tlane, svs):
                                sim_action = 0
                            else:
                                self._lc_claims.append((tlane, ego.s))
                        self.last_rl_actions[ego.id] = sim_action

                # execute currently held action
                ego.set_marl_action(self.last_rl_actions[ego.id])
                ego.setCommand(svs, self.tls, self.t, self.dt)

            else:
                ego.setCommand(svs, self.tls, self.t, self.dt)

        # propagate one simulator step
        self.step_dynamics_only(ext_cav=ext_cav)

        # team reward after joint consequence
        reward = self.compute_team_reward(debug)

        self.sim_step_count += 1
        return reward
        

    

