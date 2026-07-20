#! /usr/bin/env python3

from math import sin, cos, sqrt, tan, atan, atan2, fmod
import numpy as np
import math
import os

from _constants import *

# Continuous-lateral perception (CONT_PERCEPTION=1, eval-time; default off = legacy integer-lane
# match). When on, car-following treats a vehicle as "in lane L" once its continuous lateral
# position l is within 0.6 of L (i.e., over the lane line) -- so a follower perceives and brakes for
# a merging vehicle DURING the crossing, not only after its integer lane flips at ~85% across.
_CONT_PERCEPTION = os.environ.get("CONT_PERCEPTION", "0") == "1"
_LANE_TOL = float(os.environ.get("LANE_TOL", "0.6"))

def _in_lane(sv, lane):
    if _CONT_PERCEPTION:
        return abs(getattr(sv, "l", sv.lane) - lane) < _LANE_TOL
    return lane == sv.lane

# Functions
def getRoadPos(x, y, v, theta, gamma, x0, y0, lw):
    ''' Function to calculate s, l, and ldot for the ego ||
    x0 + s cos(gamma) - l sin(gamma) = x ||
    y0 + s sin(gamma) + l cos(gamma) = y ||
    
    z0 + n R(gamma) = z ||

    gamma = atan2(y1-y0, x1-x0) # Straight road orientation angle
    '''
    
    # Calculate s, l
    t = gamma

    b = np.array([
        x-x0, 
        y-y0
    ])

    A = np.array([
        [cos(t), -sin(t)],
        [sin(t), cos(t)]
    ])

    z = np.linalg.solve(A, b)
    
    s = z[0]
    l = z[1]
    ldot = v*sin(theta-gamma)
    
    # Normalize such that l = 1.0 is the centerline of first lane, l = 2.0 is centerline of second lane, l = 3.0 is ...
    l = l/lw + 1
    ldot = ldot/lw

    return s, l, ldot

def distance(x, x2, y, y2, theta):
    '''Euclidean distance between points in forward direction from theta'''
    return (x2 - x)*cos(theta) + (y2 - y)*sin(theta)

def getFrontObs(s, l, lane, obs):
    '''Get distance and index of nearest obstacle in front of position s+l in given lane'''
    ds = 2000.
    i = -1
    for k in range(0, len(obs[0])):
        if obs[2][k] >= lane and lane <= obs[3][k]:
            d00 = obs[0][k] - (s + l)

            if d00 > -5 and d00 < ds: # Behind obstacle
                ds = d00
                i = k
            elif (s + l) <= obs[1][k] and (s + l) >= obs[0][k]: # On top of obstacle
                ds = 0.
                i = k

    return ds, i

def getFrontTL(s, l, tls):
    '''Get distance, status, and index of nearest traffic light in front of position s+l'''
    ds = 2000.
    ls = LIGHTSTATUS.GREEN
    i = -1
    for k in range(0, len(tls)):
        d0 = tls[k].s - (s + l)
        if d0 > -5 and d0 < ds:
            ds, ls, i = d0, tls[k].status, k

    return ds, ls, i

def getnFrontTLs(n, s, l, tls):
    '''Get vector of n nearest traffic lights in front of position s+l'''
    # Get all traffic light distances from ego vehicle front bumper
    d0s = [tls[k].s - (s + l) for k in range(0, len(tls))]
    is_fronts = [d0 > -5 for d0 in d0s]

    # Sort the distances from smallest -> farthest
    d0s, is_fronts, tls = zip(*sorted(zip(d0s, is_fronts, tls)))

    # Get sorted traffic light objects for all those in front
    ntls = [tls[k] for k in range(0, len(tls)) if is_fronts[k]]

    # Only provide first n tls in front of ego
    if len(ntls) > n:
        ntls = ntls[0:n]
    
    return ntls

def getnLatestTLs(n, s, l, tls):
    '''Get vector of n latest traffic lights, preferring those in front of position s+l'''
    # Get all traffic light distances from ego vehicle front bumper
    d0s = [tls[k].s - (s + l) for k in range(0, len(tls))]
    is_fronts = [d0 > -5 for d0 in d0s]

    # Sort the distances from smallest -> farthest
    d0s, is_fronts, tls = zip(*sorted(zip(d0s, is_fronts, tls)))

    # Get sorted traffic light objects for all those in front
    rtls = [tls[k] for k in range(0, len(tls)) if not is_fronts[k]]
    ftls = [tls[k] for k in range(0, len(tls)) if is_fronts[k]]

    # Only provide first n tls in front of ego
    len_ftls = len(ftls)
    if len_ftls > n:
        ftls = ftls[0:n]

    # Prepend with latest m tls behind of ego if not enough lights are in front
    ntls = ftls
    if len_ftls < n:
        m = n-len_ftls
        ntls = rtls[-m::] + ftls

    assert len(ntls) == n, 'Incorrect number of traffic lights found!'

    return ntls

def getFrontVeh(svs, x, y, l, theta, lane):
    '''Get sv immediately in front and in the given lane'''
    nvds = 2000.
    nvvr = 20.
    i = -1
    for k in range(0, len(svs)):
        if _in_lane(svs[k], lane):
            d0 = distance(x+l*cos(theta), svs[k].x, y+l*sin(theta), svs[k].y, theta)
            if d0 > -2-l and d0 < nvds: # Behind another neighboring vehicle in current lane
                nvds, nvvr, i = d0, svs[k].v, k
    return nvds, nvvr, i

def getRearVeh(svs, x, y, theta, lane):
    '''Get sv immediately behind and in the given lane'''
    nvds = -2000.
    nvvr = 20.
    i = -1
    for k in range(0, len(svs)):
        if _in_lane(svs[k], lane):
            d0 = distance(x, svs[k].x+svs[k].len*cos(svs[k].theta), y, svs[k].y+svs[k].len*sin(svs[k].theta), theta)
            if d0 < 2+svs[k].len and d0 > nvds: # In front of another neighboring vehicle in current lane
                nvds, nvvr, i = d0, svs[k].v, k

    return nvds, nvvr, i

def collision_detect(svs, x, lane):
    h0 = None
    for k in range(0, len(svs)):
        if lane == svs[k].lane:
            h0 = abs(x - svs[k].x)
            if h0 < VEHLENGTH:
                break
    return h0, k



def getSideVeh(svs, x, y, theta, lane):
    '''Return true/false if an sv immediately next to and in the given lane'''
    safe = True
    for k in range(0, len(svs)):
        if lane == svs[k].lane:
            d0 = distance(x, svs[k].x+svs[k].len*cos(svs[k].theta), y, svs[k].y+svs[k].len*sin(svs[k].theta), theta)
            if d0 < 2+svs[k].len and d0 > -2:
                safe = False
                break

    return safe

def getOpenLanes(s, numLanes, obs = [], l = 0, divs = []):
    lc = list(range(1, numLanes+1))

    # Check if near a blocked lane from obstacles
    if any(obs):
        lc = [lc[i] 
            for i in range(0, len(lc)) 
            for k in range(0, len(obs[0])) 
            if (not (s > obs[0][k] and s < obs[1][k] and lc[i] >= obs[2][k] and lc[i] <= obs[3][k]))
        ]

    # Check if near a blocked lane from lane divider # s_start, s_end, l_lower, l_upper
    if any(divs):
        lc = [lc[i] 
            for i in range(0, len(lc)) 
            for k in range(0, len(divs[0])) 
            if ( not (
                s > divs[0][k] and # Next to divider
                s < divs[1][k] and 
                ( 
                    (lc[i] >= 0.5*(divs[2][k]+divs[3][k]) and l < 0.5*(divs[2][k]+divs[3][k]) ) or #
                    (lc[i] <= 0.5*(divs[2][k]+divs[3][k]) and l > 0.5*(divs[2][k]+divs[3][k]) ) # 
                )
            ) )
        ]

    return lc







