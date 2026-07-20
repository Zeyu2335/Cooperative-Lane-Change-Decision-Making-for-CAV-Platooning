#! /usr/bin/env python3
### Enums
def enum(**enums):
    return type('Enum', (), enums)

# Different traffic configurations
ROUTE = enum(NONE=0, SYNTHETIC=1, PEACHTREE=2, CONSTRUCTION=3, OFFSITE=4, SIM1=5) # Enums for route scenario that codes TL placement and timings as assigned in RD matrix
GEOMETRY = enum(NONE=0, MERGE1=2, SINGLELANE=1, TWOLANE=10, FOURLANE=11, FIVELANE=12) # Enums for road geometry scenario that codes obstacle and divider placement
TRAFFIC = enum(NONE=-1, RANDOM=0, MERGE1=1, SIM1=2, SINGLECAV=9, MIXEDRANDOM=10) # Enums for road traffic scenario that codes initial positioning and desired speed

# Different object types and statuses
LIGHTSTATUS = enum(GREEN=2, AMBER=1, RED=0) # ANL enums for light status
LIGHTTYPE = enum(LIGHT=10, STOP=20) # ANL enums for intersection type in RD matrix - traffic light or stop sign

VEHSTATUS = enum(NONE=0, FREE=1, LOWUTIL=2, TLINTAC=3) # Enums for MOBIL status on its current driving pattern
VEHTYPE = enum(NONE=-1, MOBIL=0, CAV=1, PCC=2, DRLCAV=3) # Enums for type of vehicle as assigned in SIM matrix

### Settings
# Vehicle parameter settings
VEHWIDTH = 1.90 # assumed vehicle width [m]
VEHLENGTH = 3.25 # assumed vehicle length [m]

# Simulation settings
RANDSEED = 20 # Random seed used to fix the pseudo random generation. None to use system time

TEND = 80.0 # time til simulation ends automatically [s]

# Road parameter settings
SHOULDERRUNUP = 21.0 # distance ahead of obstacle s that shoulder will ramp into next lane [m]
LANEWIDTH = 3.70 # width of lanes [m]

# Randomization settings
S0 = 80.0 # start position of surrounding traffic in NONE scenario [m]
DS = 20.0 # position increment of consecutive surrounding vehicles - mean increment in random generation [m]
DSVAR = 6.0 # bound on position increment variance in random generation 

VD = 20 # desired speed of surrounding traffic - peak of desired speed density function in random generation [m/s]
VDLOWER = 22 # lower bound on desired speed of surrounding traffic in random generation
VDUPPER = 18 # upper bound on desired speed of surrounding traffic in random generation