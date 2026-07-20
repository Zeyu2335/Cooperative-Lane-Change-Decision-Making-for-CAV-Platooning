#! /usr/bin/env python3

from math import sin, cos, sqrt, tan, atan, atan2, fmod, pi
import numpy as np

from _constants import * 

def stanley(v, Ye, Le):
    '''Run Stanley-like controller as based from Hoffmann, Gabriel M., Claire J. Tomlin, Michael Montemerlo, and Sebastian Thrun. "Autonomous Automobile Trajectory Tracking for Off-Road Driving: Controller Design, Experimental Validation and Racing." American Control Conference. 2007, pp. 2296–2301. doi:10.1109/ACC.2007.4282788'''
    k1 = 1.05
    k2 = 2.50
    k3 = 1.00
    e = 1.10

    return min( max(
        k1*Ye + atan(k2*Le / (e + k3*v)),
        -35.*pi/180.),
        35.*pi/180. )

def dyn(t, x, u):
    '''x = [x, y, v, theta], u = a, d'''
    Lf, Lr = 0.5*VEHLENGTH, 0.5*VEHLENGTH # Lengths of center of rotation to front and rear axle
    beta = atan( Lf/(Lr+Lf)*tan(u[1]) )
    
    return np.array([
        x[2]*cos(x[3]+beta),
        x[2]*sin(x[3]+beta),
        u[0],
        x[2]/Lf*sin(beta)
    ])

def RK2(f, dt, t, x, u):
    '''Single-step second-order Runge-Kutta integrator'''
    k1 = dt*f(t, x, u)
    k2 = dt*f(t+0.5*dt, x+0.5*k1, u)

    t = t+dt
    x = x+0.5*(k1+k2)
    return x

def RK4(f, dt, t, x, u):
    '''Single-step fourth-order Runge-Kutta integrator'''
    k1 = dt*f(t, x, u)
    k2 = dt*f(t+0.5*dt, x+0.5*k1, u)
    k3 = dt*f(t+0.5*dt, x+0.5*k2, u)
    k4 = dt*f(t+dt, x+k3, u)
    
    t = t+dt
    x = x+0.1666667*(k1+2.*k2+2.*k3+k4)
    return x
