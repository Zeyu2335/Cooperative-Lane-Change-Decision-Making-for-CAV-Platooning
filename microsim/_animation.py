#! /usr/bin/env python3

import os, shutil
import time
from math import sin, cos, sqrt, tan, atan, atan2, fmod, nan

import imageio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

from _constants import *

class animation:
    '''Animation class to hold and update an animation figure of the traffic simulation'''
    def getLane(self, s0, s1, l, step=[]):
        '''Gives [[x], [y]] road-frame from transform of input s, l'''
        S = [ max(s0, self.SRTK[0]), min(s1, self.SRTK[1]) ]
        L = [ max(-1, l-1), min(self.numLanes, l-1) ]

        x = [self.XRTK[0] + S[i]*cos(self.gamma) - L[i]*sin(self.gamma)*self.laneWidths[0] for i in range(0, 2)]
        y = [self.YRTK[0] + S[i]*sin(self.gamma) + L[i]*cos(self.gamma)*self.laneWidths[0] for i in range(0, 2)]

        # Interpolate to increase resolution of coordinates
        if step:
            iL = interp1d(S, L)

            S = np.arange(S[0], S[-1], step)
            L = [iL(s) for s in S]
            x = [self.XRTK[0] + S[i]*cos(self.gamma) - L[i]*sin(self.gamma)*self.laneWidths[0] for i in range(0, len(S))]
            y = [self.YRTK[0] + S[i]*sin(self.gamma) + L[i]*cos(self.gamma)*self.laneWidths[0] for i in range(0, len(S))]

        return x, y, S, L

    def getShoulder(self, s0, s1, l0, l1):
        '''Gives [[x], [y]] road-frame from transform of input s, l'''
        sp0 = s1 - SHOULDERRUNUP
        sp1 = s0 + SHOULDERRUNUP

        S = [ max( min(s0, sp0) , self.SRTK[0]), min( min(s1, sp1) , self.SRTK[1]), min( max(s1, sp1) , self.SRTK[1]) ]
        L = [ max(-0.5, l0-1 - 0.5), min(self.numLanes+0.5, l1-1 + 0.5), min(self.numLanes+0.5, l1-1 + 0.5) ]

        x = [self.XRTK[0] + S[i]*cos(self.gamma) - L[i]*sin(self.gamma)*self.laneWidths[0] for i in range(0, len(S))]
        y = [self.YRTK[0] + S[i]*sin(self.gamma) + L[i]*cos(self.gamma)*self.laneWidths[0] for i in range(0, len(S))]

        return x, y, S, L

    def getLaneWithShoulder(self, s0, s1, l, obs):
        '''Gives [[x], [y]] road-frame from transform of input s, l\n
        Replaces coordinates where a shoulder is adjacent with nan
        '''
        x, y, S, _ = self.getLane(s0, s1, l, step=0.50)

        for k in range(0, self.numObs):
            _, _, Ss, Ls = self.getShoulder(obs[0][k], obs[1][k], obs[2][k], obs[3][k]) # obs = [[s_start], [s_end], [l_start], [l_end]]

            for j in range(0, len(Ls)):
                dir = Ls[-1]-Ls[0]

                x = [nan if (dir*l <= dir*(Ls[j]+1.0) and S[i] > Ss[0] and S[i] <= Ss[-1]) else x[i] for i in range(0, len(S))]
                y = [nan if (dir*l <= dir*(Ls[j]+1.0) and S[i] > Ss[0] and S[i] <= Ss[-1]) else y[i] for i in range(0, len(S))]

        return x, y

    def getVeh(self, x, y, t, len, wid):
        '''Gives [x0,... ], [y0, ...] as corners of the bounding rectangle of vehicle from input coordinates and orientation'''
        ct, st = cos(t), sin(t)
        xp, yp = [x, x+len*ct], [y, y+len*st]

        x0 = xp[0]+0.5*wid*st
        x1 = xp[1]+0.5*wid*st
        x2 = xp[1]-0.5*wid*st
        x3 = xp[0]-0.5*wid*st

        y0 = yp[0]-0.5*wid*ct
        y1 = yp[1]-0.5*wid*ct
        y2 = yp[1]+0.5*wid*ct
        y3 = yp[0]+0.5*wid*ct

        return [x0, x1, x2, x3, x0], [y0, y1, y2, y3, y0]

    def __init__(self, XRTK = [0., 1.], YRTK = [0., 1.], laneWidths=[LANEWIDTH], laneOrient=0., numTLs=0, numEgos=0, numSvs=0, numObs=0, numDivs=0, useVisual = False, recordVisual = False, HZ = 10, followId=0, doblit = True):
        # Geometry parameters
        self.XRTK = XRTK # Rightmost lane x centerline
        self.YRTK = YRTK # Rightmost lane y centerline
        self.SRTK = [-10, sqrt( (XRTK[-1]-XRTK[0])**2+(YRTK[-1]-YRTK[0])**2 )+10]
        self.laneWidths = laneWidths
        self.numLanes = len(self.laneWidths)
        self.numTLs = numTLs # Traffic lights
        self.numEgos = numEgos # The externally controlled ego vehicles
        self.numSvs = numSvs # The virtual traffic
        self.numObs = numObs # Lane obstacles
        self.numDivs = numDivs # Lane dividers
        self.gamma = laneOrient # The orientation of the straight-road wrt the XY frame
        self.HZ = HZ # Framerate
        self.followId = followId # Id of Sv to follow in camera frame
        
        # Visualization parameters
        self.recordVisual = recordVisual
        self.useVisual = (useVisual or recordVisual)
        
        self.doblit = doblit

        self.lines = [None]*((1+self.numLanes) + (self.numObs+self.numDivs) + self.numTLs + 3*self.numEgos + 2*self.numSvs)
        self.setStatic = True # Set this true in constructor as a flag to draw all non-moving obstacles - set false after first frame

        # Timer parameters
        self.timer = 0. # [s]

        # Drawing
        if self.useVisual:
            ### Setup figure
            self.fig = plt.figure(figsize=(8., 5.))

            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')

            self.ax.set_xlim(min(XRTK)-10, max(XRTK)+10) # Fixed plotting frame
            self.ax.set_ylim(min(YRTK)-10, max(YRTK)+10)
            
            self.ax.tick_params(axis='x', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            self.ax.tick_params(axis='y', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            
            ### Setup artists
            # Lanes
            i = 0
            self.lines[i] = self.ax.plot([nan], [nan], 'k', linewidth = 0.8, animated = True)[0]  # Right shoulder
            
            for _ in range(1, self.numLanes): # Dashed lane dividers
                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], 'k', linewidth = 0.4, linestyle = 'dashed', dashes=(20, 80), animated = True)[0] # Lane divider
            
            i+=1
            self.lines[i] = self.ax.plot([nan], [nan], 'k', linewidth = 0.8, animated = True)[0] # Left shoulder

            # Geometry
            for _ in range(0, self.numObs):
                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], 'k', linewidth = 0.8, animated = True)[0]

            for _ in range(0, self.numDivs): # Solid lane dividers
                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], 'k', linewidth = 0.8, animated = True)[0]

            # Intersections
            for _ in range(0, self.numTLs):
                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], 'g', linewidth = 3.0, animated = True)[0]

            # Ext Driver
            for _ in range(0, self.numEgos):
                color = 'r'

                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], color, linewidth = 1.2, animated = True)[0]
                
                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], marker = "o", markersize = 4, markeredgecolor=color, markerfacecolor=color, animated = True)[0]
                
                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], color, linewidth = 0.8, linestyle = 'dotted', animated = True)[0]

            # SVs
            for _ in range(0, self.numSvs):
                color = 'k'

                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], color, linewidth = 1.2, animated = True)[0]
                
                i+=1
                self.lines[i] = self.ax.plot([nan], [nan], marker = "o", markersize = 4, markeredgecolor=color, markerfacecolor=color, animated = True)[0]

            # Check to follow a vehicle perspective
            xego, yego = None, None

            if self.followId:
                xego, yego = 0, 0

            if xego is not None and yego is not None: # If the frame should follow a vehicle perspective
                skew = 55 # Lead distance in front of ego
                side = 100 # Frame width around ego

                self.ax.set_xlim(xego-side-skew*cos(self.gamma), xego+side+skew*cos(self.gamma))
                self.ax.set_ylim(yego-side-skew*sin(self.gamma), yego+side+skew*sin(self.gamma))

            ### Error check
            assert all(self.lines), 'Lines not fully initialized.'
            assert self.numEgos == 0 or self.numEgos == 1, 'Multiple egos not yet supported.'

            ### Setup doblit
            plt.show(block=False)
            plt.pause(0.1)

            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

            for line in self.lines:
                self.ax.draw_artist(line)

            self.fig.canvas.blit(self.fig.bbox)

        # Recording
        if self.recordVisual:
            # Setup frame grabbing
            self.frames_index = 0

            # Directory for files
            self.frames_cleanup = True

            self.frames_folder = 'imgio/'
            if not os.path.exists(self.frames_folder):
                # Create a new directory because it does not exist
                os.makedirs(self.frames_folder)

                print(f'Made directory {self.frames_folder} to store temporary files.')

    def save(self):
        '''Save current frames to gif'''
        if self.recordVisual:
            # Write gif
            print('Creating gif... ', end='', flush=True)

            self.frames = [None]*(int)(self.frames_index) # Maximum number of frames possible are preallocated

            for i in range(0, self.frames_index):
                image = imageio.v3.imread(f'imgio/img_{i}.png')
                self.frames[i] = image
            
            del self.frames[self.frames_index:]

            imageio.mimsave('sim.gif', self.frames, duration = 1000*1/self.HZ)
            print('Done.')

            # Remove temporary png files
            if self.frames_cleanup:
                for filename in os.listdir(self.frames_folder):
                    file_path = os.path.join(self.frames_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)

                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))

    def draw(self, xego = [], yego = [], tego = [], svs = [], tls = [], obs = [], divs = [], Xego = [], Yego = []):
        if self.useVisual:
            tic = time.time()

            i = 0

            # Plot road
            for k in range(0, self.numLanes+1):
                if self.setStatic:
                    l = (k+0.5)
                    x, y = self.getLaneWithShoulder(self.SRTK[0], self.SRTK[1], l, obs)

                    self.lines[i].set_xdata(x)
                    self.lines[i].set_ydata(y)
                i+=1

            # Plot geometry
            for k in range(0, self.numDivs):
                if self.setStatic:
                    l = 0.5*(divs[2][k]+divs[3][k]) # Divs specifies upper and lower lanes so marking to draw is their average
                    x, y, _, _ = self.getLane(divs[0][k], divs[1][k], l) # div = [[s_start], [s_end], [l_start], [l_end]]

                    self.lines[i].set_xdata(x)
                    self.lines[i].set_ydata(y)
                i+=1

            for k in range(0, self.numObs):
                if self.setStatic:
                    x, y, _, _ = self.getShoulder(obs[0][k], obs[1][k], obs[2][k], obs[3][k]) # obs = [[s_start], [s_end], [l_start], [l_end]]
                    
                    self.lines[i].set_xdata(x)
                    self.lines[i].set_ydata(y)
                i+=1
                
            # Plot lights/intersections
            w = sum(self.laneWidths)+0.2
            for tl in tls:
                color = 'g'
                if tl.status is LIGHTSTATUS.AMBER:
                    color = 'y'
                elif tl.status is LIGHTSTATUS.RED:
                    color = 'r'

                if self.setStatic:
                    x, y = [tl.x+w*(-sin(tl.theta)), tl.x-w*(-sin(tl.theta))], [tl.y+w*(cos(tl.theta)), tl.y-w*(cos(tl.theta))]

                    self.lines[i].set_xdata(x)
                    self.lines[i].set_ydata(y)

                self.lines[i].set_color(color)
                i+=1

            # Plot ego/external vehicles
            for _ in range(0, self.numEgos):
                if xego and yego and tego:
                    x, y = self.getVeh(xego, yego, tego, VEHLENGTH, VEHWIDTH)

                    self.lines[i].set_xdata(x)
                    self.lines[i].set_ydata(y)
                    i+=1

                    self.lines[i].set_xdata([xego])
                    self.lines[i].set_ydata([yego])
                    i+=1

                # Plot ego planned trajectories
                if any(Xego) and any(Yego): 
                    self.lines[i].set_xdata(Xego)
                    self.lines[i].set_ydata(Yego)
                i+=1

            # Plot surrounding vehicles
            for sv in svs:
                x, y = self.getVeh(sv.x, sv.y, sv.theta, sv.len, sv.wid)
                color = 'k'
                if sv.type == VEHTYPE.CAV or sv.type == VEHTYPE.DRLCAV:
                    color = 'b'
                elif sv.type == VEHTYPE.PCC:
                    color = 'g'

                self.lines[i].set_xdata(x)
                self.lines[i].set_ydata(y)
                self.lines[i].set_color(color)
                i+=1

                self.lines[i].set_xdata([sv.x])
                self.lines[i].set_ydata([sv.y])
                self.lines[i].set_color(color)
                self.lines[i].set_markeredgecolor(color)
                self.lines[i].set_markerfacecolor(color)
                i+=1

            ### Set camera frame
            self.fig.canvas.restore_region(self.bg)

            # Check to follow a vehicle perspective
            if (not xego and not yego) and self.followId: # Empty xego and yego arguments so check the followId instead
                svind = [i for i in range(0, self.numSvs) if self.followId == svs[i].id]

                if svind:
                    i = svind[0]
                    xego, yego = svs[i].x, svs[i].y

                else:
                    raise ValueError(f'Did not find vehicle with ID {self.followId} to follow its perspective!')

            if xego or yego: # If the frame should follow a vehicle perspective
                skew = 25 # Lead distance in front of agent the animation is following
                side = 45 # Frame width around of agent the animation is following

                self.ax.set_xlim(xego-side-skew*cos(self.gamma), xego+side+skew*cos(self.gamma))
                self.ax.set_ylim(yego-side-skew*sin(self.gamma), yego+side+skew*sin(self.gamma))

            # Draw all lines
            for line in self.lines:
                self.ax.draw_artist(line)
            
            self.fig.canvas.blit(self.ax.bbox)    
            
            self.fig.canvas.flush_events()

            # Cleanup
            self.setStatic = False

            if self.recordVisual:
                # Record temporary frame - save as png
                self.fig.savefig(f'imgio/img_{self.frames_index}.png', 
                    transparent = False,  
                    facecolor = 'white'
                )
                
                self.frames_index += 1

            self.timer = time.time()-tic
