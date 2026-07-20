#! /usr/bin/env python3

import numpy as np
from math import nan

import pandas as pd
import matplotlib.pyplot as plt
from _constants import *
from itertools import groupby
import math

class logger:
    '''Logging class for microsimulation that keeps the vehicle and traffic light data'''
    def __init__(self, dt, t_max, numTLs, numSVs, filename='data.csv'):
        # Handle input arguments
        self.ts = np.arange(0, t_max, step=dt) # Vector of expected simulation times
        self.n_steps = len(self.ts) # Number of expected steps in simulation to record

        self.numTLs = numTLs # Number of traffic lights
        self.numSVs = numSVs # Number of simulated vehicles

        # Reserve memory for each of the objects for storage
        self.s_tl = dict() # Position of traffic lights
        self.t_tl = dict() # Status of traffic lights

        self.s_sv = dict() # Position of simulated vehicles
        self.v_sv = dict() # Velocity
        self.a_sv = dict() # Acceleration
        self.l_sv = dict() # Lateral position of simulated vehicles
        self.type_sv = dict() # Type of simulated vehicles
        self.lane = dict() # Lane ID
        self.platoon = dict() # Lane ID

        for k in range(0, self.numTLs):
            id = k+1

            self.s_tl[id] = [nan]*self.n_steps
            self.t_tl[id] = [nan]*self.n_steps

        for k in range(0, self.numSVs):
            id = k+1

            self.s_sv[id] = [nan]*self.n_steps
            self.v_sv[id] = [nan]*self.n_steps
            self.a_sv[id] = [nan]*self.n_steps
            self.l_sv[id] = [nan]*self.n_steps
            self.type_sv[id] = [nan]*self.n_steps
            self.lane[id] = [nan]*self.n_steps
            self.platoon[id] = [nan]*self.n_steps

        # Filename
        self.folder = '' # Folder to save filename under
        self.filename = filename # Filename to save data to - the char array <self.folder+self.filename> is the full path to the file
        
        # Error check
        assert dt > 0, 'Logging expects positive dt.'
        assert t_max > 0, 'Logging expects positive t_max.'

    def step(self, t, TLs, SVs):
        '''Get traffic intersection status'''
        # Get the index in the dictionary lists to write the current simulation status to
        t_inds = np.where( abs(self.ts - t) < 1e-6 )[0]

        # Write simulation status if time index is unique and within preallocated buffer
        if len(t_inds) == 1:
            i = t_inds[0]

            for tl in TLs:
                id = tl.id
                # if id > 5:
                #     id = id % 5

                self.s_tl[id][i] = tl.s
                self.t_tl[id][i] = tl.status

            for sv in SVs:
                id = sv.id

                self.s_sv[id][i] = sv.s
                self.v_sv[id][i] = sv.v
                self.a_sv[id][i] = sv.ua
                self.l_sv[id][i] = sv.l
                
                self.type_sv[id][i] = sv.type
                self.lane[id][i] = sv.lane
                self.platoon[id][i] = sv.in_platoon

    def save(self):
        '''Save the simulation data stored in class'''
        # Initialize data to list and header string
        data = []
        header = ''

        # Write simulation data to tuple and header string
        data.append(self.ts) # Time
        header += 't, '

        data.append([self.numTLs]*self.n_steps) # Number of traffic lights
        header += 'ntls, '

        data.append([self.numSVs]*self.n_steps) # Number of simulated vehicles
        header += 'nvehs, '

        for k in range(0, self.numTLs): # Position and status of each traffic light
            id = k+1

            data.append(self.s_tl[id])
            data.append(self.t_tl[id])
            
            header += 'tl{:d}_s, '.format(id)
            header += 'tl{:d}_t, '.format(id)

        for k in range(0, self.numSVs): # Position velocity acceleration lateral position and type of each vehicle
            id = k+1

            data.append(self.s_sv[id])
            data.append(self.v_sv[id])
            data.append(self.a_sv[id])
            data.append(self.l_sv[id])
            data.append(self.type_sv[id])
            data.append(self.lane[id])
            data.append(self.platoon[id])

            header += 'sv{:d}_s, '.format(id)
            header += 'sv{:d}_v, '.format(id)
            header += 'sv{:d}_a, '.format(id)
            header += 'sv{:d}_l, '.format(id)
            header += 'sv{:d}_type, '.format(id)
            header += 'sv{:d}_lane,'.format(id)
            header += 'sv{:d}_platoon'.format(id)

            # End of line logic
            if k < self.numSVs-1: # Don't want the last header entry to have a trailing ','
                header += ', '
            
        # Make data into a tuple
        savedata = tuple(data)

        # Write simulation data to file
        savepath = self.folder+self.filename

        np.savetxt(savepath, np.column_stack(savedata), delimiter=', ', fmt='%0.2f', header=header, comments='')

        # Success
        print('Saved log data to file {:s}'.format(savepath))

class plotting:
    '''Plotting class to read data stored from traffic simulation and plot traffic flow metrics'''
    def getColor(self, veh_type):
        c = 'k'

        if veh_type == VEHTYPE.NONE: # From _CONSTANTS -> NONE=-1, MOBIL=0, CAV=1, PCC=2
            c = 'k'
            linewidth_veh = 0.5
        elif veh_type == VEHTYPE.MOBIL:
            c = 'k'
            linewidth_veh = 0.5
        elif veh_type == VEHTYPE.CAV or veh_type == VEHTYPE.DRLCAV:
            c = 'b'
            linewidth_veh = 1
        elif veh_type == VEHTYPE.PCC:
            c = 'g'
            linewidth_veh = 1
        else:
            raise ValueError('Unknown veh type in plotting!')
        
        return c, linewidth_veh
    
    def getline(self, lane):
        linetype = None
        if lane == 1:
            linetype = 'dotted'
        elif lane == 2:
            linetype = 'solid'
        elif lane == 3:
            linetype = 'dashed'
        else:
            raise ValueError('Unknown lane in plotting!')
        return linetype
        
    def getLabel(self, veh_type, labels):
        label = None

        if veh_type == VEHTYPE.NONE:
            pass
        elif veh_type == VEHTYPE.MOBIL:
            label = 'HV'
        elif veh_type == VEHTYPE.CAV:
            label = 'CAV'
        elif veh_type == VEHTYPE.DRLCAV:
            label = 'DRLCAV'
        elif veh_type == VEHTYPE.PCC:
            label = 'AV'
        else:
            raise ValueError('Unknown veh type in plotting!')

        if veh_type in labels.keys():
            label = None

        else:
            labels[veh_type] = label

        return label, labels

    def __init__(self, filename = 'data.csv',limit_l = 1, show_figure=True):
        ### Settings
        LINEWIDTH = 1

        ### Read the stored csv file from filename
        data = pd.read_csv(filename, sep=', ', engine='python')

        n_rows = data.shape[0]

        # Get number vehicles and number traffic lights
        ntls = int(data['ntls'][0])
        nvehs = int(data['nvehs'][0])

        ### Extract data
        t = data['t']

        # Traffic light red status
        tls_s = []
        tls_is_red = []
        
        for n in range(1, ntls+1):
            # Dictionary key for lookup
            s_key = 'tl{:d}_s'.format(n)
            t_key = 'tl{:d}_t'.format(n)

            # Get data
            is_red = [s == LIGHTSTATUS.RED for s in data[t_key]]
            tl_s = [data[s_key][k] if is_red[k] else nan for k in range(0, n_rows)]
            
            tls_is_red.append(is_red)
            tls_s.append(tl_s)

        # Vehicle status
        svs_s = []
        svs_v = []
        svs_t = []
        svs_l = []
        svs_a = []
        svs_lane = []
        svs_type = []

        for n in range(1, nvehs+1):
            # Dictionary key for lookup
            s_key = 'sv{:d}_s'.format(n)
            v_key = 'sv{:d}_v'.format(n)
            t_key = 'sv{:d}_type'.format(n)
            l_key = 'sv{:d}_l'.format(n)
            a_key = 'sv{:d}_a'.format(n)
            lane_key = 'sv{:d}_lane'.format(n)
            type_key = 'sv{:d}_type'.format(n)

            # Get data
            svs_s.append(data[s_key])
            svs_v.append(data[v_key])
            svs_t.append(data[t_key][1])
            svs_l.append(data[l_key])
            svs_a.append(data[a_key])
            svs_lane.append(data[lane_key])
            svs_type.append(data[lane_key])
    

        if show_figure == False:
            # number of lane change
            num_LC = 0
        
            for n in range(0, nvehs):
                for value, group in groupby(svs_lane[n]):
                    if math.isnan(value):
                        continue
                    if svs_type[n][1] == 3:
                        num_LC += 1
                if svs_type[n][1] == 3:
                    num_LC -= 1
        else:
            ### Plot time position trajectories through corridor
            plt.figure()

            for n in range(0, ntls):
                # Plot red lights only
                plt.plot(t, tls_s[n], color='r', label=None, linewidth=LINEWIDTH)
            num_LC = 0
            for n in range(0, nvehs):
                # Select line color based on vehicle type
                veh_type = svs_t[n]
                c, linewidth_veh = self.getColor(veh_type)

                # select line linetype based on lane
                start = 0
                
                for value, group in groupby(svs_lane[n]):
                    if math.isnan(value):
                        continue
                    if svs_type[n][1] == 3:
                        num_LC += 1
                    length = len(list(group))
                    index = list(range(start, start + length))
                    linetype = self.getline(int(value))
                    start += length
                    # Plot vehicle trajectory
                    plt.plot(t[index], svs_s[n][index], color = c, linestyle=linetype, label=f'Lane{value}', linewidth = linewidth_veh)
                if svs_type[n][1] == 3:
                    num_LC -= 1
                # if n == 0:
                    # plt.scatter(t[69], svs_s[n][69],color='red')
                plt.xlabel('Time [s]')
                plt.ylabel('Position [m]')

            plt.draw()
            plt.show()

            ### Plot velocity trajectories through corridor
            plt.figure()
        
            for n in range(0, nvehs):
                # Select line color based on vehicle type
                veh_type = svs_t[n]
                c, linewidth = self.getColor(veh_type)
                
                # Plot vehicle trajectory
                plt.plot(svs_s[n], svs_v[n], color = c, label=None, linewidth = LINEWIDTH)

                plt.xlabel('Position [m]')
                plt.ylabel('Velocity [m/s]')

            plt.draw()


            plt.legend()
            # Show plots
            plt.show()
        # lane = np.unique(svs_l)
        # numl = len(lane[~np.isnan(lane)])
        # nvehs_l = int(nvehs/numl)
        # for i in range(0, numl):
        #     if i < limit_l:
        #         pass
        
        # printing
        energy = []
        speed = []
        for n in range(0, nvehs):
            energy.append(np.sum(abs(np.multiply(svs_a[n][1:]**2,np.diff(t)))))
            speed.append(np.average(svs_v[n][1:]))
        print('Energy:', np.average(energy),', Speed:', np.average(speed),', numLC:', num_LC)
        self.avr_energy = np.average(energy)
        self.avr_speed = np.average(speed)
        self.num_LC = num_LC

    def result_calculation(self):
        return self.avr_energy, self.avr_speed, self.num_LC

        

        