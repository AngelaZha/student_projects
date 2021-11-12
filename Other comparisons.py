#!/usr/bin/env python

'''
estimate attitude from an ArduPilot replay log using a python state estimator
'''
from __future__ import print_function
from builtins import range

import os

from MAVProxy.modules.lib import mp_util
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("log", metavar="LOG")
parser.add_argument("--debug", action='store_true')
parser.add_argument('--param', type= str, default = 'true', help='type true to set param same as EKF')

args = parser.parse_args()
param = args.param

from pymavlink import mavutil
from pymavlink.rotmat import Vector3, Matrix3
from pymavlink.mavextra import expected_earth_field_lat_lon
import math
from math import degrees

GRAVITY_MSS = 9.80665

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.linalg import logm
from math import pi
import random
import utm

from scipy.interpolate import interp1d

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#plt.close("all")

from statistics import mean

def skew_to_vec (skew):
    return np.array([[skew[2][1]],[skew[0][2]],[skew[1][0]]])

###Load SITL, EqF and EKF pickle data for comparison
import pickle

if param == 'true':
    print("Using EKF3 parameter")
elif param =='false':
    print("Using default parameter")
else:
    print('Error! Not sure what\'s the param setting')

f = open('SITLoutput_' + args.log + '.pckl', 'rb')
sitl = pickle.load(f)
f.close()
f = open('EqFoutput_' + args.log + ' EKparam ' + param + '.pckl', 'rb')
EqF = pickle.load(f)
f.close()
f = open('EKFoutput_' + args.log + '.pckl', 'rb')
ekf = pickle.load(f)
f.close()

output = {**sitl, **EqF, **ekf}
filename = args.log

if "nobias" in args.log:
    EqFbias = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    EKFinternal_bias = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    print("Using 0 for true bias")
else:
    EqFbias = np.array([[-0.01],[-0.02],[-0.03],[0.1],[0.2],[0.2]])
    dtavg = output['EK3.dtaverage'][0]
    EKFinternal_bias = dtavg * np.array([[-0.01],[-0.02],[-0.03],[0.1],[0.2],[0.2]])
    print("Setting true bias value")

# # graph all the fields we've output
# import matplotlib.pyplot as plt
# datatype = ["Roll", "Pitch", "Yaw"]
# fig, axs = plt.subplots(len(datatype),1)
# for i in range(len(datatype)):
#     for k in output.keys():        
#         if datatype[i] in k:
#             t = [ v[0] for v in output[k] ]
#             y = [ v[1] for v in output[k] ]
#             axs[i].plot(t, y, label=k)
#             maxt = t[-1]
# plt.legend(loc='upper left')
# fig.suptitle('Estimated attitude with baro+mag for %s log using WMM' %filename)
# plt.setp(axs[-1], xlabel='Time(s)')
# plt.setp(axs[0], ylabel='Roll(Deg)')
# plt.setp(axs[1], ylabel='Pitch(Deg)')
# plt.setp(axs[2], ylabel='Yaw(Deg)')
# # axs[0].set_ylim([-180, 180])
# # axs[1].set_ylim([-90, 90])
# axs[2].set_ylim([0, 360])
# plt.setp(axs, xlim=[0,maxt])
# plt.show()

#Compare relative N,E,D positions

def rotmat_xyz2NED(lat, long):
    R1 = np.array([-math.sin(math.radians(lat)) * math.cos(math.radians(long)),
                    -math.sin(math.radians(lat)) * math.sin(math.radians(long)),
                    math.cos(math.radians(lat))
                    ])
    R2 = np.array([-math.sin(math.radians(long)),
                    math.cos(math.radians(long)),
                    0
                    ])
    R3 = np.array([-math.cos(math.radians(lat)) * math.cos(math.radians(long)),
                    -math.cos(math.radians(lat)) * math.sin(math.radians(long)),
                    -math.sin(math.radians(lat))
                    ])
    
    R_0 = np.array([R1,R2,R3])
    return R_0

def lla2xyz(lat,long,alt,earthradius): #converts lat,long,alt coord to x,y,z coord
    p = (alt+ earthradius) * np.array([
            np.cos(np.radians(lat)) * np.cos(np.radians(long)),
            np.cos(np.radians(lat)) * np.sin(np.radians(long)),
            np.sin(np.radians(lat))
            ])
    return p

R0 = rotmat_xyz2NED(output['SIM.lat'][0],output['SIM.long'][0])

earthrad = 6.3781*1.0e6 #mean radius is 6371km, but here follows convention used in mp_util.py from Mavproxy module
p = lla2xyz(output['SIM.lat'],output['SIM.long'],np.array(output['SIM.alt']),earthrad)
p_rel = R0 @ p

#using AP default SIM2 PN PE and PD as comparison
sim2_N = np.array(output['SIM2.Xi_p'])[:,0,0]
sim2_N = sim2_N - sim2_N[0]
sim2_E = np.array(output['SIM2.Xi_p'])[:,1,0]
sim2_E = sim2_E - sim2_E[0]
sim2_D = np.array(output['SIM2.Xi_p'])[:,2,0]
sim2_D = sim2_D - sim2_D[0]
p_rel1 = np.vstack((sim2_N,sim2_E,sim2_D))##########

pos_rel = R0 @ (p.T - p[:,0]).T #(p_rel.T - p_rel[:,0]).T

# fig9, ax9 = plt.subplots(3)
# fig9.suptitle('Estimated position with baro+mag on nolag2 log')
# # ax9[0].plot(output['EK3.time'], EK3_N)
# ax9[0].plot(output['SIM.time'], pos_rel[0,:],'-.')
# ax9[0].plot(output['SIM2.time'], sim2_N)
# # ax9[0].plot(output['EqF.time'], EqF_N, ':')
# ax9[0].set_ylabel('North (m)')
# # ax9[1].plot(output['EK3.time'], EK3_E)
# ax9[1].plot(output['SIM.time'], pos_rel[1,:],'-.')
# ax9[1].plot(output['SIM2.time'], sim2_E)
# # ax9[1].plot(output['EqF.time'], EqF_E, ':')
# ax9[1].set_ylabel('East (m)')
# # ax9[2].plot(output['EK3.time'],EK3_D,label='EK3')
# ax9[2].plot(output['SIM.time'], pos_rel[-1,:],'-.',label='SITL using lla to ned')
# ax9[2].plot(output['SIM2.time'], sim2_D, label='SITL from SIM2')
# # ax9[2].plot(output['EqF.time'], EqF_D, ':', label='EqF')
# ax9[2].legend(loc='best')
# ax9[2].set_xlabel('Time(s)')
# ax9[2].set_ylabel('Down (m)')
    
#Calculate origin of EKF3 in x,y,z coordinate
EK3_org = lla2xyz(output['EK3org.lat'][0], output['EK3org.long'][0], output['EK3org.alt'][0], earthrad)
R0_EK3 = rotmat_xyz2NED(output['EK3org.lat'][0], output['EK3org.long'][0])

diff = R0_EK3 @ EK3_org #- R0 @ p[:,0]
EK3_N = output['EK3.north'] + diff[0]
EK3_E = output['EK3.east'] + diff[1]
EK3_D = output['EK3.down'] + diff[2]

EqF_org = lla2xyz(output['EqF.origin'][0].x,output['EqF.origin'][0].y,output['EqF.origin'][0].z,earthrad)
R0_EqF = rotmat_xyz2NED(output['EqF.origin'][0].x, output['EqF.origin'][0].y)

diff2 = R0_EqF @ EqF_org #- R0 @ p[:,0]
EqF_N = output['EqF.posx'] + diff2[0]
EqF_E = output['EqF.posy'] + diff2[1]
EqF_D = output['EqF.posz'] + diff2[2]

####
# diff3 = R0_EqF @ EqF_org - R0 @ p[:,0]
# pos_rel = (pos_rel.T + diff3).T
# print('size of diff3 is ', diff3.shape)

# #plot = ["North","East","Down"]
# fig1, ax1 = plt.subplots(3)
# fig1.suptitle('Estimated position with baro+mag on nolag2 log')
# ax1[0].plot(output['EK3.time'], EK3_N)
# ax1[0].plot(output['SIM.time'], p_rel[0,:],'-.')
# ax1[0].plot(output['EqF.time'], EqF_N, ':')
# ax1[0].set_ylabel('North (m)')
# ax1[1].plot(output['EK3.time'], EK3_E)
# ax1[1].plot(output['SIM.time'], p_rel[1,:],'-.')
# ax1[1].plot(output['EqF.time'], EqF_E, ':')
# ax1[1].set_ylabel('East (m)')
# ax1[2].plot(output['EK3.time'],EK3_D,label='EK3')
# ax1[2].plot(output['SIM.time'],p_rel[-1,:],'-.',label='SITL')
# ax1[2].plot(output['EqF.time'], EqF_D, ':', label='EqF')
# ax1[2].legend(loc='best')
# ax1[2].set_xlabel('Time(s)')
# ax1[2].set_ylabel('Down (m)')
# plt.setp(ax1, xlim=[0,output['SIM.time'][-1]])

#RMSE of position, using interpolation to match size
def interp_func(x,y,newx):
    f = interp1d(x,y)
    newy = f(newx)
    return newy

i = np.searchsorted(output['SIM.time'], output['EK3.time'][0])
l = np.searchsorted(output['SIM.time'],output['EK3.time'][-1],side ='right')
EK3_N_new =interp_func(output['EK3.time'],EK3_N,output['SIM.time'][i:l])
EK3_E_new =interp_func(output['EK3.time'],EK3_E,output['SIM.time'][i:l])
EK3_D_new =interp_func(output['EK3.time'],EK3_D,output['SIM.time'][i:l])
rse = np.sqrt(np.square(p_rel[0,i:l] - EK3_N_new) + np.square(p_rel[1,i:l] - EK3_E_new) + 
              np.square(p_rel[2,i:l] - EK3_D_new))

j = np.searchsorted(output['EqF.time'],output['SIM.time'][0])
k = np.searchsorted(output['EqF.time'],output['SIM.time'][-1],side ='right')
sitl_N =interp_func(output['SIM.time'],p_rel[0,:],output['EqF.time'][j:k])
sitl_E =interp_func(output['SIM.time'],p_rel[1,:],output['EqF.time'][j:k])
sitl_D =interp_func(output['SIM.time'],p_rel[2,:],output['EqF.time'][j:k])
eqf_rse = np.sqrt(np.square(sitl_N - EqF_N[j:k]) + np.square(sitl_E - EqF_E[j:k]) + 
              np.square(sitl_D - EqF_D[j:k]))

# plt.figure() # Here's the part I need
# plt.plot(output['SIM.time'][i:l], (rse),label = 'EK3')
# plt.plot(output['EqF.time'][j:k],(eqf_rse), label = 'EqF')
# plt.yscale("log")
# plt.legend(loc='best')
# plt.title('NED estimated positions RSE for %s using WMM, using lla converted position as ground truth' %filename)
# plt.xlabel('Time(s)')
# plt.ylabel('m')
# plt.show()


# import pickle
# f = open('With_b_mat.pckl', 'rb')
# eqf_bmat_rse = pickle.load(f)
# f.close()

# diff = eqf_bmat_rse - eqf_rse
# plt.figure(3)
# plt.plot(output['EqF.time'][j:k], (diff))
# plt.yscale("log")
# plt.title('RSE diff comparison with/without bias (+ve if improvement)')
# plt.xlabel('Time(s)')
# plt.ylabel('m')

# plt.figure(4)
# plt.plot(output['EqF.time'][j:k], eqf_bmat_rse, label = 'EqF with B matrix')
# plt.plot(output['EqF.time'][j:k], eqf_rse, label = 'EqF with bias')
# #plt.plot(output['SIM.time'][i:], rse,label = 'EK3')
# plt.yscale("log")
# plt.title('RSE estimated position comparison for adding bias estimation')
# plt.xlabel('Time(s)')
# plt.ylabel('m')
# plt.legend(loc='best')

# plt.figure(5)
# plt.plot(output['baro.time'],output['plot.baro'])
# plt.title('Baro measurements')
# plt.xlabel('Time(s)')
# plt.ylabel('m')

# fig2, ax2 = plt.subplots(3)
# ax2[0].plot(output['plot.magtime'], output['plot.magX'])
# fig2.suptitle('Mag output measurements')
# ax2[0].set_title('magX')
# ax2[1].plot(output['plot.magtime'], output['plot.magY'])
# ax2[1].set_title('magY')
# ax2[2].plot(output['plot.magtime'], output['plot.magZ'])
# ax2[2].set_title('magZ')
# ax2[2].set_xlabel('Time(s)')
# ax2[2].set_ylabel('(milliGauss?)')


sitl_alt = output['SIM.alt'][0] - np.array(output['SIM.alt']) 
plt.figure()
plt.plot(output['SIM.time'],sitl_alt,'-.',label='SITL altitude')
plt.plot(output['baro.time'],np.negative(output['plot.baro']),label = 'Baro altitude')
plt.title('SITL vs barometric relative altitude measurements')
plt.legend(loc='best')
plt.xlabel('Time(s)')
plt.ylabel('m')

# avg_N = mean(output['plot.normmagN'])
# avg_E = mean(output['plot.normmagE'])
# avg_D = mean(output['plot.normmagD'])
# print('avg of normmagN is', avg_N, '\n')
# print('avg of normmagE is', avg_E, '\n')
# print('avg of normmagD is', avg_D, '\n')

# print('WMM expected magN is', output['mag.earthN'][-1], '\n')
# print('WMM expected magE is',output['mag.earthE'][-1], '\n')
# print('WMM expected magD is', output['mag.earthD'][-1], '\n')

# print('SITL expected magN is', inertial_NEDmag[0][0], '\n')
# print('SITL expected magE is', inertial_NEDmag[1][0], '\n')
# print('SITL expected magD is', inertial_NEDmag[2][0], '\n')

# #Comparing EKF3 NED mag estimates, EqF NED mag estimates using averaged inertial mag; reference line as WMM inertial mag and average inertial mag
# fig3, ax3 = plt.subplots(3)
# ax3[0].plot(output['EKF.magtime'], output['EK3.magN'],'--')
# ax3[0].plot(output['plot.magtime'], output['plot.normmagN'],'-.')
# ax3[0].plot(output['plot.magtime'], output['mag.earthN'], c = "green")
# ax3[0].axhline(y=0.395346176504478,xmin=0,xmax= output['plot.magtime'][-1],c="red")
# ax3[0].axhline(y=inertial_NEDmag[0][0],xmin=0,xmax= output['plot.magtime'][-1],c="blue")
# plt.suptitle('Normalised mag in inertial NED frame on nolag2 log')
# ax3[0].set_title('Mag N')
# ax3[1].plot(output['EKF.magtime'], output['EK3.magE'], '--', label = 'EK3 NED mag strength')
# ax3[1].plot(output['plot.magtime'], output['plot.normmagE'], '-.', label='R_Ahat @ norm_mag using avg mag')
# ax3[1].plot(output['plot.magtime'], output['mag.earthE'], c = "green", label = 'WMM expected mag readings')
# ax3[1].axhline(y=0.11012878308931125,xmin=0,xmax= output['plot.magtime'][-1],c="red", label='Average mag readings on bias log')
# ax3[1].axhline(y=inertial_NEDmag[1][0],xmin=0,xmax= output['plot.magtime'][-1],c="blue", label='SITL inertial NED mag')
# ax3[1].set_title('Mag E')
# ax3[1].legend(loc='best')
# ax3[2].plot(output['EKF.magtime'], output['EK3.magD'], '--')
# ax3[2].plot(output['plot.magtime'], output['plot.normmagD'],'-.')
# ax3[2].plot(output['plot.magtime'], output['mag.earthD'],c = "green")
# ax3[2].axhline(y=-0.9114719248336718,xmin=0,xmax= output['plot.magtime'][-1],c="red")
# ax3[2].axhline(y=inertial_NEDmag[2][0],xmin=0,xmax= output['plot.magtime'][-1],c="blue")
# ax3[2].set_title('Mag D')
# ax3[2].set_xlabel('Time(s)')
# ax3[2].set_ylabel('N,E,D')

#Plot gyro bias
fig4, ax4 = plt.subplots(3)
ax4[0].plot(output['EK3.time'], np.array(output['EK3.biasgx'])*(pi/180),c = 'mediumblue')
ax4[0].plot(output['EqF.time'], output['EqF.biasgx'],c = 'orangered', ls='--')
ax4[0].axhline(y=EqFbias[0][0],xmin=0,xmax= output['EqF.time'][-1], c = 'g', ls = '-.', linewidth=1)
fig4.suptitle('Gyroscope Bias Estimate')
# ax4[0].set_title('Gyroscope Bias In Body X-axis')
ax4[0].set_ylabel('X axis')
ax4[1].plot(output['EK3.time'], np.array(output['EK3.biasgy'])*(pi/180),c = 'mediumblue')
ax4[1].plot(output['EqF.time'], output['EqF.biasgy'],c = 'orangered', ls='--')
ax4[1].axhline(y=EqFbias[1][0],xmin=0,xmax= output['EqF.time'][-1],c='g',ls = '-.',linewidth=1)
# ax4[1].set_title('Gyroscope Bias In Body Y-Axis')
ax4[1].set_ylabel('Y axis')
ax4[2].plot(output['EK3.time'], np.array(output['EK3.biasgz'])*(pi/180),c = 'mediumblue',label='EKF3')
ax4[2].plot(output['EqF.time'], output['EqF.biasgz'],c = 'orangered', ls='--',label='EqF')
ax4[2].axhline(y=EqFbias[2][0],xmin=0,xmax= output['EqF.time'][-1],c='g',ls = '-.',linewidth=1,label='True Gyroscope Bias')
# ax4[2].set_title('Gyroscope Bias In Body Z-Axis')
ax4[2].legend(loc='best', prop={'size': 8})
ax4[2].set_xlabel('Time (s)')
ax4[2].set_ylabel('Z axis Estimate (rad$s^{-1}$)')
plt.setp(ax4, xlim=[0,output['EK3.time'][-1]])
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18), originally 19.2, 9.67
fig4.savefig('Graphs for simulation set 2/'+ args.log+'/Gyro Bias ' + args.log + '.pdf', bbox_inches='tight')

#Plot accel bias
fig5, ax5 = plt.subplots(3)
ax5[0].plot(output['XKF.time'], output['EK3.biasax'],c = 'mediumblue')
ax5[0].plot(output['EqF.time'], output['EqF.biasax'],c = 'orangered', ls='--')
fig5.suptitle('Accelerometer Bias Estimate')
ax5[0].axhline(y=EqFbias[3][0],xmin=0,xmax= output['EqF.time'][-1],c = 'g', ls = '-.', linewidth=1)
# ax5[0].set_title('Accel Bias X')
ax5[0].set_ylabel('X axis')
ax5[1].plot(output['XKF.time'], output['EK3.biasay'],c = 'mediumblue')
ax5[1].plot(output['EqF.time'], output['EqF.biasay'],c = 'orangered', ls='--')
ax5[1].axhline(y=EqFbias[4][0],xmin=0,xmax= output['EqF.time'][-1],c = 'g', ls = '-.', linewidth=1)
# ax5[1].set_title('Accel Bias Y')
ax5[1].set_ylabel('Y axis')
ax5[2].plot(output['XKF.time'], output['EK3.biasaz'],c = 'mediumblue',label='EKF3')
ax5[2].plot(output['EqF.time'], output['EqF.biasaz'],c = 'orangered', ls='--',label='EqF')
ax5[2].axhline(y=EqFbias[5][0],xmin=0,xmax= output['EqF.time'][-1],c = 'g', ls = '-.', linewidth=1,label='True Accelerometer Bias')
# ax5[2].set_title('Accel Bias Z')
ax5[2].legend(loc='best', prop={'size': 8})
ax5[2].set_xlabel('Time(s)')
ax5[2].set_ylabel('Z axis Estimate ($ms^{-2}$)')
plt.setp(ax5, xlim=[0,output['EK3.time'][-1]])
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
fig5.savefig('Graphs for simulation set 2/'+ args.log+'/Accel Bias ' + args.log + '.pdf', bbox_inches='tight')

fig6, ax6 = plt.subplots(3)
ax6[0].plot(output['EqF.time'], output['EqF.velx'])
ax6[0].plot((output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,0,0])
ax6[0].plot((output['EK3.time']),output['EK3.VN'])
fig6.suptitle('Velocity NED')
ax6[0].set_title('Velocity North')
ax6[1].plot(output['EqF.time'], output['EqF.vely'])
ax6[1].plot((output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,1,0])
ax6[1].plot((output['EK3.time']),output['EK3.VE'])
ax6[1].set_title('Velocity East')
ax6[2].plot(output['EqF.time'], output['EqF.velz'],label='EqF estimated velocity')
ax6[2].plot((output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,2,0], label = 'SITL velocity')
ax6[2].plot((output['EK3.time']),output['EK3.VD'], label='EKF estimated velocity')
ax6[2].set_title('Velocity Down')
ax6[2].legend(loc='best')
ax6[2].set_xlabel('Time(s)')
# ax6[2].set_ylabel('$ms^{-1}$')
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
fig6.savefig('Graphs for simulation set 2/'+ args.log+'/EqF estimated velocity ' + args.log + '.pdf', bbox_inches='tight')

##############################################
#Plot NEES
#First compute EqF NEES
j = np.searchsorted(output['EqF.GPStime'],output['SIM.time'][0])
k = np.searchsorted(output['EqF.GPStime'],output['SIM.time'][-1],side ='right')
Xi_p = np.empty([3,len(output['EqF.GPStime'][j:k])])
Xi_v = np.empty([3,len(output['EqF.GPStime'][j:k])])

for i in range(3):
    #Correct SIM2 PN, PE and PD which is from home to a relative position 
    # sim2_pos = np.array(output['SIM2.Xi_p'])[:,i,0]
    # sim2_pos = sim2_pos - sim2_pos[0]
    # Xi_p[i,:] = interp_func(np.array(output['SIM2.time']),sim2_pos, np.array(output['EqF.GPStime'][j:k]))  
    
    #interpolate SITL position to match EqF time
    Xi_p[i,:] = interp_func(np.array(output['SIM.time']),pos_rel[i,:], np.array(output['EqF.GPStime'][j:k]))         
    #interpolate SITL velocity to match EqF time
    Xi_v[i,:] = interp_func(np.array(output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,i,0], np.array(output['EqF.GPStime'][j:k]))         
p_inv = np.array(output['EqF.p_inv'][j:k])
v_inv = np.array(output['EqF.v_inv'][j:k])

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
#Interpolate SITL rotation to match EqF time
def slerp_interp(x,y,newx):
    rotation = R.from_matrix(np.array(y))
    slerp = Slerp(np.array(x), rotation)
    interp_rot = slerp(newx)
    return interp_rot.as_matrix()
Xi_R = slerp_interp(output['SIM.time'],output['SIM.Xi_R'], output['EqF.GPStime'][j:k])
R_inv = np.array(output['EqF.R_inv'][j:k])

#Apply right action for rotation and apply error map
R_action = Xi_R @ R_inv
epsilon_R = [logm(i) for i in R_action]
epsilon_R = np.array([skew_to_vec(i) for i in epsilon_R]).T.reshape(3,-1)
#Apply right action to position and velocity, error map they are kept the same
epsilon_p = Xi_p + (Xi_R @ p_inv).T.reshape(3,-1)
epsilon_v = Xi_v + (Xi_R @ v_inv).T.reshape(3,-1)

epsilon = np.vstack((epsilon_R, epsilon_p, epsilon_v))
bias_error = EqFbias - (np.array(output['EqF.bhat'][j:k])).T.reshape(6,-1)
error = np.vstack((epsilon, bias_error))
ric = np.linalg.inv(np.array(output['EqF.Riccati'][j:k]))
eqf_e = error.T.reshape(len(output['EqF.GPStime'][j:k]),len(error),1)
eqf_eT = error.T.reshape(len(output['EqF.GPStime'][j:k]),1,len(error))
EqF_NEES = (eqf_eT @ ric @ eqf_e).reshape(k-j)

# error2 = np.vstack((epsilon_R, epsilon_v, bias_error)) #check NEES without position error
# ric2 = ric[:,[*range(0,3), *range(6,15)],:]
# ric2 = ric2[:,:,[*range(0,3), *range(6,15)]]
# eqf_e2 = error2.T.reshape(len(output['EqF.GPStime'][j:k]),len(error2),1)
# eqf_e2T = error2.T.reshape(len(output['EqF.GPStime'][j:k]),1,len(error2))
# EqF2_NEES = (eqf_e2T @ ric2 @ eqf_e2).reshape(k-j)

#Compute EKF NEES
#Use SVAA.time as base time
m = np.searchsorted(output['SVAA.time'],output['SIM.time'][0])
n = np.searchsorted(output['SVAA.time'],output['SIM.time'][-1],side ='right')

#Interpoalte SITL position and velocity to match SVAA.time
#Because number of SVAA time stamp is large, reduce by cutting in timesteps
step = 10
Xi2_p = np.empty([3,len(output['SVAA.time'][m:n:step])])
Xi2_v = np.empty([3,len(output['SVAA.time'][m:n:step])])
for i in range(3):
    #Correct SIM2 PN, PE and PD which is from home to a relative position 
    #sim2_pos = np.array(output['SIM2.Xi_p'])[:,i,0]
    #sim2_pos = sim2_pos - sim2_pos[0]
    #Xi2_p[i,:] = interp_func(np.array(output['SIM2.time']),sim2_pos, np.array(output['SVAA.time'][m:n:step]))
    
    #interpolate SITL position
    Xi2_p[i,:] = interp_func(np.array(output['SIM.time']),pos_rel[i,:], np.array(output['SVAA.time'][m:n:step]))         
    #interpolate SITL velocity
    Xi2_v[i,:] = interp_func(np.array(output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,i,0], np.array(output['SVAA.time'][m:n:step]))         

#Interpolate SITL quaternion to match SVAA.time
def slerp_interp_quat(x,y,newx):
    y = np.roll(np.array(y),-1, axis =1) #rotation object store real (scalar) components last
    quaternions = R.from_quat(np.array(y))
    slerp = Slerp(np.array(x), quaternions)
    interp_quat = slerp(newx)
    return np.roll(interp_quat.as_quat(),1, axis = 1)
SIM_quat = slerp_interp_quat(output['SIM.time'],output['SIM.quaternion'], output['SVAA.time'][m:n:step])

#EKF NEES
EKFstate_0_14 = np.array(output['SVAA.state'][m:n:step]) #Cut EKF states to match SVAA time
EKFstate_15 = np.array(output['SVAB.state'][m:n:step])
length = len(range(m,n,step)) #work out the length of new time
#Compute the quaternion difference
ekf_quat = EKFstate_0_14[:,0:4]
ekf_quat = R.from_quat(np.roll(ekf_quat,-1,axis=1)) #roll to have quaternion in the correct order (x,y,z,w) real part last
SITL_quat= R.from_quat(np.roll(SIM_quat,-1,axis=1))
SITL_quat_inv = SITL_quat.inv()

quat_diff = ekf_quat * SITL_quat_inv
quat_diff = np.roll(quat_diff.as_quat(),1,axis=1) #roll to make quaternion real part come first (column 0)
#quaternion_diff = abs(abs(quat_diff) - np.array([1,0,0,0]))
positive_real_quat_diff = np.sign(quat_diff[:,0])[:,None] * quat_diff
diff = (positive_real_quat_diff - np.array([1,0,0,0]))
quaternion_diff = diff

#Compute difference for vel, pos and bias
bias = np.tile(EKFinternal_bias,length)
SITL_velposbias = np.vstack((Xi2_v, Xi2_p, bias))
EKF_velposbias = np.vstack((EKFstate_0_14[:,4:].T, EKFstate_15.T))
err_velposbias = EKF_velposbias - SITL_velposbias

#Error for all 16 states and inverse the covariance
er = np.vstack((quaternion_diff.T, err_velposbias))
covariance = np.array(output['EKF.Covariances'][m:n:step])
cov_trim = np.linalg.pinv(covariance[:,0:16,0:16]) #what to do if can't do inverse of covariance? Currently just doing pseudo-inverse

ekf_e = er.T.reshape(length, len(er), 1)
ekf_eT = er.T.reshape(length, 1, len(er))
ekf_NEES = (ekf_eT @ cov_trim @ ekf_e).reshape(length) #EKF NEES for the first 16 states

er2 = er[[*range(4,16)],40:] #check NEES altitude error
cov_trim2 = np.linalg.inv(covariance[40:,4:16,4:16])
# cov_trim2 = cov_trim[:,[*range(4,16)],:]
# cov_trim2 = cov_trim2[:,:,[*range(4,16)]]
ekf_e2 = er2.T.reshape(length-40,len(er2),1)
ekf_e2T = er2.T.reshape(length-40,1,len(er2))
ekf2_NEES = (ekf_e2T @ cov_trim2 @ ekf_e2).reshape(length-40)

EKFtime = np.array(output['SVAA.time'][m:n:step])
# plt.figure()
# plt.plot(output['EqF.GPStime'][j:k], EqF_NEES, label = 'EqF NEES (all 15 states)')
# EKFtime_trim = EKFtime[40:]
# # trim = np.logical_and(ekf2_NEES >0, ekf2_NEES>0) #cut values too big or negative
# # EKFtime_trim = EKFtime_trim[trim]
# # ekf2_NEES = ekf2_NEES[trim]
# plt.plot(EKFtime[40:], ekf2_NEES, label = 'EKF NEES (pos,vel and bias)')
# plt.title('Normalised estimation error squared (NEES) (on %s log)' %filename)
# plt.legend(loc='best')
# #plt.ylim([0, 8.5])
# plt.xlabel('Time(s)')
# plt.ylabel('NEES')

####################################

#Plot state estimate error squared
error_sqr = (eqf_eT @ eqf_e).reshape(k-j)
EKF_error_sqr = (ekf_eT @ ekf_e).reshape(length)

#separate out EKF error squared to attitude, vel, pos
EKF_att = (ekf_eT[:,:,0:4] @ ekf_e[:,0:4,:]).reshape(length)
EKF_vel = (ekf_eT[:,:,4:7] @ ekf_e[:,4:7,:]).reshape(length)
EKF_pos = (ekf_eT[:,:,7:10] @ ekf_e[:,7:10,:]).reshape(length)

#separate out EKF error squared to attitude, vel, pos
EqF_att = (eqf_eT[:,:,0:3] @ eqf_e[:,0:3,:]).reshape(k-j)
EqF_pos = (eqf_eT[:,:,3:6] @ eqf_e[:,3:6,:]).reshape(k-j)
EqF_vel = (eqf_eT[:,:,6:9] @ eqf_e[:,6:9,:]).reshape(k-j)

# plt.figure()
# plt.plot(output['EqF.GPStime'][j:k], error_sqr, label = 'EqF error squared')
# plt.plot(output['EqF.GPStime'][j:k], EqF_att, label = 'EqF attitude error')
# plt.plot(output['EqF.GPStime'][j:k], EqF_pos, label = 'EqF pos error')
# plt.plot(output['EqF.GPStime'][j:k], EqF_vel, label = 'EqF vel error')
# plt.plot(EKFtime, EKF_error_sqr, label = 'EKF error squared')
# #plt.plot(EKFtime, EKF_att, label = 'EKF attitude error')
# #plt.plot(EKFtime, EKF_vel, label = 'EKF vel error')
# #plt.plot(EKFtime, EKF_pos, label = 'EKF pos error')
# plt.title('Error squared over time')
# plt.legend(loc='best')
# plt.yscale("log")
# plt.xlabel('Time(s)')

#Plot EKF positions
fig9, ax9 = plt.subplots(3)
ax9[0].plot(EKFtime, EKFstate_0_14[:,7])
ax9[0].plot(output['EqF.GPStime'][j:k], Xi_p[0,:])
fig9.suptitle('EKF position vs True position for %s' %filename)
ax9[0].set_title('Position North')
ax9[1].plot(EKFtime, EKFstate_0_14[:,8])
ax9[1].plot(output['EqF.GPStime'][j:k], Xi_p[1,:])
ax9[1].set_title('Position East')
ax9[2].plot(EKFtime, EKFstate_0_14[:,9],label='EKF positions')
ax9[2].plot(output['EqF.GPStime'][j:k], Xi_p[2,:],label='SITL position')
ax9[2].set_title('Position Down')
ax9[2].legend(loc='best')
ax9[2].set_xlabel('Time(s)')
ax9[2].set_ylabel('m')
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
fig9.savefig('Graphs for simulation set 2/'+ args.log+'/True position ' + args.log + '.pdf', bbox_inches='tight')

##For presentation
#plot only the position
plt.figure()
plt.plot(output['EqF.GPStime'][j:k], Xi_p[0,:],label='SITL position')
plt.legend(loc='best')
plt.title('Simulation true position')
plt.xlim(left=100, right=250)
plt.xlabel('Time (s)')
plt.ylabel('North (m)')
plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 3) # set figure's size manually to your full screen (32x18)
plt.savefig('Graphs for simulation set 2/'+ args.log+'/True North pos' + args.log + '.pdf', bbox_inches='tight')

#plot only the velocity
plt.figure()
plt.plot(output['EqF.GPStime'][j:k], Xi_v[0,:],label='SITL velocity')
plt.legend(loc='best')
plt.title('Simulation true velocity')
plt.xlim(left=100, right=250)
plt.xlabel('Time (s)')
plt.ylabel('North (m$s^{-1}$)')
plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 3) # set figure's size manually to your full screen (32x18)
plt.savefig('Graphs for simulation set 2/'+ args.log+'/True North vel' + args.log + '.pdf', bbox_inches='tight')


#Plot the true position vs EqF projected position
fig7, ax7 = plt.subplots(3)
ax7[0].plot(output['EqF.GPStime'][j:k], Xi_p[0,:])
ax7[0].plot(output['EqF.GPStime'][j:k], -(Xi_R @ p_inv).T.reshape(3,-1)[0,:])
fig7.suptitle('True position vs projected position ($R_{Xi} R_X^T p_X$)')
ax7[0].set_title('Position North')
ax7[1].plot(output['EqF.GPStime'][j:k], Xi_p[1,:])
ax7[1].plot(output['EqF.GPStime'][j:k], -(Xi_R @ p_inv).T.reshape(3,-1)[1,:])
ax7[1].set_title('Position East')
ax7[2].plot(output['EqF.GPStime'][j:k], Xi_p[2,:],label='True Xi position')
ax7[2].plot(output['EqF.GPStime'][j:k], -(Xi_R @ p_inv).T.reshape(3,-1)[2,:],label='Projected position')
ax7[2].set_title('Position Down')
ax7[2].legend(loc='best')
ax7[2].set_xlabel('Time(s)')
ax7[2].set_ylabel('m')

# fig8, ax8 = plt.subplots(3)
# ax8[0].plot(output['EqF.GPStime'][j:k], epsilon_p[0,:])
# fig8.suptitle('Position error between true and projected position')
# ax8[0].set_title('Pos North')
# ax8[1].plot(output['EqF.GPStime'][j:k], epsilon_p[1,:])
# ax8[1].set_title('Pos East')
# ax8[2].plot(output['EqF.GPStime'][j:k], epsilon_p[2,:])
# ax8[2].set_title('Pos Down')
# ax8[2].set_xlabel('Time(s)')
# ax8[2].set_ylabel('m')

# plt.figure()
# eigRiccati = np.array(output['Riccati.eig'][j:k])
# for i in range(15):
#     plt.plot(output['EqF.GPStime'][j:k], eigRiccati[:,i],label = 'State %i' %i)
# plt.xlabel('Time(s)')
# plt.legend(loc='best')
# plt.suptitle('Eigenvalues of Riccati matrix (nolag2 log)')

# #Most uncertain eigenvector decomposed to different parts
# plt.figure()
# eigvect = np.array(output['eigenvect'][j:k])
# for i in range(15):
#     plt.plot(output['eigen.time'][j:k], eigvect[:,i], label = 'State %i' %i)
# plt.xlabel('Time(s)')
# plt.legend(loc='best')
# plt.suptitle('Eigenvector for largest eigenvalue')


##Compute EKF3 NIS and plot together with EqF NIS 

# #Plot NIS
# plt.figure()
# plt.plot(output['EqF.GPStime'],output['EqF.NIS'],label = 'EqF GPS NIS')
# plt.plot(output['INAC.time'], output['EKF.nis'],label = 'EKF3 GPS NIS')
# plt.title('Normalised innovation error squared (NIS) (%s log)' %filename)
# plt.legend(loc='best')
# plt.yscale("log")
# plt.xlabel('Time(s)')
# plt.ylabel('NIS')

# #Plot position error
# pos_error_eqf = (error[3:6,:])
# pos_error_ekf = (err_velposbias[3:6,:])
# plt.figure()
# pos_eqf_rse = np.sqrt(np.square(pos_error_eqf[0,:]) + np.square(pos_error_eqf[1,:]) + 
#               np.square(pos_error_eqf[2,:]))
# pos_ekf_rse = np.sqrt(np.square(pos_error_ekf[0,:]) + np.square(pos_error_ekf[1,:]) + 
#               np.square(pos_error_ekf[2,:]))
# plt.plot(output['EqF.GPStime'][j:k], pos_eqf_rse,label = 'EqF position error')
# plt.plot(EKFtime, pos_ekf_rse,label = 'EKF3 position error')
# plt.title('Error in position %s' %filename)
# plt.legend(loc='best')
# plt.yscale("log")
# plt.xlabel('Time(s)')
# plt.ylabel('m')

# #Plot velocity error
# vel_error_eqf = (error[6:9,:])
# vel_error_ekf = (err_velposbias[0:3,:])
# plt.figure()
# vel_eqf_rse = np.sqrt(np.square(vel_error_eqf[0,:]) + np.square(vel_error_eqf[1,:]) + 
#               np.square(vel_error_eqf[2,:]))
# vel_ekf_rse = np.sqrt(np.square(vel_error_ekf[0,:]) + np.square(vel_error_ekf[1,:]) + 
#               np.square(vel_error_ekf[2,:]))
# plt.plot(output['EqF.GPStime'][j:k], vel_eqf_rse,label = 'EqF velocity error')
# plt.plot(EKFtime, vel_ekf_rse,label = 'EKF3 velocity error')
# plt.title('Error in velocity %s' %filename)
# plt.legend(loc='best')
# plt.yscale("log")
# plt.xlabel('Time(s)')
# plt.ylabel('m/s')


SITL_pos = np.empty([3,len(output['EqF.GPStime'][j:k])])
calculated_pos = np.empty([3,len(output['EqF.GPStime'][j:k])])
#Calculate the position discrepancy between SITL PN PE PD and what I used using lla/ned conversion
for i in range(3):
    #Correct SIM2 PN, PE and PD which is from home to a relative position 
    sim2_pos = np.array(output['SIM2.Xi_p'])[:,i,0]
    sim2_pos = sim2_pos - sim2_pos[0]
    SITL_pos[i,:] = interp_func(np.array(output['SIM2.time']),sim2_pos, np.array(output['EqF.GPStime'][j:k]))  
    
    #interpolate SITL position to match EqF time
    calculated_pos[i,:] = interp_func(np.array(output['SIM.time']),pos_rel[i,:], np.array(output['EqF.GPStime'][j:k]))         

fig10, ax10 = plt.subplots(3)
fig10.suptitle('Position comparison')
ax10[0].plot(output['EqF.GPStime'][j:k], SITL_pos[0,:],'-.')
ax10[0].plot(output['EqF.GPStime'][j:k], calculated_pos[0,:])
ax10[0].set_ylabel('North (m)')
ax10[1].plot(output['EqF.GPStime'][j:k], SITL_pos[1,:], '-.')
ax10[1].plot(output['EqF.GPStime'][j:k], calculated_pos[1,:])
ax10[1].set_ylabel('East (m)')
ax10[2].plot(output['EqF.GPStime'][j:k], SITL_pos[2,:],'-.',label='SITL true position')
ax10[2].plot(output['EqF.GPStime'][j:k], calculated_pos[2,:], label='position using my conversion')
ax10[2].legend(loc='best')
ax10[2].set_xlabel('Time(s)')
ax10[2].set_ylabel('Down (m)')
   