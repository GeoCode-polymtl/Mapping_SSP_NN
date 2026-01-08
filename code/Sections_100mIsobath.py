import copy

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from scipy.ndimage import gaussian_filter,gaussian_filter1d
from obspy import Trace, Stream
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
import sys
import argparse
import pandas as pd
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch


# %% Parameters
parser = argparse.ArgumentParser()
parser.add_argument('-ln','--linenumber',type=str,default='04-01',
                    help='''Line number in the format 05-06 (ARAC survey number - line number), choose between
                            ['04-01', '04-02', '04-08', '04-09', '04-10', '04-11', '05-01', '05-03', '05-05', '05-06', 
                             '05-07', '05-08', '05-11', '05-12', '05-14', '05-15', '05-16', '05-17']''')

args = parser.parse_args()
linenumber = args.linenumber

print('Line number:',linenumber)

line_names = {'04-01':'ARA04C_line01','04-02':'ARA04C_line02','04-08':'ARA04C_line08',
              '04-09':'ARA04C_line09','04-10':'ARA04C_line10','04-11':'ARA04C_line11',
              '05-01':'ARA05C_line01','05-03':'ARA05C_line03','05-05':'ARA05C_line05',
              '05-06':'ARA05C_line06','05-07':'ARA05C_line07','05-08':'ARA05C_line08',
              '05-11':'ARA05C_line11','05-12':'ARA05C_line12','05-14':'ARA05C_line14',
              '05-15':'ARA05C_line15','05-16':'ARA05C_line16','05-17':'ARA05C_line17'}

'Isobath lims'
indx_lims = {'ARA04C_line01': [236,1260,-1],
             'ARA04C_line02': [0,-1],'ARA04C_line08': [0,965,-1],'ARA04C_line09': [0,-1],
             'ARA04C_line10': [0,1098,-1],'ARA04C_line11': [0,425,-1],
             'ARA05C_line01': [0,842,6852],
             'ARA05C_line03': [972,2058],
             'ARA05C_line05': [0,851],
             'ARA05C_line06': [116,802],
             'ARA05C_line07': [0,763],
             'ARA05C_line08': [15,515],
             'ARA05C_line11': [0,2024,-1],
             'ARA05C_line16': [615,1315],
             'ARA05C_line17': [0,1026]}
'Velocity files'
vel_files = {'ARA04C_line01':'../../InvertedModels/TL/ARA04C_line01_int2_bp_TL_vel.mat',
             'ARA04C_line02':'../../InvertedModels/TL/ARA04C_line02_int2_bp_TL_vel.mat',
             'ARA04C_line08':'../../InvertedModels/TL/ARA04C_line08_int2_bp_TL_vel.mat',
             'ARA04C_line09':'../../InvertedModels/TL/ARA04C_line09_int2_bp_TL_vel.mat',
             'ARA04C_line10':'../../InvertedModels/TL/ARA04C_line10_int2_bp_TL_vel.mat',
             'ARA04C_line11':'../../InvertedModels/TL/ARA04C_line11_int2_bp_TL_vel.mat',
             'ARA05C_line01':'../../InvertedModels/TL/ARA05C_line01_ts_bp_TL_vel.mat',
             'ARA05C_line03':'../../InvertedModels/TL/ARA05C_line03_ts_bp_TL_vel.mat',
             'ARA05C_line05':'../../InvertedModels/TL/ARA05C_line05_ts_bp_TL_vel.mat',
             'ARA05C_line06':'../../InvertedModels/TL/ARA05C_line06_ts_bp_TL_vel.mat',
             'ARA05C_line07':'../../InvertedModels/TL/ARA05C_line07_ts_bp_TL_vel.mat',
             'ARA05C_line08':'../../InvertedModels/TL/ARA05C_line08_bp_TL_vel.mat',
             'ARA05C_line11':'../../InvertedModels/TL/ARA05C_line11_ts_bp_TL_vel.mat',
             'ARA05C_line16':'../../InvertedModels/TL/ARA05C_line16_ts_bp_TL_vel.mat',
             'ARA05C_line17':'../../InvertedModels/TL/ARA05C_line17_ts_bp_TL_vel.mat'}

'Coordinates files'
cords_folder = "../../DataPreprocessed/CMPs_coords"
file_coords = {'ARA04C_line01': "%s/ARA04C_line01_int2_cmps.csv"%cords_folder,
               'ARA04C_line02': "%s/ARA04C_line02_int2_cmps.csv"%cords_folder,
               'ARA04C_line08': "%s/ARA04C_line08_int2_cmps.csv"%cords_folder,
               'ARA04C_line09': "%s/ARA04C_line09_int2_cmps.csv"%cords_folder,
               'ARA04C_line10': "%s/ARA04C_line10_int2_cmps.csv"%cords_folder,
               'ARA04C_line11': "%s/ARA04C_line11_int2_cmps.csv"%cords_folder,
               'ARA05C_line01': "%s/ARA05C_line01_ts_cmps.csv"%cords_folder,
               'ARA05C_line03': "%s/ARA05C_line03_ts_cmps.csv"%cords_folder,
               'ARA05C_line05': "%s/ARA05C_line05_ts_cmps.csv"%cords_folder,
               'ARA05C_line06': "%s/ARA05C_line06_ts_cmps.csv"%cords_folder,
               'ARA05C_line07': "%s/ARA05C_line07_ts_cmps.csv"%cords_folder,
               'ARA05C_line08': "%s/ARA05C_line08_cmps.csv"%cords_folder,
               'ARA05C_line11': "%s/ARA05C_line11_ts_cmps.csv"%cords_folder,
               'ARA05C_line16': "%s/ARA05C_line16_ts_cmps.csv"%cords_folder,
               'ARA05C_line17': "%s/ARA05C_line17_ts_cmps.csv"%cords_folder}

# line = 'ARA04C_line09'
line = line_names[linenumber]
less_pars_NN = False
dt = 2000 # in ms

if less_pars_NN:
    vel_files[line] = '%s_less_pars%s' %(vel_files[line][:-8], vel_files[line][-8:])

line_vel_file = h5.File(vel_files[line],'r')
line_vel_data = {ol:line_vel_file.get(ol)[()] for ol in ['vp','vs','1/q']}
line_vel_file.close()
line_vel_av = {ol:np.average(line_vel_data[ol],axis=0).T for ol in line_vel_data}

outlabels = ['vp','vs','1/q']
lims = [[1.5,3.5],[0,1.5,],[0,0.07]]

# %% Functions
def plot_permafrost(permafrost, indx_lims, line):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    # im = ax.imshow(permafrost[:160, indx_lims[line][0]:indx_lims[line][1]], aspect='auto', cmap='jet', vmin=0,vmax=1,
    #                extent=[0, (indx_lims[line][1]-indx_lims[line][0]) * 50e-3, 400, 0])
    im = ax.imshow(permafrost[:, indx_lims[line][0]:indx_lims[line][1]], aspect='auto', cmap='jet', vmin=0, vmax=1,
                   extent=[0, (indx_lims[line][1] - indx_lims[line][0]) * 50e-3, (permafrost.shape[0]+1)*2.5, 0])

    ax.set_ylabel('depth (m)')
    ax.set_xlabel('distance (km)')
    plt.suptitle('%s Permafrost interpretation' % line)
    fig.colorbar(im, ax=ax)
    plt.show()

def write_segy(data,coords,filename):
    coords_cmps, coords_x, coords_y = coords[:, 0], coords[:, 1], coords[:, 2]
    stream = Stream()
    traces_write = [Trace(data=data[:,i]) for i in range(data.shape[1])]
    for i,trace_w in enumerate(traces_write):
        if not hasattr(trace_w.stats, 'segy.trace_header'):
            trace_w.stats.segy = {}
        trace_w.stats.delta = 2500/10**6
        trace_w.stats.segy.trace_header = SEGYTraceHeader()
        trace_w.stats.segy.trace_header.trace_sequence_number_within_segy_file = i
        trace_w.stats.segy.trace_header.ensemble_number = int(coords_cmps[i])
        trace_w.stats.segy.trace_header.group_coordinate_x = int(coords_x[i])
        trace_w.stats.segy.trace_header.group_coordinate_y = int(coords_y[i])
        trace_w.stats.segy.trace_header.number_of_samples_in_this_trace = data.shape[0]
        trace_w.stats.segy.trace_header.sample_interval_in_ms_for_this_trace = 2500
        stream.append(trace_w)

    stream.stats = AttribDict()
    stream.stats.binary_file_header = SEGYBinaryFileHeader()
    stream.stats.binary_file_header.number_of_data_traces_per_ensemble = 1
    print(stream)

    stream.write(filename, format="SEGY", data_encoding=5, byteorder=sys.byteorder)
def first_nonzero(arr, axis, invalid_val=np.nan):
    mask = arr>0.25
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=np.nan):
    mask = arr>0.25
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

# %% Permafrost distribution interpretation

##########
vpl, vph = 2.25, 3.0
vsl, vsh = .8, 1.1
ql = 0.025
##########
cleanup = True

'Ice free conditions if no condition applies'
'Note, better definition of the lower permafrost layers, some bugs when expected ice bonded conditions'
permafrost = np.zeros(line_vel_av['1/q'].shape)
mask = (line_vel_av['1/q'] < ql) & ((line_vel_av['vp']>=vph) | (line_vel_av['vs'] > vsh))
mask = ((line_vel_av['vp']>=vph) | (line_vel_av['vs'] >= vsh))
permafrost[mask] = 1
mask2 = ((line_vel_av['1/q'] > ql) & (((vpl<=line_vel_av['vp']) & (line_vel_av['vp']<vph)) |
                                      ((vsl<=line_vel_av['vs']) & (line_vel_av['vs'] < vsh))))
permafrost[mask2] = 0.5

if indx_lims[line][1] == -1: indx_lims[line][1] = permafrost.shape[1]
plot_permafrost(gaussian_filter(permafrost,sigma=(0,3))[:160,:], indx_lims, line)

permafrost_c = copy.copy(permafrost)
if cleanup:
    if line == 'ARA05C_line17':
        permafrost_c[52:,:250] = 0; permafrost_c[55:,250:550] = 0; permafrost_c[60:,550:700]=0; permafrost_c[52:,750:] = 0
        permafrost_c[70:,700:750]=0
    if line == 'ARA05C_line16': permafrost_c[60:,:1287] = 0; permafrost_c[46:,:1100] = 0; permafrost_c[120:,1287:]=0
    if line == 'ARA05C_line11': permafrost_c[60:,:] = 0
    if line == 'ARA05C_line08':
        permafrost_c[52:,:420]=0; permafrost_c[60:,420:450]=0; permafrost_c[52:,450:]=0
    if line == 'ARA05C_line07':
        permafrost_c[52:,:300] = 0; permafrost_c[70:,300:370]=0; permafrost_c[110:,370:450]=0; permafrost_c[72:,400:455]=0
        permafrost_c[60:,455:]=0
    if line == 'ARA05C_line06':
        permafrost_c[50:,:470] = 0; permafrost_c[90:,470:700] = 0; permafrost_c[95:,700:730]=0; permafrost_c[120:,730:770] = 0
    if line == 'ARA05C_line05':
        permafrost_c[62:,:300] = 0; permafrost_c[50:,300:380]=0; permafrost_c[58:,380:450]=0; permafrost_c[50:,450:]=0
    if line == 'ARA05C_line03':
        permafrost_c[52:,:1230]= 0; permafrost_c[54:,1230:1330] = 0
        permafrost_c[65:,1330:1600]=0; permafrost_c[70:,1600:1635]=0; permafrost_c[60:,1635:]=0
    if line == 'ARA05C_line01':
        #permafrost_c[55:,:95] = 0; permafrost_c[50:,95:200] = 0; #permafrost_c[50:,200:220] = 0
        permafrost_c[62:,:450] = 0
        permafrost_c[63:,450:570]=0
        permafrost_c[52:,380:420]=0
        permafrost_c[85:,570:750]=0
        permafrost_c[70:,750:]=0
        #permafrost_c[62:,300:390] = 0; permafrost_c[53:,390:420]=0; permafrost_c[52:,420:600] = 0; permafrost_c[60:,600:] = 0
    if line == 'ARA04C_line11': permafrost_c[:,:] = 0
    if line == 'ARA04C_line10':
        permafrost_c[58:,:520] = 0; #permafrost_c[75:,520:560] = 0; #permafrost_c[150:,560:700] = 0
        permafrost_c[55:,670:] = 0
    if line == 'ARA04C_line09': permafrost_c[58:,:] = 0; permafrost_c[55:,120:130] = 0; permafrost_c[57:,160:] = 0
    if line == 'ARA04C_line08':
        permafrost_c[58:,625:] = 0; permafrost_c[55:,:625]=0
        # permafrost_c[50:,580:675] = 0;
        # permafrost_c[64:,400:580] = 0
        # permafrost_c[57:,250:400] = 0;
        # permafrost_c[50:,:250] = 0
    if line == 'ARA04C_line02': permafrost_c[69:,:50] = 0; permafrost_c[60:,50:] = 0
    if line == 'ARA04C_line01': permafrost_c[60:,400:] = 0; permafrost_c[52:, -80:] = 0; permafrost_c[75:,:400]=0
    permafrost_c = gaussian_filter(permafrost_c,sigma=(3,3))
permafrost = gaussian_filter(permafrost,sigma=(3,3))
plot_permafrost(permafrost_c[:160,:], indx_lims, line),

# %% Read Coordinates
coords = np.genfromtxt(file_coords[line],delimiter=',')
coords_cmps,coords_x,coords_y = coords[:,0],coords[:,1],coords[:,2]

# %% Generating SEGY file
write_file = False
if write_file:
    filename = '../SSPInterpretation/%s_permafrost_dist2_smooth.sgy'%line
    write_segy(permafrost[:160, indx_lims[line][0]:indx_lims[line][1]].astype(np.float32),
               coords[indx_lims[line][0]:indx_lims[line][1], :], filename=filename)
write_file_ul = False
if write_file_ul:
    filename_ul = '../SSPInterpretation/%s_permafrost_uplay_smooth.sgy'%line
    write_segy(permafrost_c[:160, indx_lims[line][0]:indx_lims[line][1]].astype(np.float32),
               coords[indx_lims[line][0]:indx_lims[line][1], :], filename=filename_ul)
write_vel_files = False
if write_vel_files:
    filename_vp = '../InvertedModels/VP_models/%s_vp_400m.sgy'%line
    write_segy((line_vel_av['vp'])[:160,indx_lims[line][0]:indx_lims[line][1]].astype(np.float32),
               coords[indx_lims[line][0]:indx_lims[line][1],:],filename=filename_vp)
    filename_vs = '../InvertedModels/VS_models/%s_vs_400m.sgy' % line
    write_segy((line_vel_av['vs'])[:160, indx_lims[line][0]:indx_lims[line][1]].astype(np.float32),
               coords[indx_lims[line][0]:indx_lims[line][1], :], filename=filename_vs)
    filename_1q = '../InvertedModels/1Q_models/%s_1q_400m.sgy' % line
    write_segy((line_vel_av['1/q'])[:160, indx_lims[line][0]:indx_lims[line][1]].astype(np.float32),
               coords[indx_lims[line][0]:indx_lims[line][1], :], filename=filename_1q)

# %% Seafloor

sediments = np.ones(line_vel_av['vp'].shape)
mask = (line_vel_av['vp'] < 1.6) & (line_vel_av['vs'] < 0.4)
sediments[mask] = 0

sf_idx = first_nonzero(sediments[:160,:], axis=0, invalid_val=np.nan).astype(int)[indx_lims[line][0]:indx_lims[line][1]]
sf = (sf_idx*2.5)
# %% Permafrost boundaries

top_idx = first_nonzero(permafrost_c[:160,:], axis=0, invalid_val=0).astype(int)[indx_lims[line][0]:indx_lims[line][1]]
bop_idx = last_nonzero(permafrost_c[:160,:], axis=0, invalid_val=-1).astype(int)[indx_lims[line][0]:indx_lims[line][1]]

top = (top_idx*2.5)
bop = (bop_idx*2.5)
xs = np.arange(0,(indx_lims[line][1] - indx_lims[line][0]) * 50e-3,50e-3)
mask = top>0

fig, ax = plt.subplots(nrows=4,figsize=(9, 3.5*4))
im0 = ax[0].imshow(gaussian_filter(line_vel_av['vp'][:160, indx_lims[line][0]:indx_lims[line][1]],sigma=(3,3)),
                   aspect='auto', cmap='jet', vmin=1.5,
                   vmax=3.5, extent=[0, (indx_lims[line][1] - indx_lims[line][0]) * 50e-3, 400, 0])
fig.colorbar(im0, ax=ax[0], label='$V_p$ (km/s)')

im1 = ax[1].imshow(gaussian_filter(line_vel_av['vs'][:160, indx_lims[line][0]:indx_lims[line][1]],sigma=(3,3)),
                   aspect='auto', cmap='jet', vmin=0.4,
                   vmax=1.5, extent=[0, (indx_lims[line][1] - indx_lims[line][0]) * 50e-3, 400, 0])
fig.colorbar(im1, ax=ax[1], label='$V_s$ (km/s)')

im2 = ax[2].imshow(gaussian_filter(line_vel_av['1/q'][:160, indx_lims[line][0]:indx_lims[line][1]],sigma=(3,3)),
                   aspect='auto', cmap='plasma', vmin=0.005,
                   vmax=0.04, extent=[0, (indx_lims[line][1] - indx_lims[line][0]) * 50e-3, 400, 0])

cmap = pl.cm.jet
mycmap = cmap(np.arange(cmap.N))
mycmap[:20,-1] = 0
mycmap[20:40,-1] = np.arange(0,1,.05)
mycmap = ListedColormap(mycmap)
fig.colorbar(im2, ax=ax[2], label='1/Q')
im3 = ax[3].imshow(permafrost[:160, indx_lims[line][0]:indx_lims[line][1]], aspect='auto', cmap=mycmap, vmin=0, vmax=1,
                   extent=[0, (indx_lims[line][1] - indx_lims[line][0]) * 50e-3, 400, 0])
cbar = fig.colorbar(im3, ax=ax[3],ticks=[0,.5,1]) #,label='Permafrost distribution')
cbar.ax.set_yticklabels(['Ice free','Ice\nbearing','Ice\nbonded'])


ax[3].set_xlabel('Distance (km)')
[axi.plot(xs,gaussian_filter1d(sf,sigma=3),'silver',label='Sea floor') for axi in ax[:]]
[axi.plot(xs[mask],top[mask],'g.',label='ToP',ms=3.5) for axi in ax[:]]
[axi.plot(xs[mask],bop[mask],'k.',label='BoP (first layer)',ms=3.5) for axi in ax[:]]
[axi.legend(facecolor='w',framealpha=1) for axi in ax[:-1]]
[axi.set_ylabel('Depth (m)') for axi in ax]
[axi.set_title(s) for axi,s in zip(ax,['a)','b)','c)','d)'])]
if linenumber == '04-01':
    ax[0].annotate('SE',xy=(51,25),xytext=(46,25),ha='center',va='center',c='w', fontsize=14,
                   arrowprops=dict(arrowstyle='->',color='w',lw=3))
    ax[0].annotate('NW',xy=(0,25),xytext=(5,25),ha='center',va='center',c='w', fontsize=14,
                   arrowprops=dict(arrowstyle='->',color='w',lw=3))
    # ax[0].arrow(x=20,y=380,dx=2,dy=0,fc='w',ec='w',shape='right',lw=1)
    ax[0].annotate('',xy=(22,380),xytext=(18,380),ha='center',va='center',c='darkgreen', fontsize=14,
                   arrowprops=dict(arrowstyle='->',color='darkgreen',lw=3))
    ax[1].annotate('', xy=(22, 380), xytext=(18, 380), ha='center', va='center', c='darkgreen', fontsize=14,
                   arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3))
    ax[2].annotate('', xy=(39, 380), xytext=(35, 380), ha='center', va='center', c='darkgreen', fontsize=14,
                   arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3))
    ax[3].annotate('', xy=(22, 380), xytext=(18, 380), ha='center', va='center', c='darkgreen', fontsize=14,
                   arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3))
    plt.savefig('figs/Inverted04-01.png',dpi=300,bbox_inches='tight')
plt.show()

# %% Thickness and velocities
pthick = bop-top
pthick[~mask] = 0
mid_idx = ((bop_idx-top_idx)/2+top_idx).astype(int)

max_vp,min_vp = np.zeros(top.shape), np.zeros(top.shape)
max_vs,min_vs = np.zeros(top.shape), np.zeros(top.shape)
mid_1q,mid_vp,mid_vs = np.zeros(top.shape),np.zeros(top.shape),np.zeros(top.shape)

if linenumber=='04-01':
    fig, ax = plt.subplots(figsize=(9, 3.5*4),nrows=4,sharex=True)
    ax[0].plot(xs,pthick,'-',label='Permafrost thickness')
    ax[0].set_ylabel('Thickness (m)',size=12)

    for i in range(len(top)):
        if top_idx[i] == 0:
            max_vp[i],min_vp[i],max_vs[i],min_vs[i] = [line_vel_av[key][:160,indx_lims[line][0]:indx_lims[line][1]][40,i]
                                                       for key in ['vp','vp','vs','vs']]
            mid_vp[i], mid_vs[i], mid_1q[i] = [line_vel_av[key][:160,indx_lims[line][0]:indx_lims[line][1]][40,i]
                                                       for key in ['vp','vs','1/q']]
        elif top_idx[i] !=bop_idx[i]:
            max_vp[i],max_vs[i] = [line_vel_av[key][:160,indx_lims[line][0]:indx_lims[line][1]][top_idx[i]:
                                                                                                bop_idx[i],i].max()
                                   for key in ['vp','vs']]
            min_vp[i],min_vs[i] = [line_vel_av[key][:160,indx_lims[line][0]:indx_lims[line][1]][top_idx[i]:
                                                                                                bop_idx[i],i].min()
                                   for key in ['vp','vs']]
            mid_vp[i],mid_vs[i],mid_1q[i] = [line_vel_av[key][:160, indx_lims[line][0]:indx_lims[line][1]][mid_idx[i],i]
                                             for key in ['vp','vs','1/q']]
        else:
            max_vp[i],min_vp[i],max_vs[i],min_vs[i] = [line_vel_av[key][:160,indx_lims[line][0]:
                                                                             indx_lims[line][1]][top_idx[i],i]
                                                       for key in ['vp','vp','vs','vs']]
            mid_vp[i], mid_vs[i], mid_1q[i] = [line_vel_av[key][:160,indx_lims[line][0]:indx_lims[line][1]][top_idx[i],i]
                                               for key in ['vp','vs','1/q']]

    ax[1].plot(xs,mid_vp,'-',label='$V_P$ at permafrost midpoint')
    ax[1].plot([xs.min(),xs.max()],[vpl,vpl],'--k')
    ax[1].plot([xs.min(),xs.max()],[vph,vph],'--k')
    ax[1].text(1*xs.max(),vpl,'\nIce Free',va='top',size=12,ha='left')
    ax[1].text(1*xs.max(),vpl,'Ice bearing\n',va='bottom',size=12,ha='left')
    ax[1].text(1*xs.max(),vph,'Ice bonded',va='bottom',size=12,ha='left')
    ax[1].set_ylabel('$V_P$ at Pfr midpoint',size=12)

    ax[2].plot(xs,mid_vs,'-',label='$V_S$ at permafrost midpoint')
    ax[2].plot([xs.min(),xs.max()],[vsl,vsl],'--k')
    ax[2].plot([xs.min(),xs.max()],[vsh,vsh],'--k')
    ax[2].text(1*xs.max(),vsl,'\nIce Free',va='top',size=12,ha='left')
    ax[2].text(1*xs.max(),vsl,'Ice bearing\n',va='bottom',size=12,ha='left')
    ax[2].text(1*xs.max(),vsh,'Ice bonded\n',va='bottom',size=12,ha='left')
    ax[2].set_ylabel('$V_S$ at Pfr midpoint',size=12)

    ax[3].plot(xs,mid_1q,'-',label='1/Q at permafrost midpoint')
    ax[3].plot([xs.min(),xs.max()],[ql,ql],'--k')
    ax[3].text(1*xs.max(),ql,'\nIce Free +\nIce bonded',va='top',size=12,ha='left')
    ax[3].text(1*xs.max(),ql,'Ice bearing\n',va='bottom',size=12,ha='left')
    ax[3].set_xlabel('Distance (km)',size=12)
    ax[3].set_ylabel('1/Q at Pfr midpoint',size=12)

    [axi.set_title(t,size=14) for axi,t in zip(ax,['a)','b)','c)','d)'])]

    con1 = ConnectionPatch(xyA=(0,90), coordsA=ax[0].transData,
                          xyB=(0,0.0175), coordsB=ax[3].transData,
                          color='y',ls='--')
    fig.add_artist(con1)
    con2 = ConnectionPatch(xyA=(10, 90), coordsA=ax[0].transData,
                          xyB=(10, 0.0175), coordsB=ax[3].transData,
                          color='y', ls='--')
    fig.add_artist(con2)
    ax[0].plot([0,10],[90,90],color='y',ls='--')
    ax[3].plot([0,10], [.0175,.0175], color='y', ls='--')

    con3 = ConnectionPatch(xyA=(15, 90), coordsA=ax[0].transData,
                           xyB=(15, 0.0175), coordsB=ax[3].transData,
                           color='r', ls='--')
    fig.add_artist(con3)
    con4 = ConnectionPatch(xyA=(45, 90), coordsA=ax[0].transData,
                           xyB=(45, 0.0175), coordsB=ax[3].transData,
                           color='r', ls='--')
    fig.add_artist(con4)
    ax[0].plot([15, 45], [90, 90], color='r', ls='--')
    ax[3].plot([15, 45], [.0175, .0175], color='r', ls='--')

    plt.savefig('figs/Figure_PermafrostParameters.png',dpi=300,bbox_inches='tight')
    plt.show()

top[~mask] = np.nan
bop[~mask] = np.nan

# %% Permafrost state
permafrost_state = np.repeat(['Ice bearing'],permafrost_c.shape[1])
mask_fp = [np.any(permafrost_c[:,i] >= .75) for i in range(permafrost_c.shape[1])]
mask_if = [np.all(permafrost_c[:,i] <= .25) for i in range(permafrost_c.shape[1])]
permafrost_state[mask_fp] = 'Frozen'
permafrost_state[mask_if] = 'Ice Free'

permafrost_par = {'cmp' : coords[indx_lims[line][0]:indx_lims[line][1],0],
                  'x' : coords[indx_lims[line][0]:indx_lims[line][1],1],
                  'y' : coords[indx_lims[line][0]:indx_lims[line][1],2],
                  'top' : top, 'bop' : bop, 'pthick' : pthick,
                  'mid_vp' : mid_vp, 'mid_vs' : mid_vs, 'mid_1q' : mid_1q,
                  'max_vp' : max_vp, 'max_vs' : max_vs
                  }
permafrost_dist = {'cmp': coords[indx_lims[line][0]:indx_lims[line][1],0],
                   'x':coords[indx_lims[line][0]:indx_lims[line][1],1],
                   'y':coords[indx_lims[line][0]:indx_lims[line][1],2],
                   'permafrost_state' : permafrost_state[indx_lims[line][0]:indx_lims[line][1]]
                   }
sea_floor = {'Line' : line.replace('_line','-'),
             'x' : coords[indx_lims[line][0]:indx_lims[line][1],1],
             'y' : coords[indx_lims[line][0]:indx_lims[line][1],2],
             'seafloor' : gaussian_filter1d(sf,sigma=3),
             'cmp': coords[indx_lims[line][0]:indx_lims[line][1],0],}
# %% Writing csv files
write_csv = True
if write_csv:
    df = pd.DataFrame(permafrost_par)
    df.to_csv('../SSPInterpretation/%s_permafrost_parameters.csv'%line,index=False)
    df2 = pd.DataFrame(permafrost_dist)
    df2.to_csv('../SSPInterpretation/%s_permafrost_distribution.csv'%line,index=False)
    df3 = pd.DataFrame(sea_floor)
    df3.to_csv('../SeaFloor/%s_SeaFloor.csv'%line,index=False)
