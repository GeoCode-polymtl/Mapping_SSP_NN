import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.ndimage import gaussian_filter
from obspy import Trace, Stream
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
import sys

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
vel_files = {'ARA04C_line01':'../InvertedModels/TL/ARA04C_line01_int2_bp_TL_vel.mat',
             'ARA04C_line02':'../InvertedModels/TL/ARA04C_line02_int2_bp_TL_vel.mat',
             'ARA04C_line08':'../InvertedModels/TL/ARA04C_line08_int2_bp_TL_vel.mat',
             'ARA04C_line09':'../InvertedModels/TL/ARA04C_line09_int2_bp_TL_vel.mat',
             'ARA04C_line10':'../InvertedModels/TL/ARA04C_line10_int2_bp_TL_vel.mat',
             'ARA04C_line11':'../InvertedModels/TL/ARA04C_line11_int2_bp_TL_vel.mat',
             'ARA05C_line01':'../InvertedModels/TL/ARA05C_line01_ts_bp_TL_vel.mat',
             'ARA05C_line03':'../InvertedModels/TL/ARA05C_line03_ts_bp_TL_vel.mat',
             'ARA05C_line05':'../InvertedModels/TL/ARA05C_line05_ts_bp_TL_vel.mat',
             'ARA05C_line06':'../InvertedModels/TL/ARA05C_line06_ts_bp_TL_vel.mat',
             'ARA05C_line07':'../InvertedModels/TL/ARA05C_line07_ts_bp_TL_vel.mat',
             'ARA05C_line08':'../InvertedModels/TL/ARA05C_line08_bp_TL_vel.mat',
             'ARA05C_line11':'../InvertedModels/TL/ARA05C_line11_ts_bp_TL_vel.mat',
             'ARA05C_line16':'../InvertedModels/TL/ARA05C_line16_ts_bp_TL_vel.mat',
             'ARA05C_line17':'../InvertedModels/TL/ARA05C_line17_ts_bp_TL_vel.mat'}
'Coordinates files'
cords_folder = "../DataPreprocessed/CMPs_coords"
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

line = 'ARA04C_line09'
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


# %% Accepting inversion values conditioned to 1/q, vp, vs for Ice bearing
temp1 = np.ones(line_vel_av['1/q'].shape)
mask = (line_vel_av['1/q'] < .025) & (line_vel_av['vp']<=2.25) | (line_vel_av['vs'] < .8)
temp1[mask] = 0

# %% Accepting inversion values conditioned to 1/q, Vp, Vs for Ice bonded conditions
temp2 = np.ones(line_vel_av['1/q'].shape)
mask = (line_vel_av['1/q'] < .025) & (line_vel_av['vp']<=2.25)
temp2[mask] = 0
mask2 = (line_vel_av['vp']<=3.0) & (line_vel_av['vs']<=1.0)
temp2[mask2] = 0

# %% Permafrost distribution interpretation
smoothing = True
permafrost = np.zeros(line_vel_av['1/q'].shape)
mask = (line_vel_av['1/q'] < .025) & (line_vel_av['vp']<=2.25) | (line_vel_av['vs'] < .8)
permafrost[np.logical_not(mask)] = 0.5

mask2_1 = (line_vel_av['1/q'] < .025) & (line_vel_av['vp']<=2.25)
mask2_2 = (line_vel_av['vp']<=3.0) & (line_vel_av['vs']<=1.1)
mask2 = np.logical_or(mask2_1,mask2_2)
permafrost[np.logical_not(mask2)] = 1

if smoothing:
    if line == 'ARA05C_line05': permafrost[100:,:] = 0
    if line == 'ARA05C_line01': permafrost[110:,:] = 0
    if line == 'ARA05C_line03': permafrost[90:,1300:] = 0
    if line == 'ARA05C_line11': permafrost[95:,:1300] = 0 ; permafrost[120:,1300:] = 0
    if line == 'ARA05C_line07': permafrost[140:, :] = 0; permafrost[100:, 400:] = 0
    if line == 'ARA05C_line17': permafrost[60:, 100:600] = 0; permafrost[80:, 600:] = 0
    if line == 'ARA04C_line09': permafrost[60:, :100] = 0; permafrost[60:, 150:] = 0
    if line == 'ARA04C_line08': permafrost[120:, :] = 0
    if line == 'ARA04C_line10': permafrost[65:, :540] = 0; permafrost[82:, 700:] = 0
    if line == 'ARA04C_line02': permafrost[80:, :] = 0
    if line == 'ARA04C_line01': permafrost[100:, 400:1220] = 0; permafrost[60:, 700:1100] = 0
    permafrost = gaussian_filter(permafrost,sigma=(0,3))

fig,ax = plt.subplots(figsize=(9,3.5))
im = ax.imshow(permafrost[:160, indx_lims[line][0]:indx_lims[line][1]], aspect='auto', cmap='jet', vmin=0,vmax=1,
               extent=[0, (indx_lims[line][1]-indx_lims[line][0]) * 50e-3, 400, 0])
if line == 'ARA05C_line01': ax.plot((29.5,29.5),(50,400),'--w')
if line == "ARA05C_line03": ax.plot((24,24),(50,400),'--w'); ax.plot((7,7),(50,400),'--w')
if line == 'ARA05C_line05':
    ax.plot((21,21),(50,400),'--w'); ax.plot((35,35),(50,400),'--w')
    ax.plot((38.5,38.5),(50,400),'--w'); ax.plot((41.5,41.5),(50,400),'--w')
if line == 'ARA05C_line06':
    ax.plot((30,30),(50,400),'--w')
    ax.plot((25,25),(50,400),'--w'); ax.plot((22.5,22.5),(50,400),'--w')
    ax.plot((13,13),(50,400),'--w'); ax.plot((16,16),(50,400),'--w')
    ax.plot((4,4),(50, 400), '--w'); ax.plot((7,7), (50, 400), '--w')
if line == 'ARA05C_line07': ax.plot((26,26),(50,400),'--w')
if line == 'ARA05C_line08': ax.plot((8.5,8.5),(50,400),'--w'); ax.plot((11,11),(50,400),'--w')
if line == 'ARA05C_line11': ax.plot((10,10),(50,400),'--w'); ax.plot((97,97),(50,400),'--w')
if line == 'ARA05C_line16':
    ax.plot((30,30),(50,400),'--w')
    ax.plot((12,12),(50,400),'--w'); ax.plot((13.5,13.5),(50,400),'--w')
if line == 'ARA05C_line17': ax.plot((6.5, 6.5), (50, 400), '--w')
if line == 'ARA04C_line01': ax.plot((20, 20), (50, 400), '--w')
if line == 'ARA04C_line08':
    ax.plot((43, 43), (50, 400), '--w')
    ax.plot((14,14),(50,400), '--w'); ax.plot((26,26),(50,400), '--w')
if line == 'ARA04C_line10': ax.plot((33,33),(50,400), '--w'); ax.plot((52,52),(50,400), '--w')
if line == 'ARA04C_line11': ax.plot((5,5),(50,400), '--w'); ax.plot((14,14),(50,400), '--w')

ax.set_ylabel('depth (m)')
ax.set_xlabel('distance (km)')
plt.suptitle('%s Permafrost interpretation'%line)
fig.colorbar(im,ax=ax)
plt.show()

# %% Read Coordinates
coords = np.genfromtxt(file_coords[line],delimiter=',')
coords_cmps,coords_x,coords_y = coords[:,0],coords[:,1],coords[:,2]


# %% Generating SEGY file
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

write_file = True
if write_file:
    filename = '../SSPInterpretation/%s_permafrost_dist2_smooth.sgy'%line
    write_segy(permafrost[:160, indx_lims[line][0]:indx_lims[line][1]].astype(np.float32),
               coords[indx_lims[line][0]:indx_lims[line][1], :], filename=filename)
    filename_vp = '../InvertedModels/VP_models/%s_vp_400m.sgy'%line
    write_segy((line_vel_av['vp'])[:160,indx_lims[line][0]:indx_lims[line][1]].astype(np.float32),
               coords[indx_lims[line][0]:indx_lims[line][1],:],filename=filename_vp)

