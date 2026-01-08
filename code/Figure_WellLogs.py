import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

vel_files = {'ARA05C_11':'../../InvertedModels/TL/ARA05C_line11_ts_bp_TL_vel.mat',
             'ARA05C_08':'../../InvertedModels/TL/ARA05C_line08_bp_TL_vel.mat',
             'ARA05C_06':'../../InvertedModels/TL/ARA05C_line06_ts_bp_TL_vel.mat',
             'ARA04C_10': '../../InvertedModels/TL/ARA04C_line10_int2_bp_TL_vel.mat',
             'ARA05C_17':'../../InvertedModels/TL/ARA05C_line17_ts_bp_TL_vel.mat',
             'ARA05C_03': '../../InvertedModels/TL/ARA05C_line03_ts_bp_TL_vel.mat',
             'ARA04C_01':'../../InvertedModels/TL/ARA04C_line01_int2_bp_TL_vel.mat'
             }

isobath_lims = {'ARA05C_06': [116,-1],'ARA05C_08': [15,515], 'ARA05C_11': [0,-1], 'ARA04C_10': [0,-1],
                'ARA05C_17': [0,1026],'ARA05C_03': [972,2058],'ARA04C_01': [236,-1]}


def read_velocities(file,isobath):
    ii0,ii1 = isobath
    line_vel_file = h5.File(file,'r')
    line_vel_data = {ol:line_vel_file.get(ol)[()] for ol in ['vp','vs','1/q']}
    line_vel_file.close()
    line_vel_av = {ol:np.average(line_vel_data[ol],axis=0).T[:,ii0:ii1] for ol in line_vel_data}
    return line_vel_av

data = dict()
for line in ['ARA05C_11','ARA05C_08','ARA05C_06','ARA04C_10','ARA05C_17','ARA05C_03','ARA04C_01']:
    data[line] = read_velocities(vel_files[line],isobath_lims[line])
data['ARA05C_08'] = {ol:data['ARA05C_08'][ol][:,::-1] for ol in data['ARA05C_08']}
data['ARA04C_10'] = {ol:data['ARA04C_10'][ol][:,::-1] for ol in data['ARA04C_10']}

for key in data['ARA05C_06']:
    data['ARA05C_06'][key] = np.pad(data['ARA05C_06'][key],((0,0),(0,20)),mode='edge')

# for line in data:
#     plt.imshow(data[line]['vp'][:160,],aspect='auto',cmap='jet',vmin=1.4,vmax=3.5)
#     plt.title(line)
#     plt.show()

# %% Well logs:
vps, ys = {}, {}
'B35'
vps['B35'] = np.array([1.78,1.40,3.25,1.65,1.47,1.37,2.41,1.87,2.75,1.52,2.25,1.71,2.18,1.66,2.21,1.92,1.93,2.59,2.43,
                       1.75,2.62,1.68,1.79])
ys['B35'] = np.array([64.50,79.57,94.41,110.76,124.14,139.56,155.45,185.72,201.92,216.63,232.21,245.32,262.46,276.47,
                      292.55,307.68,323.63,338.19,353.75,368.41,383.62,398.00,413.00])
'M-13 no processing 5 points'
vps['M13'] = np.array([2.375283342,2.102223896,2.115742028,2.233636913,2.657715046,4.691736013-1,3.427191226,
                       2.124021554,1.943859697,1.961107218,1.946040645,1.967586622,1.617849956,1.536057525,1.70866447,
                       1.745989305,1.882978723,1.534886966,1.438718374,1.379573258,1.444972849,1.648528525,1.748956552,
                       1.811580377,1.799114555,1.787216381,1.785255405,1.6327766])
ys['M13'] = np.array([107.4,128.6,150,171.4,186.6,201.8,217,232.2,247.4,262.6,277.8,293.2,308.4,323.6,338.8,354.2,369.4,
                      387.6,406,424.4,442.6,460.8,476.2,491.4,506.6,521.8,540.2,558.4])-60
'K59'
vps['K59'] = np.array([1.43,1.429055599,1.535779523,1.72353197,1.80892063,1.965893966,1.91960806,1.72999755,
                       1.599653714])
ys['K59'] = np.array([261.2136,276.4536,291.6936,309.9816,328.2696,349.6056,370.9416,392.2776,410.5656])

vps['K59_c'] = np.array([1.595811518,1.78245614,1.693333333])
ys['K59_c'] = np.array([170,200,231])

K59_5C11 = (187 - isobath_lims['ARA05C_11'][0])
K59_5C08 = (len(data['ARA05C_08']['vp'][1]) - (493 - isobath_lims['ARA05C_08'][0]))
M13_5C06 = (803 - isobath_lims['ARA05C_06'][0])
M13_4C10 = (len(data['ARA04C_10']['vp'][1])-(587 - isobath_lims['ARA04C_10'][0]))
B35_5C17 = (819 - isobath_lims['ARA05C_17'][0])
B35_5C03 = (1792 - isobath_lims['ARA05C_03'][0])
B35_4C01 = (1081 - isobath_lims['ARA04C_01'][0])

# %% Seafloor
'Seafloor files'
sf_folder = '../SeaFloor'
sf_files = {key: '%s/%s'%(sf_folder,key.replace('C_','C_line')+'_SeaFloor.csv')
            for key in vel_files}

nearcmp = {'K59_0511':1604,'K59_0508':4548,
        'M13_0506':6524,'M13_0410':4756,
        'B35_0517':6676,'B35_0503':14452,'B35_0401':8764}

# %% Plot 2
fig, ax = plt.subplots(ncols=7,sharey='all', sharex='all', figsize=(8,3))
lc = 'w'#'silver'
'K59 vs 5C-11'
sf_0511 = np.genfromtxt(sf_files['ARA05C_11'], delimiter=',')
sf_0511[np.isnan(sf_0511)] = 0
sf_idx = np.argmin(np.abs(sf_0511[:,4] - nearcmp['K59_0511']))
ax[0].imshow(data['ARA05C_11']['vp'][:160,K59_5C11-5:K59_5C11+5],aspect='auto',vmin=1.4,vmax=3.5,
             extent=[0,0.5,400,0],cmap='jet',interpolation='bilinear')
temp = (2*(vps['K59']-1.4)/(3.7-1.4)-1)*.2+.25
ax[0].plot(temp,ys['K59'],lc)
tempc = (2*(vps['K59_c']-1.4)/(3.7-1.4)-1)*.2+.25
ax[0].plot(tempc,ys['K59_c'],lc,ls=':')
ax[0].plot([0,.5], [sf_0511[sf_idx,3], sf_0511[sf_idx,3]],'grey',ls='--',lw=.75)
ax[0].plot([0.25,0.25],[64,159.95],lw=1,c='k',ls='--')
ax[0].set_ylim([400,0])
ax[0].set_ylabel('Depth (m)',size=8)
ax[0].set_yticks(np.arange(0,401,100))
ax[0].set_yticklabels(np.arange(0,401,100),fontsize=8)
# ax[0].set_xlim([1.3,3.7])

'K59 vs 5C-08'
sf_0508 = np.genfromtxt(sf_files['ARA05C_08'], delimiter=',')
sf_0508[np.isnan(sf_0508)] = 0
sf_idx = np.argmin(np.abs(sf_0508[:,4] - nearcmp['K59_0508']))
ax[1].imshow(data['ARA05C_08']['vp'][:160,K59_5C08-5:K59_5C08+5],aspect='auto',vmin=1.4,vmax=3.5,
             extent=[0,0.5,400,0],cmap='jet',interpolation='bilinear')
ax[1].plot(temp,ys['K59'],lc)
ax[1].plot(tempc,ys['K59_c'],lc,ls=':')
ax[1].plot([0,.5], [sf_0508[sf_idx,3], sf_0508[sf_idx,3]],'grey',ls='--',lw=.75)
ax[1].plot([0.25,0.25],[64,159.95],lw=1,c='k',ls='--')
# [axi.text(.05,250,'1.4\n|', ha='center', c=lc,size=8) for axi in ax[:2]]
# [axi.text(.45,250,'3.7\n|', ha='center', c=lc,size=8) for axi in ax[:2]]
# [axi.text(.25,200,'K-59\nVp (km/s)', ha='center', c=lc,size=8) for axi in ax[:2]]
[axi.text(.05,75,'1.4\n|', ha='center', c=lc,size=8) for axi in ax[:2]]
[axi.text(.45,75,'3.7\n|', ha='center', c=lc,size=8) for axi in ax[:2]]
[axi.text(.25,35,'K-59\nVp (km/s)', ha='center', c=lc,size=8) for axi in ax[:2]]

'M13 vs 5C-06'
sf_0506 = np.genfromtxt(sf_files['ARA05C_06'], delimiter=',')
sf_0506[np.isnan(sf_0506)] = 0
sf_idx = np.argmin(np.abs(sf_0506[:,4] - nearcmp['M13_0506']))
ax[2].imshow(data['ARA05C_06']['vp'][:160,M13_5C06-5:M13_5C06+5],aspect='auto',vmin=1.4,vmax=3.5,
             extent=[0,0.5,400,0],cmap='jet',interpolation='bilinear')
ax[2].plot([0.25,0.25],[57,171],lw=1,c='k',ls='--')
temp = (2*(vps['M13']-1.4)/(3.7-1.4)-1)*.2+.25
ax[2].plot(temp,ys['M13'],lc)
ax[2].plot([0,.5], [sf_0506[sf_idx,3], sf_0506[sf_idx,3]],'grey',ls='--',lw=.75)

'M13 vs 4C-10'
sf_0410 = np.genfromtxt(sf_files['ARA04C_10'], delimiter=',')
sf_0410[np.isnan(sf_0410)] = 0
sf_idx = np.argmin(np.abs(sf_0410[:,4] - nearcmp['M13_0410']))
ax[3].imshow(data['ARA04C_10']['vp'][:160,M13_4C10-5:M13_4C10+5],aspect='auto',vmin=1.4,vmax=3.5,
             extent=[0,0.5,400,0],cmap='jet',interpolation='bilinear')
ax[3].plot([0.25,0.25],[57,171],lw=1,c='k',ls='--')
ax[3].plot(temp,ys['M13'],lc)
ax[3].plot([0,.5], [sf_0410[sf_idx,3], sf_0410[sf_idx,3]],'grey',ls='--',lw=.75)
[axi.text(.05,85,'|\n1.4', ha='center', c=lc,size=8) for axi in ax[2:4]]
[axi.text(.45,85,'|\n3.7', ha='center', c=lc,size=8) for axi in ax[2:4]]
[axi.text(.25,35,'M-13\nVp (km/s)', ha='center', c=lc,size=8) for axi in ax[2:4]]
#
'B35 vs 5C-17'
sf_0517 = np.genfromtxt(sf_files['ARA05C_17'], delimiter=',')
sf_0517[np.isnan(sf_0517)] = 0
sf_idx = np.argmin(np.abs(sf_0517[:,4] - nearcmp['B35_0517']))
ax[4].imshow(data['ARA05C_17']['vp'][:160,B35_5C17-5:B35_5C17+5],aspect='auto',vmin=1.4,vmax=3.5,
             extent=[0,0.5,400,0],cmap='jet',interpolation='bilinear')
ax[4].plot([0.25,0.25],[56,101.8],lw=1,c='k',ls='--')
temp = (2*(vps['B35']-1.4)/(3.7-1.4)-1)*.2+.25
ax[4].plot(temp,ys['B35'],lc)
ax[4].plot([0,.5], [sf_0517[sf_idx,3], sf_0517[sf_idx,3]],'grey',ls='--',lw=.75)
#
'B35 vs 5C-03'
sf_0503 = np.genfromtxt(sf_files['ARA05C_03'], delimiter=',')
sf_0503[np.isnan(sf_0503)] = 0
sf_idx = np.argmin(np.abs(sf_0503[:,4] - nearcmp['B35_0503']))
ax[5].imshow(data['ARA05C_03']['vp'][:160,B35_5C03-5:B35_5C03+5],aspect='auto',vmin=1.4,vmax=3.5,
             extent=[0,0.5,400,0],cmap='jet',interpolation='bilinear')
ax[5].plot([0.25,0.25],[56,101.8],lw=1,c='k',ls='--')
ax[5].plot(temp,ys['B35'],lc)
ax[5].plot([0,.5], [sf_0503[sf_idx,3], sf_0503[sf_idx,3]],'grey',ls='--',lw=.75)
#
'B35 vs 4C-01'
sf_0401 = np.genfromtxt(sf_files['ARA04C_01'], delimiter=',')
sf_0401[np.isnan(sf_0401)] = 0
sf_idx = np.argmin(np.abs(sf_0401[:,4] - nearcmp['B35_0401']))
im = ax[6].imshow(data['ARA04C_01']['vp'][:160,B35_5C03-5:B35_5C03+5],aspect='auto',vmin=1.4,vmax=3.5,
             extent=[0,0.5,400,0],cmap='jet',interpolation='bilinear')
ax[6].plot([0.25,0.25],[56,101.8],lw=1,c='k',ls='--')
ax[6].plot(temp,ys['B35'],lc)
ax[6].plot([0,.5], [sf_0401[sf_idx,3], sf_0401[sf_idx,3]],'grey',ls='--',lw=.75)
[axi.text(.05,75,'1.4\n|', ha='center', c=lc,size=8) for axi in ax[4:]]
[axi.text(.45,75,'3.7\n|', ha='center', c=lc,size=8) for axi in ax[4:]]
[axi.text(.25,35,'B-35\nVp (km/s)', ha='center', c=lc,size=8) for axi in ax[4:]]

cbar_ax = fig.add_axes([0.805,-0.075,0.1,0.03])
cbar = fig.colorbar(im,cax=cbar_ax,orientation='horizontal')
cbar.set_label('$V_P$ (km/s)', fontsize=8)
cbar.set_ticks([2,3])
cbar.ax.tick_params(labelsize=8)
# cbar.set_label(units[j], fontsize=8)
#
# ax[0].set_ylabel('Depth (m)')
[axi.set_title(s) for axi,s in zip(ax,['(a)','(b)','(c)','(d)','(e)','(f)','(g)']) ]
[axi.set_xlabel('Distance (km)',size=8) for axi in ax ]
[axi.set_xticks([0,0.5]) for axi in ax]
[axi.set_xticklabels([0,0.5],fontsize=8) for axi in ax]
plt.savefig('figs/Figure_WellLogs.png',dpi=300,bbox_inches='tight')
plt.show()