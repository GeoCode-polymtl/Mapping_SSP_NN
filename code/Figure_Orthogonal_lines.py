"""Plot inverted Vp Vs and 1/Q ofr lines ARA04C-11, ARA04C-10 m abd ARA04C-02 parallel to the shelf"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

vel_files = {'ARA04C_11':'../InvertedModels/TL/ARA04C_line11_int2_bp_TL_vel.mat',
             'ARA05C_08':'../InvertedModels/TL/ARA05C_line08_bp_TL_vel.mat',
             'ARA05C_06':'../InvertedModels/TL/ARA05C_line06_ts_bp_TL_vel.mat',
             'ARA05C_16':'../InvertedModels/TL/ARA05C_line16_ts_bp_TL_vel.mat',
             'ARA05C_05':'../InvertedModels/TL/ARA05C_line05_ts_bp_TL_vel.mat',
             'ARA04C_09':'../InvertedModels/TL/ARA04C_line09_int2_bp_TL_vel.mat',
             'ARA05C_01':'../InvertedModels/TL/ARA05C_line01_ts_bp_TL_vel.mat',
             'ARA04C_01':'../InvertedModels/TL/ARA04C_line01_int2_bp_TL_vel.mat',
             'ARA05C_03':'../InvertedModels/TL/ARA05C_line03_ts_bp_TL_vel.mat'}
isobath_lims = {'ARA04C_11': [0,-1],'ARA05C_08': [15,515],'ARA05C_06': [116,-1],
                'ARA05C_16': [615,1315],'ARA05C_05': [0,851],'ARA04C_09': [0,-1],
                'ARA05C_01': [0,842],'ARA04C_01': [236,-1],'ARA05C_03': [972,2058]}

intercepts = dict()
intercepts['ARA04C_11'] = {'ARA05C_07': [54,55],'ARA05C_08': [202,203]}
intercepts['ARA05C_08'] = {'ARA04C_11':[164,165,166],'ARA05C_07':[21,22],'ARA05C_11':[485,486,487]}
intercepts['ARA05C_06'] = {'ARA04C_08':[420,421],'ARA04C_10':[787],'ARA05C_07':[679,680],'ARA05C_11':[157,158]}
intercepts['ARA05C_16'] = {'ARA04C_08':[928,929],'ARA04C_10':[1281,1282],'ARA05C_11':[636,637,638],
                           'ARA05C_17':[1175,1176]}
intercepts['ARA05C_05'] = {'ARA04C_08':[470,471],'ARA04C_10':[122,123],'ARA05C_11':[807,808,809],
                           'ARA05C_17':[240,241,242]}
intercepts['ARA04C_09'] = {'ARA05C_01': [189,190,191],'ARA05C_17': [165,166]}
intercepts['ARA05C_01'] = {'ARA04C_09':[44],'ARA05C_11':[653,654],'ARA05C_17':[68,69,70]}
intercepts['ARA04C_01'] = {'ARA05C_03': [1080,1081],'ARA05C_11': [450,451],'ARA05C_17': [1067,1068]}
intercepts['ARA05C_03'] = {'ARA04C_01':[1792,1793],'ARA05C_11':[1161,1162],'ARA05C_17':[1779,1780,1781]}

def read_velocities(file,isobath=[0,-1]):
    ii0, ii1 = isobath
    line_vel_file = h5.File(file,'r')
    line_vel_data = line_vel_file.get('vp')[()]
    line_vel_file.close()
    line_vel_av = np.average(line_vel_data, axis=0).T[:,ii0:ii1]
    return line_vel_av

data = dict()
for line in vel_files:
    data[line] = read_velocities(vel_files[line],isobath_lims[line])
data['ARA04C_11'] = data['ARA04C_11'][:,::-1]
# data['ARA04C_01'] = data['ARA04C_01'][:,236:-1]
data['ARA05C_08'] = data['ARA05C_08'][:,::-1]
data['ARA05C_05'] = data['ARA05C_05'][:,::-1]
data['ARA05C_01'] = data['ARA05C_01'][:,::-1]


inters_pos = dict()
inters_pos['ARA04C_11'] = {line2: (len(data['ARA04C_11'][1])-np.average(intercepts['ARA04C_11'][line2]))*50e-3
                           for line2 in intercepts['ARA04C_11']}
inters_pos['ARA05C_08'] = {line2: (len(data['ARA05C_08'][1]) -
                                   (np.average(intercepts['ARA05C_08'][line2])-isobath_lims['ARA05C_08'][0])
                                   )*50e-3 for line2 in intercepts['ARA05C_08']}
inters_pos['ARA05C_06'] = {line2: (np.average(intercepts['ARA05C_06'][line2])-
                                   isobath_lims['ARA05C_06'][0])*50e-3 for line2 in intercepts['ARA05C_06']}

inters_pos['ARA05C_16'] = {line2: (np.average(intercepts['ARA05C_16'][line2])-
                                   isobath_lims['ARA05C_16'][0])*50e-3 for line2 in intercepts['ARA05C_16']}
inters_pos['ARA04C_09'] = {line2: np.average(intercepts['ARA04C_09'][line2])*50e-3 for line2 in intercepts['ARA04C_09']}
inters_pos['ARA05C_05'] = {line2: (len(data['ARA05C_05'][1]) - np.average(intercepts['ARA05C_05'][line2]))*50e-3
                           for line2 in intercepts['ARA05C_05']}

inters_pos['ARA05C_01'] = {line2: (len(data['ARA05C_01'][1]) - np.average(intercepts['ARA05C_01'][line2]))*50e-3
                           for line2 in intercepts['ARA05C_01']}
inters_pos['ARA04C_01'] = {line2: (np.average(intercepts['ARA04C_01'][line2])-236)*50e-3
                           for line2 in intercepts['ARA04C_01']}
inters_pos['ARA05C_03'] = {line2: (np.average(intercepts['ARA05C_03'][line2])-972)*50e-3
                           for line2 in intercepts['ARA05C_03']}

data['ARA05C_06'] = np.pad(data['ARA05C_06'],((0,0),(0,20)),mode='edge')

# %% Plotting

fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(6.5,4.5), sharey='all')
lims = {'vp': [1.4,3.5], 'vs':[0.5,1.5], '1/q':[0,0.075]}
plt.subplots_adjust(wspace=0.075,hspace=0.5)
props = dict(boxstyle='round', facecolor='white', alpha=0.25)

ds1 = 50
x = (4,4.7,6.4)
for j,line in enumerate(['ARA04C_11','ARA05C_08','ARA05C_06']):
    im0 = ax[0,j].imshow(data[line][:160,:],aspect='auto',cmap='jet',vmin=lims['vp'][0],vmax=lims['vp'][1],
                         extent=[0,data[line].shape[1]*ds1*1e-3,400,0])
    cbaxes = inset_axes(ax[0,j], width="30%", height="5%", loc=3)
    cbar1 = fig.colorbar(im0, cax=cbaxes, orientation='horizontal', ticks=[lims['vp'][0], lims['vp'][1]])
    cbar1.ax.tick_params(labelsize=6, color='w')
    cbar1.outline.set_edgecolor('k')
    cbar1.ax.xaxis.set_label_coords(2, -0.04)
    cbar1.ax.xaxis.set_ticks_position("top")
    cbar1.ax.tick_params(labelsize=4, color='k')
    ax[0,j].text(x[j], 325, '$V_P$ (km s$^{-1}$)', size=5, ha='center')
    if line in inters_pos:
        [ax[0,j].plot([inters_pos[line][line2],inters_pos[line][line2]],[50,395],'--w') for line2 in inters_pos[line]]
ax[0,0].text(21,350,'ARA04C_11',bbox=props, ha='right',size=6)
ax[0,1].text(24.5,350,'ARA05C_08',bbox=props, ha='right',size=6)
ax[0,2].text(27,350,'ARA05C_06',bbox=props, ha='right',size=6)

ax[0,0].annotate('SE',xytext=(17,30),xy=(21.2,30),va='center', color='white',fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[0,0].annotate('NW', xytext=(2.75, 30), xy=(0, 30), va='center', color='white', fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[0,1].annotate('SE',xytext=(20,30),xy=(25,30),va='center', color='white',fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[0,1].annotate('NW', xytext=(3.25, 30), xy=(.0, 30), va='center', color='white', fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[0,2].annotate('SE',xytext=(28,30),xy=(35.3,30),va='center', color='white',fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[0,2].annotate('NW', xytext=(4.5, 30), xy=(0, 30), va='center', color='white', fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))

x = (6.5,8.2,2)
for j,line in enumerate(['ARA05C_16','ARA05C_05','ARA04C_09']):
    im0 = ax[1,j].imshow(data[line][:160,:],aspect='auto',cmap='jet',vmin=lims['vp'][0],vmax=lims['vp'][1],
                         extent=[0,data[line].shape[1]*ds1*1e-3,400,0])
    cbaxes = inset_axes(ax[1,j], width="30%", height="5%", loc=3)
    cbar1 = fig.colorbar(im0, cax=cbaxes, orientation='horizontal', ticks=[lims['vp'][0], lims['vp'][1]])
    cbar1.ax.tick_params(labelsize=6, color='w')
    cbar1.outline.set_edgecolor('k')
    cbar1.ax.xaxis.set_label_coords(2, -0.04)
    cbar1.ax.xaxis.set_ticks_position("top")
    cbar1.ax.tick_params(labelsize=4, color='k')
    ax[1,j].text(x[j], 325, '$V_P$ (km s$^{-1}$)', size=5, ha='center')
    if line in inters_pos:
        [ax[1,j].plot([inters_pos[line][line2],inters_pos[line][line2]],[50,395],'--w') for line2 in inters_pos[line]]
ax[1,0].text(34,350,'ARA05C_16',bbox=props, ha='right',size=6)
ax[1,1].text(42,350,'ARA05C_05',bbox=props, ha='right',size=6)
ax[1,2].text(10.2,350,'ARA04C_09',bbox=props, ha='right',size=6)

ax[1,0].annotate('SE',xytext=(28,30),xy=(35,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[1,0].annotate('NW', xytext=(4.25, 30), xy=(0, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[1,1].annotate('SE',xytext=(34,30),xy=(42.5,30),va='center', color='white',fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[1,1].annotate('NW', xytext=(6, 30), xy=(.5, 30), va='center', color='white', fontsize=7,
                 arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[1,2].annotate('SE',xytext=(8.2,30),xy=(10.4,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[1,2].annotate('NW', xytext=(1.25, 30), xy=(0, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))

x = (8.2,9.5,10)
for j,line in enumerate(['ARA05C_01','ARA04C_01','ARA05C_03']):
    im0 = ax[2,j].imshow(data[line][:160,:],aspect='auto',cmap='jet',vmin=lims['vp'][0],vmax=lims['vp'][1],
                         extent=[0,data[line].shape[1]*ds1*1e-3,400,0])
    cbaxes = inset_axes(ax[2,j], width="30%", height="5%", loc=3)
    cbar1 = fig.colorbar(im0, cax=cbaxes, orientation='horizontal', ticks=[lims['vp'][0], lims['vp'][1]])
    cbar1.ax.tick_params(labelsize=6, color='w')
    cbar1.outline.set_edgecolor('k')
    cbar1.ax.xaxis.set_label_coords(2, -0.04)
    cbar1.ax.xaxis.set_ticks_position("top")
    cbar1.ax.tick_params(labelsize=4, color='k')
    ax[2,j].text(x[j], 325, '$V_P$ (km s$^{-1}$)', size=5, ha='center')
    if line in inters_pos:
        [ax[2,j].plot([inters_pos[line][line2],inters_pos[line][line2]],[50,395],'--w') for line2 in inters_pos[line]]
ax[2,0].text(41,350,'ARA05C_01',bbox=props, ha='right',size=6)
ax[2,1].text(50,350,'ARA04C_01',bbox=props, ha='right',size=6)
ax[2,2].text(53.2,350,'ARA05C_03',bbox=props, ha='right',size=6)

[axi.set_ylabel('Depth (m)',size=8) for axi in ax[:,0]]
[axi.set_xlabel('Distance (km)',size=8) for axi in ax[-1,:]]
[axi.tick_params(axis='both',labelsize=7) for axi in ax.flatten()]
[axi.set_title(s,size=10) for axi,s in zip(ax.flatten(),['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)'])]

ax[2,0].annotate('SE',xytext=(33.5,30),xy=(42,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[2,0].annotate('NW', xytext=(5.8, 30), xy=(.5, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[2,1].annotate('SE',xytext=(41,30),xy=(51,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[2,1].annotate('NW', xytext=(6.5, 30), xy=(0, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[2,2].annotate('SE',xytext=(42.8,30),xy=(54.2,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))
ax[2,2].annotate('NW', xytext=(6.5, 30), xy=(0., 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))

'WelLogs'
tempx = (len(data['ARA05C_08'][1]) - (493 - isobath_lims['ARA05C_08'][0]))*50e-3
ax[0,1].plot([tempx,tempx],[50,350], c='k',ls=':')
ax[0,1].set_ylim([400,0])
ax[0,1].text(tempx,100,'K-59',c='k',ha='left',size=5)

tempx = (803 - isobath_lims['ARA05C_06'][0])*50e-3
ax[0,2].plot([tempx,tempx],[50,350], c='k',ls=':')
ax[0,2].set_ylim([400,0])
ax[0,2].text(tempx,395,'M-13',c='k',ha='center',size=5)

tempx = (1081 - isobath_lims['ARA04C_01'][0])*50e-3
ax[2,1].plot([tempx,tempx],[50,350], c='k',ls=':')
ax[2,1].set_ylim([400,0])
ax[2,1].text(tempx,395,'B-35',c='k',ha='center',size=5)

tempx = (1792 - isobath_lims['ARA05C_03'][0])*50e-3
ax[2,2].plot([tempx,tempx],[50,350], c='k',ls=':')
ax[2,2].set_ylim([400,0])
ax[2,2].text(tempx,395,'B-35',c='k',ha='center',size=5)


plt.savefig('figs/Figure_Orthogonal_lines.png',dpi=300,bbox_inches='tight')
plt.show()

