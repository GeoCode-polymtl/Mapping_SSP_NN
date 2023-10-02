"""Plot inverted Vp Vs and 1/Q ofr lines ARA05C-11, ARA04C-08 m abd ARA05C-07 parallel to the shelf"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

vel_files = {'ARA05C_11':'../InvertedModels/TL/ARA05C_line11_ts_bp_TL_vel.mat',
             'ARA04C_08':'../InvertedModels/TL/ARA04C_line08_int2_bp_TL_vel.mat',
             'ARA05C_07':'../InvertedModels/TL/ARA05C_line07_ts_bp_TL_vel.mat'}
isobath_lims = {'ARA05C_11': [0,-1],'ARA04C_08': [0,-1],'ARA05C_07': [0,763]}

intercepts = dict()
intercepts['ARA05C_11'] = {'ARA04C_01':[1626],'ARA05C_01':[1475,1476],'ARA05C_03':[1725,1726],
                           'ARA05C_05':[1085,1086],'ARA05C_06':[775,776],'ARA05C_08':[195,196],
                           'ARA05C_16':[900,901,902]}
intercepts['ARA04C_08'] = {'ARA05C_05': [742,743],'ARA05C_06': [402,403],'ARA05C_16': [532,533]}
intercepts['ARA05C_07'] = {'ARA04C_11':[652,653,654],'ARA05C_06':[109,110],'ARA05C_08':[667,668]}

def read_velocities(file,isobath):
    """Read velocity files"""
    ii0,ii1 = isobath
    line_vel_file = h5.File(file,'r')
    line_vel_data = {ol:line_vel_file.get(ol)[()] for ol in ['vp','vs','1/q']}
    line_vel_file.close()
    line_vel_av = {ol:np.average(line_vel_data[ol],axis=0).T[:,ii0:ii1] for ol in line_vel_data}
    return line_vel_av

data = dict()
for line in ['ARA05C_11','ARA04C_08','ARA05C_07']:
    data[line] = read_velocities(vel_files[line],isobath_lims[line])
data['ARA05C_07'] = {ol:data['ARA05C_07'][ol][:,::-1] for ol in data['ARA05C_07']}

inters_pos = dict()
inters_pos['ARA05C_11'] = {line2: np.average(intercepts['ARA05C_11'][line2])*50e-3
                           for line2 in intercepts['ARA05C_11']}
inters_pos['ARA04C_08'] = {line2: np.average(intercepts['ARA04C_08'][line2])*50e-3 for line2 in intercepts['ARA04C_08']}
inters_pos['ARA05C_07'] = {line2: (len(data['ARA05C_07']['vp'][1])-np.average(intercepts['ARA05C_07'][line2]))*50e-3
                           for line2 in intercepts['ARA05C_07']}


# %% Plotting
fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(6.5,3.5), sharey='all')
lims = {'vp': [1.4,3.5], 'vs':[0.5,1.5], '1/q':[0,0.075]}
plt.subplots_adjust(wspace=0.075,hspace=0.3)
props = dict(boxstyle='round', facecolor='white', alpha=0.25)

ds1 = 50
x = (18.5,9.5,7.5)
for j,line in enumerate(['ARA05C_11','ARA04C_08','ARA05C_07']):
    for i,ol in enumerate(['vp','vs','1/q']):
        i0,i1 = isobath_lims[line]
        im0 = ax[j,i].imshow(data[line][ol][:160,:],aspect='auto',cmap='jet',vmin=lims[ol][0],vmax=lims[ol][1],
                             extent=[0,data[line][ol].shape[1]*ds1*1e-3,400,0])
        cbaxes = inset_axes(ax[j,i], width="30%", height="5%", loc=3)
        cbar1 = fig.colorbar(im0, cax=cbaxes, orientation='horizontal', ticks=[lims[ol][0], lims[ol][1]])
        cbar1.outline.set_edgecolor('k')
        cbar1.ax.xaxis.set_label_coords(2, -0.04)
        cbar1.ax.xaxis.set_ticks_position("top")
        cbar1.ax.tick_params(labelsize=4, color='k')
        if ol != '1/q': ax[j,i].text(x[j], 325, '%s (km s$^{-1}$)' % ol.capitalize(), size=5, ha='center')
        else: ax[j,i].text(x[j], 325, '%s' % ol.upper(), size=5, ha='center')
        if line in inters_pos:
            [ax[j,i].plot([inters_pos[line][line2], inters_pos[line][line2]], [50, 395], '--w') for line2 in
             inters_pos[line]]
[ax[0,i].text(99,350,'ARA05C_11',bbox=props, ha='right',size=6) for i in range(3)]
[ax[1,i].text(47,350,'ARA04C_08',bbox=props, ha='right',size=6) for i in range(3)]
[ax[2,i].text(37.25,350,'ARA05C_07',bbox=props, ha='right',size=6) for i in range(3)]


[ax[0,j].annotate('NE',xytext=(80,30),xy=(101,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]
[ax[0,j].annotate('SW', xytext=(12, 30), xy=(0, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]
[ax[-1,j].annotate('NE',xytext=(30.0,30),xy=(38,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]
[ax[-1,j].annotate('SW', xytext=(5, 30), xy=(.25, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]
[ax[1,j].annotate('NE',xytext=(38.,30),xy=(48,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]
[ax[1,j].annotate('SW', xytext=(6., 30), xy=(0, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]

[axi.text(x,200,s,size=10,va='center') for axi,s,x in zip(ax[:,0],['(a)','(b)','(c)'],[-50,-24,-19])]
[axi.set_ylabel('Depth (m)',size=8) for axi in ax[:,0]]
[axi.set_xlabel('Distance (km)',size=8) for axi in ax[-1,:]]
[axi.tick_params(axis='both',labelsize=7) for axi in ax.flatten()]

'well logs'
vps = np.array([1.43,1.429055599,1.535779523,1.72353197,1.80892063,1.965893966,1.91960806,1.72999755,1.599653714])
ys = np.array([261.2136,276.4536,291.6936,309.9816,328.2696,349.6056,370.9416,392.2776,410.5656])

tempx = (187 - isobath_lims['ARA05C_11'][0])*50e-3
ax[0,0].plot([tempx,tempx],[50,350], c='k',ls=':')
ax[0,0].set_ylim([400,0])
ax[0,0].text(tempx,100,'K-59',c='k',ha='left',size=5)

'Titles'
[axi.set_title(title,fontsize=10) for axi,title in zip(ax[0,:],['$V_P$','$V_S$','$1/Q$'])]

plt.savefig('figs/Figure_Parallel_lines_1.png',dpi=300,bbox_inches='tight')
plt.show()