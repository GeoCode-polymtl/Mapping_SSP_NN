"""Plot inverted Vp Vs and 1/Q ofr lines ARA05C-17, ARA04C-10 m abd ARA04C-02 parallel to the shelf"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

vel_files = {'ARA05C_17':'../InvertedModels/TL/ARA05C_line17_ts_bp_TL_vel.mat',
             'ARA04C_10':'../InvertedModels/TL/ARA04C_line10_int2_bp_TL_vel.mat',
             'ARA04C_02':'../InvertedModels/TL/ARA04C_line02_int2_bp_TL_vel.mat'}
isobath_lims = {'ARA05C_17': [0,1026],'ARA04C_10': [0,-1],'ARA04C_02': [0,-1]}

intercepts = dict()
intercepts['ARA05C_17'] = {'ARA04C_01':[808],'ARA04C_09':[599,600],'ARA05C_01':[604,605,606],
                           'ARA05C_03':[810,811],'ARA05C_05':[313,314,315],'ARA05C_16':[100,101]}
intercepts['ARA04C_10'] = {'ARA05C_05': [239,240],'ARA05C_06': [586,587],'ARA05C_16': [459,460]}

def read_velocities(file,isobath):
    """Read velocity files"""
    ii0,ii1 = isobath
    line_vel_file = h5.File(file,'r')
    line_vel_data = {ol:line_vel_file.get(ol)[()] for ol in ['vp','vs','1/q']}
    line_vel_file.close()
    line_vel_av = {ol:np.average(line_vel_data[ol],axis=0).T[:,ii0:ii1] for ol in line_vel_data}
    return line_vel_av

data = dict()
for line in ['ARA05C_17','ARA04C_10','ARA04C_02']:
    data[line] = read_velocities(vel_files[line],isobath_lims[line])

data['ARA04C_02'] = {key:data['ARA04C_02'][key][:,::-1] for key in data['ARA04C_02']}
data['ARA04C_10'] = {key:data['ARA04C_10'][key][:,::-1] for key in data['ARA04C_10']}

inters_pos = dict()
inters_pos['ARA05C_17'] = {line2: np.average(intercepts['ARA05C_17'][line2])*50e-3 for line2 in intercepts['ARA05C_17']}
inters_pos['ARA04C_10'] = {line2: (len(data['ARA04C_10']['vp'][1])-np.average(intercepts['ARA04C_10'][line2]))*50e-3
                           for line2 in intercepts['ARA04C_10']}

# %% Plotting
fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(6.5,3.5), sharey='all')
lims = {'vp': [1.4,3.5], 'vs':[0.5,1.5], '1/q':[0,0.075]}
plt.subplots_adjust(wspace=0.075,hspace=0.3)
props = dict(boxstyle='round', facecolor='white', alpha=0.25)

ds1 = 50
x = (9.5,10,3.75)
for j,line in enumerate(['ARA05C_17','ARA04C_10','ARA04C_02']):
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
[ax[0,i].text(50,300,'ARA05C_17',bbox=props, ha='right',size=6) for i in range(3)]
[ax[1,i].text(54,350,'ARA04C_10',bbox=props, ha='right',size=6) for i in range(3)]
[ax[2,i].text(19,350,'ARA04C_02',bbox=props, ha='right',size=6) for i in range(3)]


[ax[0,j].annotate('NE',xytext=(40,30),xy=(50.5,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))  for j in range(3)]
[ax[0,j].annotate('SW', xytext=(6, 30), xy=(.0, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))  for j in range(3)]
[ax[1,j].annotate('NE',xytext=(44.,30),xy=(54.9,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]
[ax[1,j].annotate('SW', xytext=(6.5, 30), xy=(0, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]
[ax[2,j].annotate('NE',xytext=(15.,30),xy=(19,30),va='center', color='white',fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]
[ax[2,j].annotate('SW', xytext=(2.25, 30), xy=(0, 30), va='center', color='white', fontsize=7,
                   arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5)) for j in range(3)]

[axi.text(x,200,s,size=10,va='center') for axi,s,x in zip(ax[:,0],['(a)','(b)','(c)'],[-25,-27,-9.5])]
[axi.set_ylabel('Depth (m)',size=8) for axi in ax[:,0]]
[axi.set_xlabel('Distance (km)',size=8) for axi in ax[-1,:]]
[axi.tick_params(axis='both',labelsize=7) for axi in ax.flatten()]

'well logs'
tempx = (819 - isobath_lims['ARA05C_17'][0])*50e-3
ax[0,0].plot([tempx,tempx],[50,350], c='k',ls=':')
ax[0,0].set_ylim([400,0])
ax[0,0].text(tempx,395,'B-35',c='k',ha='center',size=5)

tempx = (len(data['ARA04C_10']['vp'][1])-(587 - isobath_lims['ARA04C_10'][0]))*50e-3
ax[1,0].plot([tempx,tempx],[50,350], c='k',ls=':')
ax[1,0].set_ylim([400,0])
ax[1,0].text(tempx,395,'M-13',c='k',ha='center',size=5)

'Titles'
[axi.set_title(title,fontsize=10) for axi,title in zip(ax[0,:],['$V_P$','$V_S$','$1/Q$'])]

plt.savefig('figs/Figure_Parallel_lines_2.png',dpi=300,bbox_inches='tight')
plt.show()
