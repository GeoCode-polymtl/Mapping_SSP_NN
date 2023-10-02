import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

intersects = dict()

intersects['ARA04C_01'] = {'ARA05C_03':[1080,1081],
                           'ARA05C_11':[450,451],
                           'ARA05C_17':[1067,1068]}
intersects['ARA04C_08'] = {'ARA05C_05':[742,743],
                           'ARA05C_06':[402,403],
                           'ARA05C_16':[532,533]}
intersects['ARA04C_10'] = {'ARA05C_05':[239,240],
                           'ARA05C_06':[586,587],
                           'ARA05C_16':[459,460]}
intersects['ARA04C_11'] = {'ARA05C_07':[54,55],
                           'ARA05C_08':[202,203]}

intersects['ARA05C_01'] = {'ARA04C_09':[44],
                           'ARA05C_11':[653,654],
                           'ARA05C_17':[68,69,70]}
intersects['ARA05C_03'] = {'ARA04C_01':[1792,1793],
                           'ARA05C_11':[1161,1162],
                           'ARA05C_17':[1779,1780,1781]}
intersects['ARA05C_05'] = {'ARA04C_08':[470, 471],
                           'ARA04C_10':[122,123],
                           'ARA05C_11':[807,808,809],
                           'ARA05C_17':[240,241,242]}
intersects['ARA05C_06'] = {'ARA04C_08':[420,421],
                           'ARA04C_10':[787],
                           'ARA05C_07':[679,680],
                           'ARA05C_11':[157,158]}
intersects['ARA05C_07'] = {'ARA04C_11':[652,653,654],
                           'ARA05C_06':[109,110],
                           'ARA05C_08':[667,668]}
intersects['ARA05C_08'] = {'ARA04C_11':[164,165,166],
                           'ARA05C_07':[21,22],
                           'ARA05C_11':[485,486,487], }
intersects['ARA05C_16'] = {'ARA04C_08':[928,929],
                           'ARA04C_10':[1281,1282],
                           'ARA05C_11':[636,637,638],
                           'ARA05C_17':[1175,1176]}
intersects['ARA05C_17'] = {'ARA04C_01':[808],
                           'ARA04C_09':[599,600],
                           'ARA05C_01':[604,605,606],
                           'ARA05C_03':[810,811],
                           'ARA05C_05':[313,314,315],
                           'ARA05C_16':[100,101]}
intersects['ARA05C_11'] = {'ARA04C_01':[1626],
                           'ARA05C_01':[1475,1476],
                           'ARA05C_03':[1725,1726],
                           'ARA05C_05':[1085,1086],
                           'ARA05C_06':[775,776],
                           'ARA05C_08':[195,196],
                           'ARA05C_16':[900,901,902]}

vel_files = {'ARA04C_01':'../InvertedModels/TL/ARA04C_line01_int2_bp_TL_vel.mat',
             'ARA04C_02':'../InvertedModels/TL/ARA04C_line02_int2_bp_TL_vel.mat',
             'ARA04C_08':'../InvertedModels/TL/ARA04C_line08_int2_bp_TL_vel.mat',
             'ARA04C_09':'../InvertedModels/TL/ARA04C_line09_int2_bp_TL_vel.mat',
             'ARA04C_10':'../InvertedModels/TL/ARA04C_line10_int2_bp_TL_vel.mat',
             'ARA04C_11':'../InvertedModels/TL/ARA04C_line11_int2_bp_TL_vel.mat',
             'ARA05C_01':'../InvertedModels/TL/ARA05C_line01_ts_bp_TL_vel.mat',
             'ARA05C_03':'../InvertedModels/TL/ARA05C_line03_ts_bp_TL_vel.mat',
             'ARA05C_05':'../InvertedModels/TL/ARA05C_line05_ts_bp_TL_vel.mat',
             'ARA05C_06':'../InvertedModels/TL/ARA05C_line06_ts_bp_TL_vel.mat',
             'ARA05C_07':'../InvertedModels/TL/ARA05C_line07_ts_bp_TL_vel.mat',
             'ARA05C_08':'../InvertedModels/TL/ARA05C_line08_bp_TL_vel.mat',
             'ARA05C_11':'../InvertedModels/TL/ARA05C_line11_ts_bp_TL_vel.mat',
             'ARA05C_14':'../InvertedModels/TL/ARA05C_line14_ts_bp_TL_vel.mat',
             'ARA05C_15':'../InvertedModels/TL/ARA05C_line15_ts_bp_TL_vel.mat',
             'ARA05C_16':'../InvertedModels/TL/ARA05C_line16_ts_bp_TL_vel.mat',
             'ARA05C_17':'../InvertedModels/TL/ARA05C_line17_ts_bp_TL_vel.mat'}

i = 0
fig,ax = plt.subplots(ncols=8,nrows=4,sharey='all',sharex='all',figsize=(6.5,4.5))
plt.subplots_adjust(wspace=0.05,hspace=0.05)
ys = np.arange(0,400,2.5)
for line1 in ['ARA05C_11','ARA04C_08','ARA05C_07','ARA05C_17','ARA04C_10']:
    line1_vel_file = h5.File(vel_files[line1],'r')
    line1_vel_data = {ol:line1_vel_file.get(ol)[()] for ol in ['vp','vs','1/q']}
    line1_vel_file.close()
    line1_vel_av = {ol:np.average(line1_vel_data[ol],axis=0).T for ol in line1_vel_data}
    for j,line2 in enumerate(['ARA05C_08','ARA04C_11','ARA05C_06','ARA05C_16','ARA05C_05','ARA05C_01','ARA05C_03',
                              'ARA04C_01']):
        if line2 in intersects[line1]:
            line2_vel_file = h5.File(vel_files[line2], 'r')
            line2_vel_data = {ol: line2_vel_file.get(ol)[()] for ol in ['vp', 'vs', '1/q']}
            line2_vel_file.close()
            line2_vel_av = {ol: np.average(line2_vel_data[ol], axis=0).T for ol in line2_vel_data}
            [ax[i, j].plot(line1_vel_av['vp'][:160, indx], ys,'g', label='%s %i' % (line1, indx),linewidth=0.75)
             for indx in intersects[line1][line2]]
            [ax[i, j].plot(line2_vel_av['vp'][:160, indx], ys,'r', label='%s %i' % (line2, indx), linewidth=0.75)
             for indx in intersects[line2][line1]]
        elif line1 not in ['ARA05C_07','ARA05C_17']:
            if j==0 and i==3: ax[i,j].spines[['right','top']].set_visible(False)
            elif j==0:
                ax[i,j].spines[['right','top','bottom']].set_visible(False)
                ax[i,j].axes.get_xaxis().set_visible(False)
            elif i==3:
                ax[i, j].spines[['right', 'top', 'left']].set_visible(False)
                ax[i,j].axes.get_yaxis().set_visible(False)
            else:ax[i,j].axis('off')
    if line1 != 'ARA05C_07':i +=1
ax[0,0].invert_yaxis()
ax[0,0].set_xticks([1.5,2.5])
[axi.set_title(s,size=7,c='r') for axi,s in zip(ax[0,:],['ARA05C_08','ARA04C_11','ARA05C_06','ARA05C_16','ARA05C_05',
                                                   'ARA05C_01','ARA05C_03','ARA04C_01'])]
[axi.text(4.7,200,s,c='g',size=7,ha='right',va='center')
 for axi,s in zip(ax[:,-1],['ARA05C_11','ARA04C_08','ARA05C_07\nARA05C_17','ARA04C_10'])]
[axi.set_ylabel('Depth (m)',size=6) for axi in ax[:,0]]
[axi.set_xlabel('$V_P$ (km s$^{-1}$)',size=6) for axi in ax[-1,:]]
[axi.tick_params(axis='both',labelsize=6) for axi in ax.flatten()]
ax[-1,-1].annotate('NE',xytext=(2.5,100),xy=(1.5,100),va='center', color='b',fontsize=7,
                 arrowprops=dict(arrowstyle="<|-", color='b', lw=2.5))
ax[-1,-1].annotate('SE',xytext=(1.6,350),xy=(1.6,100),ha='center', color='b',fontsize=7,
                 arrowprops=dict(arrowstyle="<|-", color='b', lw=2.5))
plt.savefig('figs/Figure_Intersects.png',dpi=300,bbox_inches='tight')
plt.show()

