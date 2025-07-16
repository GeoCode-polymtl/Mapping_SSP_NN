import numpy as np
import tensorflow as tf
import tf_keras as k3
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from CNN_Utilities import test_loss,read_inp2#,load_dataset_test
from importlib import import_module
import h5py as h5
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
from skimage.metrics import structural_similarity as ssim

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# %%
datatype = ['shotgather','dispersion','radon','fft_radon']
dhmins = ['all']
ntrains =2

batch_size = 10
outlabel = ['vp','vs','1/q']
categorical = False
mult_outputs = True

depth_train = 700
output_depth = '1500m'

dataset_module = import_module("DefinedDataset." + 'DatasetPermafrost_simpler_40dhmin_%s' % output_depth)
dataset = getattr(dataset_module, 'DatasetPermafrost_simpler_40dhmin_%s' % output_depth)()
dataset.generate_dataset(ngpu=1)

indx_p = {i: j for j, i in enumerate(outlabel)}
indx_l = {i: j for j, i in enumerate(outlabel)}
indx_l['1/q'] = 3

lims = {'vp': [dataset.model.properties['vp'][0] / 1000, dataset.model.properties['vp'][1] / 1000],
        'vs': [dataset.model.properties['vs'][0] / 1000, dataset.model.properties['vs'][1] / 1000],
        'q': [dataset.model.properties['q'][0], dataset.model.properties['q'][1]]}
lims['1/q'] = [1 / lims['q'][1], 1 / lims['q'][0]]

# checkpoint_dir_tl = '../../NN/Results/1500m/TL/Checkpoints'
# core_path_tl='../../NN/TransferLearning/Datasets/noisy_data/DatasetPermafrost'
checkpoint_dir_tl = '/mnt/d/JeffersonNN/with10dhmin/2D/Checkpoints'
core_path_tl='/mnt/d/JeffersonNN/2D/Datasets/noisy_data/DatasetPermafrost'

output_depth = '1500m'
folder_path_list = {'deeper_%s' % output_depth: '%s_2D_deeper_%s' % (core_path_tl, output_depth),
                    '40dhmin_%s' % output_depth: '%s_2D_40dhmin_%s' % (core_path_tl, output_depth),
                    '30dhmin_%s' % output_depth: '%s_2D_30dhmin_%s' % (core_path_tl, output_depth),
                    '20dhmin_%s' % output_depth: '%s_2D_20dhmin_%s' % (core_path_tl, output_depth),
                    '20dhmin_hv_%s' % output_depth: '%s_2D_20dhmin_hv_%s' % (core_path_tl, output_depth),
                    '10dhmin_%s' % output_depth: '%s_2D_10dhmin_%s' % (core_path_tl, output_depth),
                    '10dhmin_hv_%s' % output_depth: '%s_2D_10dhmin_hv_%s' % (core_path_tl, output_depth)
                    }
gen_file = False
if gen_file:
    predicted_values_tl = dict()
    label_values_tl = dict()
    # for key in ['40dhmin_1500m']:
    for key in list(folder_path_list):
        print('...............getting %s'%key)
        input_2D = {key: [] for key in datatype}
        label_2D = []
        for i in range(1,16):
            for j in range(23):
                h5file = h5.File(folder_path_list[key]+'/test/%i_%i.mat'%(i,j))
                temp_data = h5file.get('inputs/')
                input_2D['shotgather'].append(temp_data['shotgather'][()])
                input_2D['radon'].append(temp_data['radon'][()])
                input_2D['dispersion'].append(temp_data['dispersion'][()])
                input_2D['fft_radon'].append(temp_data['fft_radon'][()])
                temp_label = h5file.get('labels/')[()]
                label_2D.append(temp_label)
        input_2D = {key : np.array(input_2D[key]) for key in datatype}
        label_2D = np.array(label_2D)

        ##############
        ncmps = 11
        data_rec_ad = {key: np.tile(input_2D[key], [1, 1, 1, ncmps]) for key in input_2D}

        """The arange of ncmps is generated in the loop"""
        for key_d_t in data_rec_ad:
            for ii, i in enumerate(np.arange(ncmps // 2, 0, -1)):
                data_rec_ad[key_d_t][i:, :, :, ii] = data_rec_ad[key_d_t][:-i, :, :, ii]
                for j in range(i): data_rec_ad[key_d_t][j, :, :, ii] = data_rec_ad[key_d_t][0, :, :, ii]
                data_rec_ad[key_d_t][:-i, :, :, -ii - 1] = data_rec_ad[key_d_t][i:, :, :, -ii - 1]
                for j in range(i, 0, -1): data_rec_ad[key_d_t][-j, :, :, -ii - 1] = data_rec_ad[key_d_t][-1, :, :, -ii - 1]
        ########

        predicted_tl = []
        ntrains=64
        latest_epoch_tl = 8
        for j in range(ntrains):
            cp_path_core = checkpoint_dir_tl + '/' + '_'.join(datatype) + '/' + str(j) # + '/all'
            if latest_epoch_tl == None:
                latest_epoch = max([int(i[-4:]) for i in os.listdir(cp_path_core)])  # For load whole model
            else:
                latest_epoch = int(latest_epoch_tl)
            latest_path = cp_path_core + '/cp%s_%04d' %(output_depth[:-1],latest_epoch)

            print('\t train: %i, latest epoch: %i' % (j, latest_epoch))
            # cnn_model_tl = tf.keras.models.load_model(latest_path, custom_objects={"test_loss": test_loss})
            cnn_model_tl = k3.models.load_model(latest_path, custom_objects={"test_loss": test_loss})
            # pred = cnn_model_tl.predict(input_2D)
            pred = cnn_model_tl.predict(data_rec_ad)

            predicted_tl.append(pred)

        predicted_values_tl[key] = {ol: [i[indx_p[ol]][:,:,0] * (lims[ol][1] - lims[ol][0]) + lims[ol][0]
                                         for i in predicted_tl] for j,ol in enumerate(outlabel)}
        label_values_tl[key] = {ol: label_2D[:, :, indx_l[ol]] * (lims[ol][1] - lims[ol][0]) + lims[ol][0]
                                for ol in outlabel}

    # %% Save file
    h5file = h5.File('2Dtesting.mat','a')
    # for key1 in ['40dhmin_1500m']:
    for key1 in list(predicted_values_tl):
        for key2 in list(predicted_values_tl[key1]):
            h5file.create_dataset('predicted_values_tl/%s/%s'%(key1,key2),
                                  data=np.array(predicted_values_tl[key1][key2]))
    for key1 in list(label_values_tl):
        for key2 in list(label_values_tl[key1]):
            h5file.create_dataset('label_values_tl/%s/%s'%(key1,key2),data=np.array(label_values_tl[key1][key2]))
    h5file.close()

# %% Read files
print("Reading Files")
h5file = h5.File('2Dtesting.mat','r')
predicted_values_tl = {key:{ol:h5file.get('predicted_values_tl/%s/%s'%(key,ol))[()] for ol in outlabel}
                       for key in folder_path_list}
label_values_tl = {key:{ol:np.split(h5file.get('label_values_tl/%s/%s'%(key,ol))[()],15) for ol in outlabel}
                       for key in folder_path_list}
h5file.close()

predicted_av_tl = {key:{ol:np.split(np.average(predicted_values_tl[key][ol],axis=0),15) for ol in outlabel}
                       for key in folder_path_list}
predicted_sd_tl = {key:{ol:np.split(np.std(predicted_values_tl[key][ol],axis=0),15) for ol in outlabel}
                   for key in folder_path_list}

vlims = {'vp':[1.5,4.0],'vs':[0,1.5],'1/q':[0,0.075]}
vlims_sd = {'vp': [0,0.35], 'vs': [0,0.25], '1/q': [0,0.02]}

ssim_db = {}
l2norm = {}
mistrust = {}
for i, key in enumerate(label_values_tl):
    ssim_db[key] = {}
    l2norm[key] = {}
    mistrust[key] = {}
    for ol in outlabel:
        ssim_db[key][ol] = []
        l2norm[key][ol] = []
        mistrust[key][ol] = []
        # fig, ax = plt.subplots(nrows=2, ncols=len(label_values_tl),sharey='all',sharex='all')
        fig, ax = plt.subplots(nrows=2, ncols=len(label_values_tl[key][ol]), sharey='all', sharex='all')
        for ii in range(len(label_values_tl[key][ol])):
            ax[0,ii].imshow(label_values_tl[key][ol][ii].T,aspect='auto',vmin=vlims[ol][0],vmax=vlims[ol][1])
            ax[1,ii].imshow(predicted_av_tl[key][ol][ii].T, aspect='auto',vmin=vlims[ol][0],vmax=vlims[ol][1])
            temp_ssim = ssim(label_values_tl[key][ol][ii][5:-5,:280],predicted_av_tl[key][ol][ii][5:-5,:],
                             data_range=(predicted_av_tl[key][ol][ii].max()-predicted_av_tl[key][ol][ii].min()))
            ssim_db[key][ol].append(temp_ssim)
            ax[1,ii].text(22,22,'SSIM: %1.2f'%temp_ssim,ha='center',c='w')
            temp_l2 = np.linalg.norm(label_values_tl[key][ol][ii][5:-5,:280]-predicted_av_tl[key][ol][ii][5:-5,:],ord=2,axis=-1)
            temp_l2 /= (lims[ol][1] - lims[ol][0])
            temp_l2 /= (np.array(predicted_av_tl[key][ol]).shape[-1])**.5
            temp_l2 = np.average(temp_l2)
            l2norm[key][ol].append(temp_l2)
            temp_mt = np.linalg.norm(predicted_av_tl[key][ol][ii] - np.average(predicted_av_tl[key][ol],axis=0),
                                     ord=2,axis=-1)
            temp_mt /= (lims[ol][1] - lims[ol][0])
            temp_mt /= (np.array(predicted_av_tl[key][ol]).shape[-1]) ** .5
            temp_mt = np.average(temp_mt)
            mistrust[key][ol].append(temp_mt)
            ax[1, ii].text(22, 52, 'L2: %.2f' %(temp_l2*100) + '$\%$', ha='center', c='w')
            # ax[0,i].set_title(key,size=6)
        plt.suptitle('%s, %s'%(key,ol))
        plt.show()

# %% 2D test Paper2
print("2D test Paper2")
fig,ax = plt.subplots(ncols=3,nrows=4,sharey=True,sharex=True)
models = [3,1,2,3]
titles = ['$V_P$','$V_S$','1/Q']
units = ['km/s','km/s','']
cbar_pos = [0.14,0.415,0.69]
# for i,key in enumerate(list(label_values_tl)[:-1]):
for i, key in enumerate(['deeper_1500m','40dhmin_1500m','30dhmin_1500m','20dhmin_1500m']):
    for j,ol in enumerate(outlabel):
        temp_plot = np.concatenate(
            (label_values_tl[key][ol][models[i]][5:-5,:280],predicted_av_tl[key][ol][models[i]][5:-5,:]),axis=0).T
        im = ax[i,j].imshow(temp_plot,aspect='auto',vmin=vlims[ol][0],vmax=vlims[ol][1],extent=[0,2,700,0])
        ssim_i = np.average(ssim_db[key][ol])
        ax[i,j].text(1.5,100,'SSIM: %.2f'%ssim_i,c='w',ha='center',size=6)
        l2_i = np.average(l2norm[key][ol])
        ax[i,j].text(1.5,200,'L$_2$: %.2f' %(l2_i*100) + '$\%$',ha='center',c='w',size=6)
        if i==0: ax[i,j].set_title(titles[j])
        if i==3:
            cbar_ax = fig.add_axes([cbar_pos[j],0.025,0.2,0.01])
            cbar = fig.colorbar(im,cax=cbar_ax,orientation='horizontal')
            cbar.set_label(titles[j])
            cbar.set_label(units[j], fontsize=8)
[axi.tick_params(axis='y',labelsize=8) for axi in ax.flatten()]
[axi.set_xticks(np.arange(0.5,2.5)) for axi in ax.flatten()]
[axi.set_xticklabels(['Label','Prediction']) for axi in ax.flatten()]
minor_locator = AutoMinorLocator(2)
[axi.xaxis.set_minor_locator(minor_locator) for axi in ax.flatten()]
[axi.grid(which='minor',c='w') for axi in ax.flatten()]
[axi.grid(which='minor', c='w') for axi in ax.flatten()]
[axi.set_ylabel('Depth (m)') for axi in ax[:,0]]
[axi.text(-.8,350,s,va='center',ha='right',fontsize=10) for axi,s in zip(ax[:,0],('a)','b)','c)','d)'))]
plt.savefig('figs/2Dtest.png',dpi=300,bbox_inches='tight')
plt.show()

# %% 2D test Paper2 version 2
# fig,ax = plt.subplots(ncols=3,nrows=4,sharey=True,sharex=True,figsize=[8.512,4.8])
print("2D test Paper2 version 2")
fig = plt.figure(figsize=[8.512,4.8])
gs = fig.add_gridspec(nrows=4,ncols=8,width_ratios=[2,1,.2,2,1,.2,2,1])
ax = np.array([[fig.add_subplot(gs[i,j]) for j in range(8)] for i in range(4)])

models = [3,1,2,3]
titles = ['$V_P$','$V_S$','1/Q']
units = ['km/s','km/s','']
cbar_pos = [0.14,0.415,0.69]
cbar_pos_sd = [0.2875,0.56,0.835]
# for i,key in enumerate(list(label_values_tl)[:-1]):
for i, key in enumerate(['deeper_1500m','40dhmin_1500m','30dhmin_1500m','20dhmin_1500m']):
    for j,ol in enumerate(outlabel):
        temp_plot = np.concatenate(
            (label_values_tl[key][ol][models[i]][:,:280],predicted_av_tl[key][ol][models[i]]),axis=0).T
        im = ax[i,j*3].imshow(temp_plot,aspect='auto',vmin=vlims[ol][0],vmax=vlims[ol][1],extent=[0,2,700,0])
        im_sd = ax[i,j*3+1].imshow(predicted_sd_tl[key][ol][models[i]].T, aspect='auto',
                                   vmin=vlims_sd[ol][0], vmax=vlims_sd[ol][1], extent=[0, 1, 700, 0])
        # ssim_i = np.average(ssim_db[key][ol])
        # ax[i,j*3].text(1.5,100,'SSIM: %.2f'%ssim_i,c='w',ha='center',size=6)
        l2_i = np.average(l2norm[key][ol])
        ax[i,j*3].text(1.5,100,'L$_2$: %.2f' %(l2_i*100) + '$\%$',ha='center',c='w',size=6)
        if i==0: ax[i,j*3].set_title(titles[j])
        if i==3:
            cbar_ax = fig.add_axes([cbar_pos[j],0.025,0.11,0.01])
            cbar = fig.colorbar(im,cax=cbar_ax,orientation='horizontal')
            cbar.set_label(titles[j])
            cbar.set_label(units[j], fontsize=8)

            cbar_ax_sd = fig.add_axes([cbar_pos_sd[j], 0.025, 0.06, 0.01])
            cbar_sd = fig.colorbar(im_sd, cax=cbar_ax_sd, orientation='horizontal')
            cbar_sd.set_label(titles[j])
            cbar_sd.set_label(units[j], fontsize=8)
[axi.tick_params(axis='y',labelsize=8) for axi in ax.flatten()]
[axi.set_xticks(np.arange(0.5,2.5)) for axi in ax[:,::3].flatten()]
[axi.set_xticklabels(['Label','Prediction']) for axi in ax[-1,::3].flatten()]
[axi.set_xticks([0.5]) for axi in ax[:,1::3].flatten()]
[axi.set_xticklabels(['$\sigma$']) for axi in ax[:,1::3].flatten()]
[axi.set_xticklabels([]) for axi in ax[:,2::3].flatten()]
[axi.set_yticklabels([]) for axi in ax[:,1::3].flatten()]
[axi.set_yticks([]) for axi in ax[:,1::3].flatten()]
[axi.set_yticklabels([]) for axi in ax[:,2::3].flatten()]
[axi.set_axis_off() for axi in ax[:,2::3].flatten()]

[axi.set_xticklabels([]) for axi in ax[:-1,:].flatten()]

minor_locator = AutoMinorLocator(2)
[axi.xaxis.set_minor_locator(minor_locator) for axi in ax[:,::3].flatten()]
[axi.grid(which='minor',c='w') for axi in ax.flatten()]
[axi.grid(which='minor', c='w') for axi in ax.flatten()]
[axi.set_ylabel('Depth (m)') for axi in ax[:,0]]
[axi.text(-1,350,s,va='center',ha='right',fontsize=10) for axi,s in zip(ax[:,0],('a)','b)','c)','d)'))]
plt.savefig('figs/2Dtest_v2.png',dpi=300,bbox_inches='tight')
plt.show()

# %% 2D test Paper2 dissertation
print("2D test Paper2 dissertation")
fig = plt.figure(figsize=[6,4.8])
gs = fig.add_gridspec(nrows=3,ncols=2,width_ratios=[2,1])
ax = np.array([[fig.add_subplot(gs[i,j]) for j in range(2)] for i in range(3)])

models = [3,1,2,3]
titles = ['$V_P$','$V_S$','1/Q']
units = ['km/s','km/s','']
cbar_pos = [0.31,0.6,0.89][::-1]
cbar_pos_sd = [0.35,0.56,0.835]
key = '40dhmin_1500m'
# for i,key in enumerate(list(label_values_tl)[:-1]):
for i, key in enumerate(['deeper_1500m','40dhmin_1500m','30dhmin_1500m','20dhmin_1500m']):
    for j,ol in enumerate(outlabel):
        print(j,ol)
        temp_plot = np.concatenate(
            (label_values_tl[key][ol][models[i]][:,:280],predicted_av_tl[key][ol][models[i]]),axis=0).T
        im = ax[j,0].imshow(temp_plot,aspect='auto',vmin=vlims[ol][0],vmax=vlims[ol][1],extent=[0,2,700,0])
        im_sd = ax[j,1].imshow(predicted_sd_tl[key][ol][models[i]].T, aspect='auto',
                                   vmin=vlims_sd[ol][0], vmax=vlims_sd[ol][1], extent=[0, 1, 700, 0])
#         ssim_i = np.average(ssim_db[key][ol])
#         ax[i,j*3].text(1.5,100,'SSIM: %.2f'%ssim_i,c='w',ha='center',size=6)
        l2_i = np.average(l2norm[key][ol])
        ax[j,0].text(1.5,200,'L$_2$: %.2f' %(l2_i*100) + '$\%$',ha='center',c='w',size=6)
        # if i==0: ax[i,j*3].set_title(titles[j])
#         if i==3:
        cbar_ax = fig.add_axes([.31,cbar_pos[j],0.11,0.01])
        cbar = fig.colorbar(im,cax=cbar_ax,orientation='horizontal')
        cbar_ax.xaxis.set_ticks_position('top')
        cbar_ax.xaxis.set_ticks(vlims[ol])
        cbar_ax.xaxis.set_ticklabels(vlims[ol],fontsize=6)
        # cbar.set_label(titles[j])
        # cbar.set_label(units[j], fontsize=8)
#
        cbar_ax_sd = fig.add_axes([0.85,cbar_pos[j], 0.06, 0.01])
        cbar_sd = fig.colorbar(im_sd, cax=cbar_ax_sd, orientation='horizontal')
        cbar_ax_sd.xaxis.set_ticks_position('top')
        cbar_ax_sd.xaxis.set_ticks(vlims_sd[ol])
        cbar_ax_sd.xaxis.set_ticklabels(vlims_sd[ol], fontsize=6)
        # cbar_sd.set_label(titles[j])
        # cbar_sd.set_label(units[j], fontsize=8)
[axi.tick_params(axis='y',labelsize=8) for axi in ax.flatten()]
[axi.set_xticks(np.arange(0.5,2.5)) for axi in ax[:,::3].flatten()]
[axi.set_xticklabels(['Label','Prediction']) for axi in ax[:,0].flatten()]
[axi.set_xticks([0.5]) for axi in ax[:,1::3].flatten()]
[axi.set_xticklabels(['$\sigma$']) for axi in ax[:,1::3].flatten()]
[axi.set_xticklabels([]) for axi in ax[:,2::3].flatten()]
[axi.set_yticklabels([]) for axi in ax[:,1::3].flatten()]
[axi.set_yticks([]) for axi in ax[:,1::3].flatten()]
[axi.set_yticklabels([]) for axi in ax[:,2::3].flatten()]
[axi.set_axis_off() for axi in ax[:,2::3].flatten()]
#
# [axi.set_xticklabels([]) for axi in ax[:-1,:].flatten()]

# minor_locator = AutoMinorLocator(2)
[axi.xaxis.set_minor_locator(minor_locator) for axi in ax[:,::3].flatten()]
[axi.grid(which='minor',c='w') for axi in ax.flatten()]
[axi.grid(which='minor', c='w') for axi in ax.flatten()]
[axi.set_ylabel('Depth (m)') for axi in ax[:,0]]
[axi.text(-.5,350,s,va='center',ha='right',fontsize=10) for axi,s in zip(ax[:,0],('$V_P$','$V_S$','1/Q'))]
plt.subplots_adjust(wspace=0.025,hspace=.55)
plt.savefig('figs/2Dtest_v3_dissertation.png',dpi=300,bbox_inches='tight')
# plt.subplots_adjust(wspace=0.025,hspace=.55)
plt.show()

# %% Paper 3
print("Paper3")
fig,ax = plt.subplots(ncols=3,nrows=1,sharey=True,sharex=True,figsize=[6,1.75])
plt.subplots_adjust(wspace=0.05)
# models = [3,1,2,3]
models = [8,6,-1,12,10]
titles = ['$V_P$','$V_S$','$1/Q$']
units = ['$V_P$ (km/s)','$V_S$ (km/s)','$1/Q$']
cbar_pos = [0.14,0.415,0.69]
# for i,key in enumerate(list(label_values_tl)[:-1]):
key,i = '40dhmin_1500m',1
for j,ol in enumerate(outlabel):
    temp_plot = np.concatenate(
        (label_values_tl[key][ol][models[i]][5:-5,:280],predicted_av_tl[key][ol][models[i]][5:-5,:]),axis=0).T
    im = ax[j].imshow(temp_plot,aspect='auto',vmin=vlims[ol][0],vmax=vlims[ol][1],extent=[0,2,700,0])
    ssim_i = np.average(ssim_db[key][ol])
    # ax[j].text(1.5,100,'SSIM: %.2f'%ssim_i,c='w',ha='center',size=6)
    l2_i = np.average(l2norm[key][ol])
    # ax[j].text(1.5,200,'L$_2$: %.2f' %(l2_i*100) + '$\%$',ha='center',c='w',size=6)
    # ax[j].set_title(titles[j])
    cbar_ax = fig.add_axes([cbar_pos[j],-0.05,0.2,0.01])
    cbar = fig.colorbar(im,cax=cbar_ax,orientation='horizontal')
    cbar.set_label(titles[j])
    cbar.set_label(units[j], fontsize=8)
[axi.tick_params(axis='y',labelsize=8) for axi in ax.flatten()]
[axi.set_xticks(np.arange(0.5,2.5)) for axi in ax.flatten()]
[axi.set_xticklabels(['Label','Prediction']) for axi in ax.flatten()]
minor_locator = AutoMinorLocator(2)
[axi.xaxis.set_minor_locator(minor_locator) for axi in ax.flatten()]
[axi.grid(which='minor',c='w') for axi in ax.flatten()]
[axi.grid(which='minor', c='w') for axi in ax.flatten()]
ax[0].set_ylabel('Depth (m)')
# [axi.text(-.8,350,s,va='center',ha='right',fontsize=10) for axi,s in zip(ax[:,0],('a)','b)','c)','d)'))]
plt.savefig('figs/2Dtest_paper3.png',dpi=300,bbox_inches='tight')
plt.show()

# %% l2norm vs standard deviation
print("l2norm vs standard deviation")
fig,ax = plt.subplots(ncols=3,figsize=[6,2.])
plt.subplots_adjust(wspace=0.6)
for i,ol in enumerate(outlabel):
    if i == 1:
        [ax[i].scatter(np.array(l2norm[key][ol])*(lims[ol][1] - lims[ol][0])*1e3,
                       np.mean(np.mean(predicted_sd_tl[key][ol],axis=-1),axis=-1)*1e3,s=10,label=key[:-6])
         for key in list(mistrust) if key != '20dhmin_hv_1500m']
    else:
        [ax[i].scatter(np.array(l2norm[key][ol])*(lims[ol][1] - lims[ol][0])*1e3,
                       np.mean(np.mean(predicted_sd_tl[key][ol],axis=-1),axis=-1)*1e3,s=10)
         for key in list(mistrust) if key != '20dhmin_hv_1500m']
    ax[i].set_xlabel('L2 norm (m/s)')
    # ax[i].set_ylabel('Mistrust')
    # ax[i].set_title(ol.capitalize())
    # ax[i].set_xlim([0,28])
    # ax[i].set_ylim([0,15])
    ax[i].plot([0,500],[0,500],'k--')
[axi.set_xlim([0,i]) for axi,i in zip(ax,[500,400,40])]
[axi.set_ylim([0,i]) for axi,i in zip(ax,[200,130,15])]
ax[0].set_ylabel('$\sigma_{V_P}$ (m/s)')
ax[1].set_ylabel('$\sigma_{V_S}$ (m/s)')
ax[2].set_ylabel('$\sigma_{1/Q} (x10^{-3})$')
ax[1].legend(fontsize=6)
plt.savefig('figs/sd_l2norm.png',dpi=300,bbox_inches='tight')
plt.show()

# %% l2norm vs mistrust
print("l2norm vs mistrust")
predicted_NN = {key:{ol: np.split(predicted_values_tl[key][ol],15,axis=1)
                     for ol in outlabel} for key in list(folder_path_list)}

l2norm_nns = {}
mistrust = {}
for i, key in enumerate(label_values_tl):
    ssim_db[key] = {}
    l2norm_nns[key] = {}
    mistrust[key] = {}
    for ol in outlabel:
        l2norm_nns[key][ol] = []
        mistrust[key][ol] = []
        for ii in range(len(label_values_tl[key][ol])):
            temp_l2 = [np.average(np.linalg.norm(label_values_tl[key][ol][ii][:,:280]-predicted_NN[key][ol][ii][jj,:,:],
                                                 ord=2,axis=-1)/(lims[ol][1]-lims[ol][0])/(280**.5))
                       for jj in range(64)]
            l2norm_nns[key][ol].append(np.array(temp_l2))
            temp_mt = [np.average(np.linalg.norm(np.average(predicted_NN[key][ol][ii],axis=0)-predicted_NN[key][ol][ii][jj,:,:],
                                                 ord=2,axis=-1)/(lims[ol][1]-lims[ol][0])/(280**.5))
                       for jj in range(64)]
            mistrust[key][ol].append(np.array(temp_mt))

fig,ax = plt.subplots(ncols=3,figsize=[6,3.],sharey=True)
plt.subplots_adjust(wspace=0.05)
for i,ol in enumerate(outlabel):
    if i == 1:
        [ax[i].scatter(np.concatenate(l2norm_nns[key][ol])*100,np.concatenate(mistrust[key][ol])*100,s=10,
                       label=key[:-6])
         for key in list(mistrust) if key != '20dhmin_hv_1500m']
    else:
        [ax[i].scatter(np.concatenate(l2norm_nns[key][ol])*100,np.concatenate(mistrust[key][ol])*100,s=10)
         for key in list(mistrust) if
         key != '20dhmin_hv_1500m']
    ax[i].set_xlabel('L2 error (%)')
    # ax[i].set_ylabel('Mistrust')
    ax[i].set_title(ol.capitalize())
    ax[i].set_xlim([0,28])
    ax[i].set_ylim([0,15])
    ax[i].plot([0,28],[0,28],'k--')
ax[0].set_ylabel('L2 prediction (%)')
ax[1].legend()
plt.savefig('figs/mistrust_l2norm.png',dpi=300,bbox_inches='tight')
plt.show()

# _,ax = plt.subplots(ncols=3,nrows=2)
# for i,ol in enumerate(outlabel):
#     ax[0,i].imshow(label_values_tl[key][ol][0][:,:280].T,aspect='auto')
#     ax[1,i].imshow(predicted_NN[key][ol][0][0,:,:280].T,aspect='auto')
# plt.show()


# predicted_NN = {key:{ol: np.array([np.split(predicted_values_tl[key][ol][i,:,:],5) for i in range(64)])
#                      for ol in outlabel}
#                 for key in list(folder_path_list)}
#
# l2norm_nns = {}
# mistrust = {}
# for i, key in enumerate(label_values_tl):
#     ssim_db[key] = {}
#     l2norm_nns[key] = {}
#     mistrust[key] = {}
#     for ol in outlabel:
#         l2norm_nns[key][ol] = []
#         mistrust[key][ol] = []
#         for ii in range(len(label_values_tl[key][ol])):
#             temp_l2 = [np.average(np.linalg.norm(label_values_tl[key][ol][ii][:,:280]-predicted_NN[key][ol][jj,ii,:,:],
#                                                  ord=2,axis=-1)/(lims[ol][1]-lims[ol][0])/(280**.5))
#                        for jj in range(64)]
#             l2norm_nns[key][ol].append(np.array(temp_l2))
#             temp_mt = [np.average(np.linalg.norm(predicted_av_tl[key][ol][ii][:,:280]-predicted_NN[key][ol][jj,ii,:,:],
#                                                  ord=2,axis=-1)/(lims[ol][1]-lims[ol][0])/(280**.5))
#                        for jj in range(64)]
#             mistrust[key][ol].append(np.array(temp_mt))

# fig,ax = plt.subplots(ncols=3,figsize=[6,3.],sharey=True)
# plt.subplots_adjust(wspace=0.05)
# for i,ol in enumerate(outlabel):
#     if i == 1:
#         [ax[i].scatter(np.concatenate(l2norm_nns[key][ol])*100,np.concatenate(mistrust[key][ol])*100,s=10,
#                        label=key[:-6])
#          for key in list(mistrust) if key != '20dhmin_hv_1500m']
#     else:
#         [ax[i].scatter(np.concatenate(l2norm_nns[key][ol])*100,np.concatenate(mistrust[key][ol])*100,s=10)
#          for key in list(mistrust) if
#          key != '20dhmin_hv_1500m']
#     ax[i].set_xlabel('L2 norm (%)')
#     # ax[i].set_ylabel('Mistrust')
#     ax[i].set_title(ol.capitalize())
#     ax[i].set_xlim([0,28])
#     ax[i].set_ylim([0,15])
#     ax[i].plot([0,28],[0,28],'k--')
# ax[0].set_ylabel('Mistrust (%)')
# ax[1].legend()
# plt.savefig('figs/mistrust_l2norm.png',dpi=300,bbox_inches='tight')
# plt.show()

temp_vp, temp_vs, temp_1q = [], [], []
for key in l2norm: temp_vp.append(l2norm[key]['vp'])
for key in l2norm: temp_vs.append(l2norm[key]['vs'])
for key in l2norm: temp_1q.append(l2norm[key]['1/q'])

print('Vp average L2 norm: %.4f%%'%((np.array(temp_vp)).flatten().mean()*100))
print('Vs average L2 norm: %.4f%%'%((np.array(temp_vs)).flatten().mean()*100))
print('1/Q average L2 norm: %.4f%%'%((np.array(temp_1q)).flatten().mean()*100))
