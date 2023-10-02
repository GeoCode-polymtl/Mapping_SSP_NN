from CNN_Utilities import *
from TL_Utilities import *
import argparse
import matplotlib.pyplot as plt


# %% Parameters
parser = argparse.ArgumentParser()
parser.add_argument('-ln','--linenumber',type=str,default='05-06',
                    help='''Line number in the format 05-06 (ARAC survey number - line number), choose between
                          ['04-01', '04-02', '04-08', '04-09', '04-10', '04-11', '05-01', '05-03', '05-05', '05-06', 
                           '05-07', '05-08', '05-11', '05-12', '05-14', '05-15', '05-16', '05-17']''')
parser.add_argument('-gpu','--gpu',type=str,default=['0'],nargs='*',help='id number of the gpu(s) to use')

args = parser.parse_args()
linenumber = args.linenumber
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu)     #"0"

line_names = {'04-01':'ARA04C_line01_int2_bp','04-02':'ARA04C_line02_int2_bp','04-08':'ARA04C_line08_int2_bp',
              '04-09':'ARA04C_line09_int2_bp','04-10':'ARA04C_line10_int2_bp','04-11':'ARA04C_line11_int2_bp',
              '05-01':'ARA05C_line01_ts_bp','05-03':'ARA05C_line03_ts_bp','05-05':'ARA05C_line05_ts_bp',
              '05-06':'ARA05C_line06_ts_bp','05-07':'ARA05C_line07_ts_bp','05-08':'ARA05C_line08_bp',
              '05-11':'ARA05C_line11_ts_bp','05-12':'ARA05C_line12_ts_bp','05-14':'ARA05C_line14_ts_bp',
              '05-15':'ARA05C_line15_ts_bp','05-16':'ARA05C_line16_ts_bp','05-17':'ARA05C_line17_ts_bp'}


output_depth = '1500m'
mult_outputs=True
outlabel = ['vp','vs','1/q']
datatype = ['shotgather','dispersion','radon','fft_radon']
key_nn = 'all'
preprocessing = False
line = line_names[linenumber]
itrains,ntrains = 0,64
checkpoint_dir_tl = '../CheckpointsTL'

file_pars = {'ARA04C_line01_int2_bp': ["../DataPreprocessed/ARA04C_line01_ts_int2_cmds.sgy",
                                      '../DataPreprocessed/MultiInput/ARA04C_line01_int2_cmps_bp.mat', 'cmp', 50,True],
             'ARA04C_line02_int2_bp': ["../DataPreprocessed/ARA04C_line02_ts_int2_cmds.sgy",
                                      '../DataPreprocessed/MultiInput/ARA04C_line02_int2_cmps_bp.mat', 'cmp', 50,True],
             'ARA04C_line08_int2_bp': ["../DataPreprocessed/ARA04C_line08_ts_int2_cmds.sgy",
                                      '../DataPreprocessed/MultiInput/ARA04C_line08_int2_cmps_bp.mat', 'cmp', 50,True],
             'ARA04C_line09_int2_bp': ["../DataPreprocessed/ARA04C_line09_ts_int2_cmds.sgy",
                                      '../DataPreprocessed/MultiInput/ARA04C_line09_int2_cmps_bp.mat', 'cmp', 50,True],
             'ARA04C_line10_int2_bp': ["../DataPreprocessed/ARA04C_line10_ts_int2_cmds.sgy",
                                      '../DataPreprocessed/MultiInput/ARA04C_line10_int2_cmps_bp.mat', 'cmp', 50,True],
             'ARA04C_line11_int2_bp': ["../DataPreprocessed/ARA04C_line11_ts_int2_cmds.sgy",
                                      '../DataPreprocessed/MultiInput/ARA04C_line11_int2_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line01_ts_bp': ["../DataPreprocessed/ARA05C_line01_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line01_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line03_ts_bp': ["../DataPreprocessed/ARA05C_line03_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line03_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line05_ts_bp': ["../DataPreprocessed/ARA05C_line05_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line05_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line06_ts_bp': ["../DataPreprocessed/ARA05C_line06_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line06_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line07_ts_bp': ["../DataPreprocessed/ARA05C_line07_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line07_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line08_bp': ["../DataPreprocessed/ARA05C_line08_cmds.sgy",
                                  '../DataPreprocessed/MultiInput/ARA05C_line08_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line11_ts_bp': ["../DataPreprocessed/ARA05C_line11_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line11_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line12_ts_bp': ["../DataPreprocessed/ARA05C_line12_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line12_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line14_ts_bp': ["../DataPreprocessed/ARA05C_line14_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line14_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line15_ts_bp': ["../DataPreprocessed/ARA05C_line15_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line15_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line16_ts_bp': ["../DataPreprocessed/ARA05C_line16_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line16_ts_cmps_bp.mat', 'cmp', 50,True],
             'ARA05C_line17_ts_bp': ["../DataPreprocessed/ARA05C_line17_ts_cmds.sgy",
                                     '../DataPreprocessed/MultiInput/ARA05C_line17_ts_cmps_bp.mat', 'cmp', 50,True],
             }

# %% Evaluating
print('################## Evaluating ##################')
print('line: %s'%line)
print('Loading dataset')
filein, fileout, gather, ds, bp = file_pars[line]   # recorded, multi-input domain files, shot spacing, bandpass
predicted = []

offmin,dg = 85,12.5  # In meters.
offmax = offmin + 120*dg
offset = np.arange(offmin, offmax, dg)
data_rec = get_dataset_rec2(datatype,offset,filein=filein,fileout=fileout,gather=gather,bp=bp)
nsplits = data_rec['shotgather'].shape[0]//900+1    # Define the number of splits of the shot gather for memory handling
data_rec_split = { l:np.array_split(data_rec[l],nsplits,axis=0) for l in datatype} # splitting the shot gathers

ncmps = 11
predicted_values = []
for split in range(nsplits):
    print('Split %i/%i, tiling recorded data'%(split,nsplits-1))
    data_rec_ad = {key:np.tile(data_rec_split[key][split],[1,1,1,ncmps]) for key in data_rec}

    """The arange of ncmps is generated in the loop"""
    for key in data_rec_ad:
        for ii, i in enumerate(np.arange(ncmps // 2, 0, -1)):
            data_rec_ad[key][i:,:,:,ii] = data_rec_ad[key][:-i,:,:,ii]
            for j in range(i): data_rec_ad[key][j,:,:,ii] = data_rec_ad[key][0,:,:,ii]
            data_rec_ad[key][:-i,:,:,-ii-1] = data_rec_ad[key][i:,:,:,-ii- 1]
            for j in range(i, 0, -1): data_rec_ad[key][-j,:,:,-ii- 1] = data_rec_ad[key][-1,:,:,-ii- 1]

    """Evaluating the splitted data"""
    data_rec_ad = {l: tf.convert_to_tensor(data_rec_ad[l], dtype=tf.float32) for l in datatype}

    predicted_values.append(evaluate(data_rec_ad, itrains, ntrains, checkpoint_dir_tl, datatype, outlabel, output_depth,
                                   preprocessing,mult_outputs))
"""Concatenating the splitted data"""
predicted_values = {ol:np.concatenate([pv[ol] for pv in predicted_values],axis=1) for ol in outlabel}

"""Plotting"""
plot_lims = {'vp':(1.4,4),'vs':(0,1.5),'1/q':(1e-2,1e-1)}
fig, ax = plt.subplots(nrows=3, figsize=(3.5, 9),sharex='all')
for i, ol in enumerate(outlabel):
    im0 = ax[i].imshow(gaussian_filter(np.average(predicted_values[ol], axis=0)[:, :].T,sigma=1), aspect='auto',
                       cmap='jet', vmin=plot_lims[ol][0], vmax=plot_lims[ol][1],
                       extent=[0, predicted_values[ol][0].shape[0] * ds / 1000, 700, 0])
    fig.colorbar(im0, ax=ax[i], label=ol.capitalize())
    ax[i].set_title(ol.capitalize())
    ax[i].set_ylabel('Depth (m)')
ax[-1].set_xlabel('Distance (km)')
plt.show()

# %% Generating velocity file
savefile = True
if savefile:
    vel_file = '../InvertedModels/TL/'+line + "_TL_vel.mat"
    if not os.path.isfile(vel_file):
        h5file = h5.File(vel_file,'w')
        [h5file.create_dataset(ol,data=np.array(predicted_values[ol])) for ol in predicted_values]
        h5file.close()
