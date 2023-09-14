# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from importlib import import_module
from hyperbolic_radon import hyperbolic_radon
from dispersion import dispersion
from GeoFlow.SeismicUtilities import random_noise
from tqdm import tqdm
import h5py as h5
from NN_permafrost.CNN_Utilities import test_loss

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def shot2cmps_geometry(traces, sx, gx):
    """
    Transform an array of traces from shot gathers to a list of cmps gathers

    Args:
        traces (): A 2 dimensional matrix containing all the traces. Dimensions 0 and 1 are time and traces
        sx (): shot positions
        gx (): geophones positions

    Returns:
        traces_cmp, cmps_gat

    """
    cmps = ((sx + gx) / 2).astype(int)
    offsets = sx - gx

    'Counting cmps'
    unique_cmps, unique_inverse, unique_counts = np.unique(cmps, return_inverse=True, return_counts=True)
    unique_counts = unique_counts[unique_inverse]

    'removing cmps with count lower than the max'
    mask = unique_counts == unique_counts.max()
    traces = traces[:, mask]
    cmps = cmps[mask]
    offsets = offsets[mask]

    'Sorting with cmp location'
    mask = np.argsort(cmps)
    traces = traces[:, mask]
    cmps = cmps[mask]
    offsets = offsets[mask]

    'Generating CMP gathers and sorting by offset'
    i, ii = 0, 120
    traces_cmp, cmps_gat, offsets_cmp = [], [], []
    while ii <= len(cmps):
        mask = np.argsort(offsets[i:ii])
        temp = traces[:, i:ii]
        traces_cmp.append(temp[:, mask])
        cmps_gat.append(cmps[i:ii][mask])
        offsets_cmp.append(offsets[i:ii][mask])
        i += 120
        ii += 120

    return traces_cmp, cmps_gat,offsets_cmp

def cnn_sg(ncmps=11):
    """Preprocesing for 2D input NN"""
    l2 = 1e-7
    preprocessing_input = tf.keras.layers.Input(shape=(1000,120,ncmps),name='shotgather')
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(preprocessing_input)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    preprocessing_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 3), activation=tf.nn.leaky_relu,
                                                  padding='same', name='preprocessing_shotgather',
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(preprocessing_input, preprocessing_output, name='preprocessing_shotgather')

def autoencoder_sg(ncmps=11):
    """Preprocessing for 2D input NN"""
    l2 = 1e-7
    preprocessing_input = tf.keras.layers.Input(shape=(1000,120,ncmps),name='shotgather')
    'Encoder'
    x = tf.keras.layers.Conv2D(filters=11, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(preprocessing_input)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    'Decoder'
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 2))(x)
    preprocessing_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 3), activation=tf.nn.leaky_relu,
                                                  padding='same', name='preprocessing_shotgather',
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(preprocessing_input,preprocessing_output,name='preprocessing_shotgather')

def cnn_radon(ncmps=11):
    """Preprocesing for 2D input NN"""
    l2 = 1e-7
    preprocessing_input = tf.keras.layers.Input(shape=(1000,200,ncmps),name='radon')
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(preprocessing_input)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    preprocessing_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 3), activation=tf.nn.leaky_relu,
                                                  padding='same', name='preprocessing_radon',
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(preprocessing_input, preprocessing_output, name='preprocessing_radon')

def autoencoder_radon(ncmps=11):
    """Preprocessing for 2D input NN"""
    l2 = 1e-7
    preprocessing_input = tf.keras.layers.Input(shape=(1000,200,ncmps),name='radon')
    'Encoder'
    x = tf.keras.layers.Conv2D(filters=11, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(preprocessing_input)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(4, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 5))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    'Decoder'
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 5))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(4, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 2))(x)
    preprocessing_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 3), activation=tf.nn.leaky_relu,
                                                  padding='same', name='preprocessing_radon',
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(preprocessing_input,preprocessing_output,name='preprocessing_radon')

def cnn_disp(ncmps=11):
    """Preprocesing for 2D input NN"""
    l2 = 1e-7
    preprocessing_input = tf.keras.layers.Input(shape=(200,200,ncmps),name='dispersion')
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(preprocessing_input)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    preprocessing_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 3), activation=tf.nn.leaky_relu,
                                                  padding='same', name='preprocessing_dispersion',
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(preprocessing_input, preprocessing_output, name='preprocessing_dispersion')

def autoencoder_disp(ncmps=11):
    """Preprocessing for 2D input NN"""
    l2 = 1e-7
    preprocessing_input = tf.keras.layers.Input(shape=(200,200,ncmps),name='dispersion')
    'Encoder'
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(preprocessing_input)
    x = tf.keras.layers.MaxPool2D(pool_size=(4, 4))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 5))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    'Decoder'
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 5))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(4, 4))(x)
    preprocessing_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 3), activation=tf.nn.leaky_relu,
                                                  padding='same', name='preprocessing_dispersion',
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(preprocessing_input,preprocessing_output,name='preprocessing_dispersion')

def cnn_fftradon(ncmps=11):
    """Preprocesing for 2D input NN"""
    l2 = 1e-7
    preprocessing_input = tf.keras.layers.Input(shape=(200,200,ncmps),name='fft_radon')
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(preprocessing_input)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    preprocessing_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 3), activation=tf.nn.leaky_relu,
                                                  padding='same', name='preprocessing_fft_radon',
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(preprocessing_input, preprocessing_output, name='preprocessing_fft_radon')

def autoencoder_fftradon(ncmps=11):
    """Preprocessing for 2D input NN"""
    l2 = 1e-7
    preprocessing_input = tf.keras.layers.Input(shape=(200,200,ncmps),name='fft_radon')
    'Encoder'
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(preprocessing_input)
    x = tf.keras.layers.MaxPool2D(pool_size=(4, 4))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 5))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    'Decoder'
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 5))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(4, 4))(x)
    preprocessing_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(17, 3), activation=tf.nn.leaky_relu,
                                                  padding='same', name='preprocessing_fft_radon',
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(preprocessing_input,preprocessing_output,name='preprocessing_fft_radon')

# %% "Generating shotgathers"
def generate_noisy_dataset_2d(case: str, trainsize: int, testsize: int, ncmps: int):
    """
    Generate noisy files containing information of the cmps centered and adjacent in the shotgather (time offset),
    dispersion, radon, and fft_radon domains and the corresponding labels.
    Args:
        case (): 
        trainsize (): 
        testsize (): 
        ncmps (): 

    Returns:
        object: 
    """
    dataset_module = import_module("DefinedDataset."+case)
    dataset = getattr(dataset_module, case)()
    dataset.trainsize = trainsize
    dataset.validatesize = validatesize = (dataset.trainsize // 10)
    dataset.testsize = testsize
    
    fmax = 100
    dt = dataset.acquire.dt*dataset.acquire.resampling
    nt = dataset.acquire.NT//dataset.acquire.resampling
    t = np.arange(0, nt*dt, dt)
    
    c = np.linspace(1000, 4500, 200)
    c_radon = np.linspace(1000, 3000, 200)
    
    print('Generating shots')
    dataset.generate_dataset(ngpu=1)
    # sizes_lab = dataset.generator.read(dataset.files['train'][0])[1]['vpdepth'].shape[0]
    lims = {'q': [dataset.model.properties['q'][0], dataset.model.properties['q'][1]]}
    lims['1/q'] = [1 / lims['q'][1], 1 / lims['q'][0]]
    
    for phase,size in zip(['train','validate','test'],[trainsize,validatesize,testsize]):
        print('\n_________________Generating %s data_________________'%phase)
        base_folder = 'Datasets/noisy_data/' + case + '/' + phase
        if not os.path.exists(base_folder) or not os.listdir(base_folder):
            os.makedirs(base_folder, exist_ok=True)
            for s in range(size):
                data = dataset.get_example(phase=phase)
                sizes_lab = data[1]['vpdepth'].shape[0]
                "Getting cmps"
                sx = np.array([dataset.generator.seismic.src_pos_all[0, int(srcid)]
                               for srcid in dataset.generator.seismic.rec_pos_all[3, :]])
                gx = dataset.generator.seismic.rec_pos_all[0, :]
    
                traces = np.reshape(data[0]['shotgather'][...,-1],(1000,-1),'F') # Fortran format seems to do the trick
                traces_cmp, cmps_gat, offsets_cmp = shot2cmps_geometry(traces, sx, gx)
    
                shot_all,disp_all,radon_all,fft_radon_all = [],[],[],[]
                for i,cmp in tqdm(enumerate(traces_cmp),desc='Transforming model %i to different domains' %s):
                    'Transforming to different domains'
                    x = offsets_cmp[i]
                    d = traces_cmp[i]
                    disp = dispersion(d.T, dt, x, c, fmax=fmax, epsilon=1e-6).numpy().T
                    disp = random_noise(disp, .005)
                    radon0 = hyperbolic_radon(d.T, t, x, c_radon).numpy().T
                    d = random_noise(d, .005)
                    radon = hyperbolic_radon(d.T, t, x, c_radon).numpy().T
                    freq = np.fft.fftfreq(radon.shape[0], dt)
                    mask = (freq >= 0) & (freq < fmax)
                    fft_radon = np.abs(np.fft.fft(radon0, axis=0))[mask, :]
                    'Normalizing'
                    disp = np.expand_dims((disp-np.min(disp))/(np.max(disp)-np.min(disp)),axis=-1)
                    d = np.expand_dims(d/np.max(np.abs(d)),axis=-1)
                    radon = np.expand_dims(radon/np.max(np.abs(radon)), axis=-1)
                    fft_radon = np.expand_dims((fft_radon-np.min(fft_radon)) / (np.max(fft_radon) - np.min(fft_radon)),
                                               axis=-1)
                    'Appending'
                    shot_all.append(d)
                    disp_all.append(disp)
                    radon_all.append(radon)
                    fft_radon_all.append(fft_radon)
    
                'Grouping'
                inputs = {'shotgather': np.array(shot_all), 'dispersion': np.array(disp_all),
                          'radon': np.array(radon_all), 'fft_radon': np.array(fft_radon_all)}
                inputs_ad = {key: np.tile(inputs[key], [1, 1, 1, ncmps]) for key in inputs}
                for key in inputs_ad:
                    for ii, i in enumerate(np.arange(ncmps // 2, 0, -1)):
                        inputs_ad[key][i:, :, :, ii] = inputs_ad[key][:-i, :, :, ii]
                        for j in range(i): inputs_ad[key][j, :, :, ii] = inputs_ad[key][0, :, :, ii]
                        inputs_ad[key][:-i, :, :, -ii - 1] = inputs_ad[key][i:, :, :, -ii - 1]
                        for j in range(i, 0, -1): inputs_ad[key][-j, :, :, -ii - 1] = inputs_ad[key][-1, :, :, -ii - 1]
    
                'saving'
                fnl = len('%s/Datasets/%s/%s/example_' % (os.getcwd(), case, phase))  # file name length
                if phase == 'train': file_num = int(data[-1][fnl:])
                elif phase == 'validate': file_num = int(data[-1][fnl:]) - (trainsize - 1)
                else: file_num = int(data[-1][fnl:]) - (trainsize + validatesize - 1)
    

                if not os.path.exists(base_folder):
                    os.makedirs(base_folder,exist_ok=True)
                # base_name = 'Datasets/noisy_data/' + case + '/' + phase + '/' + str(file_num)
                for i in tqdm(range(ncmps//2,inputs_ad['shotgather'].shape[0] -ncmps//2),
                              desc='Writing files model %i'%s):
                    labels = np.empty([4, sizes_lab])
                    idx = int(np.mean(cmps_gat[i])//2.5)
                    labels[0] = data[1]['vpdepth'][:,idx]
                    labels[1] = data[1]['vsdepth'][:,idx]
                    labels[2] = data[1]['qdepth'][:, idx]
                    q = labels[2]*(lims['q'][1]-lims['q'][0]) + lims['q'][0]
                    labels[3] = (1/q - lims['1/q'][0])/(lims['1/q'][1]-lims['1/q'][0])
    
                    file_name = '%s/%i_%i.mat'%(base_folder,file_num,i)
                    noisy_file = h5.File(file_name,'a')
                    [noisy_file.create_dataset('inputs/' + inp, data=inputs_ad[inp][i,...]) for inp in inputs_ad]
                    noisy_file.create_dataset('labels', data=labels.T)
                    noisy_file.close()
        else: continue

def cnn_preprocessing(cnn_model,preprocessing_nn,datatype=['shotgather','dispersion','radon','fft_radon'],
                      outlabel=['vp','vs','1/q'],key_nn='all'):
    """Generating the TL NN

    Args:
        cnn_model (): The already trainned NN
        preprocessing_nn (): List of preprocessing NN for the shotgather, dispersion, radon and fft_radon images
        datatype ():
        outlabel ():
        key_nn ():
    """

    "Modifying the NN"
    "Extract the configuration https://stackoverflow.com/questions/49546922/keras-replacing-input-layer"
    cnn_model_config = cnn_model.get_config()

    "Change the configuration"
    for i in range(len(cnn_model_config['layers'])):
        if cnn_model_config['layers'][i]['class_name'] == 'InputLayer':
            "If layer is Input, we change the name to read the preprocessed image"
            layer = cnn_model_config['layers'][i]
            cnn_model_config['layers'][i]['name'] = 'preprocessing_%s' % cnn_model_config['layers'][i]['name']
            cnn_model_config['layers'][i]['config']['name'] = 'preprocessing_%s' \
                                                              % cnn_model_config['layers'][i]['config']['name']
        elif cnn_model_config['layers'][i]['inbound_nodes'][0][0][0] in datatype:
            "If the layer takes as input the InputLayer we update the name of the input layer"
            cnn_model_config['layers'][i]['inbound_nodes'][0][0][0] = 'preprocessing_%s' \
                                                                      % cnn_model_config['layers'][i]['inbound_nodes'] \
                                                                          [0][0][0]
    for i in range(len(cnn_model_config['input_layers'])):
        "We update the name of the inputlayers outside the layers in the config dictionary"
        cnn_model_config['input_layers'][i][0] = 'preprocessing_%s' % cnn_model_config['input_layers'][i][0]

    "Create new model"
    cnn_model2 = cnn_model.__class__.from_config(cnn_model_config, custom_objects={"test_loss": test_loss})
    tf.keras.utils.plot_model(cnn_model2, to_file="model2_%s.png" % key_nn, show_shapes=True, expand_nested=True)

    "Including preprocessing"
    cnn_model2.trainable = False
    combined_model = cnn_model2([preprocessing_nn[pre].output for pre in preprocessing_nn])
    combined_model[0]._name = combined_model[0].name[6:-20]
    combined_model[1]._name = combined_model[1].name[6:-20]
    combined_model[2]._name = combined_model[2].name[6:-20]

    cnn_comb_model = tf.keras.models.Model(inputs=[preprocessing_nn[l].input for l in datatype],
                                           outputs=combined_model)
    cnn_comb_model.output_names[0] = cnn_comb_model.output[0].name
    cnn_comb_model.output_names[1] = cnn_comb_model.output[1].name
    cnn_comb_model.output_names[2] = cnn_comb_model.output[2].name
    tf.keras.utils.plot_model(cnn_comb_model, to_file="comb_model_%s.png" % key_nn, show_shapes=True,
                              expand_nested=False, show_layer_activations=True)
    cnn_comb_model.summary()

    "compiling"
    loss = test_loss
    if '1/q' in outlabel:
        loss_weights = {ol: (1 - .02) / (len(outlabel) - 1) for ol in outlabel if ol != '1/q'}
        loss_weights['1/q'] = 0.02
    else:
        loss_weights = {ol: 1 / len(outlabel) for ol in outlabel}
    cnn_comb_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, loss_weights=loss_weights)
    return cnn_comb_model

def cnn_nopreprocessing(cnn_model,ncmps,datatype=['shotgather','dispersion','radon','fft_radon'],
                        outlabel=['vp','vs','1/q'],key_nn='all'):
    """
    Modify the first layer of the 1D NN and its weights to match the 2D input

    Args:
        cnn_model ():
        ncmps ():
        datatype ():
        outlabel ():
        key_nn ():
    """
    "Extract the configuration https://stackoverflow.com/questions/49546922/keras-replacing-input-layer"
    # weights = cnn_model.get_weights()
    cnn_model_config = cnn_model.get_config()
    # inputs_summary = [(layer['name'], layer['config']['batch_input_shape'])
    #                   for layer in cnn_model_config['layers'] if layer['class_name'] == 'InputLayer']

    "Change the configuration"
    for i in range(len(cnn_model_config['layers'])):
        if cnn_model_config['layers'][i]['class_name'] == 'InputLayer':
            # If layer is Input, we change the shape and the name read the preprocessed image
            layer = cnn_model_config['layers'][i]
            cnn_model_config['layers'][i]['config']['batch_input_shape'] = (
            *layer['config']['batch_input_shape'][:-1], ncmps)
        else:
            if cnn_model_config['layers'][i]['inbound_nodes'][0][0][0] not in datatype:
                cnn_model_config['layers'][i]['config']['trainable'] = False
            if cnn_model_config['layers'][i]['class_name'] == 'Functional':
                for nested_layer in cnn_model_config['layers'][i]['config']['layers']:
                    if nested_layer['class_name'] != 'InputLayer':
                        nested_layer['config']['trainable'] = False
    # inputs_summary2 = [(layer['name'], layer['config']['batch_input_shape'])
    #                    for layer in cnn_model_config['layers'] if layer['class_name'] == 'InputLayer']

    "Create new model"
    cnn_model2 = cnn_model.__class__.from_config(cnn_model_config, custom_objects={"test_loss": test_loss,
                                                                                   "leaky_relu":tf.nn.leaky_relu})
    tf.keras.utils.plot_model(cnn_model2, to_file="model2_%s.png"%key_nn, show_shapes=True, expand_nested=True)

    "Weights"
    weigths = []
    for layer in cnn_model.layers:
        layer_weight = layer.get_weights()
        if layer.name != 'tf.concat' and layer.input.name in datatype:
            # if layer.name != 'tf.concat' and layer.input.name in ['preprocessing_%s' %key for key in datatype]:
            if bool(layer_weight):
                print(layer_weight[0].shape)
                ncmp = ncmps
                layer_weight[0] = np.tile(layer_weight[0], [1, 1, ncmp, 1])
                # The weight is the same in the center and .1 of the trained weights in the edges
                layer_weight[0][:, :, : ncmp // 2, :] *= .1
                layer_weight[0][:, :, -ncmp // 2 + 1:, :] *= .1
                print(layer_weight[0].shape)
        weigths.append(layer_weight)

    'Loading weights'
    for layer, weigth in zip(cnn_model2.layers, weigths):
        layer.set_weights(weigth)

    "compiling"
    loss = test_loss
    if '1/q' in outlabel:
        loss_weights = {ol: (1 - .02) / (len(outlabel) - 1) for ol in outlabel if ol != '1/q'}
        loss_weights['1/q'] = 0.02
    else:
        loss_weights = {ol: 1 / len(outlabel) for ol in outlabel}
    cnn_model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, loss_weights=loss_weights)
    cnn_model2.summary()

    return cnn_model2


def evaluate(data_rec_ad, itrains, ntrains, checkpoint_dir_tl, datatype, outlabel, output_depth,
             preprocessing,mult_outputs):
    predicted = []
    for j in range(itrains, itrains + ntrains):
        cp_path_core_tl = checkpoint_dir_tl + '/' + '_'.join(datatype) + '/' + str(j)
        # checkpoint_path_tl = cp_path_core_tl + '/cp%s_{epoch:04d}' % output_depth[:-1]
        # cp_callback_tl = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_tl, save_weights_only=False,
        #                                                     verbose=1, save_freq='epoch')
        # tb_callback_tl = tf.keras.callbacks.TensorBoard(cp_path_core_tl + 'TB')
        # datatypestr = '_'.join(datatype)

        # latest_epoch_tl = 0
        if os.path.isdir(cp_path_core_tl):
            latest_epoch_tl = max([int(i[-4:]) for i in os.listdir(cp_path_core_tl)])
            latest_path_tl = cp_path_core_tl + '/cp%s_%04d' % (output_depth[:-1], latest_epoch_tl)
            print('NN: %i, latest epoch tl: %i' % (j, latest_epoch_tl))
            cnn_comb_model = tf.keras.models.load_model(latest_path_tl, custom_objects={"test_loss": test_loss})
            if preprocessing:
                cnn_comb_model.output_names[0] = cnn_comb_model.output[0].name[6:-20]
                cnn_comb_model.output_names[1] = cnn_comb_model.output[1].name[6:-20]
                cnn_comb_model.output_names[2] = cnn_comb_model.output[2].name[6:-20]
            else:
                cnn_comb_model.output_names[0] = cnn_comb_model.output[0].name[:-20]
                cnn_comb_model.output_names[1] = cnn_comb_model.output[1].name[:-20]
                cnn_comb_model.output_names[2] = cnn_comb_model.output[2].name[:-20]

        pred = cnn_comb_model.predict(data_rec_ad, batch_size=10)
        if mult_outputs:
            pred = np.transpose(np.array(pred), axes=[-1, 1, 2, 0])[0]
        predicted.append(pred)

    dataset_module = import_module("DefinedDataset." + 'DatasetPermafrost_2D_40dhmin_1500m')
    dataset = getattr(dataset_module, 'DatasetPermafrost_2D_40dhmin_1500m')()
    lims = {'vp': [dataset.model.properties['vp'][0] / 1000, dataset.model.properties['vp'][1] / 1000],
            'vs': [dataset.model.properties['vs'][0] / 1000, dataset.model.properties['vs'][1] / 1000],
            'q': [dataset.model.properties['q'][0], dataset.model.properties['q'][1]]}
    lims['1/q'] = [1 / lims['q'][1], 1 / lims['q'][0]]

    indx = {i: j for j, i in enumerate(outlabel)}
    predicted_values = {ol: [i[:, :, indx[ol]] * (lims[ol][1] - lims[ol][0]) + lims[ol][0] for i in predicted]
                        for ol in outlabel}
    return predicted_values

# %%
# trainsize, testsize = 1,0
# ncmps = 11
# case = 'DatasetPermafrost_2D_40dhmin_1500m'
# generate_noisy_dataset_2d(case,trainsize,testsize,ncmps)