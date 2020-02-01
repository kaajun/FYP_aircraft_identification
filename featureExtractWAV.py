import os
import sys
import librosa
import librosa.display
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy.io import wavfile
from math import floor
from numpy import *
import speechpy
from sklearn.preprocessing import LabelBinarizer
from waveform_analysis import C_weight
from analyzeWAV import CLASS_TYPE,WORKING_DIRECTORY
from function_researchpaper import energy_profile_E,zero_crossing_profile_Z,get_Tmid

stats = []

NPY_FOLDER_DIR = './YoutubeAudio/NPY_DATA_segment/'

if not os.path.exists(NPY_FOLDER_DIR):
    if not os.path.exists('./YoutubeAudio/'):
        os.mkdir("YoutubeAudio")
    os.mkdir(NPY_FOLDER_DIR)

SAMPLE_RATE = 16000 
FRAME_LENGTH = 4000
STRIDE = 1600
N_MELS = 60
N_MFCC = 20
N_WIDTH = 9
N_CHROMA = 13
N_BANDS = 6
DATA_POINT_WINDOW = 250
DATA_POINT_STRIDE = 100

# SAMPLE_RATE = 44100
# FRAME_LENGTH = 11025
# STRIDE = 4410
# N_MELS = 60
# N_MFCC = 20
# N_WIDTH = 9
# N_CHROMA = 13
# N_BANDS = 6
# DATA_POINT_WINDOW = 680
# DATA_POINT_STRIDE = 275

df = pd.read_csv("lib2.csv")
aircraft_count = df['aircraft_type'].value_counts().index.tolist()[:5]
df = df[df['aircraft_type'].isin(CLASS_TYPE)]
CLASS_DICT = dict(zip(aircraft_count,[np.array([n+1]) for n in range (5) ]))

###additional filter
# df = df[df['Duration']>240]
# print(df['aircraft_type'].value_counts())
# exit(1)

files_to_process = df['Title'].tolist()

def get_aircraft_type_title(wav):
    splits = wav[:-4].split(' ')
    if 'Airbus' in splits:
        aircraft_type = splits[splits.index('Airbus')+1]
    elif 'Boeing' in splits:
        aircraft_type = splits[splits.index('Boeing')+1]
    else:
        aircraft_type = ''
    return aircraft_type

def apply_cmvn(npy):
    return speechpy.processing.cmvnw(npy)

def apply_c_weight(y,fs):
    return C_weight(y,fs)

def audio_last_n_sec(y,fs,last_n_second):
    return y[-last_n_second*fs:]

def audio_segment_zeroCrossxEnergy(y,fs,segment_s_length,FL= 200,OP = 0.2):
    energy = energy_profile_E(y,FL,OP,fs)
    energy_n = (energy - np.min(energy))/np.ptp(energy)
    zero_cross = zero_crossing_profile_Z(y,FL,OP,fs)
    # print('length of y',len(y))
    zero_cross_n = (zero_cross - np.min(zero_cross))/np.ptp(zero_cross)
    Tmid_e,Tmid = get_Tmid(zero_cross_n,energy_n,FL,OP,fs)
    # print('tmid',Tmid)
    start_sound = int(Tmid-2*segment_s_length*fs)
    end_sound = int(Tmid+2*segment_s_length*fs)
    # print('start1',start_sound)
    # print('end1',end_sound)
    if start_sound < 0:
        start_sound = 0
    if end_sound > len(y):
        end_sound = len(y)
    # print('start',start_sound)
    # print('end',end_sound)
    y = y[start_sound:end_sound]
    
    return y,Tmid/fs,start_sound/fs,end_sound/fs

def feature_extract_audio(y,
                            sample_rate=SAMPLE_RATE,
                            frame_length=FRAME_LENGTH,
                            stride=STRIDE,
                            n_mels=N_MELS,
                            n_mfcc=N_MFCC,
                            width=N_WIDTH,
                            n_chroma=N_CHROMA,
                            n_bands=N_BANDS):
    #_clip, _ = librosa.load(os.path.join(dir,wav))
    _clip = y
    _mels = librosa.feature.melspectrogram(_clip,n_fft=frame_length, hop_length=stride, n_mels=n_mels,sr=sample_rate)
    
    _mfcc = librosa.feature.mfcc(y=_clip,sr=sample_rate,n_mfcc=n_mfcc,n_fft=frame_length,hop_length=stride)
    _mfcc_d = librosa.feature.delta(_mfcc,width=width,order=1)
    _mfcc_dd = librosa.feature.delta(_mfcc,width=width,order=2)
    _chroma = librosa.feature.chroma_stft(y=_clip,sr=sample_rate,n_fft=frame_length,hop_length=stride,n_chroma=n_chroma)
    _spec_cons = librosa.feature.spectral_contrast(y=_clip,sr=sample_rate,n_fft=frame_length,hop_length=stride,n_bands=n_bands)
    _res = np.concatenate((_mels,_mfcc,_mfcc_d,_mfcc_dd,_chroma,_spec_cons),axis=0)
    return(_res)

def generate_data_points(npy,wav,window=DATA_POINT_WINDOW,stride=DATA_POINT_STRIDE):
    _seg_data = []
    _seg_label = []
    label_text = get_aircraft_type_title(wav[:-4])
    label_class = CLASS_DICT[label_text]
    npy_array = npy.transpose()
    npy_array = apply_cmvn(npy_array)
    dt_points = int(floor((npy_array.shape[0]-window)/stride) + 1)
    
    for ii in range(dt_points):
        _seg_array = npy_array[ii*stride:ii*stride+window,:]
        _seg_array = _seg_array[newaxis,:,:]
        if len(_seg_data)==0 :
            _seg_data = _seg_array
            _seg_label = label_class
        else:
            _seg_data = np.concatenate([_seg_data,_seg_array],axis=0)
            _seg_label = np.concatenate([_seg_label,label_class],axis=0)
    _seg_array = npy_array[npy_array.shape[0]-window:npy_array.shape[0],:]
    _seg_array = _seg_array[newaxis,:,:]
    
    _seg_data = np.concatenate([_seg_data,_seg_array],axis=0)
    _seg_label = np.concatenate([_seg_label,label_class],axis=0)
    # print(_seg_data)
    # print(_seg_label)
    # print(_seg_data.shape)
    # print(_seg_label.shape)
    return(_seg_data,_seg_label)
    #np.save("{0}_fin_data/{1}_data.npy".format(folder,npy),_seg_data)
    #np.save("{0}_fin_label/{1}_label.npy".format(folder,npy),_seg_label)

if not os.path.exists(NPY_FOLDER_DIR+'data/'):
    os.mkdir(NPY_FOLDER_DIR+'data/')
    os.mkdir(NPY_FOLDER_DIR+'label/')

for wavfile in tqdm(files_to_process):
    y, fs = librosa.load(os.path.join(WORKING_DIRECTORY,wavfile),sr=44100)
    y = apply_c_weight(y,fs)
    # dur = len(y)/fs
    # y = audio_last_n_sec(y,fs,140)
    # y_after_n_sec = len(y)/fs
    # print(y_after_n_sec)
    y,Tm,_s,_einsum_path_dispatcher = audio_segment_zeroCrossxEnergy(y,fs,15)
    y_after_audio_seg = len(y)/fs
    npy = feature_extract_audio(y)
    data_npy,label_npy = generate_data_points(npy,wavfile)
    np.save(NPY_FOLDER_DIR+'data/'+'{}_dt.npy'.format(wavfile[:-4]),data_npy)
    np.save(NPY_FOLDER_DIR+'label/'+'{}_lb.npy'.format(wavfile[:-4]),label_npy)
    stats.append({'title':wavfile[:-4],
                # 't_mid':Tm,
                # 'start_s':s,
                # 'end_s':e,
                # 'after_last_n_sec':y_after_n_sec,
                'after_audio_segment':y_after_audio_seg,
                'data_shape':data_npy.shape,
                'label_shape':label_npy[0],
                'aircraft_type':get_aircraft_type_title(wavfile)})
    # except:
    #     print("There's error processing file {}".format(wavfile))
    #     continue

df = pd.DataFrame(stats)
df.to_csv("extract_result_youtube_segment.csv",index=False)

