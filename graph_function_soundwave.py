import os
import numpy as np 
import pandas as pd 
from waveform_analysis import C_weight
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from function_researchpaper import energy_profile_E,zero_crossing_profile_Z,get_Tmid
from analyzeWAV import WORKING_DIRECTORY
from tqdm import tqdm

WORKING_DIRECTORY = './CHANGE_NAME_RECORDINGS'

PLOT_INDIVIDUAL = False
FRAME_LENGTH = 200
OVERLAPPING_PERCENTAGE = 0.2
SEGMENT_LENGTH = 2000

def plotout_C(y,y_c,aircraft_type,wav_file,fs):
    period = 1/fs
    t = np.arange(0, len(y)*period,period)
    if len(t) != len(y):
        t = t[:-1]
    fig, ax = plt.subplots(2,1, figsize=(18,12))
    title_png = "Sound File for {}-{}".format(aircraft_type,wav_file)
    ax[0].plot(t,y)
    ax[0].set_title('{}'.format(aircraft_type))
    ax[1].plot(t,y_c)
    ax[1].set_title('{}-with C weight filter'.format(aircraft_type))
    fig.suptitle(title_png)
    plt.savefig("PLOTTING_RESULT/Rec_"+title_png+"_with_Cweight.png")

def plot_sf(y,aircraft_type,wav_file,fs):
    period = 1/fs
    t = np.arange(0, len(y)*period,period)
    if len(t) != len(y):
        t = t[:-1]
    plt.figure(figsize=(18,12))
    title_png = "Sound File for {}-{}".format(aircraft_type,wav_file)
    plt.title(title_png)
    plt.plot(t[:],y)
    plt.savefig("PLOTTING_RESULT/Rec_"+title_png+".png")

def plot_energy_zerocross(energy,zero_cross,wav_file,fs,aircraft_type,Tmid_e):
    fig, ax = plt.subplots(1,1, figsize=(18,12))
    plt.plot(energy, 'b', label='Energy')
    plt.plot(zero_cross, 'g', label='Zero Cross')
    plt.axvline(x=Tmid_e,color='r',label='Tmid')
    plt.legend()
    title_png = "Sound File for {}-{}".format(aircraft_type,wav_file)
    fig.suptitle(title_png)
    plt.savefig("PLOTTING_RESULT/Rec_ZE_"+title_png+".png")

def plot_sf_Tmid_segment(y,Tmid,segment_length,wav_file,fs,aircraft_type):
    period = 1/fs
    segment_length = segment_length/1000*fs*period
    Tmid = Tmid*period
    t = np.arange(0, len(y)*period,period)
    if len(t) != len(y):
        t = t[:-1]
    plt.figure(figsize=(18,12))
    title_png = "{}".format(wav_file)
    plt.title(title_png)
    plt.plot(t[:],y)
    plt.legend()
    plt.axvline(x=Tmid,color='r',label='Tmid',linewidth=2)
    plt.axvline(x=Tmid-segment_length,color='r',label='-S',linewidth=0.5)
    plt.axvline(x=Tmid-2*segment_length,color='r',label='-2S',linewidth=0.5)
    plt.axvline(x=Tmid+2*segment_length,color='r',label='2S',linewidth=0.5)
    plt.axvline(x=Tmid+segment_length,color='r',label='S',linewidth=0.5)
    plt.savefig("PLOTTING_RESULT/Rec_SEG_"+title_png+".png")

def plot_multiple_sf_Tmid_segment(y_list,
                                Tmid_list,
                                segment_length,
                                wav_file_list,
                                aircraft_type_list,fs_list):
    fig, ax = plt.subplots(len(y_list),1, figsize=(18,10))
    title_png = "Segment for various Aircraft Sound"
    
    for ii,y in enumerate(y_list):
        fs = fs_list[ii]
        Tmid = Tmid_list[ii]
        wav_file = wav_file_list[ii]
        aircraft_type = aircraft_type_list[ii]
        period = 1/fs
        segment_length_used = segment_length/1000*fs*period
        t = np.arange(0, len(y)*period,period)
        if len(t) != len(y):
            t = t[:-1]
        Tmid = Tmid*period
        ax[ii].plot(t,y)
        ax[ii].set_title('{}<><><>{}'.format(aircraft_type,wav_file))
        ax[ii].axvline(x=Tmid,color='r',label='Tmid',linewidth=2)
        ax[ii].axvline(x=Tmid-segment_length_used,color='r',label='-S',linewidth=0.5)
        ax[ii].axvline(x=Tmid-2*segment_length_used,color='r',label='-2S',linewidth=0.5)
        ax[ii].axvline(x=Tmid+2*segment_length_used,color='r',label='2S',linewidth=0.5)
        ax[ii].axvline(x=Tmid+segment_length_used,color='r',label='S',linewidth=0.5)
        #print((Tmid,Tmid-segment_length,Tmid-2*segment_length,Tmid+2*segment_length,Tmid+segment_length))
    fig.suptitle(title_png)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("PLOTTING_RESULT/REC_"+title_png+".png")
        
if not(os.path.exists("PLOTTING_RESULT")):
    os.mkdir("PLOTTING_RESULT")

#df = pd.read_csv("lib2.csv")
df = pd.read_csv("lib_recorded.csv")
top10aircraft_type = df['aircraft_type'].value_counts()[:10].index.tolist()
df = df[df['aircraft_type'].isin(top10aircraft_type)]
idx = np.random.permutation(np.arange(len(df)))
df = df.iloc[idx].drop_duplicates('aircraft_type')
# df = df.drop_duplicates('aircraft_type')
soundfile_tograph = df['Title'].tolist()

print(len(soundfile_tograph))
print(soundfile_tograph)

wav_file = soundfile_tograph[0]

y_list = []
Tmid_list = []
wav_file_list = []
aircraft_type_list = []
fs_list = []

for wav_file in tqdm(soundfile_tograph):
    # splits = wav_file[:-4].split(' ')
    # if 'Airbus' in splits:
    #     aircraft_type = splits[splits.index('Airbus')+1]
    #     reg_no = splits[splits.index('Airbus')+2]
    # elif 'Boeing' in splits:
    #     aircraft_type = splits[splits.index('Boeing')+1]
    #     reg_no = splits[splits.index('Boeing')+2]
    # else:
    #     aircraft_type = ''
    #     reg_no = ''
    aircraft_type = wav_file.split("_")[0]

    file_dir = os.path.join(WORKING_DIRECTORY,wav_file)
    fs, y = wavfile.read(file_dir)

    N_channels = len(y.shape)
    if N_channels == 2:
        y = y.sum(axis=1)/2

    y_c = C_weight(y,fs)

    energy = energy_profile_E(y_c,FRAME_LENGTH,OVERLAPPING_PERCENTAGE,fs)
    energy_n = (energy - np.min(energy))/np.ptp(energy)
    zero_cross = zero_crossing_profile_Z(y_c,FRAME_LENGTH,OVERLAPPING_PERCENTAGE,fs)
    zero_cross_n = (zero_cross - np.min(zero_cross))/np.ptp(zero_cross)
    Tmid_e,Tmid = get_Tmid(zero_cross_n,energy_n,FRAME_LENGTH,OVERLAPPING_PERCENTAGE,fs)

    if PLOT_INDIVIDUAL == True:
        plot_sf(y,aircraft_type,wav_file,fs)
        plotout_C(y,y_c,aircraft_type,wav_file,fs)
        plot_energy_zerocross(energy_n,zero_cross_n,wav_file,fs,aircraft_type,Tmid_e)
        plot_sf_Tmid_segment(y,Tmid,SEGMENT_LENGTH,wav_file,fs,aircraft_type)
    else:
        y_list.append(y_c)
        Tmid_list.append(Tmid)
        wav_file_list.append(wav_file)
        aircraft_type_list.append(aircraft_type)
        fs_list.append(fs)

if PLOT_INDIVIDUAL == False:
    plot_multiple_sf_Tmid_segment(y_list[:5],Tmid_list[:5],SEGMENT_LENGTH,wav_file_list[:5],aircraft_type_list[:5],fs_list[:5])


exit(1)