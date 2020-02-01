import os
import sys
import pandas as pd
from tqdm import tqdm
import scipy.io.wavfile as wavfile

WORKING_DIRECTORY = './youtube-dl/Output'
wav_files = os.listdir(WORKING_DIRECTORY)
print("Total Audio File downloaded",len(wav_files))

takeoff_wav_files = [x for x in wav_files if ("Takeoff" in x and "Landing" not in x)]
print("Total take off audio file",len(takeoff_wav_files))

stats = []


def get_duration():
    for wav_file in tqdm(takeoff_wav_files):
        splits = wav_file[:-4].split(' ')
        if 'Airbus' in splits:
            aircraft_type = splits[splits.index('Airbus')+1]
            reg_no = splits[splits.index('Airbus')+2]
        elif 'Boeing' in splits:
            aircraft_type = splits[splits.index('Boeing')+1]
            reg_no = splits[splits.index('Boeing')+2]
        else:
            aircraft_type = ''
            reg_no = ''
        file_dir = os.path.join(WORKING_DIRECTORY,wav_file)
        fs, y = wavfile.read(file_dir)
        N_channels = len(y.shape)
        if N_channels == 2:
            y = y.sum(axis=1)/2
        # try:
        # if y.dtype == 'int16':
        #     nb_bits = 16
        # elif y.dtype == 'int32':
        #     nb_bits = 32
        # max_nb_bits = float(2**(nb_bits-1))
        # y = y / (max_nb_bits+1.0)
        N_samples = y.shape[0]
        duration = N_samples/fs
        stats.append({'Title':wav_file,'Duration':duration,'aircraft_type':aircraft_type,'reg_no':reg_no})
        # except:
        #     print("this file name error-{}".format(wav_file))
    df = pd.DataFrame(stats)
    df.to_csv("lib.csv",index=False)

# get_duration()
df = pd.read_csv("lib.csv")

print(df['Duration'].describe())

df = df[df['Duration']<=300]
# df['filename'] = [x[:-4] for x in df['Title'].tolist()]
# print(len(df))
print(df.aircraft_type.value_counts()[:10])
CLASS_TYPE = df.aircraft_type.value_counts()[:5].index.tolist()
# masterdf = pd.read_csv('D:\FYP\master_youtube_akino33.csv')
# fn_regnumber_dict = pd.Series(masterdf.reg_no.values,index=masterdf.title).to_dict()
# aircraft_uq = masterdf.aircraft_type.unique().tolist()
# print(aircraft_uq)
# aircraft_dict = {x:x for x in aircraft_uq}
# exit(1)


# df['reg_no']=df['filename'].map(fn_regnumber_dict)
df.to_csv("lib2.csv",index=False)





