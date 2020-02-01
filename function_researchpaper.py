import numpy as np 
from waveform_analysis import C_weight
from tqdm import tqdm
#import librosa.load
import os

def energy_profile_E(signal,frame_length,overlapping_percentage,fs=16000):
    """
    overlapping_percentage is in the format of decimal
    """
    frame_length_sample = frame_length/1000*fs
    k_total = int((signal.size-frame_length_sample)/(frame_length_sample*(1-overlapping_percentage)))
    result = []
    for ii in range(k_total):
        start_id = int(ii*((1-overlapping_percentage)*frame_length_sample))
        end_id = int(start_id+frame_length_sample)
        signal_foc = signal[start_id:end_id]
        signal_foc = np.absolute(signal_foc)
        signal_abs = np.square(signal_foc)
        energy = np.sum(signal_abs)
        result.append(energy)
    return np.asarray(result)

def zero_crossing_profile_Z(signal,frame_length,overlapping_percentage,fs=16000):
    """
    overlapping_percentage is in the format of decimal
    """
    frame_length_sample = frame_length/1000*fs
    s3 = np.sign(signal)
    s3[s3==0] = -1
    zero_crossings = np.where(np.diff(s3))[0].tolist()
    F_foc = np.zeros((signal.size-1,), dtype=int)
    F_foc[zero_crossings] = 1
    k_total = int((signal.size-frame_length_sample)/
    (frame_length_sample*(1-overlapping_percentage)))
    result = []
    for ii in range(k_total):
        start_id = int(ii*((1-overlapping_percentage)*frame_length_sample))
        end_id = int(start_id+frame_length_sample-1)
        _Z = F_foc[start_id:end_id]
        Z_foc = np.sum(_Z)
        result.append(Z_foc)
    return np.asarray(result)

def get_Tmid(Z_prof,E_prof,frame_length,overlapping_percentage,fs=16000):
    frame_length_sample = frame_length/1000*fs
    Emax = np.amax(E_prof)
    T_mid_possible = np.argwhere(E_prof > (Emax*0.85))
    Z_select = Z_prof[T_mid_possible]
    Zmax = np.amax(Z_select)
    Tmid_e = np.where(Z_prof == Zmax)[0]
    if not(len(Tmid_e)==1):
        Tmid_e = Tmid_e[0]
    Tmid = int(Tmid_e*((1-overlapping_percentage)*frame_length_sample))
    return (Tmid_e,Tmid)

