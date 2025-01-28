import pandas as pd
import mne
import numpy as np 

def load_and_preprocess_eeg(df):
    
    # Supprimer les colonnes non EEG, comme 'timestamp'
    df = df.loc[:, ~df.columns.str.contains('timestamps', case=False)]
    df = df.loc[:, ~df.columns.str.contains('Right AUX', case=False)]
    df = df.fillna(df.mean())  # Remplacement des valeurs nulles par la moyenne
    mne.set_log_level('WARNING') 
    # Convertir en format MNE
    info = mne.create_info(list(df.columns), ch_types=['eeg'] * len(df.columns), sfreq=256)
    info.set_montage('standard_1020')
    data = mne.io.RawArray(df.T, info)
    data.set_eeg_reference()
    
    # Découper en epochs
    epochs = mne.make_fixed_length_epochs(data, duration=1.0, overlap=0.0)  # Durée de 1 seconde, pas de chevauchement
    eeg_data = epochs.get_data()
    
    # Prendre seulement la première époque
    eeg_data = eeg_data[0:1, :, :, np.newaxis]  # Garder seulement la première époque
    
    return eeg_data