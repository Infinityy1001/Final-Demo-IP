import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import mne
from pylsl import StreamInlet, resolve_byprop  
from model.eegitnet import EEGITNet
from model.initial_parameters import *
from model.load_process_eeg import load_and_preprocess_eeg
import time
import keyboard

# Charger le modèle pré-entraîné
model = EEGITNet()  
model.load_state_dict(torch.load('model/eegitnet_model.pth'))
model = model.to(device)
model.eval()


# STREAM EEG

# Trouver le stream EEG
print("Recherche des streams LSL...")
streams = resolve_byprop('type', 'EEG')

# Se connecter au stream EEG
inlet = StreamInlet(streams[0])
print("Connecté au stream EEG.")

sampling_frequency = 256  
num_channels = 5  
buffer_duration = 1  
buffer_size = int(buffer_duration * sampling_frequency)  
data_buffer = np.zeros((buffer_size, num_channels))  
timestamps = []  
current_index = 0  

try:
    while True:
        # Obtenir un échantillon de données
        sample, timestamp = inlet.pull_sample()
        data_buffer[current_index] = sample  # Ajouter l'échantillon au buffer
        timestamps.append(timestamp)  # Enregistrer le timestamp
        current_index += 1
        
        # Si le buffer est plein, créer un DataFrame et traiter les données
        if current_index == buffer_size:
            # Créer un DataFrame avec les données EEG et les timestamps
            df = pd.DataFrame(data_buffer, columns=['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'])
            
            eeg_data = load_and_preprocess_eeg(df)
            
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)

            # Faire une prédiction
            with torch.no_grad():
                output = model(eeg_tensor)
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

            # Afficher la prédiction
            if predicted.item() == 0:
                print(f"Prédiction : Gauche (Confiance : {probabilities[0, 0].item() * 100:.2f}%)")
                keyboard.press('left')
                time.sleep(0.1)
                keyboard.release('left')
                keyboard.press('up')
                time.sleep(0.1)  # Attente courte pour simuler la durée de la pression
                keyboard.release('up')
            else:
                print(f"Prédiction : Droite (Confiance : {probabilities[0, 1].item() * 100:.2f}%)")
                keyboard.press('right')
                time.sleep(0.1)
                keyboard.release('right')
                keyboard.press('up')
                time.sleep(0.1)  # Attente courte pour simuler la durée de la pression
                keyboard.release('up')
            
            # Réinitialiser le buffer et les indices
            data_buffer = np.zeros((buffer_size, num_channels))
            timestamps = []
            current_index = 0
        
        time.sleep(1 / sampling_frequency)  # Attendre la prochaine itération

except KeyboardInterrupt:
    print("Arrêt de la lecture des données.")


