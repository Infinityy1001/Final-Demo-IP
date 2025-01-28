import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Fs = 256  # Fréquence d'échantillonnage
n_channels = 4  # Nombre de canaux
Wn = 1  # Durée de la fenêtre d'échantillonnage
n_samples = Wn * Fs  # Longueur de la fenêtre d'échantillonnage
batch_size = 64
epochs = 500
n_ff = [2, 4, 8, 16]  # Nombre de filtres de fréquence pour chaque module d'inception
n_sf = [1, 1, 1, 1]  # Nombre de filtres spatiaux dans chaque sous-bande de fréquence
