
import torch
import torch.nn as nn

class EEGITNet(nn.Module):
    def __init__(self, num_classes=3, n_channels=4):
        super(EEGITNet, self).__init__()
        
        # Couche de convolution temporelle
        self.temporal_conv = nn.Conv2d(in_channels=n_channels, out_channels=16, kernel_size=(1, 32), padding=(0, 16))
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.activation1 = nn.ELU()
        
        # Couche de convolution spatiale
        self.spatial_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(n_channels, 1), padding=(1, 0), groups=16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.activation2 = nn.ELU()
        
        # Couche de pooling
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2))  # Réduire le kernel_size
        
        # Couche de convolution temporelle profonde
        self.temporal_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 16), padding=(0, 12))
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.activation3 = nn.ELU()
        
        # Couche de pooling
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))  # Réduire le kernel_size
        
        # Couche fully connected
        self.fc1 = nn.Linear(81600, 128)  # Ajustez la taille en fonction de la sortie de la dernière couche de pooling
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # print("Input shape:", x.shape)  # Afficher la forme de l'entrée
        
        # Couche de convolution temporelle
        x = self.temporal_conv(x)
        # print("After temporal_conv:", x.shape)
        
        x = self.batch_norm1(x)
        x = self.activation1(x)
        
        # Couche de convolution spatiale
        x = self.spatial_conv(x)
        # print("After spatial_conv:", x.shape)
        
        x = self.batch_norm2(x)
        x = self.activation2(x)
        
        # Couche de pooling
        x = self.pool1(x)
        # print("After pool1:", x.shape)
        
        # Couche de convolution temporelle profonde
        x = self.temporal_conv2(x)
        # print("After temporal_conv2:", x.shape)
        
        x = self.batch_norm3(x)
        x = self.activation3(x)
        
        # Couche de pooling
        x = self.pool2(x)
        # print("After pool2:", x.shape)
        
        # Aplatir la sortie pour la couche fully connected
        x = x.view(x.size(0), -1)
        # print("After flattening:", x.shape)
        
        # Couche fully connected
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
