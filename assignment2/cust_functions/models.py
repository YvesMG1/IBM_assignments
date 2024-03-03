import torch.nn as nn
import torch.nn.functional as F

class BaselineModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaselineModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Final layer to get the required channels
        self.final_conv = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)  # 2 channels for 2 classes
        
    def forward(self, x):
        # Downsampling
        x1 = F.relu(self.conv1(x))
        x2 = F.max_pool2d(x1, 2)
        
        x2 = F.relu(self.conv2(x2))
        x3 = F.max_pool2d(x2, 2)
        
        x3 = F.relu(self.conv3(x3))
        x4 = F.max_pool2d(x3, 2)
        
        x4 = F.relu(self.conv4(x4))
        
        # Upsampling
        x = F.relu(self.upconv1(x4))
        x = x + x3  # Add the feature map from downsampling
        
        x = F.relu(self.upconv2(x))
        x = x + x2  # Add the feature map from downsampling
        
        x = F.relu(self.upconv3(x))
        x = x + x1  # Add the feature map from downsampling
        
        # Final layer
        x = self.final_conv(x)
        
        return x
    


class AdvancedCNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(AdvancedCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up1 = nn.BatchNorm2d(256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up2 = nn.BatchNorm2d(128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up4 = nn.BatchNorm2d(32)
        
        # Final layer to get the required channels
        self.final_conv = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Downsampling
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.max_pool2d(x1, 2)
        
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = F.max_pool2d(x2, 2)
        
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = F.max_pool2d(x3, 2)
        
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = F.max_pool2d(x4, 2)
        
        x5 = F.relu(self.bn5(self.conv5(x5)))
        
        # Upsampling
        x = self.dropout(F.relu(self.bn_up1(self.upconv1(x5))))
        x = x + x4  # Shortcut connection
        
        x = self.dropout(F.relu(self.bn_up2(self.upconv2(x))))
        x = x + x3  # Shortcut connection
        
        x = self.dropout(F.relu(self.bn_up3(self.upconv3(x))))
        x = x + x2  # Shortcut connection
        
        x = self.dropout(F.relu(self.bn_up4(self.upconv4(x))))
        x = x + x1  # Shortcut connection
        
        # Final layer
        x = self.final_conv(x)
        
        return x
