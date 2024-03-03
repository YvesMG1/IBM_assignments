import torch.nn as nn
import torch.nn.functional as F
import torch

class Unet_DW_baseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Unet_DW_baseline, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Upsampling layers
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Final layer to get the required channels
        self.final_conv = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)  # output_dim: 10 channels for 10 classes
        
    def forward(self, x):
        # Downsampling
        x1 = F.relu(self.conv1(x))
        x2 = F.max_pool2d(x1, 2)
        
        x2 = F.relu(self.conv2(x2))
        x3 = F.max_pool2d(x2, 2)
        
        x3 = F.relu(self.conv3(x3))
        
        # Upsampling
        x = F.relu(self.upconv2(x3))
        x = x + x2  # Add the feature map from downsampling
        
        x = F.relu(self.upconv3(x))
        x = x + x1  # Add the feature map from downsampling
        
        # Final layer
        x = self.final_conv(x)
        
        return x
    
class Unet_DW(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Unet_DW, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Final layer to get the required channels
        self.final_conv = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)  # output_dim: 10 channels for 10 classes
        
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
    

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Unet_DW_attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Unet_DW_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Attention blocks
        self.attention_block1 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.attention_block2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.attention_block3 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        
        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Final layer to get the required channels
        self.final_conv = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)  # output_dim: 10 channels for 10 classes
        
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
        g1 = self.upconv1(x4)
        x3 = self.attention_block1(g1, x3)
        x = F.relu(g1 + x3)
        
        g2 = self.upconv2(x)
        x2 = self.attention_block2(g2, x2)
        x = F.relu(g2 + x2)
        
        g3 = self.upconv3(x)
        x1 = self.attention_block3(g3, x1)
        x = F.relu(g3 + x1)
        
        # Final layer
        x = self.final_conv(x)
        return x
    
class ConvReductionBlock_simple(nn.Module):
    def __init__(self, input_channels):
        super(ConvReductionBlock_simple, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)  # Output: 32 x 128 x 128
        self.pool = nn.MaxPool2d(8, stride=8)  # Output: 32 x 16 x 16

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        return x
    

class FullyConnectedFusion(nn.Module):
    def __init__(self, input_channels, output_size, hidden_size=256, num_fc_layers=3, time_steps=3):
        super(FullyConnectedFusion, self).__init__()
        self.reduction_block = ConvReductionBlock_simple(input_channels)
        self.output_size = output_size
        self.fc_layers = nn.ModuleList()
        # Input layer
        flattened_size = 32 * 16 * 16 * time_steps  # Adjusted for 3 time steps
        self.fc_layers.append(nn.Linear(flattened_size, hidden_size))
        # Hidden layers
        for _ in range(1, num_fc_layers - 1):  # Adjust for the output layer
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        # Output layer
        self.fc_layers.append(nn.Linear(hidden_size, output_size * 256 * 256))

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        x_reduced = []
        for t in range(time_steps):
            # Process each time step through the reduction block
            x_t = self.reduction_block(x[:, t])
            x_reduced.append(x_t.view(batch_size, -1))
        # Concatenate along the feature dimension
        x = torch.cat(x_reduced, dim=1)
        # Pass through fully connected layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:  # Apply ReLU non-linearity on all but last layer
                x = F.relu(x)
        # Reshape to the output dimensions
        out = x.view(batch_size, self.output_size, 256, 256)
        return out