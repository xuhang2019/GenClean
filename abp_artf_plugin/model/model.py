import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
        cmode: control mode
        0b00: vanilla
        0b01: residual connections
    """
    def __init__(self, input_shape=1200, latent_dim=20, norm_non_stationary=True, cmode=0b00):
        super().__init__()
        self.encoder = VAEEncoder(input_shape=input_shape, latent_dim=latent_dim)
        self.decoder = VAEDecoder(input_shape=input_shape, latent_dim=latent_dim)
        self.norm_non_stationary = norm_non_stationary
        self.cmode = cmode
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        if self.norm_non_stationary:
            # 32,1,1200
            # B, D, L
            means = x.mean(2, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            
        z_mean, z_log_var, encoder_outputs = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decoder(z, encoder_outputs) if self.cmode & 0b01 else self.decoder(z) # add residual
         
        if self.norm_non_stationary:
            x_hat = x_hat * stdev + means
        
        return x_hat, z_mean, z_log_var
    
class VAEEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VAEEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7, stride=1)
        self.mp1 = nn.MaxPool1d(5, padding=2) 
        self.conv2 = nn.Conv1d(8, 16, kernel_size=15, padding=7, stride=1)
        self.mp2 = nn.MaxPool1d(5, padding=2)
        self.dropout1 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=15, padding=7, stride=1)
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout(0.1)
        # Here, flatten -> channels * current size, currtent size = input_shape//25
        self.fc = nn.Linear(16 * (input_shape // 25), 16)
        self.fc_mean = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)

    def forward(self, x):
        # x = torch.unsqueeze(x, 1)  # Add channel dimension
        x1 = self.conv1(x) # 8, 1200
        x = self.mp1(F.relu(x1)) 
        x2 = self.conv2(x)  # 16, 240
        x = self.mp2(F.relu(x2))
        x = self.dropout1(x)
        x3 = self.conv3(x) # 16, 48
        x = F.relu(x3)
        x = self.flatten(x)
        x = self.dropout2(x)
        x = F.relu(self.fc(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var, [x1, x2, x3]
    
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, input_shape=1200):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 16)
        self.fc2 = nn.Linear(16, 16 * (input_shape // 25)) 
        self.dropout1 = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(16, 16, kernel_size=15, padding=7, stride=1)
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        # self.crop1 = nn.Identity()  # Replace with Cropping1D when available
        self.dropout2 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=15, padding=7, stride=1)
        self.upsample2 = nn.Upsample(scale_factor=5, mode='nearest')
        # self.crop2 = nn.Identity()  # Replace with Cropping1D when available
        self.conv3 = nn.Conv1d(8, 1, kernel_size=15, padding=7, stride=1)

    def forward(self, z, encoder_outputs=None):
        if encoder_outputs is not None:
            assert len(encoder_outputs) == 3, "Encoder outputs must have 3 elements"
        
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = x.view(-1,16,48)
        x = F.relu(self.conv1(x)) if encoder_outputs is None else F.relu(self.conv1(x) + encoder_outputs[2])
        x = self.upsample1(x) if encoder_outputs is None else self.upsample1(x) + encoder_outputs[1]
        # x = self.crop1(x)
        x = self.dropout2(x) 
        x = F.relu(self.conv2(x)) 
        x = self.upsample2(x) if encoder_outputs is None else self.upsample2(x) + encoder_outputs[0]
        # x = self.crop2(x)
        x = self.conv3(x) 
        return x
    
def load_ckpt(model, ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(ckpt['state_dict'])
    except Exception as e:
        encoder_state_dict = {}
        decoder_state_dict = {}
        
        prefix_mapping = { # startwith: (target_dict, replacement)
            'model.0.': (encoder_state_dict, ''),
            'model.encoder.': (encoder_state_dict, ''),
            'model.1.': (decoder_state_dict, ''),
            'model.decoder.': (decoder_state_dict, '')
        }

        for key, value in ckpt['state_dict'].items():
            for prefix, (target_dict, replacement) in prefix_mapping.items():
                if key.startswith(prefix):
                    new_key = key.replace(prefix, replacement)
                    target_dict[new_key] = value

        # Load the state dicts into the model's encoder and decoder
        model.encoder.load_state_dict(encoder_state_dict)
        model.decoder.load_state_dict(decoder_state_dict)
    return model