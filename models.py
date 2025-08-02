import os
from typing import Any, Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import Config, Logger
from tqdm import trange, tqdm

from datasets import normalize_tensor

ENCODER_DECODER= ['VAE', 'FCN-VAE']
END_TO_END = ['VAE_raw', 'LSTM', 'Transformer']


def model_provider(model_name, **kwargs):
    if model_name == 'VAE':
        return GencleanModel(latent_dim=10, input_shape=1200)
    else:
        raise ValueError(f"Model {model_name} not found")
    
def norm_layer(ch, kind='bn'):
    return nn.BatchNorm1d(ch) if kind=='bn' else nn.GroupNorm(8, ch)

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, norm_kind):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, k, stride=s, padding=k//2)
        self.bn   = norm_layer(out_c, norm_kind)
    def forward(self,x): return F.relu(self.bn(self.conv(x)))

class UpBlockTranspose(nn.Module):
    def __init__(self,in_c,out_c,k,s,norm_kind):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_c,out_c,k,stride=s,padding=k//2,output_padding=s-1)
        self.bn     = norm_layer(out_c, norm_kind)
    def forward(self,x): return F.relu(self.bn(self.deconv(x)))

class PixelShuffle1D(nn.Module):          # r = upscale factor
    def __init__(self,r): super().__init__(); self.r=r
    def forward(self,x):
        b,c,l = x.shape; assert c%self.r==0
        x = x.view(b, c//self.r, self.r, l)          # (B,C/r,r,L)
        return x.permute(0,1,3,2).reshape(b, c//self.r, l*self.r)

class UpBlockSubPixel(nn.Module):
    def __init__(self,in_c,out_c,r,norm_kind):
        super().__init__()
        self.expand = nn.Conv1d(in_c, out_c*r, 1)
        self.ps     = PixelShuffle1D(r)
        self.bn     = norm_layer(out_c, norm_kind)
    def forward(self,x): return F.relu(self.bn(self.ps(self.expand(x))))
    
class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1, norm='bn'):
        super().__init__()
        norm_layer = nn.BatchNorm1d if norm=='bn' else lambda c: nn.GroupNorm(8, c)
        self.net = nn.Sequential(
            norm_layer(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, kernel, padding=dilation*(kernel-1)//2,
                       dilation=dilation),
            norm_layer(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 1)
        )
    def forward(self, x): return x + self.net(x)
    

class Encoder1DLarge(nn.Module):
    def __init__(self, latent_dim, norm_kind='bn'):
        super().__init__()
        cfg = [(1,32,7,2),(32,64,7,2),(64,128,5,2),(128,256,5,2)]
        self.blocks = nn.Sequential(*[DownBlock(*c,norm_kind) for c in cfg])
        self.fc_mu  = nn.Linear(256*75, latent_dim)
        self.fc_lv  = nn.Linear(256*75, latent_dim)
    def forward(self,x):
        x = self.blocks(x).flatten(1)
        return self.fc_mu(x), self.fc_lv(x)

class Decoder1DLarge(nn.Module):
    def __init__(self, latent_dim, up_mode='deconv', norm_kind='bn', oact='tanh'):
        super().__init__()
        self.fc     = nn.Linear(latent_dim, 256*75)
        self.oact = oact
        up = UpBlockTranspose if up_mode=='deconv' else UpBlockSubPixel
        
        # cfg = [(256,128,5,2),(128,64,5,2),(64,32,5,2),(32,1,5,2)]
        cfg = [(256,128,5,2),(128,64,5,2),(64,32,5,2),(32,16,5,2),(16,1,3,1)]
        self.blocks = nn.Sequential(*[
            up(*c, norm_kind) if i<len(cfg)-1 else nn.Conv1d(c[0],1, c[2],padding=c[2]//2)
            for i,c in enumerate(cfg)])
    def forward(self,z):
        x = self.fc(z).view(z.size(0),256,75)
        if self.oact == 'tanh':
            return torch.tanh(self.blocks(x))
        elif self.oact == 'mlp':
            return self.blocks(x)
        else:
            raise ValueError(f"Invalid output activation: {self.oact}")

class Encoder1DSmall(nn.Module):
    """
    down_cfg: [(out_channels, kernel, stride), ...]
    1200 -> 600 -> 300 -> 150  (3× stride-2)
    """
    def __init__(self, latent_dim=10, norm='bn'):
        super().__init__()
        down_cfg = [(8, 7, 2), (16, 7, 2), (16, 5, 2)]
        layers, in_c = [], 1
        for out_c, k, s in down_cfg:
            layers += [
                nn.Conv1d(in_c, out_c, k, stride=s, padding=k//2, bias=False),
                norm_layer(out_c, norm), nn.ReLU(inplace=True)
            ]
            in_c = out_c
        self.backbone = nn.Sequential(*layers)            # L=150, C=16
        self.gap      = nn.AdaptiveAvgPool1d(1)           # → (B,16,1)
        self.fc_mu    = nn.Linear(16, latent_dim)
        self.fc_lv    = nn.Linear(16, latent_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).squeeze(-1)      # (B,16)
        return self.fc_mu(x), self.fc_lv(x)
    
class Decoder1DSmall(nn.Module):
    """
    latent → 16×150 → PixelShuffle ×3 → length 1200
    up_cfg: [(out_ch, r), ...]  # r = upscale factor (2)
    """
    def __init__(self, latent_dim=10, norm='bn'):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 16 * 150)       # 15 k 参数
        up_cfg = [(16, 2), (8, 2), (4, 2)]                # 150→300→600→1200
        layers = []
        in_c = 16
        for out_c, r in up_cfg:
            # 1×1 conv 先扩张到 out_c*r，再 PixelShuffle1D(r)
            layers += [
                nn.Conv1d(in_c, out_c * r, 1, bias=False),
                PixelShuffle1D(r),
                norm_layer(out_c, norm), nn.ReLU(inplace=True)
            ]
            in_c = out_c
        layers.append(nn.Conv1d(in_c, 1, 3, padding=1))   # 输出层
        self.up_path = nn.Sequential(*layers)

    def forward(self, z):
        x = self.proj(z).view(z.size(0), 16, 150)         # (B,16,150)
        return torch.tanh(self.up_path(x))                # (B,1,1200)



class VAEEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VAEEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7, stride=1)
        self.mp1 = nn.MaxPool1d(5, padding=2) # default stride value is kernel_size
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
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout2(x)
        x = F.relu(self.fc(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var
    
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

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = x.view(-1,16,48)
        x = F.relu(self.conv1(x))
        x = self.upsample1(x)
        # x = self.crop1(x)
        x = self.dropout2(x)
        x = F.relu(self.conv2(x))
        x = self.upsample2(x)
        # x = self.crop2(x)
        x = self.conv3(x)
        return x



# 新增GencleanModel类，封装所有模型相关逻辑
class GencleanModel(nn.Module):
    def __init__(self, input_shape=1200, latent_dim=10, model_str='raw', use_revin=False):
        super().__init__()
        self.revin = None
        self._init_coder(model_str, latent_dim, input_shape, use_revin)
    
    def _init_coder(self, model_str, latent_dim, input_shape, use_revin):
        if model_str == 'raw':
            self.encoder = VAEEncoder(input_shape=input_shape, latent_dim=latent_dim)
            self.decoder = VAEDecoder(input_shape=input_shape, latent_dim=latent_dim)
        elif model_str == 'small':
            self.encoder = Encoder1DSmall(latent_dim=latent_dim)
            self.decoder = Decoder1DSmall(latent_dim=latent_dim)
        elif model_str == 'large':
            self.encoder = Encoder1DLarge(latent_dim=latent_dim)
            self.decoder = Decoder1DLarge(latent_dim=latent_dim)
        elif model_str == 'large_mlp':
            self.encoder = Encoder1DLarge(latent_dim=latent_dim)
            self.decoder = Decoder1DLarge(latent_dim=latent_dim, oact='mlp')
            
        if use_revin:
            self.revin = RevIN(num_features=1, affine=True)
            print("RevIN initialized")
    
    def forward(self, x):
        if self.revin is not None:
            x = x.permute(0, 2, 1)
            x = self.revin(x, mode='norm')
            x = x.permute(0, 2, 1)
            
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decoder(z)
        
        if self.revin is not None:
            x_hat = x_hat.permute(0, 2, 1)
            x_hat = self.revin(x_hat, mode='denorm')
            x_hat = x_hat.permute(0, 2, 1)
            
        return x_hat, z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def reconstruction_loss(self, x_input, x_output):
        return F.mse_loss(x_input, x_output, reduction='sum')

    def kl_loss(self, z_mean, z_log_var):
        return 0.5 * torch.sum(torch.exp(z_log_var) + z_mean**2 - 1 - z_log_var)

# 重构Gencleaner为纯代理结构
class Gencleaner(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.ckpt_dir = config.ckpt_dir
        self.save_name = config.save_name
        self.model_name = config.model_name
        self.verbose = config.verbose
        
        self.model = GencleanModel(input_shape=config.input_shape, latent_dim=config.latent_dim, model_str=config.model_str, use_revin=config.use_revin).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.best_val_loss = float('inf')
    

    def training_step(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(self.device)
        if self.model_name in ENCODER_DECODER:
            x_input = x
            x_output, z_mean, z_log_var = self.model(x_input)
            loss_kl = self.model.kl_loss(z_mean, z_log_var)
            loss_recon = self.model.reconstruction_loss(x_input, x_output)
            loss = loss_kl + loss_recon
            return loss, {'recon_loss': loss_recon.item(), 'kl_loss': loss_kl.item(), 'total_loss': loss.item()}
        else:
            x_recon = self.model(x)
            loss = self.model.reconstruction_loss(x, x_recon)
            return loss, {'total_loss': loss.item()}

    def validation_step(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(self.device)
        if self.model_name in ENCODER_DECODER:
            x_input = x
            x_output, z_mean, z_log_var = self.model(x_input)
            loss_kl = self.model.kl_loss(z_mean, z_log_var)
            loss_recon = self.model.reconstruction_loss(x_input, x_output)
            loss = loss_kl + loss_recon
            return loss.item(), {'recon_loss': loss_recon.item(), 'kl_loss': loss_kl.item(), 'total_loss': loss.item()}
        else:
            x_recon = self.model(x)
            loss = self.model.reconstruction_loss(x, x_recon)
            return loss.item(), {'total_loss': loss.item()}

    def fit(self, X, X_val, logger: Logger):
        # create ckpt dir when training
        os.makedirs(self.ckpt_dir, exist_ok=True)
        batch_size = self.config.batch_size
        
        # TODO: did not use dataloader for the small dataset
        X = X.to(self.device)
        X_val = X_val.to(self.device)
        num_epochs = self.config.num_epoch
        num_train_steps = (len(X) + batch_size - 1) // batch_size
        num_val_steps = (len(X_val) + batch_size - 1) // batch_size
        
        for epoch in trange(num_epochs, desc='Epochs', position=0):
            self.train()
            train_losses = []
            step_iter = tqdm(enumerate(range(0, len(X), batch_size)), total=num_train_steps, desc=f'Train Epoch {epoch}', leave=False, position=1)
            for idx, batch_idx in step_iter:
                num_step = num_train_steps * epoch + idx
                data = X[batch_idx:batch_idx+batch_size]
                self.optimizer.zero_grad()
                loss, log_dict = self.training_step(data)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                step_iter.set_postfix({"loss": loss.item()})
                
                train_msg = {f"train/{k}": v for k, v in log_dict.items()}
                logger.log(wandb_msg=train_msg)
                
                if self.verbose:
                    logger.log(f"[Train] Epoch {epoch} Step {num_step} Loss: {loss.item():.4f}", **{f"train/{k}": v for k, v in log_dict.items()})
            avg_train_loss = sum(train_losses) / len(train_losses)

            self.eval()
            val_losses = []
            val_iter = tqdm(enumerate(range(0, len(X_val), batch_size)), total=num_val_steps, desc=f'Val Epoch {epoch}', leave=False, position=2)
            with torch.no_grad():
                for idx, batch_idx in val_iter:
                    data = X_val[batch_idx:batch_idx+batch_size]
                    val_loss, val_log_dict = self.validation_step(data)
                    val_losses.append(val_loss)
                    val_iter.set_postfix({"val_loss": val_loss})
                    val_msg = {f"val/{k}": v for k, v in val_log_dict.items()}
                    logger.log(wandb_msg=val_msg)
                    
                    if self.verbose:
                        logger.log(f"[Val] Epoch {epoch} Step {idx} Loss: {val_loss:.4f}", **{f"val/{k}": v for k, v in val_log_dict.items()})
            avg_val_loss = sum(val_losses) / len(val_losses)

            # Save best checkpoint
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_ckpt(epoch, best=True)
            # Save every epoch
            if epoch % 25 == 0:
                self.save_ckpt(epoch, best=False)
            if epoch == num_epochs - 1:
                self.save_ckpt(epoch, best=False)

    def save_ckpt(self, epoch, best=False):
        fname = f"{self.save_name}_best.pth" if best else f"{self.save_name}_epoch_{epoch}.pth"
        ckpt_path = os.path.join(self.ckpt_dir, fname)
        torch.save(self.model.state_dict(), ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    def inference(self, x):
        """
            input:
                - x: (N, L)
                
            output:
                - x_output: (N, L)
                - z_mean: (N, latent_dim)
                - z_log_var: (N, latent_dim)
        """
        
        
        # Convert numpy to tensor and move to device
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x.detach().unsqueeze(1)
        x = normalize_tensor(x, data_norm=self.config.data_norm)
        self.model.eval()
        with torch.no_grad():
            x_output, z_mean, z_log_var = self.model(x)
            x_output = x_output.squeeze(1)
        return x_output.cpu().numpy(), z_mean.cpu().numpy(), z_log_var.cpu().numpy()
    
    
class Encoder1DAudio(nn.Module):
    def __init__(self, in_ch=1, base=64, latent_dim=128, norm='bn'):
        super().__init__()
        self.blocks = nn.ModuleList([])
        ch = in_ch
        for stride, mult in zip([4,4,4,2],[1,2,4,8]):     # 256× 下采样
            self.blocks.append(nn.Conv1d(ch, base*mult, 4, stride, 1))
            ch = base*mult
            self.blocks.append(ResBlock1D(ch, norm=norm))
        self.to_latent = nn.Conv1d(ch, 2*latent_dim, 1)

    def forward(self,x):
        for blk in self.blocks: x = blk(x)
        mu, logvar = torch.chunk(self.to_latent(x).mean(-1), 2, dim=1)
        return mu, logvar                      # (B, latent_dim)

class Decoder1DAudio(nn.Module):
    def __init__(self, latent_dim=128, out_len=48000, up_mode='pxsh'):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * (out_len//256))
        self.up_mode = up_mode
        self.resblocks = nn.ModuleList([])
        ch = 512
        for r in [2,4,4,4]:                    # 256× 上采样
            if up_mode=='deconv':
                self.resblocks.append(nn.ConvTranspose1d(ch, ch//r, r, r//2))
            else:  # sub-pixel
                self.resblocks.append(nn.Sequential(
                    nn.Conv1d(ch, ch*r, 1), PixelShuffle1D(r)))
            ch//=r
            self.resblocks.append(ResBlock1D(ch))
        self.out = nn.Conv1d(ch, 1, 3, padding=1)

    def forward(self,z):
        x = self.fc(z).view(z.size(0),512,-1)
        for blk in self.resblocks: x = blk(x)
        return torch.tanh(self.out(x))




if __name__ == '__main__':
    import fire
    from utils import Config, Logger
    
    def main(run_name='test_0725', debug=True, device='cuda', use_revin=True):
    
        cfg = Config(run_name=run_name, debug=debug, device=device, use_revin=use_revin)
        logger = Logger(cfg)
        G = Gencleaner(cfg)
        
        mock_train_data = torch.randn(32, 1, 1200)
        mock_val_data = torch.randn(32, 1, 1200)
        G.fit(mock_train_data, mock_val_data, logger)
        
    fire.Fire(main)