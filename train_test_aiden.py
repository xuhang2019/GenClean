import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from models.model_zoo import VAELoss
import wandb
from typing import List


class Logger:
    def log(*args, **kwargs):
        pass

class WandbLogger(Logger):
    def __init__(self, project, config, name):
        
        self.wandb = wandb.init(project=project, config=config, name=name)
        
    def log(self, *args, **kwargs):
        """
            usage:
                log({'train/loss':0.5}, step = idx)
            ref: https://docs.wandb.ai/ref/python/log/
        """
        self.wandb.log(*args, **kwargs)

def train(model, device, X, Y, V_X=None, V_Y=None, lr=0.001, batch_size=256, num_epoch=16, use_tqdm=True, logger=Logger(), special_case=None, epoch_per_validate=1,config=dict()):
    model = model.to(device)   
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    criterion = VAELoss()
    use_validation = V_X is not None and V_Y is not None
    if use_validation:
        V_X = torch.tensor(V_X).to(device).float()
        V_Y = torch.tensor(V_Y).to(device).float()
    
    if use_tqdm:
        pbar = tqdm(range(0, num_epoch))
    else:
        pbar = range(0, num_epoch)
        
    for epoch in pbar:
        for idx, batch_idx in enumerate(range(0, len(X), batch_size)):
            num_step = (len(X) // batch_size)*epoch + idx
            data = X[batch_idx:batch_idx+batch_size]
            target = Y[batch_idx:batch_idx+batch_size]

            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
            
            optimizer.zero_grad()
            output = model(data)
            # special case for VAE
            if special_case == "VAE":
                x_hat, z1, z2 = model(data)
                loss = criterion(x_hat, target, z1, z2, return_mode='two')
                recon_loss, kl_loss = loss
                loss = loss[0] + loss[1]
                logger.log({"train/recon_loss": recon_loss.item(), 
                            "train/kl_loss": kl_loss.item(), 
                            "train/total_loss": loss.item(),
                            "epoch": epoch}, step=num_step)
                
            else:
                loss = criterion(output, target)
                
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
            if use_tqdm:
                pbar.set_description(f"loss: {loss.item():.5f}")   
            
        if (epoch+1) % epoch_per_validate == 0:
            # validation
            if use_validation:
                with torch.no_grad():
                    loss_sum = 0
                    for batch_idx in range(0, len(X), batch_size):
                        data = V_X[batch_idx:batch_idx+batch_size]
                        target = V_Y[batch_idx:batch_idx+batch_size]
                        output = model(data)
                        
                        if special_case == "VAE":
                            # only record the first one image
                            x_hat, z1, z2 = model(data)
                            ori = np.squeeze(data[0].cpu().numpy())
                            recon = np.squeeze(x_hat[0].cpu().numpy())
                            fig = val_img(ori, recon, epoch)
                            logger.log({"val/img": wandb.Image(fig)}, step=epoch)
                            break
                            
                            # loss = criterion(x_hat, target, z1, z2, return_mode='two')
                            # loss = loss[0] + loss[1]
                            # logger.log({"val/validation_loss": loss.item()}, step=epoch)
                        else:
                            loss = criterion(output, target)
                            
                        # loss_sum += loss.cpu().numpy()
                    # logger.log({"validation_loss": loss_sum.item()}, step=epoch)

    return model

def posttrain(model, file_path='model.pth'):
    # raise NotImplementedError
    torch.save(model.state_dict(), file_path)

def evaluate(model, device, X, special_case='VAE') -> List:
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = []
        for i in range(0, len(X)):
            data = X[i]
            data = torch.tensor(data).unsqueeze(1).to(device).float()
            
            if special_case == 'VAE':
                output, z1, z2 = model(data)
                output = output.cpu().numpy()
                z1 = z1.cpu().numpy()
                z2 = z2.cpu().numpy()
                outputs.append((output, z1, z2))
            else:
                output = model(data).cpu().numpy()
                outputs.append(output)

    # outputs = np.concatenate(outputs, axis=0)
    return outputs

def postevaluate(outputs):
    raise NotImplementedError

def val_img(ori, recon, epoch):
    import matplotlib.pyplot as plt
    plt.close('all')
    fig, ax = plt.subplots(2,1)
    ax[0].plot(ori, color='blue', label='Original')
    ax[0].plot(recon, color='red', label='Reconstructed')
    ax[1].plot(recon-ori, color='green', label='Difference')
    mse = np.mean((ori - recon)**2)
    mae = np.mean(np.abs(ori - recon))
    ax[0].legend()
    ax[0].set_title(f'EPOCH: {epoch}, MSE: {mse:.2f}, MAE: {mae:.2f}')
    plt.tight_layout()
    return fig
