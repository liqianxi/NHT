from pytorch_lightning.callbacks import Callback
import torch

class LipschitzRegularizer(Callback):

    def __init__(self, lipschitz_coeff, p=2):
        self.lipschitz_coeff = lipschitz_coeff
        self.p = p
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.h.named_parameters():
            if 'weight' in name:
                with torch.no_grad():
                    rescale_factor = torch.norm(param.data, p=self.p) / self.lipschitz_coeff 
                    param.data = param.data /  max(1, rescale_factor)