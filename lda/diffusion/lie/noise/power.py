import numpy as np
import torch
from .base import BaseVENoiseSchedule


class PowerNoiseSchedule(BaseVENoiseSchedule):
    def __init__(
        self,
        alpha_start: float = 1e-8,
        alpha_end: float = 1.0,
        timesteps: int = 500,
        power: float = 3.0,
    ):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.power = power
        super().__init__(timesteps)
        
    def create_alphas(self, timesteps):
        return (
            np.linspace(
                self.alpha_start ** (1 / self.power),
                self.alpha_end ** (1 / self.power),
                timesteps,
            )
            ** self.power
        )
    
    def get_gradient(self, timesteps):
        time_arr = torch.linspace(0, timesteps, timesteps + 1, requires_grad=True)[:timesteps]
        alpha_start = torch.tensor(self.alpha_start, requires_grad=True)
        alpha_end = torch.tensor(self.alpha_end, requires_grad=True)
        base = (alpha_start ** (1/self.power) - alpha_end ** (1/self.power)) / (timesteps-1)

        alphas = (alpha_end ** (1/self.power) + time_arr * base) ** self.power
        gradients = -torch.autograd.grad(outputs=alphas.sum(), inputs=time_arr, create_graph=True)[0]
        return gradients