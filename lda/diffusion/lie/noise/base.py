import numpy as np

class BaseVENoiseSchedule():
    def __init__(self, timesteps: int = 500):
        self.timesteps = timesteps

        alphas = self.create_alphas(timesteps + 1)
        betas = np.diff(alphas, prepend=0)
        alphas_prev = np.append(0.0, alphas[:-1])

        self.betas = np.array(betas, dtype=np.float32)
        self.alphas = np.array(alphas, dtype=np.float32)
        self.alphas_prev = np.array(alphas_prev, dtype=np.float32)

        self.sqrt_betas = np.array(np.sqrt(betas), dtype=np.float32)
        self.sqrt_alphas = np.array(np.sqrt(alphas), dtype=np.float32)
        self.sqrt_alphas_prev = np.array(np.sqrt(alphas_prev), dtype=np.float32)

        self.coef1 = self.sqrt_betas / self.sqrt_alphas * self.sqrt_betas
        self.coef2 = self.sqrt_alphas_prev / self.sqrt_alphas * self.sqrt_betas

    def create_alphas(self, timesteps):
        return np.linspace(0.01, 1.0, timesteps)