import math
import torch
import numpy as np
import theseus as th
from theseus.geometry import SO3

class NormalSO3_Flat():
    """NormalSO3_Flat: Concentrated Gaussian SO3"""

    _min_scale = 1e-4

    def __init__(self, mean: SO3, scale: float):
        # assert isinstance(mean, SO3)
        self.mean = mean
        self.scale = np.maximum(scale, self._min_scale)

    # def prob(self, x: SO3) -> np.array:
    #     logp = self.log_prob(x)
    #     return math.exp(logp)

    # def log_prob(self, x: SO3) -> np.array:
    #     var = self.scale**2
    #     log_scale = math.log(self.scale)
    #     z = math.log(math.sqrt(2 * math.pi))
    #     dff = (self.mean.inverse() @ x).log()
    #     return -((dff**2) / (2 * var) - log_scale - z).sum()


    def _sample(self, n):
        shape = n + (3,)
        tan = torch.randn(shape)
        tan = tan * self.scale
        # tan = SO3.exp_map(tan).log_map()
        return tan

    # def sample(self, n) -> SO3:
    #     """Sample n rotations"""
    #     # n = tuple(n) if hasattr(n, "__iter__") else (n,)
    #     shape = n + (3,)
        
    #     tan = torch.randn(shape)
    #     tan = tan * self.scale
    #     so3 = SO3.exp_map(tan) #.log_map()

    #     # shape = tan.shape[:-1]
    #     # tan = tan.reshape(-1, 3)
    #     # quat = torch.stack([self.mean @ SO3.exp(t).wxyz for t in tan])

    #     # so3 = SO3(quat.reshape(shape + (4,)))
    #     return so3
    
    @classmethod
    def _sample_unit(cls, n) -> np.array:
        return cls(SO3(tensor=torch.eye(3,3).unsqueeze(0)), 1.0)._sample(n)
    
def main():
    print(NormalSO3_Flat._sample_unit(n=(1,)))

if __name__ == "__main__":
    main()