import math
import torch
import numpy as np
import theseus as th
from theseus.geometry import SE3, SO3

from .so3 import NormalSO3_Flat

class NormalSE3():
    _min_scale = 1e-4

    def __init__(self, mean: SE3, scale: float):
        self.mean = mean
        self.scale = np.maximum(scale, self._min_scale)

        self._so3_dist = NormalSO3_Flat(self.mean.rotation(), self.scale)

    def _sample(self, n):
        dx_rot = self._so3_dist._sample(n)
        shape = n + (3,)
        dx_tran = torch.randn(shape) * self.scale
        dx = torch.cat([dx_tran, dx_rot], dim=-1)
        return dx

    @classmethod
    def _sample_unit(cls, n) -> np.array:
        return cls(SE3(tensor=torch.eye(3, 4).unsqueeze(0)), 1.0)._sample(n)

def main():
    print(NormalSE3._sample_unit(n=(1,)))

if __name__ == "__main__":
    main()