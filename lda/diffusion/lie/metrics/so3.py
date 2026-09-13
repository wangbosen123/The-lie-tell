import torch
import theseus as th

from torch import Tensor
from theseus.geometry import SO3

def is_tan(t):
    if isinstance(t, Tensor) and t.shape[-1:] == (3, ) and not is_mat(t):
        return True
    else:
        return False
  
def is_mat(t):

    if t.shape == (3, 3):
        identity = torch.eye(3, dtype=t.dtype, device=t.device)
        is_orthogonal = torch.allclose(t.T @ t, identity, atol=1e-6)
        if is_orthogonal:
            return True
        else:
            return False

    if isinstance(t, Tensor) and t.shape[-2:] == (3, 3):
        return True
    else:
        return False
    
def is_quat(t):
    if isinstance(t, Tensor) and t.shape[-2:] == (1, 4):
        return True
    else:
        return False

def as_lie(t) -> SO3:
    if isinstance(t, SO3):
        return t
    if is_tan(t):
        # from tangent vector
        return SO3.exp_map(t)
    if is_mat(t):
        # from 3x3 matrix
        return SO3(tensor = t)
    if is_quat(t):
        return SO3.unit_quaternion_to_SO3(t)
    raise ValueError(t.shape)

def as_tan(t)-> Tensor:
    if is_mat(t):
        return as_lie(t).log_map()
    if isinstance(t, SO3):
        return t.log_map()
    if is_tan(t):
        return t
    raise ValueError(t.shape)

def as_mat(t)-> Tensor:
    if isinstance(t, SO3):
        return t.to_matrix()
    if is_mat(t):
        return t
    if is_tan(t):
        return SO3.exp_map(t).to_matrix()
    raise ValueError(t.shape)

def as_quat(t)-> Tensor:
    if isinstance(t, SO3):
        return t.to_quaternion()
    if is_quat(t):
        return t
    raise ValueError(t.shape)

def as_repr(t, repr: str):
    if repr == "lie":
        return as_lie(t)
    elif repr == "tan":
        return as_tan(t)
    elif repr == "mat":
        return as_mat(t)
    raise ValueError(repr)

def get_repr_size(repr: str):
    if repr == "tan":
        return 3
    
def get_euler_angle(t):
    t = as_mat(t)
    z_angle = torch.atan2(t[:,0,0], t[:,1,0])
    return z_angle

def chordal_distance(x1, x2):
    m1 = as_mat(x1)
    m2 = as_mat(x2)
    m = (m1 - m2) ** 2
    m = m.reshape((*m.shape[:-2], m.shape[-2] * m.shape[-1]))
    return m.sum(axis=-1)


def distance_fn(x1, x2, type="chordal"):
    if type == "chordal":
        return chordal_distance(x1, x2)


def main():
    matrix = SO3.rand(1)
    matrix2 = SO3.rand(2)
    print(matrix)
    vector = as_tan(matrix)
    # vector = vector.unsqueeze(dim=0)
    mat = as_mat(vector)
    print(vector.shape, mat.shape)
    # print(vector)
    m = as_mat(matrix)
    m = as_lie(m)
    # print(m)
    x = chordal_distance(matrix, matrix2)
    print(x)
    q = as_quat(matrix)
    print(q)
    m2 = as_lie(q)
    print(m2)

    matrix = torch.zeros((1, 3, 3))
    tan = as_lie(matrix)
    print(tan)

if __name__ == "__main__":
    main()