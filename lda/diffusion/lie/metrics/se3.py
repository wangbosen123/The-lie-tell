import torch
import theseus as th

from torch import Tensor
from theseus.geometry import SE3

def is_tan(t):
    if isinstance(t, Tensor) and t.shape[-1:] == (6, ):
        return True
    else:
        return False

def is_mat(t):
    if isinstance(t, Tensor) and t.shape[-2:] == (4, 4):
        return True
    else:  
        return False

def is_x_y_z_quat(t):
    if isinstance(t, Tensor) and t.shape[-1:] == (7, ):
        return True
    else:
        return False
    
def as_lie(t) -> SE3:
    if isinstance(t, SE3):
        return t
    if is_tan(t):
        # from tangent vector
        return SE3.exp_map(t)
    if is_mat(t):
        # from 4x4 matrix
        return SE3(tensor = t[:, :3])
    if is_x_y_z_quat(t):
        # TODO: Notice the wxyz format
        return SE3.x_y_z_unit_quaternion_to_SE3(t)
    raise ValueError(t.shape)

def as_tan(t) -> Tensor:
    if is_mat(t):
        return as_lie(t).log_map()
    if isinstance(t, SE3):
        return t.log_map()
    if is_tan(t):
        return t
    raise ValueError(t.shape)

def as_mat(t) -> Tensor:
    if isinstance(t, SE3):
        return t.to_matrix()
    if is_mat(t):
        return t
    if is_tan(t):
        return SE3.exp_map(t).to_matrix()
    raise ValueError(t.shape)

def as_quat(t) -> Tensor:
    if isinstance(t, SE3):
        return t.to_x_y_z_quaternion()
    if is_x_y_z_quat(t):
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
        return 6
    
def chordal_distance(x1, x2):
    m1 = as_mat(x1)
    m2 = as_mat(x2)
    m = (m1 - m2) ** 2
    m = m.reshape((*m.shape[:-2], m.shape[-2] * m.shape[-1]))
    return m.sum(axis=-1)

def distance_fn(x1, x2, type="chordal"):
    if type == "chordal":
        return chordal_distance(x1, x2)
    raise ValueError(type)

def main():
    matrix = SE3.rand(1)
    matrix2 = SE3.rand(2)
    print(matrix)
    vector = as_tan(matrix)
    mat = as_mat(vector)
    print(vector.shape, mat.shape)
    print(vector, mat)
    m = as_mat(matrix)
    m = as_lie(m)
    print(m)
    x = chordal_distance(matrix, matrix2)
    print("distance=", x)
    q = as_quat(matrix)
    print("q=", q)
    m2 = as_lie(q)
    print(m2)


if __name__ == "__main__":
    main()