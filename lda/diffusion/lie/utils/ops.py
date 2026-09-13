import torch
import theseus as th
from torch import Tensor
from theseus.geometry import SO3, SE3


def add(y, x):
    if isinstance(x, (SO3, SE3)):
        # return y @ x  # Matrix multiplication if x is an SO3 object
        return y.compose(x)
    else:
        return y + x  # Element-wise addition for non-SO3 types
    
def lsub(y, x):
    if isinstance(x, (SO3, SE3)):
        return y.compose(x.inverse())
    else:
        return y - x

def rsub(y, x):
    if isinstance(x, (SO3, SE3)):
        return (y.inverse()).compose(x)
    else:
        return - y + x

def main():
    matrix = SE3.rand(1)  # Generates a SO3 instance of shape (1, 3, 3)
    matrix2 = SE3.rand(1)  # Generates a SO3 instance of shape (1, 3, 3)
    print(matrix)
    print(matrix2)
    result1 = add(matrix, matrix2)
    result2 = lsub(matrix, matrix2)
    result3 = rsub(matrix, matrix2) 
    print(result1)
    print(result2)
    print(result3)
    
if __name__ == "__main__":
	main()