import itertools
from typing import Iterator
import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from typing import Callable, Iterable, Optional, Tuple
import torch
import numpy as np
from itertools import product
from gpytorch.means import ZeroMean
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
import numpy as np
from scipy.linalg import block_diag
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, Kernel
#from botorch.models import SingleTaskGP
#from botorch.fit import fit_gpytorch_mll

class GroupInvariantKernel(Kernel):
    def __init__(self, base_kernel="RBF", length_scale=0.1, smoothness=1.5, group=None):
        self.base_kernel = base_kernel
        self.length_scale = length_scale
        self.smoothness = smoothness
        self.group = group if group is not None else [lambda x: x]  # Default to identity transformation if no group given
        # print('self.group',self.group)

        # Define the base kernel as Matern or RBF
        if self.base_kernel == "Matern":
            self.kernel = Matern(length_scale=length_scale, nu=smoothness)
        elif self.base_kernel == "RBF":
            self.kernel = RBF(length_scale=length_scale)

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.group_invariant_kernel(X, Y,eval_gradient=eval_gradient)

    def group_invariant_kernel(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        # print('X',X)
        Y = np.atleast_2d(Y)
        # Ensure the dimensions are as expected
       
        # print('Y',Y)
        # Calculate the kernel averaged over the group transformations
        K = np.zeros((X.shape[0], Y.shape[0]))

        # # Loop over all transformations in the group
        for g in self.group:
            #print('g',g)
            X_transformed = np.array([g(x) for x in X])
            # print('X_transformed',X_transformed.shape)
            # print('X_transformed values',X_transformed)
            # print('Y',Y.shape)
             # Check if the kernel is producing a valid square matrix for each transformation
            transformed_kernel = self.kernel(X_transformed, Y)
            # print('Transformed kernel shape:', transformed_kernel.shape)
            
            K += transformed_kernel / len(self.group)
            #K += np.eye(X.shape[0]) * 1e-6

            #K += self.kernel(X_transformed, Y) / len(self.group)
        #print('k',K.shape)
        # Add jitter only if K is square
        if K.shape[0] == K.shape[1]:
            #print('yesss jitter')
            K += np.eye(K.shape[0]) * 1e-6  # Small jitter for stability

        
        #K += np.eye(X.shape[0]) * 1e-6
        #K = (K + K.T) / 2


        return K

    def diag(self, X):
        return np.diag(self.__call__(X))
      

    def is_stationary(self):
        return True
        #return self.kernel.is_stationary()

def construct_np_rotation_group(dim_space: int):
    assert dim_space % 2 == 0

    rotation_matrix = np.array([[0, 1], [-1, 0]])  # 90 degrees
    blocks = [rotation_matrix] * (dim_space // 2)
    block_diag_matrix = block_diag(*blocks)

    rot_group = [
        np.eye(dim_space),
        block_diag_matrix,
        np.linalg.matrix_power(block_diag_matrix, 2),
        np.linalg.matrix_power(block_diag_matrix, 3),
    ]
    # print('rot_group'),
    # print(rot_group)
    return rot_group


def construct_np_reflection_group(dim_space: int):
    assert dim_space % 2 == 0

    reflection_x = np.array([[1, 0], [0, -1]])
    reflection_y = np.array([[-1, 0], [0, 1]])

    blocks_x = [reflection_x] * (dim_space // 2)
    blocks_y = [reflection_y] * (dim_space // 2)
    # print('reflection')
    # print([block_diag(*blocks_x), block_diag(*blocks_y)])

    return [block_diag(*blocks_x), block_diag(*blocks_y)]

def get_np_group_functions(dim_space: int=14):
    matrices = construct_np_rotation_group(dim_space) + construct_np_reflection_group(dim_space)
    # print('matrices',matrices)
    return [lambda x, M=M: x @ M.T for M in matrices]


#


def main():
    dim_space = 4  # 14-dimensional vector
    group_funcs = get_np_group_functions(dim_space)

    kernel = GroupInvariantKernel(base_kernel="RBF", length_scale=0.2, smoothness=2.5, group=group_funcs)
    # print( construct_reflection_matrices(dim_space=14))
    #n_datapoints = 1
    np.random.seed(0)
    x = np.random.rand(20, dim_space)
    k_G_matrix = kernel(x, x)
   
    eigvals = np.linalg.eigvalsh(k_G_matrix)
    # print("Invariant kernel matrix:\n", k_G_matrix)
    # print("Kernel matrix eigenvalues:", eigvals)


    # torch.manual_seed(0)
    # x = torch.rand((n_datapoints, dim_space))

    # k = ScaleKernel(MaternKernel(nu=2.5))

    # k_G = InvariantKernel(
    #     base_kernel=k,
    #     transformations=apply_rotation_group,
    #     is_isotropic=True,
    #     is_group=True,
    # )

    # # manual_k_G_matrix = compute_manual_isotropic_kernel_matrix(k, x, x, apply_rotation_group)
    # # print("Manual kernel matrix:\n", manual_k_G_matrix)

    # k_G_matrix = k_G(x, x).to_dense()
    # print("Invariant kernel matrix:\n", k_G_matrix)

    # assert torch.allclose(k_G_matrix, manual_k_G_matrix, atol=1e-5)
    # if is_positive_definite(manual_k_G_matrix):
    #     print("The kernel matrix is positive definite.")
    # else:
    #     print("The kernel matrix is NOT positive definite.")
if __name__ == "__main__":
    main()

