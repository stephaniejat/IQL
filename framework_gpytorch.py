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
#from botorch.models import SingleTaskGP
#from botorch.fit import fit_gpytorch_mll


class InvariantKernel(gpytorch.kernels.Kernel):
    r"""A kernel that is invariant to a collection of transformations.

    Currently only supports group invariance.

    The invariant kernel is defined as:
    .. math::
        k_G(x, y) = \frac{1}{|G|^2} \sum_{g \in G} \sum_{h \in G} k(g(x), h(y))

    If the kernel is isotropic, we can use the simpler form:
    .. math::
        k_G(x, y) = \frac{1}{|G|} \sum_{g \in G} k(g(x), y)

    where :math:`G` is the group of transformations, :math:`k` is the base kernel, and :math:`x` and :math:`y` are the inputs.
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        transformations: Callable[[torch.tensor], torch.tensor],
        is_isotropic: bool = False,
        is_group: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.base_kernel = base_kernel
        self.transformations = transformations
        self.is_isotropic = is_isotropic
        self.is_group = is_group

        if not self.is_group:
            raise NotImplementedError("InvariantKernel only supports group invariance.")

    def forward(
        self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **kwargs
    ) -> torch.tensor:
        if last_dim_is_batch:
            raise NotImplementedError(
                "last_dim_is_batch=True not supported for GroupInvariantKernel."
            )
        print('x1',x1)
        
        x1_orbits = self.transformations(x1)  # Shape is ... x G x N x d
        print('x1_orbits',x1_orbits)
        print('x1_orbits shape',x1_orbits.shape)
        G = x1_orbits.shape[-3]
        if self.is_isotropic:
            # Sum is over a single set of orbits
            # x2_orbits should be constructed by tiling x2 along the -3 axis
            dims = [-1] * (x2.dim() + 1)
            print('dims',dims)
            dims[-3] = G
            print('x2',x2)
            x2_orbits = x2.unsqueeze(-3).expand(dims)
            print('x2_orbits',x2_orbits)
            print('x2_orbits',x2_orbits.shape)
            K_orbits = self.base_kernel.forward(x1_orbits, x2_orbits)
            print('K_orbits',K_orbits.shape)
            K = torch.mean(K_orbits, dim=-3)
            print('K',K.shape)
        else:
            # Sum is over all pairs of orbits
            # x2_orbits should be constructed by applying the transformations to x2
            x2_orbits = self.transformations(x2)  # Shape is ... x G x M x d

            if x2_orbits.shape[-3] != G or x1_orbits.shape[-3] != G:
                raise ValueError(
                    "Different numbers of orbits for x1 and x2. "
                    "Check that self.transformations returns a tensor of shape (..., G, N, d)."
                )

            # WARNING: This is quadratic in the number of orbits!
            # TODO: We should be able to parallelise this
            # TODO: Should be a way of doing it with less memory
            # Repeat each element of x1_orbits G times
            # New shape is G^2 x ... x N x d
            x1_orbits_expanded = x1_orbits.repeat_interleave(G, dim=-3)
            # Repeat the entire x2_orbits G times
            # New shape is ... x G^2 x M x d
            dims = [1] * x2_orbits.dim()
            dims[-3] = G
            x2_orbits_expanded = x2_orbits.repeat(dims)
            # Compute the kernel between each pair of expanded orbits = all combinations of orbits
            K_orbits = self.base_kernel.forward(x1_orbits_expanded, x2_orbits_expanded)
            K = torch.mean(K_orbits, dim=-3)
        print('kernel',K)

        if diag:
            return K.diag()
        else:
            return K
def permutation_group(x: torch.Tensor) -> torch.tensor:
    indices = range(x.shape[-1])
    #print('indices',indices)
    permuted_indices = [list(p) for p in itertools.permutations(indices)]
    #print('permuted indices',permuted_indices)
    permuted_x = x[..., permuted_indices]
    #print('permuted_x shape',permuted_x.shape)
    #print('permuted_x',permuted_x)
    # permuted_x is a tensor of shape (..., n, G, d)
    # Reorder the dimensions to (..., G, n, d)
    dim_indices = list(range(permuted_x.dim()))
    #print('dim_indices',dim_indices)
    dim_indices[-2], dim_indices[-3] = dim_indices[-3], dim_indices[-2]
    #print('return',permuted_x.permute(*dim_indices).shape)
    return permuted_x.permute(*dim_indices)

def compute_manual_isotropic_kernel_matrix(k, x, y, transformations):

    N = x.shape[0]
    M = y.shape[0]
    # Get the number of transformations by applying them to `x`
    G = transformations(x).shape[-3]
    manual_k_G_matrix = torch.zeros(N, M)

    for i in range(N):
        for j in range(M):
            # Apply all transformations to x[i], resulting in (G, d)
            Gx_i = transformations(x[i].unsqueeze(0)).squeeze()

            # Sum the kernel evaluations over all transformed versions of x[i] and y[j]
            for x_transformed in Gx_i:
                manual_k_G_matrix[i, j] += k(x_transformed.unsqueeze(0), y[j].unsqueeze(0)).to_dense().item()

            # Average over all transformations for isotropic kernel
            manual_k_G_matrix[i, j] /= G
    print('manual_k_G_matrix',manual_k_G_matrix)
    return manual_k_G_matrix

# Shift transformation functions for state or action
def shift_05(x: torch.Tensor) -> torch.Tensor:
    return (x + 0.5) % 1
def flip(x:torch.Tensor) -> torch.Tensor:
    return -x
# Identity transformation (no change)
identity = lambda x: x

# Define state and action transformations
state_transformations = [identity, flip]
action_transformations = [identity, flip]
# Define transformation groups for 2D (S x A) state-action pairs
def group_SA(x: torch.Tensor) -> torch.Tensor:
    """
    Apply all combinations of transformations for 2D state-action pairs.
    x: A torch.Tensor of shape (batch_size, 2), where the last dimension is [state, action].
    """
    # Ensure input x has shape (batch_size, 2)
    assert x.shape[-1] == 2, "Input must have shape (..., 2) for state-action pairs."

    # Apply each combination of state and action transformations
    transformed_pairs = []
    for g1, g2 in itertools.product(state_transformations, action_transformations):
        # Apply transformations g1 on the state part (x[..., 0]) and g2 on the action part (x[..., 1])
        transformed_pair = torch.stack([g1(x[..., 0]), g2(x[..., 1])], dim=-1)
        transformed_pairs.append(transformed_pair)

    # Stack along a new dimension to represent the group
    transformed_tensor = torch.stack(transformed_pairs, dim=0)  # Shape (|G|, batch_size, 2) where |G|=4
    return transformed_tensor

def is_positive_definite(kernel_matrix: torch.Tensor) -> bool:
    """
    Check if a kernel matrix is positive definite by examining its eigenvalues.
    Args:
        kernel_matrix (torch.Tensor): The kernel matrix (should be symmetric).
    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    # Ensure the matrix is symmetric
    if not torch.allclose(kernel_matrix, kernel_matrix.T, atol=1e-5):
        print("Matrix is not symmetric!")
       

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(kernel_matrix)
    print("Eigenvalues:", eigenvalues)

    # Check if all eigenvalues are greater than zero
    return torch.all(eigenvalues > 0)

def main():
   
    # n_datapoints = 10
    # dimension = 2
     
    # k = ScaleKernel(MaternKernel(nu=2.5))
    # torch.manual_seed(0)

    # x = torch.rand([n_datapoints, dimension])
    # # y = torch.rand([n_datapoints, dimension])

    # k_G = InvariantKernel(
    #         base_kernel=k,
    #         transformations= group_SA,
    #         is_isotropic=True,
    #         is_group=True,
    #     )

    # manual_k_G_matrix = compute_manual_isotropic_kernel_matrix(k, x, x, group_SA)
    # print('Manual kernel',manual_k_G_matrix)
    # k_G_matrix = k_G(x, x).to_dense()
    # print('Theo invariant kernel',k_G_matrix)

    # assert torch.allclose(k_G_matrix, manual_k_G_matrix, atol=1e-5)
    # if is_positive_definite(manual_k_G_matrix):
    #     print("The kernel matrix is positive definite.")
    # else:
    #     print("The kernel matrix is NOT positive definite.")

    n_datapoints = 10
    dimension = 2
     
    k = ScaleKernel(MaternKernel(nu=2.5))
    torch.manual_seed(0)

    # Generate state and action spaces
    state_space = np.linspace(-1, 1, num=10)
    action_space = np.linspace(-1, 1, num=10)

    # Create all combinations of state-action pairs
    SA = np.array(list(product(state_space, action_space)))
    # Convert to PyTorch tensor
    SA_tensor = torch.tensor(SA, dtype=torch.float32)
    print('SA tensor',SA_tensor)
    print('SA tensor',SA_tensor.shape)
    x=SA_tensor


    k_G = InvariantKernel(
            base_kernel=k,
            transformations= group_SA,
            is_isotropic=True,
            is_group=True,
        )

    manual_k_G_matrix = compute_manual_isotropic_kernel_matrix(k, x, x, group_SA)
    print('Manual kernel',manual_k_G_matrix)
    k_G_matrix = k_G(x, x).to_dense()
    print('Theo invariant kernel',k_G_matrix)

    assert torch.allclose(k_G_matrix, manual_k_G_matrix, atol=1e-5)
    if is_positive_definite(manual_k_G_matrix):
        print("The kernel matrix is positive definite.")
    else:
        print("The kernel matrix is NOT positive definite.")


    
    

if __name__ == "__main__":
    main()