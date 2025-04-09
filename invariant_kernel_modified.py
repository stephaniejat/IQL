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
from botorch.models.transforms import Normalize, Standardize
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
        is_isotropic: bool = True,
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
        #print('x1',x1)
        
        x1_orbits = self.transformations(x1)  # Shape is ... x G x N x d
        #print('x1_orbits',x1_orbits)
        #print('x1_orbits shape',x1_orbits.shape)
        G = x1_orbits.shape[-3]
        if self.is_isotropic:
            # Sum is over a single set of orbits
            # x2_orbits should be constructed by tiling x2 along the -3 axis
            dims = [-1] * (x2.dim() + 1)
            #print('dims',dims)
            dims[-3] = G
            #print('x2',x2)
            x2_orbits = x2.unsqueeze(-3).expand(dims)
            #print('x2_orbits',x2_orbits)
            #print('x2_orbits',x2_orbits.shape)
            K_orbits = self.base_kernel.forward(x1_orbits, x2_orbits)
            #print('K_orbits',K_orbits.shape)
            K = torch.mean(K_orbits, dim=-3)
            #print('K',K.shape)
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
        #print('kernel',K)

        if diag:
            return K.diag()
        else:
            return K


def construct_90deg_block_rot_groups(dim_space: int):
    assert dim_space % 2 == 0  # Ensure we have an even-dimensional space

    rotation_matrix = np.array([[0, 1], [-1, 0]]) # 90 degrees CW

    # Create block-diagonal matrix for higher dimensions
    blocks = [rotation_matrix] * (dim_space // 2)
    block_diagonal_matrix = block_diag(*blocks)

    # Rotation group elements (90, 180, 270 degrees)
    rot_group_nd = [
        np.eye(dim_space), 
        block_diagonal_matrix, 
        block_diagonal_matrix @ block_diagonal_matrix, 
        block_diagonal_matrix @ block_diagonal_matrix @ block_diagonal_matrix
    ]

    # Convert to PyTorch tensors
    rot_group_tensors = [torch.tensor(R, dtype=torch.float32) for R in rot_group_nd]

    return rot_group_tensors
# def apply_rotation_group_normalized(x: torch.Tensor, normalizer: Normalize) -> torch.Tensor:
#     """
#     Applies the rotation group transformations to x **while accounting for normalization**.

#     Args:
#         x: Normalized input tensor of shape (..., d)
#         normalizer: The Normalize transformation (from botorch)

#     Returns:
#         Tensor of shape (..., G, d) where G = 4 (identity + 90° + 180° + 270°)
#     """
#     # Get the bounds from the normalizer
#     bounds = Normalize.bounds.to(x.device)  # Shape: (2, d)
#     lower, upper = bounds[0], bounds[1]

#     # Undo normalization (map back to raw space)
#     x_raw = x * (upper - lower) + lower  # Inverse of normalization

#     # Apply rotations in the raw space
#     rotations = construct_90deg_block_rot_groups(x.shape[-1])  # Get the rotation matrices
#     transformed_raw = [x_raw @ R.T for R in rotations]  # Apply each rotation in raw space

#     # Re-normalize the transformed data
#     transformed_normalized = [(t - lower) / (upper - lower) for t in transformed_raw]

#     # Stack results along a new axis
#     return torch.stack(transformed_normalized, dim=-3)

def apply_rotation_group_normalized(x: torch.Tensor, normalizer: Normalize) -> torch.Tensor:
    """
    Applies the rotation group transformations to x **while accounting for per-dimension normalization**.

    Args:
        x: Normalized input tensor of shape (..., d)
        normalizer: The Normalize transformation (from BoTorch)

    Returns:
        Tensor of shape (..., G, d) where G = 4 (identity + 90° + 180° + 270°)
    """
    # Get the per-dimension bounds from the normalizer
    bounds = normalizer.bounds.to(x.device)  # Shape: (2, d)
    lower, upper = bounds[0], bounds[1]  # Extract lower and upper bounds per feature

    # Undo normalization (convert back to raw space)
    x_raw = x * (upper - lower) + lower  # Element-wise transformation

    # Apply rotations in the raw space
    rotations = construct_90deg_block_rot_groups(x.shape[-1])  # Get rotation matrices
    transformed_raw = [x_raw @ R.T for R in rotations]  # Apply each rotation in raw space

    # Re-normalize after applying rotations
    transformed_normalized = [(t - lower) / (upper - lower) for t in transformed_raw]

    # Stack results along a new axis (G transformations applied)
    return torch.stack(transformed_normalized, dim=-3)



def apply_rotation_group(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the rotation group transformations to x.
    
    Args:
        x: Tensor of shape (..., d), where d should be 14 in our case.
    
    Returns:
        Tensor of shape (..., G, d), where G=4 (identity + 90° + 180° + 270°).
    """
    #rotations = construct_90deg_block_rot_groups(x.shape[-1])  # Get the rotation matrices
    device = x.device
    rotations = [torch.tensor(R, dtype=torch.float32, device=device) for R in construct_90deg_block_rot_groups(x.shape[-1])]  
    # print('rotations', rotations)
    transformed = [x @ R.T for R in rotations]  # Apply each rotation
    return torch.stack(transformed, dim=-3)  # Stack along a new dimension

def main():
    dim_space = 14  # 14-dimensional vector
    n_datapoints = 10

    torch.manual_seed(0)
    x = torch.rand((n_datapoints, dim_space))

    k = ScaleKernel(MaternKernel(nu=2.5))

    k_G = InvariantKernel(
        base_kernel=k,
        transformations=apply_rotation_group_normalized,
        is_isotropic=True,
        is_group=True,
    )

    # manual_k_G_matrix = compute_manual_isotropic_kernel_matrix(k, x, x, apply_rotation_group)
    # print("Manual kernel matrix:\n", manual_k_G_matrix)

    k_G_matrix = k_G(x, x).to_dense()
    print("Invariant kernel matrix:\n", k_G_matrix)

    # assert torch.allclose(k_G_matrix, manual_k_G_matrix, atol=1e-5)
    # if is_positive_definite(manual_k_G_matrix):
    #     print("The kernel matrix is positive definite.")
    # else:
    #     print("The kernel matrix is NOT positive definite.")
if __name__ == "__main__":
    main()

