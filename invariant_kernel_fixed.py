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
        # print('x1',x1)
        
        x1_orbits = self.transformations(x1)  # Shape is ... x G x N x d
        # print('x1_orbits',x1_orbits)
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
            # print("Grad check x1_orbits requires_grad:", x1_orbits.requires_grad)
            # print("Grad check x2_orbits requires_grad:", x2_orbits.requires_grad)

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





def construct_rot_and_reflection_group(dim_space: int):
    """
    Constructs a group of 8 transformations:
    - 4 rotations (0°, 90°, 180°, 270°)
    - 4 rotations followed by reflection across the x-axis
    """
    assert dim_space % 2 == 0, "Dimension must be even (pairs of x, y)."

    # 2D 90-degree rotation matrix (counter-clockwise)
    rot_90 = np.array([[0, 1], [-1, 0]])
    rot_blocks = [rot_90] * (dim_space // 2)
    rot_matrix = block_diag(*rot_blocks)

    # 2D reflection over x-axis
    reflect_x = np.array([[1, 0], [0, -1]])
    reflect_blocks = [reflect_x] * (dim_space // 2)
    reflect_matrix_x = block_diag(*reflect_blocks)

    # Generate rotation matrices
    rotations = [np.eye(dim_space)]
    current = np.eye(dim_space)
    for _ in range(3):
        current = current @ rot_matrix
        rotations.append(current.copy())

    # Now: rotations followed by reflection in x-axis
    rot_reflections = [R @ reflect_matrix_x for R in rotations]

    # Combine all 8
    full_group = rotations + rot_reflections

    return [torch.tensor(M, dtype=torch.float32) for M in full_group]





def apply_rotation_group(x: torch.Tensor) -> torch.Tensor:
    """
    Applies 8 symmetry transformations: 4 rotations + 4 rot-reflections.

    Args:
        x: Tensor of shape (..., d), where d = 2 * num_pairs.

    Returns:
        Tensor of shape (..., 8, d), one per transformation.
    """
    device = x.device
    group = construct_rot_and_reflection_group(x.shape[-1])
    group = [g.to(device=device, dtype=torch.float32) for g in group]
    transformed = [x @ g.T for g in group]
    return torch.stack(transformed, dim=-3)

def is_positive_definite(K: torch.Tensor, tol: float = 1e-5) -> bool:
    """
    Check if a kernel matrix is positive semi-definite by checking if all its eigenvalues
    are non-negative.

    Args:
        K (torch.Tensor): The kernel matrix to check.
        tol (float): The tolerance to consider eigenvalues as non-negative.

    Returns:
        bool: True if the matrix is positive semi-definite, otherwise False.
    """
    # Eigenvalues of the kernel matrix
    eigenvalues = torch.linalg.eigvals(K)

    # Check if all eigenvalues are non-negative
    return torch.all(eigenvalues.real >= -tol)


def main():
    dim_space = 14  # 14-dimensional vector
    # print( construct_rot_and_reflection_group(dim_space=14))
    n_datapoints = 10

    torch.manual_seed(0)
    x = torch.rand((n_datapoints, dim_space))

    k = ScaleKernel(MaternKernel(nu=2.5))

    k_G = InvariantKernel(
        base_kernel=k,
        transformations=apply_rotation_group,
        is_isotropic=True,
        is_group=True,
    )

    # manual_k_G_matrix = compute_manual_isotropic_kernel_matrix(k, x, x, apply_rotation_group)
    # print("Manual kernel matrix:\n", manual_k_G_matrix)

    k_G_matrix = k_G(x, x).to_dense()
    print("Invariant kernel matrix:\n", k_G_matrix)

    # assert torch.allclose(k_G_matrix, manual_k_G_matrix, atol=1e-5)
    if is_positive_definite( k_G_matrix):
        print("The kernel matrix is positive definite.")
    else:
        print("The kernel matrix is NOT positive definite.")
if __name__ == "__main__":
    main()

