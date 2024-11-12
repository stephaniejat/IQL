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
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll


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

def compute_manual_kernel_matrix(k, x, y, transformations):
    N = x.shape[0]
    M = y.shape[0]
    G = transformations(x).shape[-3]
    manual_k_G_matrix = torch.zeros(N, M)
    print('x',x)
    print('y',y)
 
    for i in range(N):
        for j in range(M):
            print('x[i].unsqueeze(0)',x[i].unsqueeze(0))
            # Get all permutations of x[i] and y[j]
            # These are tensors of shape (G, d)
            Gx_i = transformations(x[i].unsqueeze(0)).squeeze()
            print(transformations(x[i].unsqueeze(0)).shape)
            print('Gx_i',Gx_i.shape)
            Gy_j = transformations(y[j].unsqueeze(0)).squeeze()
            print('Gy_i',Gy_j)
            # Compute the kernel for all pairs of permutations
            for x_perm in Gx_i:
                for y_perm in Gy_j:
                    print('x_perm.unsqueeze(0)',x_perm.unsqueeze(0))
                    manual_k_G_matrix[i, j] += (
                        k(x_perm.unsqueeze(0), y_perm.unsqueeze(0)).to_dense().item()
                    )
    manual_k_G_matrix /= G**2
    return manual_k_G_matrix

def reward_RKHS(P_kernel, state_space, action_space, subdir=None, alpha=0.5):
    grid_size = 10  # Grid size for fitting GP regression

    # Generate all possible input points in the grid
    values = np.linspace(0, 1, grid_size)
    X = np.array(list(product(values, repeat=2)))  # 2D grid points for state-action pairs

    # Define the group-invariant kernel based on the P_kernel parameter and group G (state-action transformations)
    if P_kernel == "Matern":
        kernel = GroupInvariantKernel(base_kernel="Matern", length_scale=0.001, smoothness=1.5, group=group_SA)
  
    elif P_kernel == "RBF":
        kernel = GroupInvariantKernel(base_kernel="RBF", length_scale=0.001, group=group_SA)

    # Sample y values from the Gaussian process with the group-invariant kernel
    gp = GaussianProcessRegressor(kernel=kernel)
    #y = np.random.randn(X.shape[0], 1)
    #y = np.random.randn(X.shape[0])  # Random values (normal distribution)
    y = gp.sample_y(X, 1)
    #print('X',X.shape)
    #print('y',y.shape)
    # Fit the Gaussian Process Regressor
    #print('alpha',alpha)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=alpha)
    K = gpr.kernel(X,X)
    eigvals = np.linalg.eigvalsh(K)
    print("Kernel matrix eigenvalues:", eigvals)

    gpr.fit(X, y)
    print('after fitting')
    
    # Generate all possible input points for prediction (across state-action space)
    values = np.linspace(0, 1, len(state_space))
    all_possible_inputs = np.array(list(product(values, repeat=2)))  # State-action pairs in 2D

    # Predict for all possible input points
    all_predictions, _ = gpr.predict(all_possible_inputs, return_std=True)
    y_pred, _ = gpr.predict(X, return_std=True)
    mse = mean_squared_error(y, y_pred)

    # Scale and normalize the predictions
    min_prediction = np.min(all_predictions)
    max_prediction = np.max(all_predictions)
    scaled_predictions = (all_predictions - min_prediction) / (max_prediction - min_prediction)
    r = scaled_predictions.reshape((len(state_space), len(action_space)))
    print(r)

    return r

#x= torch.tensor([[1,2,3]])
#x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# x = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])

# x_permuted= permutation_group(x)
# xp= torch.rand([10, 2])
# print(xp.shape)
n_datapoints = 10
dimension = 3
k = ScaleKernel(MaternKernel(nu=2.5))
torch.manual_seed(0)

x = torch.rand([n_datapoints, dimension])
y = torch.rand([n_datapoints, dimension])
k_G = InvariantKernel(
        base_kernel=k,
        transformations=permutation_group,
        is_isotropic=True,
        is_group=True,
    )
#manual_k_G_matrix = compute_manual_kernel_matrix(k, x, y, permutation_group)
k_G_matrix = k_G(x, y).to_dense()