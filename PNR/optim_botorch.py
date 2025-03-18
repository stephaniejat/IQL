from naive_reward_calculator import optim_reward

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


def objective(X):
    X = X.view(54, 2).detach().numpy()
    return torch.tensor(optim_reward(X), dtype=torch.float)

# Initial data
train_X = 840 * torch.rand(100, 108)  # 5 initial samples, each of shape (54, 2) flattened to (108,)
train_Y = torch.tensor([objective(x) for x in train_X], dtype=torch.float).unsqueeze(-1)
bounds = torch.stack([torch.zeros(108), 840* torch.ones(108)])
# Define the Gaussian Process model
gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# Define the acquisition function
UCB = UpperConfidenceBound(gp, beta=0.1)

# Optimize the acquisition function

candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=10, raw_samples=10,
)

# Evaluate the new candidate
new_y = objective(candidate)

# Update training data
# train_X = torch.cat([train_X, candidate])
# train_Y = torch.cat([train_Y, new_y.unsqueeze(-1)])

# # Refit the model
# gp = SingleTaskGP(train_X, train_Y)
# mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
# fit_gpytorch_model(mll)

print("Optimized input:", candidate.view(54, 2))
print("Objective value:", new_y.item())