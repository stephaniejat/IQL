from FrozenLakeStateWrapper import FrozenLake2DStateWrapper
import KRVI_algo_final
from KRVI_algo_final import KRVI
import argparse
import gymnasium as gym
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
def action_transformation(action_index):
    action_map = {
        0: np.array([-1, 0]),  # Left
        1: np.array([0, -1]),  # Down
        2: np.array([1, 0]),   # Right
        3: np.array([0, 1])    # Up
    }
    return action_map.get(action_index, np.array([0, 0]))  # Default to [0, 0] if invalid index

# Example usage
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run KRVI Algorithm")
    # Adding arguments for user input
    parser.add_argument("--beta", type=float, default=0.1, help="UCB coefficient")
    parser.add_argument("--horizon", type=int, default=100, help="Horizon length")
    parser.add_argument("--len_scale", type=float, default=0.1, help="Length scale for GP kernel")
    parser.add_argument("--noise_reg", type=float, default=0.1, help="Noise regularization for GP")
    parser.add_argument("--env", type=str, default="FrozenLake-v1", help="Environment name")
    parser.add_argument("--logging", type=str, default="True", help="logging metrics")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info)")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of training iterations (T)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--optim_botorch", type= int, default = 1, help ='turn on hyperparm optimization by botorch')



    args = parser.parse_args()

    env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode= "human")
    env = FrozenLake2DStateWrapper(env, rescale=True)
    optimal_V= None

    krvi = KRVI(
        kernel= RBFKernel(),
        env= env,
        beta=args.beta,
        horizon=args.horizon,
        action_transformation = action_transformation,
        len_scale=args.len_scale,
        noise_reg=args.noise_reg,
        optim_botorch=args.optim_botorch,
        optimal_V = optimal_V,
        logging=args.logging,
        verbose=args.verbose,
        seed= args.seed
    )

    krvi.train(T= args.iterations)








