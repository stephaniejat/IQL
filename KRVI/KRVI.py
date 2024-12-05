from typing import Any, ClassVar, Optional, TypeVar, Union, Callable

import numpy as np
import torch as th
from gymnasium import spaces



class KRVI:
    """
    KRVI

    Paper: "https://"

    
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param backend: the algorithm used to perform KRR when fitting Q
    :param kernel: the kernel used
    :param beta: UCB factor
    :param horizon: horizon to run
    :param penalty: penalty function for KRR
    :param exploration bonus: Exploration bonus for Q function maximization (e.g. UCB bonus)

    
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    
    :param logging: the log location (if None, no logging)
    
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    
    
    """

    def __init__(
        self,
        kernel: Union[str, type[Kernel]],
        backend: Union[str, type[GPRegressor]],
        env: Union[GymEnv, str],
        beta: float,
        horizon: int,
        penalty: Union[Callable, str], #TODO: should have a separate type other than Callable for this
        exploration_bonus: Union[Callable, str], #TODO: should have a separate type other than Callable for this
        Q: Optional[Callable], #TODO: should have a separate type other than Callable for this
        V: Optional[Callable], #TODO: should have a separate type other than Callable for this
        train_freq: Union[int, tuple[int, str]],
        logging: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
       
    ) -> None:
        """
        Boilerplate goes here
        """
        self._initialize_qv(Q,V)
        self.policy = self.get_policy()

    def _on_step(self) -> None:
        """
        What information is collected on step. Caching, logging all go here.
        """
        pass

    def _on_episode_termination(self) -> None:
        """
        What happens when an episode ends.
        """
        pass

    def _initialize_qv(self, Q: Optional[Callable] = None, V: Optional[Callable] = None) -> None:
        """
        Initilize the Q,V functions. This should be called once in the init, and can be called with values from previous runs.
        """
        pass

    def train(self, episodes: int) -> None:
        """
        Update Q and V
        """

        pass

    def get_policy(self) -> Policy:
        """
        Get policy from Q, by performing argmax. 
        """
        pass


    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state



class Kernel:
    """
    Implements our version of kernel.
    Can be a thin wrapper around other packages.
    """

    def compute(x,y) -> float: #TODO: reimplement / rename / redefine types to suit whatever backend we're using. Typing should be consistent
        pass

class GPRegressor:
    """
    Implement our version of GaussianProcessRegressor
    Can be a thin wrapper around other packages.
    """