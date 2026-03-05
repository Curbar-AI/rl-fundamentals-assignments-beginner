from rl.algorithms.common.td_agent import TemporalDifferenceAgent
import numpy as np
from collections import defaultdict    # For model, with default value of empty list
import random    # For random choice of state and action from model during planning
from typing import Union, Tuple
from rl.common.results_logger import ResultsLogger


class DynaModel:
    """
    Model for a Dyna agent, storing the (state, action) -> (reward, next_state) mapping.

    Model is a nested dictionary: first key is state, second key is action, value is (reward, next_state).
    For the purposes of this repo, we assume a deterministic environment, so the mapping is unique.

    Args:
        random_seed (int, optional): Seed for reproducibility.
    """

    def __init__(self, random_seed: Union[int, None] = None) -> None:
        self.model: defaultdict = defaultdict(dict)
        self.random_seed: Union[int, None] = random_seed
        if random_seed:
            self._set_random_seed()

    def _set_random_seed(self) -> None:
        """
        Sets the random seed for reproducibility.
        """
        random.seed(self.random_seed)

    def add(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Adds a transition to the model.

        Args:
            state (int): The state.
            action (int): The action taken from the state.
            reward (float): The reward received after taking the action.
            next_state (int): The next state after taking the action.
        """
        # HOMEWORK: Add a (r, s') transition to the model for a given (s, a) pair.
        # TODO: Implement this line

    def get(self, state: int, action: int) -> Tuple[float, int]:
        """
        Gets the reward and next state for a given state-action pair.

        Args:
            state (int): The state.
            action (int): The action taken from the state.

        Returns:
            Tuple[float, int]: A tuple containing the reward and the next state.
        """
        return self.model[state][action]

    def sample_state_action(self) -> Tuple[int, int]:
        """
        Samples a random state and action from the model.

        Returns:
            Tuple[int, int]: A tuple containing a randomly selected state and action.
        """
        # HOMEWORK: sample a random state and action from the model. Use random.choice() alongside list() to sample from
        # the keys of the model dictionary (top-level keys are states).
        state: int  = None  # TODO: Implement this assignment

        # HOMEWORK: sample a random action from the model. Use random.choice() alongside list() to sample from
        # the keys of the nested dictionary (second-level keys are actions).
        action: int  = None  # TODO: Implement this assignment

        return state, action


class Dyna(TemporalDifferenceAgent):
    """
    Dyna agent for Temporal Difference learning with planning.

    Args:
        env: The environment to interact with.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration parameter for epsilon-greedy policy.
        n_planning_steps (int): Number of planning steps to perform per real experience step.
        logger (ResultsLogger, optional): Logger for tracking results during training.
        random_seed (int, optional): Seed for reproducibility.
    """

    def __init__(
        self,
        env,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        n_planning_steps: int = 5,
        logger: Union[ResultsLogger, None] = None,
        random_seed: Union[int, None] = None
    ) -> None:
        # Initialise common Temporal Difference agent attributes: q_values, policy, episode_rewards
        super().__init__(env, gamma, alpha, epsilon, logger, random_seed)

        # Initialise Dyna-specific attributes
        self.name: str = "Dyna"
        self.n_planning_steps: int = n_planning_steps

        # Initialise attributes which reset on each new episode
        self.model: Union[DynaModel, None] = None
        self.reset()

    def reset(self) -> None:
        """
        Resets the agent's attributes, including Q-values, policy, and the model.
        """
        # Reset common Temporal Difference agent attributes: q_values, policy, episode_rewards
        super().reset()

        # Initialise Dyna-specific attributes
        self.model = DynaModel(self.random_seed)

    def learn(self, num_episodes: int = 500) -> None:
        """
        Trains the Dyna agent for a given number of episodes, performing both direct learning and planning.

        Args:
            num_episodes (int): Number of episodes to train the agent.
        """

        # HOMEWORK STARTS: Implement the Dyna algorithm (~25-30 lines).
        # Remember this is very similar to the Q-learning implementation, but with an additional internal planning loop.
        # Inspect and use the new attributes: self.model, self.n_planning_steps.
        pass  # TODO: Implement this section
        # HOMEWORK ENDS
