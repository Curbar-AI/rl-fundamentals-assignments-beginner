from rl.algorithms.planning.dyna import Dyna, DynaModel  # DynaPlus inherits from Dyna, so we can reuse a lot of the
from rl.common.q_value_table import QValueTable

import numpy as np
from typing import Optional, Tuple, Union
from rl.common.results_logger import ResultsLogger



class DynaPlusModel(DynaModel):
    """
    Model object for the Dyna-plus algorithm, storing the (state, action) -> (reward, next_state) mapping.

    Additional to the DynaModel, when this class initialises a new state, it will include transitions for all
    actions, including those not yet taken (in these instances, the modelled reward is 0 and the next state is the
    current state).

    Attributes:
        num_actions (int): The number of actions available in the environment.
    """

    def __init__(self, num_actions: int, random_seed: Optional[int] = None) -> None:
        """
        Initialise the DynaPlusModel.

        Args:
            num_actions (int): Number of actions available in the environment.
            random_seed (Optional[int]): Random seed for reproducibility.
        """
        super().__init__(random_seed)
        self.num_actions = num_actions

    def add(
            self, state: Union[int, Tuple[int, ...]], action: int,
            reward: float, next_state: Union[int, Tuple[int, ...]]
    ) -> None:
        """
        Add a transition to the model.

        Args:
            state (Union[int, Tuple[int, ...]]): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (Union[int, Tuple[int, ...]]): The next state resulting from the action.
        """
        if state not in self.model.keys():
            for a in range(self.num_actions):
                # HOMEWORK: If state is newly encountered, initialise all actions with (reward=0, next_state=state)
                # TODO: Implement this line

        # HOMEWORK: Add the actual transition
        # TODO: Implement this line


class TimeSinceLastEncountered(QValueTable):
    """
    Implements the tau(s, a) table for Dyna+ algorithm.

    This is a NumPy array, same size as the Q-value table, initialised with zeros, so can base this class on the
    QValueTable class, which comes with methods:
    - get(state, action) -> value
    - update(state, action, value)

    We can extend this class with a method to increment all values by 1, except for a single (state, action) pair, which
    is reset to 0.

    Attributes:
        values (np.ndarray): The table representing the time since each (state, action) pair was last encountered.
    """

    def __init__(self, num_states: int, num_actions: int) -> None:
        """
        Initialise the TimeSinceLastEncountered table.

        Args:
            num_states (int): The number of states in the environment.
            num_actions (int): The number of actions available in the environment.
        """
        super().__init__((num_states,), num_actions)

    def increment(self, state: Union[int, Tuple[int, ...]], action: int) -> None:
        """
        Increment all values in the table by 1, except for the specified (state, action) pair which is reset to 0.

        Args:
            state (Union[int, Tuple[int, ...]]): The state to reset.
            action (int): The action to reset.
        """
        # HOMEWORK: Increment values for all (S, A) pairs by 1
        # TODO: Implement this line

        # HOMEWORK: Except for the (state, action) pair, which is reset to 0 (use self.update with the right arguments)
        # TODO: Implement this line


class DynaPlus(Dyna):
    """
    Dyna-Q+ algorithm implementation, extending the Dyna algorithm by incorporating exploration bonuses.

    Attributes:
        kappa (float): Exploration bonus coefficient.
        model (DynaPlusModel): The model used to simulate experience for planning.
        time_since_last_encountered (TimeSinceLastEncountered): Table to track the time since each (state, action) pair
            was last encountered.
    """

    def __init__(self, env, alpha: float = 0.5, gamma: float = 1.0, epsilon: float = 0.1, n_planning_steps: int = 5,
                 kappa: float = 0.001, logger: Optional[ResultsLogger] = None, random_seed: Optional[int] = None
                 ) -> None:
        """
        Initialise the DynaPlus agent.

        Args:
            env: The environment to interact with.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Probability of choosing a random action (epsilon-greedy policy).
            n_planning_steps (int): Number of planning steps per real step.
            kappa (float): Exploration bonus coefficient.
            logger (Optional[ResultsLogger]): Logger for recording training progress.
            random_seed (Optional[int]): Random seed for reproducibility.
        """
        # Initialise attributes common to Dyna
        super().__init__(env, alpha, gamma, epsilon, n_planning_steps, logger, random_seed)

        self.name = "Dyna+"
        self.kappa = kappa
        #  when the environment is reset at the end of an episode
        self.model: Optional[DynaPlusModel] = None
        self.time_since_last_encountered: Optional[TimeSinceLastEncountered] = None
        self.reset()

    def reset(self) -> None:
        """
        Reset the agent for a new episode.
        """
        super().reset()

        self.model = DynaPlusModel(self.env.action_space.n, self.random_seed)
        self.time_since_last_encountered = TimeSinceLastEncountered(
            self.env.observation_space.n,
            self.env.action_space.n
        )

    def learn(self, num_episodes: int = 500) -> None:
        """
        Train the agent using the Dyna-Q+ algorithm.

        Args:
            num_episodes (int): Number of episodes to train the agent for.
        """
        # HOMEWORK STARTS: Implement the Dyna-Q+ algorithm (~25-30 lines).

pass  # TODO: Implement this section
        # HOMEWORK ENDS
