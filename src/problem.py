from collections import Counter
from typing import Tuple, Set

import pomdp_py
from pomdp_py.representations.distribution.histogram import Histogram

from src.agent.agent import MazeAgent
from src.agent.environment import MazeEnvironment
from src.domain.maze_state import MazeState
from src.models.observation_model import ObservationModel
from src.models.reward_model import RewardModel
from src.models.transition_model import TransitionModel
from src.models.policy_model import PolicyModel  # assuming you have this
from src.solver.factory import PlannerFactory


class MazeProblem(pomdp_py.POMDP):
    """Complete POMDP problem definition for a grid world"""

    def __init__(self,
                 planner_name: str,
                 goal_state: Tuple[int, int],
                 walls: Set[Tuple[int, int]] = None,
                 holes: Set[Tuple[int, int]] = None,
                 traps: Set[Tuple[int, int]] = None,
                 coins: Set[Tuple[int, int]] = None,
                 grid_width: int = 6,
                 grid_height: int = 6,
                 init_belief=None, init_true_state: MazeState = None,
                 obs_noise: float = 0.7):
        self.height = grid_height
        self.width = grid_width
        self.goal = goal_state
        # ----- Agent -----
        self.walls = walls if walls is not None else set()
        self.holes = holes if holes is not None else set()
        self.traps = traps if traps is not None else set()
        self.coins = coins if coins is not None else set()

        self.current_state = init_true_state
        self.policy_model = PolicyModel()
        self.transition_model = TransitionModel(grid_width, grid_height, walls=walls)
        self.observation_model = ObservationModel(width=grid_width, height=grid_height, walls=walls,
                                                  noise=obs_noise)
        self.reward_model = RewardModel(
            goal_state=goal_state,
            walls=walls,
            traps=traps,
            coins=coins,
            height=grid_height,
            width=grid_width)

        self.agent = MazeAgent(
            width=grid_width,
            height=grid_height,
            init_belief=init_belief,
            goal_pos=goal_state,
            init_pos=init_true_state,
            policy_model=self.policy_model,
            transition_model=self.transition_model,
            reward_model=self.reward_model,
            observation_model=self.observation_model,
        )
        self.env = MazeEnvironment(
            width=grid_width,
            walls=self.walls,
            holes=self.holes,
            noise=obs_noise,
            init_state=init_true_state,
            goal_state=goal_state,
            reward_model=self.reward_model,
            observation_model=self.observation_model,
            transition_model=self.transition_model,
            height=grid_height,
            traps=set()
        )

        self.planner = PlannerFactory.get_planner(planner_name, self.agent)
        super().__init__(self.agent, self.env, name="GridWorldProblem")

    def update_belief(self, action, real_observation):
        self.agent.update_history(action, real_observation)
        self.planner.update(self.agent, action, real_observation)

        if isinstance(self.planner, pomdp_py.POMCP):
            print("Num sims:", self.planner.last_num_sims)
        if isinstance(self.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                self.agent.cur_belief,
                action, real_observation,
                self.agent.observation_model,
                self.agent.transition_model
            )
            self.agent.set_belief(new_belief)

    def take_action(self):
        return self.planner.plan(self.agent)

    def get_current_belief_state(self):
        belief = self.agent.belief
        if isinstance(belief, pomdp_py.Histogram):
            return belief
        particles = self.agent.cur_belief.particles
        counts = Counter(particles)
        total = len(particles)
        histogram = Histogram({state: count / total for state, count in counts.items()})
        return histogram
        particle_list = belief.particles
        histogram = Histogram({})
        for particle in particle_list:
            histogram[particle] = histogram.get(particle, 0) + 1

        total = len(particle_list)
        for state in histogram:
            histogram[state] /= total
        return histogram
