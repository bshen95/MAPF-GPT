import ctypes
import numpy as np
from typing import Literal
from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase

from pogema import GridConfig



# used for simulating the expert data. 
class ExpertAgent:
    def __init__(self, idx):
        self._moves = GridConfig().MOVES
        self._reverse_actions = {tuple(self._moves[i]): i for i in range(len(self._moves))}

        self.idx = idx
        self.previous_goal = None
        self.path = []

    def is_new_goal(self, new_goal):
        return not self.previous_goal == new_goal
    
    def set_new_goal(self, new_goal):
        self.previous_goal = new_goal

    def set_path(self, new_path):
        self.path = new_path[::-1]

    def get_action(self):
        action = 0
        if len(self.path) > 1:
           x, y = self.path[-1]
           tx, ty = self.path[-2]
           action = self._reverse_actions[tx - x, ty - y]
           self.path.pop()
        # print(action)
        return action

    def clear_state(self):
        self.previous_goal = None
        self.path = []

class ExpertInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['Expert'] = 'Expert'
    time_limit: float = 60
    timeouts: list = [1.0, 5.0, 10.0, 60.0]
    path: str = ""

class ExpertInference:
    def __init__(self, cfg: ExpertInferenceConfig):
        self.cfg = cfg
        self.expert_agents = None
        self.path = cfg.path
        
    def _convert_path(self, agent_starts_xy, agent_targets_xy):
        """
        Converts motion strings (self.path) to actual coordinate paths
        for each agent based on their start positions.
        
        Args:
            agent_starts_xy (List[Tuple[int, int]]): Starting coordinates of agents.
            agent_targets_xy (List[Tuple[int, int]]): Target coordinates (not used here but may be useful for validation).

        Returns:
            List[List[Tuple[int, int]]]: List of paths (each a list of coordinates) for each agent.
        """
        # Define movement deltas
        
            # (0, 0): "w",
            # (-1, 0): "u",
            # (1, 0): "d",
            # (0, -1): "l",
            # (0, 1): "r",
        
        direction_delta = {
            'w': (0, 0),
            'l': (0, -1),
            'r': (0, 1),
            'd': (-1, 0),
            'u': (1, 0)
        }

        agent_motion_strings = self.path.strip().split("\n")
        all_paths = []

        for i, motion_str in enumerate(agent_motion_strings):
            x, y = agent_starts_xy[i]
            path = [(x, y)]

            for move in motion_str:
                dx, dy = direction_delta[move]
                x += dx
                y += dy
                path.append((x, y))

            all_paths.append(path)

        return all_paths
    

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        map_array = np.array(observations[0]['global_obstacles'])
        agent_starts_xy = [obs['global_xy'] for obs in observations]
        agent_targets_xy = [obs['global_target_xy'] for obs in observations]
        if self.expert_agents is None:
            self.expert_agents = [ExpertAgent(idx) for idx in range(len(observations))]
            for idx, (start_xy, target_xy) in enumerate(zip(agent_starts_xy, agent_targets_xy)):
                self.expert_agents[idx].set_new_goal(target_xy)
            converted_paths = self._convert_path(agent_starts_xy, agent_targets_xy)
            for idx, agent_path in enumerate(converted_paths):
                self.expert_agents[idx].set_path(agent_path)

        return [agent.get_action() for agent in self.expert_agents]

    def after_step(self, dones):
        pass

    def reset_states(self):
        self.expert_agents = None

    def after_reset(self):
        pass

    def get_additional_info(self):
        addinfo = {"rl_used": 0.0}
        return addinfo