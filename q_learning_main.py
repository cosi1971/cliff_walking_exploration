import rl_config
from rl_qlearning_methods import volcanoRLQL
import numpy as np

# %%
self = volcanoRLQL(number_of_episodes=rl_config.number_of_episodes,
                   gamma=rl_config.gamma,
                   alpha=rl_config.alpha,
                   epsilon=rl_config.epsilon,
                   env_dim=rl_config.env_dim)
self.qlearning(agent_position=rl_config.agent_position, cliff_positions=rl_config.cliff_positions, goal_position=rl_config.goal_position)
self.console_output(agent_position=rl_config.agent_position)

#%%
self = volcanoRLQL(number_of_episodes=rl_config.number_of_episodes*2,
                   gamma=rl_config.gamma,
                   alpha=0.001,
                   epsilon=0.5,
                   env_dim=(6,20))
cliff_positions = [(3,i) for i in range(6, 16)] + [(4,6), (4,7), (4,14), (4,15) ]
agent_position = (1,3)
goal_position = (4, 13)
self.qlearning(agent_position=agent_position, cliff_positions=cliff_positions, goal_position=goal_position)
self.console_output(agent_position=agent_position)
