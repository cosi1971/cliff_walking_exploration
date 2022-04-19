# hyperparameters set up
number_of_episodes = 10000  # Number of training episodes
gamma = 0.9  # Discount rate γ 0.9
alpha = 0.001  # Learning rate α 0.001
epsilon = 0.05  # Exploration rate ε

#env parameters
env_dim = (4, 12)

cliff_positions = [(3,i) for i in range(1, 11)]

goal_position = (3, 11)

agent_position = (3,0)
