import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import rl_config
import seaborn as sns
from loguru import logger


class volcanoRLQL(object):
    # initialize env
    def __init__(self, number_of_episodes, gamma, alpha, epsilon, env_dim=rl_config.env_dim):  # ,
        # cliff_positions=rl_config.cliff_positions,
        # goal_position=rl_config.goal_position, agent_position=rl_config.agent_position):
        self.set_up_parameters(number_of_episodes, gamma, alpha, epsilon, env_dim)
        self.set_up_environment()
        # self.init_q_table()
        # self.steps_cache = np.zeros(number_of_episodes)
        # self.rewards_cache = np.zeros(number_of_episodes)
        # self.cliff_positions = cliff_positions
        # self.goal_position = goal_position
        # self.agent_pos = agent_position

    def set_up_environment(self):
        env_dim = self.env_dim
        self.flatten_index_table = np.array(list(np.ndindex(env_dim)))  # given flattened index find array index
        self.array_index_table = np.array(list(range(np.prod(env_dim)))).reshape(env_dim)

    def set_up_states(self, agent_position, cliff_positions, goal_position):
        # self.current_state = -1

        # Initialize state-table (4 actions per state) with zeros
        # env_dim = self.env_dim
        # self.flatten_index_table = np.array(list(np.ndindex(env_dim))) # given flattened index find array index
        # self.array_index_table = np.array(list(range(np.prod(env_dim)))).reshape(env_dim) # reverse lookup array index and find flattened index

        # zero_env_table = np.zeros(self.env_dim)
        # multi_index = pd.MultiIndex.from_product([range(i) for i in (self.env_dim)], names=['pos_x', 'pos_y'])
        # self.env = pd.Series(index=multi_index, data=zero_env_table.flatten(), name='data')

        self.states = np.zeros(np.prod(self.env_dim), dtype=int)  # maintain table in flattened
        self.agent_pos = agent_position
        self.mark_path()  # agent_pos, env
        self.cliff_pos = [self.array_index_table[(x, y)] for (x, y) in cliff_positions]  # States for cliff tiles
        # self.cliff_pos = [(3,i) for i in range(1, 11)]  # States for cliff tiles
        # self.cliff_pos = np.arange(37, 47)  # States for cliff tiles
        self.goal_pos = self.array_index_table[goal_position]  # State for right-bottom corner (destination)
        # self.goal_pos = (3, 11)  # State for right-bottom corner (destination)
        self.game_over = False

    def set_up_parameters(self, number_of_episodes, gamma, alpha, epsilon, env_dim):
        self.number_of_episodes = number_of_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.env_dim = env_dim

    # # initialize agent position
    # def start_agent(self, agent_position):
    #     self.agent_pos = agent_position
    #     # self.agent_pos = (3, 0)
    #     # return 1

    def mark_path(self):
        # (posY, posX) = self.agent_pos
        self.states[self.array_index_table[self.agent_pos]] += 1
        # self.env[posY][posX] += 1

    # check if gameover, end
    def check_game_over(self, number_of_steps: int):
        """
        Function returns reward in the given state
        """
        # Game over when reached goal, fell down cliff, or exceeded 1000 steps

        self.game_over = (
            True
            if (self.next_state == self.goal_pos or self.next_state in self.cliff_pos or number_of_steps > 100)
            else False
        )
        if self.next_state == self.goal_pos and number_of_steps > 1:
            logger.info('goal reached')

        # return self.game_over
        # return state == self.N

    def env_to_text(self, agent_position):
        """
        Convert environment to text format
        Needed for visualization in console
        """
        self.states[self.cliff_pos] = -4
        self.states[self.goal_pos] = -5
        states = self.states.reshape(self.env_dim)
        states[self.agent_pos] = -6
        states[agent_position] = -7
        states = np.where(states >= 1, 1, states)

        states = np.array2string(states, precision=0, separator=" ", suppress_small=False)
        states = states.replace("[[", " |")
        states = states.replace("]]", "|")
        states = states.replace("[", "|")
        states = states.replace("]", "|")
        states = states.replace("1", "x")
        states = states.replace("0", " ")
        states = states.replace("-4", " #")
        states = states.replace("-5", " @")
        states = states.replace("-6", " ^")
        states = states.replace("-7", " Q")
        self.env_pic = states
        # return env

    # Q-table initialization
    def init_q_table(self):
        """
        Initialize Q-table to store values state-action pairs
        Set Q(s, a) = 0, for all s ? S, a ? A(s)
        """
        # Initialize Q-table (4 actions per state) with zeros
        # zero_q_table = np.zeros((4, *self.env_dim))
        # self.q_table = np.zeros((4, self.env_dim[0], self.env_dim[1]))
        # multi_index = pd.MultiIndex.from_product([range(i) for i in (4, *self.env_dim)],
        #                                          names=['action', 'pos_x', 'pos_y'])
        # flatten and reshape by shape (action, self.env_dim), q_table stays as flatten for computations
        self.q_table = np.zeros((4, np.prod(self.env_dim)))
        # self.q_table = pd.Series(index=multi_index, data=zero_q_table.flatten(), name='q_values')
        # self.q_table = np.zeros((4, x_dim * y_dim))

    # update q_table
    def update_q_table(self):
        """
        Update Q-table based on observed rewards and next state value
        For SARSA (on-policy):
        Q(S, A) <- Q(S, A) + [? * (r + (? * Q(S', A'))) -  Q(S, A)]

        For Q-learning (off-policy):
        Q(S, A) <- Q(S, A) + [? * (r + (? * max(Q(S', A*)))) -  Q(S, A)
        """
        # Compute new q-value
        new_q_value = self.q_table[(self.action, self.current_state)] + self.alpha * (
                self.reward + (self.gamma * self.maximum_state_value) - self.q_table[(self.action, self.current_state)])
        # new_q_value = self.q_table[self.action, self.state[0], self.state[1]] + self.alpha * (self.reward + (
        # self.gamma * self.maximum_state_value) - self.q_table[self.action, self.state[0], self.state[1]])

        # Replace old Q-value
        self.q_table[(self.action, self.current_state)] = new_q_value

    def epsilon_greedy_action(self):
        """
        Select action based on the ?-greedy policy
        Random action with prob. ?, greedy action with prob. 1-?
        """
        # Random uniform sample from [0,1]
        sample = np.random.random()

        if sample <= self.epsilon:  # explore
            # select random action
            self.action = np.random.choice(4)
        else:  # exploit
            # select action with largest Q-value
            # self.action = np.argmax(self.q_table.xs(self.state, level=['pos_x', 'pos_y']))
            self.action = np.argmax(self.q_table[:, self.current_state])

    def move_agent(self):
        """
        Move agent to new position based on current position and action
        """
        # Retrieve agent position
        (pos_y, pos_x) = self.agent_pos
        action = self.action
        if action == 0:  # Down
            pos_y = pos_y - 1 if pos_y > 0 else pos_y
        elif action == 1:  # Up
            pos_y = pos_y + 1 if pos_y < 3 else pos_y
        elif action == 2:  # Left
            pos_x = pos_x - 1 if pos_x > 0 else pos_x
        elif action == 3:  # Right
            pos_x = pos_x + 1 if pos_x < 11 else pos_x
        else:  # Infeasible move
            raise Exception("Infeasible move")

        self.agent_pos = (pos_y, pos_x)

    def get_state(self, next=False):
        """
        Obtain state corresponding to agent position
        """
        # self.previous_state = self.state
        if next:
            self.next_state = self.array_index_table[self.agent_pos]
            # self.next_state = x_dim * pos_x + pos_y
        else:
            self.current_state = self.array_index_table[self.agent_pos]
            # self.state = x_dim * pos_x + pos_y

        # return state

    def get_max_qvalue(self):
        """Retrieve best Q-value for state from table"""
        # self.maximum_state_value = np.amax(self.q_table.xs(self.next_state, level=['pos_x', 'pos_y']))
        self.maximum_state_value = np.amax(self.q_table[:, self.next_state])
        # return maximum_state_value

    def get_reward(self):
        """
        Compute reward for given state
        """

        # Reward of -1 for each move (including terminating)
        self.reward = -1

        # Reward of +10 for reaching goal
        if self.next_state == self.goal_pos:
            self.reward = 10

        # Reward of -100 for falling down cliff
        if self.next_state in self.cliff_pos:
            self.reward = -10

        # return reward

    # def discount(self):
    #     return 1.

    # def states(self):
    #     return range(1, self.N + 1)

    def qlearning(self, agent_position, cliff_positions, goal_position):
        """
        Q-learning algorithm
        """
        self.init_q_table()
        self.steps_cache = np.zeros(self.number_of_episodes)
        self.rewards_cache = np.zeros(self.number_of_episodes)

        # Iterate over episodes
        for episode in range(self.number_of_episodes):
            logger.info(f'current episode: {episode}')
            # Set to target policy at final episode
            if episode == len(range(self.number_of_episodes)) - 1:
                self.epsilon = 0

            # Initialize environment and agent position
            self.set_up_states(agent_position=agent_position, cliff_positions=cliff_positions,
                               goal_position=goal_position)
            number_of_steps = 0

            while not self.game_over:
                # Get state corresponding to agent position for self.agent_pos
                self.get_state(next=False)

                # Select action using ?-greedy policy for self.state, self.q_table, self.epsilon
                self.epsilon_greedy_action()

                # Move agent to next position for self.agent_pos, self.action
                self.move_agent()

                # Mark visited path, returns self.env
                self.mark_path()

                # Determine next state
                self.get_state(next=True)

                # Compute and store reward for next_state, cliff_pos, goal_pos
                self.get_reward()
                self.rewards_cache[episode] += self.reward

                # Check whether game is over
                self.check_game_over(number_of_steps=number_of_steps)

                # Determine maximum Q-value next state (off-policy) for max_qvalue_next_state
                self.get_max_qvalue()

                # Update Q-table
                self.update_q_table()

                number_of_steps += 1

            self.steps_cache[episode] = number_of_steps

        # return q_table, env, steps_cache, rewards_cache

    def console_output(self, agent_position):
        """Print path and key metrics in console"""

        self.env_to_text(agent_position)

        print("Q-learning action after {} iterations:".format(self.number_of_episodes), "\n")
        print(self.env_pic, "\n")
        print("Number of steps:", int(self.steps_cache[-1]), "(min. = 13)", "\n")
        print("Cumulative reward:", int(self.rewards_cache[-1]), "(max. = -2)", "\n")

    def plot_steps(self):
        """
        Visualize number of steps taken
        """
        mod = len(self.steps_cache) % 10
        mean_step_qlearning = np.mean(self.steps_cache[mod:].reshape(-1, 10), axis=1)

        positions = np.arange(0, len(self.steps_cache) / 10, 100)
        labels = np.arange(0, len(self.steps_cache), 1000)

        sns.set_theme(style="darkgrid")
        sns.lineplot(data=mean_step_qlearning, label="Q-learning")

        # Plot graph
        plt.xticks(positions, labels)
        plt.ylabel("# steps")
        plt.xlabel("# episodes")
        plt.legend(loc="best")
        plt.show()

    def plot_rewards(self):
        """
        Visualizes rewards
        """
        mod = len(self.rewards_cache) % 10
        mean_reward_qlearning = np.mean(
            self.rewards_cache[mod:].reshape(-1, 10), axis=1
        )

        # Set x-axis label
        positions = np.arange(0, len(self.rewards_cache) / 10, 100)
        labels = np.arange(0, len(self.rewards_cache), 1000)

        sns.set_theme(style="darkgrid")

        sns.lineplot(data=mean_reward_qlearning, label="Q-learning")

        # Plot graph
        plt.xticks(positions, labels)
        plt.ylabel("rewards")
        plt.xlabel("# episodes")
        plt.legend(loc="best")

        plt.show()

        return

    def plot_path(self):
        """Plot latest paths for SARSA and Q-learning as heatmap"""

        # Plot path Q-learning

        # Set values for cliff
        for i in range(1, 11):
            self.states[3, i] = -1

        ax = sns.heatmap(
            self.states, square=True, cbar=True, xticklabels=False, yticklabels=False
        )
        ax.set_title("Q-learning")
        plt.show()
