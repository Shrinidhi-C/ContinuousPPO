import gym
import time
import sys
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import learning
import matplotlib.pyplot as plt


class Simulation():

    def __init__(self, name_of_environment='CartPole-v0', nb_stacked_frame=1, agent_params={}):
        assert(nb_stacked_frame >= 1)
        # Main attributes of the Simulation
        self.env = gym.make(name_of_environment)   # We make the env
        self.nb_stacked_frame = nb_stacked_frame   # The number of frame observed we want to stack for the available state
        self.agent = learning.Agent(self.reset_env().shape, self.env.action_space.high, self.env.action_space.low, **agent_params)  # We create an intelligent agent

    def reset_env(self):
        """
        This function is used to get x0
        """
        return np.squeeze(np.stack([self.env.reset()] * self.nb_stacked_frame, axis=1).astype(np.float32))

    def get_next_state(self, current_state, new_obs):
        """
        This function is used to process x_k+1 based on the current observation of the environment we
        have just after taking the action. Useful mainly when we stack frames.
        """
        if self.nb_stacked_frame == 1:
            return new_obs.astype(np.float32)
        else:
            return np.append(current_state[:, 1:], np.expand_dims(new_obs, 1), axis=1).astype(np.float32)

    def test_random(self, verbose=False):
        """
        This function is used to simulate a random evolution of the environment, with random actions taken by the agent
        """
        self.env.reset()
        done = False
        while not done:
            self.env.render()
            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            if verbose:
                print(obs, rew, info)
            if done:
                self.env.render()
                break
        time.sleep(2)
        self.env.close()

    def train(self, target_score, max_episodes=1000, process_average_over=100, test_every=50, test_on=0):
        print('\n%s\n' % ('Training'.center(100, '-')))
        # Here we train our neural network with the given method
        training_score = np.empty((max_episodes,))
        training_rolling_average = np.empty((max_episodes,))
        total_rewards = deque(maxlen=process_average_over)  # We initialize the total reward list
        rolling_mean_score = -float('inf')
        ep = 0
        while rolling_mean_score < target_score and ep < max_episodes:
            state = self.reset_env()  # We get x0
            episode_reward = 0
            done = False
            visualize = (ep + 1) > test_on and (ep + 1) % test_every < test_on
            # While the game is not finished
            while not done:
                action = self.agent.take_action(state, train=True)  # we sample the action
                obs, reward, done, _ = self.env.step(action)  # We take a step forward in the environment by taking the sampled action
                if visualize:
                    # If vizualise is greater than 0, we vizualize the environment
                    self.env.render()
                episode_reward += reward
                next_state = self.get_next_state(state, obs)
                self.agent.memory.append((state, action, reward, next_state, done))
                state = next_state
            self.agent.learn_end_ep()
            total_rewards.append(episode_reward)
            rolling_mean_score = np.mean(total_rewards)
            training_score[ep] = episode_reward
            training_rolling_average[ep] = rolling_mean_score
            self.agent.print_verbose(ep, max_episodes, episode_reward, rolling_mean_score)
            self.env.close()
            ep += 1
        print('\n%s\n' % ('Training Done'.center(100, '-')))
        training_score = training_score[:ep]
        training_rolling_average = training_rolling_average[:ep]
        plt.figure()
        plt.plot(training_score, 'b', linewidth=1, label='Score')
        plt.plot(training_rolling_average, 'orange', linewidth=1, label='Rolling Average')
        plt.plot([target_score] * ep, 'r', linewidth=1, label='Target Score')
        plt.title('Evolution of the score during the Training')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    agent_params = {
        'temperature': 0.1,                             # Temperature parameter for entropy
        'epsilon': 0.2,                                   # Epsilon for PPO
        'gamma': 0.99,                                    # The discounting factor
        'lr1': 1e-2,                                      # A first learning rate
        'lr2': 1e-2,                                      # A second learning rate (equal to the first one if None)
        'hidden_conv_layers': [],                         # A list of parameters ((nb of filters, size of filter)) for each hidden convolutionnal layer
        'hidden_dense_layers': [128, 64],             # A list of parameters (nb of neurons) for each hidden dense layer
        'verbose': True                                   # Verbose to follow training
    }
    # We create a Simulation object
    sim = Simulation(name_of_environment="LunarLanderContinuous-v2", nb_stacked_frame=1, agent_params=agent_params)
    # We train the neural network
    sim.train(target_score=70, max_episodes=1000, process_average_over=100, test_every=200, test_on=5)
