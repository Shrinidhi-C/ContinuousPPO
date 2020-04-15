
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from collections import deque
import random
import itertools
import sys

tf.random.set_seed(1)

DELTA = 1e-10
NPY_SQRT1_2 = 1 / (2 ** 0.5)
NPY_PI = np.pi


def normalize(x):
    x -= x.mean()
    x /= (x.std() + DELTA)
    return x


class Agent():
    def __init__(self,
                 state_space_shape,              # The shape of the state space
                 action_space_upper_limits,      # The upper limits of the continuous action space
                 action_space_lower_limits,      # The lower limits of the continuous action space
                 epsilon=0.2,                    # Epsilon for PPO
                 temperature=0.001,              # Temperature for entropy regularization
                 gamma=0.99,                     # The discounting factor
                 lr1=1e-2,                       # A first learning rate
                 lr2=None,                       # A second learning rate
                 hidden_conv_layers=[],          # A list of parameters of for each hidden convolutionnal layer
                 hidden_dense_layers=[32],       # A list of parameters of for each hidden dense layer
                 verbose=False                   # A live status of the training
                 ):
        self.state_space_shape = state_space_shape
        self.action_space_upper_limits = action_space_upper_limits
        self.action_space_lower_limits = action_space_lower_limits
        self.action_space_size = len(action_space_upper_limits)
        self.covariance_matrix = np.diag((action_space_upper_limits - action_space_lower_limits) / 10)
        self.sigmas = np.sqrt(np.diag(self.covariance_matrix))
        self.log_sigmas = np.log(self.sigmas)
        self.epsilon = epsilon
        self.temperature = temperature
        self.gamma = gamma
        self.memory = deque()
        self.lr1 = lr1
        self.lr2 = lr1 if lr2 is None else lr2
        self.optimizer1 = tf.keras.optimizers.Adam(self.lr1)
        self.optimizer2 = tf.keras.optimizers.Adam(self.lr2)
        self.hidden_dense_layers = hidden_dense_layers
        self.hidden_conv_layers = hidden_conv_layers
        self.loss1 = - float('inf')
        self.loss2 = - float('inf')
        self.verbose = verbose
        # We build neural networks
        self.build_network()

    def get_base_architecture(self, inputs, network_name):
        x = inputs
        if len(self.state_space_shape) > 1:
            # Hidden Conv layers, relu activated
            for id_, c in enumerate(self.hidden_conv_layers):
                x = tf.keras.layers.Conv1D(filters=c[0], kernel_size=c[1], padding='same', activation='relu',
                                           kernel_initializer=tf.keras.initializers.he_normal(),
                                           name='%s_conv_%d' % (network_name, id_))(x)
            # We flatten before dense layers
            x = tf.keras.layers.Flatten(name='%s_flatten' % network_name)(x)

        for id_, d in enumerate(self.hidden_dense_layers):
            # Dense layers, relu activated
            x = tf.keras.layers.Dense(d, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(),
                                      name='%s_dense_%d' % (network_name, id_))(x)
        return x

    def build_network(self):
        """
        This function is used to build the neural networks needed for the given method
        """
        # We define the inputs of the neural network
        states = tf.keras.Input(shape=self.state_space_shape, name='states')  # The current state
        advantages = tf.keras.Input(shape=(1,), name='advantages')  # The advantage associated to the action

        x = self.get_base_architecture(states, 'actor')

        # One dense output layer, softmax activated (to get probabilities)
        means = [tf.keras.layers.Dense(1, activation='linear',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       name='mean_%d' % idx)(x) for idx in range(self.action_space_size)]

        self.policy = tf.keras.Model(inputs=states, outputs=means, name='Actor')  # One for predicting the probabilities

        self.actor = tf.keras.Model(inputs=[states, advantages], outputs=means)  # Actor for training

        self.actor.summary()

        def actor_loss(y_true, y_pred, idx):
            mu = y_pred
            action = y_true
            log_sigma = self.log_sigmas[idx]
            sigma = self.sigmas[idx]
            var = self.covariance_matrix[idx][idx]
            upper_limit = self.action_space_upper_limits[idx]
            lower_limit = self.action_space_lower_limits[idx]

            def safe_log(x):
                return K.log(tf.where(x > 0, x, DELTA))

            def cdf_gauss(a):
                x = a * NPY_SQRT1_2
                z = K.abs(x)
                half_erfc_z = 0.5 * tf.math.erf(z)
                return tf.where(
                    z < NPY_SQRT1_2,
                    0.5 + 0.5 * tf.math.erf(x),
                    tf.where(
                        x > 0,
                        1.0 - half_erfc_z,
                        half_erfc_z
                    )
                )

            def log_cdf_gauss(x):
                return tf.where(
                    x > 6,
                    -cdf_gauss(-x),
                    tf.where(
                        x > -14,
                        safe_log(cdf_gauss(x)),
                        -0.5 * K.square(x) - safe_log(-x) - 0.5 * K.log(2 * NPY_PI)
                    )
                )

            log_lik = tf.where(
                action < upper_limit,
                tf.where(
                    action > lower_limit,
                    -0.5 * K.log(2 * NPY_PI) - log_sigma - 0.5 * K.square(action - mu) / var,
                    log_cdf_gauss((lower_limit - mu) / sigma)
                ),
                log_cdf_gauss(-(upper_limit - mu) / sigma)
            )
            old_log_lik = K.stop_gradient(log_lik)
            entropy_contrib = self.temperature * old_log_lik
            ratio = K.exp(log_lik - old_log_lik)
            clipped_ratio = K.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
            return -K.mean(K.minimum(ratio * (advantages - entropy_contrib), clipped_ratio * (advantages - entropy_contrib)), keepdims=True)

        loss_dict = {'mean_%d' % idx: (lambda y_true, y_pred: actor_loss(y_true, y_pred, idx)) for idx in range(self.action_space_size)}
        loss_weights = [1 / self.action_space_size] * self.action_space_size

        self.actor.compile(loss=loss_dict,
                           optimizer=self.optimizer1,
                           loss_weights=loss_weights,
                           experimental_run_tf_function=False)

        x = self.get_base_architecture(states, 'critic')

        # One dense output layer, linear activated (to get value of state)
        values = tf.keras.layers.Dense(1, activation='linear',
                                       kernel_initializer=tf.keras.initializers.he_normal(), name='values')(x)

        self.critic = tf.keras.Model(inputs=states, outputs=values, name='Critic')  # Critic for training

        self.critic.summary()

        self.critic.compile(loss='mse', optimizer=self.optimizer2, experimental_run_tf_function=False)  # Compiling Critic for training

    def take_action(self, state, train=False):
        """
        This function is used by the agent to take an action depending on the current state
        """
        state = state[np.newaxis, :]
        means = [m.numpy()[0][0] for m in self.policy(state)]  # Sample gaussian distrib parameters
        actions = np.random.multivariate_normal(means, self.covariance_matrix)
        clipped_actions = np.clip(actions, self.action_space_lower_limits, self.action_space_upper_limits)  # We sample the action based on theseparameters of the gaussian distrib
        return clipped_actions

    def learn_end_ep(self):
        # We retrieve all states, actions and reward the agent got during the episode from the memory
        states, actions, rewards, next_states, dones = map(np.array, zip(*self.memory))
        # We process the states values with the critic network
        critic_values = self.critic(states).numpy()
        critic_next_values = self.critic(next_states).numpy()
        # We get the target reward
        targets = rewards + self.gamma * np.squeeze(critic_next_values) * np.invert(dones)
        # We get the advantage (difference between the discounted reward and the baseline)
        advantages = targets - np.squeeze(critic_values)
        # We normalize advantages
        advantages = normalize(advantages)
        # We train the two networks
        self.loss1 = self.actor.train_on_batch([states, advantages], {'mean_%d' % idx: actions[:, idx] for idx in range(self.action_space_size)})[0]
        self.loss2 = self.critic.train_on_batch(states, targets)
        self.memory.clear()

    def print_verbose(self, ep, total_episodes, episode_reward, rolling_score):
        if self.verbose == True:
            print('Episode {:3d}/{:5d} | Current Score ({:.2f}) Rolling Average ({:.2f}) | Actor Loss ({:.4f}) Critic Loss ({:.4f})'.format(ep +
                                                                                                                                            1, total_episodes, episode_reward, rolling_score, self.loss1, self.loss2), end="\r")
