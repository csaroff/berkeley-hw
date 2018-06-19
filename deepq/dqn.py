# Imports and initializations
import gym
from array2gif import write_gif
import sys
import inspect


import numpy as np
import tensorflow as tf

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.python.client import timeline


from baselines.deepq.replay_buffer import ReplayBuffer

from math import inf
from timeit import default_timer as timer

import random

import baselines.common.tf_util as U

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# Neural network for Q(s,a) -> r
# Q is typically considered as having two inputs, the state and action, and
# returns one scalar output; the expected reward received across all future
# state action pairs.  In high cardinality action spaces, this "direct"
# representation is problematic since the q function is used to predict the
# optimal action to perform in a given state.  This "argmax" of Q given s
# requires applying the q function to every possible action.  This approach is
# computationally expensive and will not scale to large action spaces.

# This alternative approach, described in the atari deepmind paper, and further
# articulated here
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26#5d63
# has the network take only a state as input and return a vector of rewards, r
# where r[i] represents the reward for taking action i in the state s. There is
# obvious difficulty extending this approach to any infinite action space, but
# we will refrain from addressing this here.
def q_nn(obs_shape, num_actions):
    # With the functional API we need to define the inputs.
    obs_input = Input(obs_shape, name='observations')
    actions_input = Input((num_actions,), name='actions_mask')

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    hidden = Dense(128, activation='relu')(obs_input)

    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = Dense(num_actions)(hidden)

    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.merge([output, actions_input], mode='mul')

    model = Model(input=[obs_input, actions_input], output=filtered_output)
    model.compile(loss=tf.losses.huber_loss, optimizer='adam', metrics=['accuracy'])
    return model

def fit_batch(model, target_model, num_actions, gamma, batch, tensorboard):
    (obs, acts, rewards, new_obs, is_dones) = batch
    # np.one_hot(acts, num_actions)
    acts = np.eye(num_actions)[acts]

    # Predict the value of the q function in the next state for each action
    # And take the outcome for the best action
    q_tp1 = target_model.predict_on_batch([new_obs, np.ones(acts.shape)])

    q_tp1_best = np.amax(q_tp1, axis=1)

    # If is_done, there is no expected future reward because our current episode
    # is finished.  This should "ground" the model since this is the only fully
    # correct(non-estimated) q value.
    q_tp1_best = (1.0 - is_dones) * q_tp1_best

    # Add actual current reward to expected future reward to generate the
    # targets for the training examples
    q_t_best = rewards + gamma * q_tp1_best
    q_true = np.reshape(q_t_best, [-1, 1]) * acts

    model.train_on_batch([obs, acts], q_true)

def clone_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model')
    return load_model('tmp_model', custom_objects={'huber_loss': tf.losses.huber_loss})

# Gameplan
# ===
#
# 1. Generate training examples by following our current policy.
# Training examples come in the form (s, a, r, s')
# 2. Add them to the replay buffer.  The buffer is required in order to
# decorelate our training examples.
# 3. Sample from the replay buffer to improve our new policy.
# 4. Periodically rotate the new policy in as the current policy
def learn(env,
          max_timesteps=100000,
          buffer_size=100000,
          epsilon_decay=0.9999,
          epsilon_minimum=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=10,
          checkpoint_freq=100,
          learning_starts=1000,
          gamma=0.999,
          target_network_update_freq=500,
          episode_render_freq=None,
          log_dir='./tensorboard'):

        tensorboard = TensorBoard(log_dir=log_dir + '/' + env.spec.id)

        epsilon = lambda t: max(epsilon_decay ** t, epsilon_minimum)

        replay_buffer = ReplayBuffer(buffer_size)
        num_actions = env.action_space.n

        # Here, we'll use a simple feed forward nn for representing
        # Q(s) -> [r_1, r_2, ..., r_n] where r_k is the reward for taking action
        # `k` in state `s`
        model = q_nn(env.observation_space.shape, num_actions)
        target_model = clone_model(model)

        # Keep some state about the current episode
        num_episodes = 0
        episode_total_reward = 0
        episode_timesteps = 0
        num_epiosodes = 0
        last_checkpoint_mean_reward = -inf
        last_episode_mean_reward = -inf

        # Start off with a fresh environment
        obs = env.reset()

        # Play breakout for max_timesteps
        for t in range(max_timesteps):
            # With probability epsilon, take a random action
            if(random.uniform(0, 1) < epsilon(t)):
                action = env.action_space.sample()
            else:
                observations = np.reshape(obs, [1, -1])
                actions = np.reshape(np.ones(num_actions), [1,-1])
                q_values = model.predict_on_batch([observations, actions])
                action = np.argmax(q_values, axis=1)[0]

            # Collect observations and store them for replay
            new_obs, reward, is_done, _ = env.step(action)
            replay_buffer.add(obs, action, reward, new_obs, is_done)
            obs = new_obs

            # Update logging info
            episode_total_reward += reward
            episode_timesteps += 1

            if t > learning_starts and t % train_freq == 0:
                start = timer()
                fit_batch(model, target_model, num_actions, gamma, replay_buffer.sample(batch_size), tensorboard)
                # print('Training for timestep ', t, ' took', timer() - start)
                # print('End timestep ', t, '-------------------------------')

            if t > learning_starts and t % target_network_update_freq == 0:
                target_model = model
                model = clone_model(model)
                print('Setting model to target model')

            if is_done and num_episodes % print_freq == 0:
                print("timesteps", t)
                print("episodes run", num_episodes)
                print("last episode reward", episode_total_reward)
                print("% time spent exploring", int(100 * epsilon(t)))

            if t % checkpoint_freq == 0 and last_episode_mean_reward > last_checkpoint_mean_reward:
                print("Saving model due to mean reward increase: ", last_checkpoint_mean_reward, " -> ", last_episode_mean_reward)
                model.save('models/' + env.spec.id + '_deepq.h5py')
                last_checkpoint_mean_reward = last_episode_mean_reward

            if is_done:
                obs = env.reset()
                last_episode_mean_reward = episode_total_reward / episode_timesteps
                episode_total_reward = 0
                episode_timesteps = 0
                num_episodes += 1

            if episode_render_freq != None and num_episodes % episode_render_freq == 0:
                env.render()

            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('timeline.ctf.json', 'w') as f:
                f.write(trace.generate_chrome_trace_format())

def play(env, model):
    for episode in range(sys.maxsize):
        obs, done = env.reset(), False
        episode_rew = 0
        frames = []
        while not done:
            frames.append(env.render(mode='rgb_array'))
            observations = np.reshape(obs, [1, -1])
            actions = np.reshape(np.ones(env.action_space.n), [1, -1])
            rewards = model.predict_on_batch([observations, actions])[0]
            action = np.argmax(rewards)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        filename = 'gifs/' + env.spec.id + '-' + str(episode) + '.gif'
        write_gif(np.asarray(frames), filename, fps=5)
        print("Episode reward", episode_rew)

def main():
    # envname = 'CartPole-v0'
    env = gym.make('Breakout-ram-v0')

    learn(env, max_timesteps=1000000, buffer_size=1000000)
    # play(env, load_model('models/' + env.spec.id + '_deepq.h5py', custom_objects={'huber_loss': tf.losses.huber_loss}))

if __name__ == '__main__':
    main()
