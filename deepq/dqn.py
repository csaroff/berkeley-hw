# Imports and initializations
import gym
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import TensorBoard

from baselines.deepq.replay_buffer import ReplayBuffer

from math import inf
from timeit import default_timer as timer

import random

# Neural network for Q(s,a) -> r
# Q is typically considered as having two inputs, the state and actions, and
# returns one scalar output, the expected reward received across all future
# state action pairs.  In high cardinality action spaces, this "direct"
# representation is problematic since the q function is used to predict the
# optimal action to perform in a given state.  This "argmax" of Q given s
# requires applying the q function to every possible action.  This approach
# will not scale to large action spaces.

# This alternative approach, described in the atari deepmind paper, and further
# articulated here
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26#5d63
# has the network take only a state as input and return a vector of rewards, r
# where r[i] represents the reward for taking action i in the state s. There is
# obvious difficulty extending this approach to any infinite action space, but
# we will refrain from addressing this here.
def simple_nn(obs_shape, num_actions):
    print('Creating model with observation shape', obs_shape)
    print('Creating model with ', num_actions, 'actions')
    model = Sequential()
    print('input_shape', obs_shape)
    model.add(Dense(8, activation='relu', input_shape=obs_shape))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss=tf.losses.huber_loss, optimizer='adam', metrics=['accuracy'])
    return model

# fit target_model using model as the predictor for future rewards
def fit_batch(model, target_model, num_acts, gamma, batch, tensorboard):
    # with tf.Session() as sess:
    (obs, acts, rewards, new_obs, is_dones) = batch
    # Predict the value of the q function in the next state for each action
    # And take the outcome for the best action
    predict_q_tp1_time = timer()
    q_tp1 = model.predict(new_obs)
    print('predict_q_tp1_time', timer() - predict_q_tp1_time)

    q_tp1_best_time = timer()
    q_tp1_bests = tf.reduce_max(q_tp1, 1)
    print('predict best q_tp1 time', timer() - q_tp1_best_time)

    # If is_done, there is no expected future reward because our current episode
    # is finished.  This should "ground" the model since this is the only fully
    # correct(non-estimated) q value.
    eoe_reward_time = timer()
    q_tp1_bests = (1.0 - is_dones) * q_tp1_bests
    print('end of episode reward time', timer() - eoe_reward_time)

    # Add actual current reward to expected future reward to generate the
    # targets for the training examples
    agg_reward_time = timer()
    q_t_bests = rewards + gamma * q_tp1_bests
    print('aggregate reward time', timer() - agg_reward_time)

    # Compute an additional forward pass to estimate the q values for the
    # actions that were not taken in order to avoid computing the error manually
    pred_q_t_time = timer()
    q_targets = model.predict(obs)
    print('predict q(t)', timer() - pred_q_t_time)

    # Replace predicted reward with observed reward
    q_t_replace_time = timer()
    q_targets = q_targets * tf.one_hot(acts, num_acts, on_value=0.0, off_value=1.0)
    q_targets = q_targets + tf.reshape(q_t_bests, [-1, 1]) * tf.one_hot(acts, num_acts, on_value=1.0, off_value=0.0)
    print('Replace pred reward with obs reward time', timer() - q_t_replace_time)

    # Fit the model using training examples
    fit_model_time = timer()
    model.fit(obs, q_targets, epochs=1, steps_per_epoch=1, verbose=True, callbacks=[tensorboard])
    print('Model fit time', timer() - fit_model_time)

def clone_model(model):
    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    clone.compile(loss=tf.losses.huber_loss, optimizer='adam', metrics=['accuracy'])
    return clone

# Gameplan
# ===
#
# 1. Generate training examples by following our current policy.
# Training examples come in the form (s, a, r, s')
# 2. Add them to the replay buffer.  The buffer is required in order to
# decorelate our training examples.
# 3. Sample from the replay buffer to improve our new policy.
# 4. Periodically rotate the new policy in as the current policy
def learn(envname,
          learning_rate=5e-4,
          max_timesteps=100000,
          buffer_size=100000,
          epsilon_decay=0.999,
          train_freq=1,
          batch_size=32,
          print_freq=10,
          checkpoint_freq=100,
          learning_starts=1000,
          gamma=0.99,
          target_network_update_freq=500,
          episode_render_freq=None,
          log_dir='./tensorboard'):

        tensorboard = TensorBoard(log_dir=log_dir + '/' + envname)

        # Create a breakout environment
        env = gym.make(envname)

        epsilon = lambda t: epsilon_decay ** t

        replay_buffer = ReplayBuffer(buffer_size)
        num_actions = env.action_space.n

        # Here, we'll use a simple feed forward nn for representing
        # Q(s) -> [r_1, r_2, ..., r_n] where r_k is the reward for taking action
        # `k` in state `s`
        model = simple_nn(env.observation_space.shape, num_actions)
        target_model = clone_model(model)

        # The current timestep, t
        # t = 0

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
                q_values = model.predict(observations)
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
                print('Training for timestep ', t, ' took', timer() - start)
                print('End timestep ', t, '-------------------------------')

            if t > learning_starts and t % target_network_update_freq == 0:
                model = target_model
                target_model = clone_model(model)
                print('Setting model to target model')

            if is_done and num_episodes % print_freq == 0:
                print("timesteps", t)
                print("episodes run", num_episodes)
                print("last episode reward", episode_total_reward)
                print("% time spent exploring", int(100 * epsilon(t)))

            if t % checkpoint_freq == 0 and last_episode_mean_reward > last_checkpoint_mean_reward:
                print("Saving model due to mean reward increase: ", last_checkpoint_mean_reward, " -> ", last_episode_mean_reward)
                model.save('models/' + envname + '_deepq.h5py')
                last_checkpoint_mean_reward = last_episode_mean_reward

            if is_done:
                obs = env.reset()
                last_episode_mean_reward = episode_total_reward / episode_timesteps
                episode_total_reward = 0
                episode_timesteps = 0
                num_episodes += 1

            if episode_render_freq != None and num_episodes % episode_render_freq == 0:
                env.render()

def main():
    # with tf.Session() as sess:
    #     # sess.run(tf.global_variables_initializer())
    #     # sess.run(tf.local_variables_initializer())
    #     # learn('Breakout-ram-v0')
    learn('CartPole-v0')

# env.unwrapped.get_action_meanings()

if __name__ == '__main__':
    main()