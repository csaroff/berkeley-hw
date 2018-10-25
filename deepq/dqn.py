# Imports and initializations
import sys
import time
import gym
from gym.spaces import Discrete, Box

import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from tensorflow.python.client import timeline

from skimage.color import rgb2gray
from skimage.transform import resize

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Input, LSTM, Lambda, Flatten, Reshape
from keras.layers.convolutional import Convolution2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, RMSprop
from keras import backend as K
K.set_image_dim_ordering('th')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.34
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from baselines.deepq.replay_buffer import ReplayBuffer
import baselines.common.tf_util as U

from math import inf
import random

from array2gif import write_gif

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom=True)
run_metadata = tf.RunMetadata()

PREPROC_OBS_SHAPE = (84, 84)

def write_log(writer, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, batch_no)
        writer.flush()

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

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
def q_nn(obs_space, num_actions, obs_hist_len):
    obs_shape = obs_space.shape

    if isinstance(obs_space, Box) and len(obs_shape) == 3:
        # Stack four frames of observations at a time so that each observation is actually a video
        # AUGMENTED_OBS_SHAPE = (obs_hist_len, obs_shape[0] // 2, obs_shape[1] // 2)
        AUGMENTED_OBS_SHAPE = (4,) + (PREPROC_OBS_SHAPE)
        # 4, 105, 80
        obs_input = Input(AUGMENTED_OBS_SHAPE, name='observations');
        # Normalize 0-255 to 0-1
        normalized = Lambda(lambda x: x / 255.0)(obs_input)
        conv_1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(normalized)
        conv_2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
        conv_3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv_2)
        conv_flattened = Flatten()(conv_3)
        hidden = Dense(512, activation='relu')(conv_flattened)
    elif isinstance(obs_space, Box) and len(obs_shape) == 1:
        input_shape = (obs_hist_len, obs_shape[0])
        obs_input = Input(input_shape, name='1dim-inputs')
        hidden = Reshape((-1,), input_shape=input_shape)(obs_input)
        hidden = Dense(128, activation='relu')(hidden)
        # hidden = Dense(128, input_shape=(obs_hist_len, obs_shape[0]), activation='relu')(obs_input)
    else:
        raise NotImplementedError('Unknown observation space')

    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = Dense(num_actions)(hidden)

    # A one_hot encoded action input. Adding this directly into our model will
    # make it a lot easier to train our model with keras
    actions_input = Input((num_actions,), name='actions_mask')
    filtered_output = keras.layers.merge([output, actions_input], mode='mul')

    # Define q as a neural net and compile it
    model = Model(input=[obs_input, actions_input], output=filtered_output)
    model.compile(RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), loss=huber_loss, metrics=['accuracy'], options=run_options, run_metadata=run_metadata)
    # model.compile(RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), loss='logcosh', metrics=['accuracy'], options=run_options, run_metadata=run_metadata)
    # model.compile(loss=tf.losses.huber_loss, optimizer='adam', metrics=['accuracy'], options=run_options, run_metadata=run_metadata)
    return model

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def _preprocess(img):
    return np.uint8(resize(rgb2gray(img), PREPROC_OBS_SHAPE, mode='constant') * 255)

def fit_batch(model, target_model, num_actions, discount_factor, batch, tensorboard, batch_no):
    (obs, acts, rewards, new_obs, is_done) = batch
    obs = np.array(obs)
    new_obs = np.array(new_obs)
    # np.one_hot(acts, num_actions)
    acts = np.eye(num_actions)[acts]

    # Predict the value of the q function in the next state for each action
    # And take the outcome for the best action
    q_tp1 = target_model.predict_on_batch([new_obs, np.ones(acts.shape)])

    q_tp1_best = np.amax(q_tp1, axis=1)

    # If is_done, there is no expected future reward because our current episode
    # is finished.  This should "ground" the model since this is the only fully
    # correct(non-estimated) q value.
    q_tp1_best = (1.0 - is_done) * q_tp1_best

    # Add actual current reward to expected future reward to generate the
    # targets for the training examples
    q_t_best = rewards + discount_factor * q_tp1_best
    q_true = np.reshape(q_t_best, [-1, 1]) * acts

    logs = model.train_on_batch([obs, acts], q_true)
    write_log(tensorboard, ['train_loss', 'train_mae'], logs, batch_no)
    if batch_no % 10 == 0:
        logs = model.test_on_batch([obs, acts], q_true)
        write_log(tensorboard, ['val_loss', 'val_mae'], logs, batch_no // 10)
    # model.fit([obs, acts], q_true, batch_size=32, verbose=0, callbacks=[tensorboard])

def clone_model(model):
    """Returns a copy of a keras model."""
    # # Must checkpoint model to avoid OOM https://github.com/keras-team/keras/issues/5345
    # clone = Model.from_config(model.get_config())
    # clone.set_weights(model.get_weights())
    # clone.compile(loss=tf.losses.huber_loss, optimizer='adam', metrics=['accuracy'], options=run_options, run_metadata=run_metadata)
    # return clone
    model.save('tmp_model')
    return load_model('tmp_model', custom_objects={'huber_loss':huber_loss})

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
          max_timesteps=50000000,
          # Human level control hyperparameters
          batch_size=32,
          buffer_size=1000000,
          agent_history_length=4,
          target_network_update_freq=10000,
          discount_factor=0.99,
          # "action repeat" handled by gym environment(equivalent to frame skip)
          train_freq=4, # agent "update frequency" in human level control paper
          initial_exploration_rate=1,
          final_exploration_rate=0.1,
          final_exploration_frame=1000000,
          replay_start_size=50000,
          print_freq=10,
          checkpoint_freq=100,
          episode_render_freq=None,
          log_dir='./tensorboard',
          start_from_checkpoint=False):

        tensorboard = TensorBoard(log_dir=log_dir + '/' + env.spec.id)
        writer = tf.summary.FileWriter(log_dir + '/' + env.spec.id)

        # Linear decay as used in the deepmind paper
        epsilon = lambda t: max(initial_exploration_rate - (t/final_exploration_frame), final_exploration_rate)
        # epsilon = lambda t: max(epsilon_decay ** t, epsilon_minimum)

        preprocess = _preprocess if len(env.observation_space.shape) == 3 else lambda x: x

        replay_buffer = ReplayBuffer(buffer_size)
        num_actions = env.action_space.n

        # Here, we'll use a simple feed forward nn for representing
        # Q(s) -> [r_1, r_2, ..., r_n] where r_k is the reward for taking action
        # `k` in state `s`
        if start_from_checkpoint:
            model = load_model('tmp_model', custom_objects={'huber_loss':huber_loss})
        else:
            model = q_nn(env.observation_space, num_actions, agent_history_length)
        target_model = clone_model(model)

        # Keep some state about the current episode
        num_episodes = 0
        episode_total_reward = 0
        episode_timesteps = 0
        episode_rewards = [0.0]

        last_checkpoint_mean_reward = -inf
        mean_100ep_reward = -inf

        # Start off with a fresh environment
        ob = preprocess(env.reset())
        obs = [ob for i in range(agent_history_length)]

        # Play breakout for max_timesteps
        for t in range(max_timesteps):
            # With probability epsilon, take a random action
            if(random.uniform(0, 1) < epsilon(t)):
                action = env.action_space.sample()
            else:
                observations = np.array([obs])
                actions = np.reshape(np.ones(num_actions), [1,-1])
                q_values = model.predict_on_batch([observations, actions])
                action = np.argmax(q_values, axis=1)[0]

            # Collect observations and store them for replay
            new_ob, reward, is_done, info = env.step(action)
            is_done = info['ale.lives'] != 5
            new_obs = list(obs)
            new_obs.pop(0)
            new_obs.append(preprocess(new_ob))

            replay_buffer.add(obs, action, reward, new_obs, is_done)
            obs = new_obs

            # Update logging info
            episode_total_reward += reward
            episode_timesteps += 1

            if t > replay_start_size and t % train_freq == 0:
                fit_batch(model, target_model, num_actions, discount_factor, replay_buffer.sample(batch_size), writer, t // train_freq)

            if t > replay_start_size and t % target_network_update_freq == 0:
                # Must checkpoint model and clear sess to avoid OOM https://github.com/keras-team/keras/issues/5345
                model.save('tmp_model')
                K.clear_session()
                target_model = load_model('tmp_model', custom_objects={'huber_loss':huber_loss})
                model = load_model('tmp_model', custom_objects={'huber_loss':huber_loss})
                print('Setting model to target model')

            if is_done:
                ob = preprocess(env.reset())
                obs = np.array([ob for i in range(agent_history_length)])
                episode_timesteps = 0
                num_episodes += 1
                episode_rewards.append(episode_total_reward)
                episode_total_reward = 0
                if len(episode_rewards) > 100:
                    episode_rewards.pop(0)
                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)

            if is_done and num_episodes % print_freq == 0:
                print("timesteps", t)
                print("episodes run", num_episodes)
                print("last episode reward", episode_rewards[-1])
                print("mean_100ep_reward", mean_100ep_reward)
                print("% time spent exploring", int(100 * epsilon(t)))

            if t % checkpoint_freq == 0 and mean_100ep_reward > last_checkpoint_mean_reward:
                print("Saving model due to mean reward increase: ", last_checkpoint_mean_reward, " -> ", mean_100ep_reward)
                model.save('models/' + env.spec.id + '_deepq.h5py')
                last_checkpoint_mean_reward = mean_100ep_reward

            if episode_render_freq != None and num_episodes % episode_render_freq == 0:
                env.render()

def play(env, model, agent_history_length=4):
    preprocess = _preprocess if len(env.observation_space.shape) == 3 else lambda x: x
    print('Begin play')
    for episode in range(sys.maxsize):
        ob, done = preprocess(env.reset()), False
        obs = [ob for i in range(agent_history_length)]
        episode_rew = 0
        frames = []
        timesteps = 0
        ale_lives = 6
        while not done:
            timesteps += 1
            # time.sleep(.02)
            # env.render()

            frames.append(env.render(mode='rgb_array'))
            observations = np.array([obs])
            actions = np.reshape(np.ones(env.action_space.n), [1, -1])
            rewards = model.predict_on_batch([observations, actions])[0]
            action = np.argmax(rewards)
            ob, rew, done, info = env.step(action)
            if info['ale.lives'] < ale_lives:
                ale_lives = info['ale.lives']
                ob, rew, done, info = env.step(1)
            episode_rew += rew
            obs.pop(0)
            obs.append(preprocess(ob))

        filename = 'gifs/' + env.spec.id + '-' + str(episode) + '.gif'
        write_gif(np.asarray(frames), filename, fps=20)
        print("Episode reward", episode_rew)
        exit(1)

def main():
    env = gym.make('BreakoutDeterministic-v4')
    # learn(env)

    # play(env, load_model('models/' + env.spec.id + '_deepq_50mil.h5py', custom_objects={'huber_loss':huber_loss}), agent_history_length=4)
    play(env, load_model('models/' + env.spec.id + '_deepq.h5py', custom_objects={'huber_loss':huber_loss}), agent_history_length=4)


if __name__ == '__main__':
    main()
