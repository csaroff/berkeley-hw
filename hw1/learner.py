#!/usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from sample_with_policy import sample_with_policy
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
import argparse

from sklearn.model_selection import train_test_split


# Loads args
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str, help='Enter a valid model: \
                        Humanoid, HalfCheetah, Hopper, Reacher, Ant, Walker2d')

    parser.add_argument('--render', action='store_true', help='Render')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()
    # v2 environment for updated gym dependency
    args.envname = args.envname + '-v2'

    return args

# Simple Feedforward Neural Network
def simple_nn(obs_size, act_size):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(obs_size, )))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(act_size, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    args = parse()

    print('Loading and building policy ...')
    expert_policy_fn = load_policy.load_policy('experts/' + args.envname + '.pkl')
    print('Loaded and built policy')

    # max_timesteps = 20

    # Choose whether to generate new observations or load them in
    # obs_data, action_data = load_pkl(args)
    obs_data, action_data, r_data = sample_with_policy(
        expert_policy_fn, args.envname, False, args.max_timesteps, 100)

    obs_train, obs_test, act_train, act_test = train_test_split(
        obs_data, action_data, test_size=0.2, shuffle=True)

    model = simple_nn(obs_data.shape[1], action_data.shape[1])
    model.fit(obs_train, act_train, batch_size=64, epochs=40, verbose=1)
    print('test loss', model.evaluate(obs_test, act_test, verbose=1))
    model.save('models/' + args.envname + '_BC_' + str(args.num_rollouts) + '.h5py')
    print('Saved model: ', args.envname)

    model = load_model('models/' + args.envname + '_BC_' + str(args.num_rollouts) + '.h5py')

    learned_policy = lambda example: model.predict(example)

    new_obs, new_act, new_returns = sample_with_policy(
        learned_policy, args.envname, args.render, args.max_timesteps, args.num_rollouts)

