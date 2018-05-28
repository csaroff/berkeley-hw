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
from keras.callbacks import TensorBoard
import argparse

from sklearn.model_selection import train_test_split

# Inputs for runs and model env
def load_pkl(envname, num_rollouts):
    filename = 'data/' + envname + str(num_rollouts) + '.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['obs'], data['act']

def save_pkl(obs, act, envname, num_rollouts):
    data = { 'obs': obs, 'act': act }
    filename = 'data/' + envname + str(num_rollouts) + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(ctx, obs_data, act_data):
    tensorboard = TensorBoard(log_dir='./tensorboard/' + ctx.algo + '_experiments/' + ctx.envname + '/' + str(ctx.run))

    obs_train, obs_test, act_train, act_test = train_test_split(
        obs_data, act_data, test_size=0.2, shuffle=False)
    ctx.model.fit(obs_train, act_train, batch_size=128, epochs=40, verbose=1, callbacks=[tensorboard])

    print('test loss', ctx.model.evaluate(obs_test, act_test, verbose=1))
    ctx.model.evaluate(obs_test, act_test, verbose=1)
    return ctx.model

def get_bc_model(ctx, use_cached_model=False, use_cached_examples=False):
    if(use_cached_model):
        return load_model('models/' + ctx.envname + '_BC_' + str(ctx.num_rollouts) + '.h5py')

    init_obs, init_act, init_r = sample_with_policy(
        policy_fn=expert_policy_fn,
        expert_policy_fn=None,
        envname=ctx.envname,
        render=False,
        max_timesteps=2,
        num_rollouts=1)

    model = simple_nn(init_obs.shape[1], init_act.shape[1])

    if(not use_cached_examples):
        print('Generating bc dataset')
        obs_data, act_data, r_data = sample_with_policy(
            policy_fn=expert_policy_fn,
            expert_policy_fn=None,
            envname=ctx.envname,
            render=False,
            max_timesteps=ctx.max_timesteps,
            num_rollouts=ctx.num_rollouts)

        print('Cacheing training examples')
        save_pkl(obs_data, act_data, ctx.envname, ctx.num_rollouts)

    else:
        obs_data, act_data, r_data = load_pkl(ctx.envname, ctx.num_rollouts)


    ctx.model = model
    ctx.algo = 'bc'
    ctx.run = 1
    model = train_model(ctx, obs_data, act_data)

    print('Cacheing model')
    model.save('models/' + ctx.envname + '_BC_' + str(ctx.num_rollouts) + '.h5py')

    return model

def generate_policies(args):
    # Generate policies
    model = get_bc_model(args)

    for i in range(args.num_rollouts):

        learned_policy = lambda example: model.predict(example)

        # dagger
        obs, acts, returns = sample_with_policy(
            policy_fn=learned_policy,
            expert_policy_fn=expert_policy_fn,
            envname=args.envname,
            render=False,
            max_timesteps=args.max_timesteps,
            num_rollouts=1)

        # update model
        args.model = model
        args.algo = 'dagger'
        args.run = i
        train_model(args, obs, acts)

        model.save('models/' + args.envname + '_DG_' + str(args.num_rollouts) + '.h5py')

def main():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        args = parse()

        # generate_policices(args)

        print('Loading and building policy ...')
        expert_policy_fn = load_policy.load_policy('experts/' + args.envname + '.pkl')

        bc_model = load_model('models/' + args.envname + '_BC_' + str(args.num_rollouts) + '.h5py')
        bc_policy_fn = lambda example: bc_model.predict(example)

        dagger_model = load_model('models/' + args.envname + '_DG_' + str(args.num_rollouts) + '.h5py')
        dagger_policy_fn = lambda example: dagger_model.predict(example)

        # Run the expert policy
        o, a, expert_returns = sample_with_policy(
            policy_fn=expert_policy_fn,
            expert_policy_fn=None,
            envname=args.envname,
            render=args.render,
            max_timesteps=args.max_timesteps,
            num_rollouts=args.num_rollouts)

        # Run the bc policy
        o, a, bc_returns = sample_with_policy(
            policy_fn=bc_policy_fn,
            expert_policy_fn=None,
            envname=args.envname,
            render=args.render,
            max_timesteps=args.max_timesteps,
            num_rollouts=args.num_rollouts)

        o, a, dagger_returns = sample_with_policy(
            policy_fn=dagger_policy_fn,
            expert_policy_fn=None,
            envname=args.envname,
            render=args.render,
            max_timesteps=args.max_timesteps,
            num_rollouts=args.num_rollouts)

        print('expert return for env', args.envname, ' ', np.sum(expert_returns))
        print('bc return for env', args.envname, ' ', np.sum(bc_returns))
        print('dagger return for env', args.envname, ' ', np.sum(dagger_returns))


if __name__ == '__main__':
    main()
