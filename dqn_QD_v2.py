"""
    This is dqn which has Q and D estimator.
    Different from V1, we give an agent penalty(reward) equal 1.0 at terminal state.
    More reward, more chance to dies (because of discount factor).
    The agent must choose the action that provides the greatest U-value.
    where
            U   =   Q - bD              ; b is constant  (change plus sign to minus)
"""

import os
import time
import random
import cv2
import argparse
import numpy as np
import tensorflow as tf

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from utils import *


class Model(object):
    def __init__(self, num_actions, death_constant, risk_constant, sess=None, multi_gpu=False, no_gpu=False):
        if multi_gpu:
            device_1 = '/device:GPU:0'
            device_2 = '/device:GPU:1'
        else:
            device_1 = '/device:GPU:0'
            device_2 = '/device:GPU:0'

        if no_gpu:
            device_1 = '/cpu:0'
            device_2 = '/cpu:0'

        with tf.variable_scope('main'):
            with tf.device(device_1):
                # make it int32 and divide by 255.
                self.x = x = tf.placeholder(tf.uint8, [None, 84, 84, 4], name="input")
                self.batch_size = tf.shape(x)[0]  # = (nenvs) if perform action || = (args.bs) if train replay buffer
                x = tf.cast(x, tf.float32)/255.

                # convolution layer
                x = tf.nn.relu( conv2d(x, 32, "l1", [8,8], [4,4]) )
                x = tf.nn.relu( conv2d(x, 64, "l2", [4,4], [2,2]) )
                conv_out = tf.nn.relu( conv2d(x, 64, "l3", [3,3], [1,1]) )

                # fully connected layer
                x = tf.nn.relu(linear(flatten(conv_out), 512, "hidden", normalized_columns_initializer(1.0)))
                self.q_values = linear(x, num_actions, "q_out", normalized_columns_initializer(1.0))

                # death predictor
                x = tf.nn.relu(linear(flatten(conv_out), 512, "hidden2", normalized_columns_initializer(1.0)))
                self.d_values = linear(x, num_actions, "d_out", normalized_columns_initializer(1.0))

                # utility value
                x = tf.nn.relu(linear(flatten(conv_out), 512, "hidden3", normalized_columns_initializer(1.0)))
                self.u_values = linear(x, num_actions, "u_out", normalized_columns_initializer(1.0))

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        with tf.variable_scope('target'):
            with tf.device(device_2):
                self.x_target = x = tf.placeholder(tf.uint8, [None, 84, 84, 4], name="input_target")
                x = tf.cast(x, tf.float32)/255.

                # convolution layer
                x = tf.nn.relu( conv2d(x, 32, "l1", [8,8], [4,4]) )
                x = tf.nn.relu( conv2d(x, 64, "l2", [4,4], [2,2]) )
                conv_out = tf.nn.relu( conv2d(x, 64, "l3", [3,3], [1,1]) )

                # fully connected layer
                x = tf.nn.relu(linear(flatten(conv_out), 512, "hidden", normalized_columns_initializer(1.0)))
                self.q_values_target = linear(x, num_actions, "q_out", normalized_columns_initializer(1.0))

                # death predictor
                x = tf.nn.relu(linear(flatten(conv_out), 512, "hidden2", normalized_columns_initializer(1.0)))
                self.d_values_target = linear(x, num_actions, "d_out", normalized_columns_initializer(1.0))

                # utility value
                x = tf.nn.relu(linear(flatten(conv_out), 512, "hidden3", normalized_columns_initializer(1.0)))
                self.u_values_target = linear(x, num_actions, "u_out", normalized_columns_initializer(1.0))

            self.var_list_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        self.sync = tf.group(
                    *(
                        [v1.assign(v2) for v1, v2 in zip(self.var_list_target, self.var_list)]
                    ))

        # act greedily with respect to the utility
        # add small random action to avoid stranded behaviour

        self.eps = eps = tf.placeholder(tf.float32, [1])
        # eps = 0.01
        deterministic_actions = tf.argmax(self.u_values, axis=1)  # greedy
        random_actions = tf.random_uniform(tf.stack([self.batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([self.batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        self.actions = tf.where(chose_random, random_actions, deterministic_actions)

        # train
        self.rewards_t = tf.placeholder(tf.float32, [None], name="reward")
        self.actions_t = tf.placeholder(tf.int32, [None], name="action")
        self.done_mask = tf.placeholder(tf.float32, [None], name="done")

        #######################################################################
        # This section compute Q-learning target and error
        #######################################################################

        # q scores for actions, we know were selected
        q_t_selected = tf.reduce_sum(self.q_values * tf.one_hot(self.actions_t, num_actions), 1)
        # target
        q_target = self.q_values_target
        u_target = self.u_values_target
        # select best action with utility head (similar to double dqn)
        tp1_best_actions = tf.argmax(u_target, 1)

        q_tp1_best = tf.reduce_sum(q_target * tf.one_hot(tp1_best_actions, num_actions), 1)
        q_tp1_best_masked = (1.0 - self.done_mask) * q_tp1_best

        # compute RHS of bellman equation
        gamma = 0.995  # was 0.99
        q_t_selected_target = self.rewards_t + gamma * q_tp1_best_masked

        # compute error
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)

        # huber loss
        delta = 1.0
        errors = tf.where(tf.abs(td_error) < delta,
                          tf.square(td_error) * 0.5,
                          delta * (tf.abs(td_error) - 0.5 * delta))

        errors = tf.reduce_mean(errors)

        ##################################################################
        # This section compute Death estimation
        ###################################################################

        # q scores for actions, we know were selected
        self.d_t_selected = tf.reduce_sum(self.d_values * tf.one_hot(self.actions_t, num_actions), 1)
        # target
        d_target = self.d_values_target

        d_tp1_best = tf.reduce_sum(d_target * tf.one_hot(tp1_best_actions, num_actions), 1)
        d_tp1_best_masked = (1.0 - self.done_mask) * d_tp1_best

        # define death reward
        r = self.done_mask

        # compute RHS of bellman equation
        gamma = 0.995  # was 0.99
        d_t_selected_target = r + gamma * d_tp1_best_masked

        # compute error
        d_td_error = self.d_t_selected - tf.stop_gradient(d_t_selected_target)

        # huber loss
        delta = 1.0
        d_errors = tf.where(tf.abs(d_td_error) < delta,
                          tf.square(d_td_error) * 0.5,
                          delta * (tf.abs(d_td_error) - 0.5 * delta))

        d_errors = tf.reduce_mean(d_errors)

        ####################################################################
        # This section compute utility value target and error
        ####################################################################
        b = death_constant  # death constant
        u_t_selected = tf.reduce_sum(self.u_values * tf.one_hot(self.actions_t, num_actions), 1)
        u_t_selected_target = q_t_selected_target - (b * d_t_selected_target)

        # compute error
        u_td_error = u_t_selected - tf.stop_gradient(u_t_selected_target)
        # huber loss
        u_errors = tf.where(tf.abs(u_td_error) < delta,
                          tf.square(u_td_error) * 0.5,
                          delta * (tf.abs(u_td_error) - 0.5 * delta))

        u_errors = tf.reduce_mean(u_errors)

        ###################################################################

        lr = 0.0001  # 1.0e-4
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.optimize_expr = optimizer.minimize(errors + d_errors + u_errors)

        self.sess = sess
        tf.global_variables_initializer().run(session=self.sess)

        tf.summary.scalar("model/q_loss", errors)
        tf.summary.scalar("model/d_loss", d_errors)
        tf.summary.scalar("model/u_loss", u_errors)
        tf.summary.scalar("model/mean_q_values", tf.reduce_mean(q_t_selected))
        tf.summary.scalar("model/mean_d_values", tf.reduce_mean(self.d_t_selected))
        tf.summary.scalar("model/mean_u_values", tf.reduce_mean(u_t_selected))
        self.summary_op = tf.summary.merge_all()

    def act(self, obs, epsilon):
        # sample an action
        return self.sess.run(self.actions,
                             feed_dict={self.x: obs, self.eps: epsilon})
                             # feed_dict={self.x : obs})


    def train(self, obses, actions, rewards, obses_tp1, dones):
        # train
        return self.sess.run([self.optimize_expr, self.summary_op],
                        feed_dict={self.x : obses,
                                   self.actions_t : actions,
                                   self.rewards_t : rewards,
                                   self.x_target : obses_tp1,
                                   self.done_mask : dones
                                  })

    def update_target(self):
        return self.sess.run(self.sync)


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size  # 200,000
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        # oldest data will be replaced
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def add_batch(self, obs_t, actions, rewards, obs_tp1, dones):
        for i in range(len(dones)):
            self.add(obs_t[i], actions[i], rewards[i], obs_tp1[i], dones[i])

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class DQN:
    def __init__(self,
              env,
              model,
              summary_writer,
              batch_size,
              max_timesteps,
              train_freq,
              learning_starts,
              target_network_update_freq,
              buffer_size,
              nstack,
              visualise=False,
              logdir=None,
              sess=None,
              saver=None,
              ):

        self.env = env
        self.nenvs = env.num_envs
        self.model = model
        nh, nw, nc = env.observation_space.shape

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Create the schedule for exploration starting from 1.
        exploration_fraction = 0.1
        self.exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                    initial_p=1.0,
                                    final_p=0.01)

        # Initialise observation
        self.obs = np.zeros((self.nenvs, nh, nw, nstack), dtype=np.uint8)
        obs = env.reset()
        self.update_obs(obs)
        self.next_obs = np.zeros((self.nenvs, nh, nw, nstack), dtype=np.uint8)
        self.update_next_obs(obs)
        self.tstart = time.time()
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq
        self.train_freq = train_freq
        self.summary_writer = summary_writer
        self.visualise = visualise
        self.saver = saver
        self.logdir = logdir
        self.previous_reward = 0.0
        self.sess = sess

    def update_obs(self, obs):
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def update_next_obs(self, obs):
        self.next_obs = np.roll(self.next_obs, shift=-1, axis=3)
        self.next_obs[:, :, :, -1] = obs[:, :, :, 0]

    def learn(self):

        # Initial setup
        episode_reward = 0.0
        episode_clip_reward = 0.0
        episode_length = 0.0

        for t in range( self.max_timesteps//self.nenvs + 1):
            # choose actions
            actions = self.model.act(self.obs, [self.exploration.value(int(t*self.nenvs))])
            # actions = self.model.act(self.obs)

            # act on env
            obs, rewards, dones, _ = self.env.step(actions)

            # clip rewards
            clip_rewards = np.sign(rewards)

            # Store transition in the replay buffer
            for n, done in enumerate(dones):
                if done:
                    self.next_obs[n] = self.next_obs[n]*0
            self.update_next_obs(obs)

            self.replay_buffer.add_batch(self.obs, actions, clip_rewards, self.next_obs, dones)

            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)

            if t > self.learning_starts and t % self.train_freq == 0:
                # Train network periodically
                train_obses_t, train_actions, train_rewards, \
                train_obses_tp1, train_dones = self.replay_buffer.sample(self.batch_size)
                _, summary = self.model.train(train_obses_t,
                                 train_actions,
                                 train_rewards,
                                 train_obses_tp1,
                                 train_dones)

                self.summary_writer.add_summary(summary, global_step=t*self.nenvs)
                self.summary_writer.flush()

            if t > self.learning_starts and t % self.target_network_update_freq == 0:
                # Update target network periodically.
                self.model.update_target()

            # collect summary
            episode_reward += rewards[0]
            episode_clip_reward += clip_rewards[0]
            episode_length += 1

            # Report summary
            if dones[0]:
                print("done : %d"%t)
                nseconds = time.time()-self.tstart
                fps = int((t*self.nenvs)/nseconds)
                # summary
                summary = tf.Summary()
                summary.value.add(tag='global/episode_reward', simple_value=episode_reward)
                summary.value.add(tag='global/episode_cliped_reward', simple_value=episode_clip_reward)
                summary.value.add(tag='global/episode_length', simple_value=episode_length)
                summary.value.add(tag='global/fps', simple_value=fps)
                self.summary_writer.add_summary(summary, global_step=t*self.nenvs)
                self.summary_writer.flush()

                # save best model
                if episode_reward > self.previous_reward:
                    self.saver.save(self.sess, self.logdir + "/best/best_model.ckpt")
                    self.previous_reward = episode_reward

                # reset episode_reward
                episode_reward = 0.0
                episode_clip_reward = 0.0
                episode_length = 0

            if t == self.max_timesteps//(2*self.nenvs):
                # save at half the training time
                self.saver.save(self.sess, self.logdir + "/half/half_model.ckpt")
                print('Save half training model')

            # visualisation for debugging process
            if self.visualise:
                vis = cv2.resize(obs[0,:,:,0] , (500,500))
                print(episode_reward)
                cv2.imshow('img', vis)
                cv2.waitKey(2)

        # save final model
        self.saver.save(self.sess, self.logdir + "/final/final_model.ckpt")
        print('Save final model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--b', help='death constant', type=float, default=1)
    parser.add_argument('--bs', help='batch size', type=int, default=512)
    parser.add_argument('--c', help='risk constant', type=float, default=0.3)
    parser.add_argument('--env', help='environment ID', default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--gpu_id', help='gpu device ID', default="0")
    parser.add_argument('--log_dir', help='experiment directory', default='./experiments')
    parser.add_argument('--multi_gpu', help='use multiple GPUs', action='store_true')
    parser.add_argument('--num_workers', help='number of workers', type=int, default=12)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--time_steps', help='max time step for training', type=int, default=int(100e6))
    parser.add_argument('--visualise', help='show game screen', action='store_true')
    args = parser.parse_args()
    print("Initialise environment...")

    # GPUs setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Experiment results
    filename = os.path.basename(__file__)[:-3]
    experiment_name = filename + "/" + args.env + '/b_' + str(args.b) + '/bs_' + str(args.bs) + '/seed_' + str(args.seed)
    log_dir = os.path.join(args.log_dir, experiment_name)

    # Create OpenNI atari-py function
    def make_env(rank):
        def _thunk():
            env = make_atari(args.env)
            env.seed(args.seed + rank)
            return wrap_deepmind(env, episode_life=False, clip_rewards=False)
        return _thunk

    # Create environments
    set_global_seeds(args.seed)
    env = SubprocVecEnv([make_env(i) for i in range(args.num_workers)])

    # Start tf session
    print("Starting session...")
    with tf.Session(config=config) as sess:
        # Create neural network models
        model = Model(num_actions=env.action_space.n,  # 18 actions for atari
                      death_constant=args.b,
                      risk_constant=args.c,
                      sess=sess,
                      multi_gpu=args.multi_gpu)

        # Start training
        summary_writer = tf.summary.FileWriter(log_dir)
        saver = tf.train.Saver()
        dqn = DQN(env,
                  model,
                  summary_writer,
                  batch_size=args.bs,
                  max_timesteps=args.time_steps,
                  train_freq=4,
                  learning_starts=10000,
                  target_network_update_freq=2000,
                  buffer_size=200000,
                  nstack=4,
                  visualise=args.visualise,
                  logdir=log_dir,
                  sess=sess,
                  saver=saver)

        print("Start!")
        print("========================================================")
        dqn.learn()

    env.close()
    print("========================================================")
    print("env closed!")
