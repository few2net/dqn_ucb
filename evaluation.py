"""
For Evaluation of the saved Agents
and Visualisation
"""
import numpy as np
import tensorflow as tf
import argparse

import os
import cv2
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

# Select which Model to use
from dqn_QD_v2_2 import Model

import matplotlib.pyplot as plt

# Create figure for plotting
xt = []
yt = []


def realtime_plot(step, d_value):
    xt.append(step)
    yt.append(d_value)
    plt.cla()
    plt.plot(xt, yt)

    # Format plot
    plt.title('D_value over Time')
    plt.ylabel('D-value')
    plt.xlabel('step')
    plt.pause(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', help='model directory')
    parser.add_argument('--b', help='death constant', type=float, default=0)
    parser.add_argument('--c', help='risk constant', type=float, default=0)
    parser.add_argument('--env', help='environment ID', default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--gpu_id', help='gpu device ID', default="0")
    parser.add_argument('--num_ep', help='number of episode', type=int, default=100)
    parser.add_argument('--no_gpu', help='do not use gpu', action='store_true')
    parser.add_argument('--visualise', help='show game screen', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    name = "half_model.ckpt"
    path = os.path.join(args.dir, name)
    file = open(args.dir + "result.txt", 'w')

    # Build environment
    env = make_atari(args.env)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_writer = None

        model = Model(env.action_space.n,
                      death_constant=args.b,
                      risk_constant=args.c,
                      sess=sess,
                      no_gpu=args.no_gpu)

        saver = tf.train.Saver()
        saver.restore(sess, path)
        nh, nw, nc = env.observation_space.shape
        nstack = 4
        print('start evaluation !!')
        print("========================================================")

        eval_rewards = []
        for i in range(args.num_ep):
            ep_reward = 0
            current_ob = np.zeros((1, nh, nw, nstack), dtype=np.uint8)
            next_ob = np.zeros((1, nh, nw, nstack), dtype=np.uint8)
            ob = env.reset()
            current_ob = np.roll(current_ob, shift=-1, axis=3)
            current_ob[0, :, :, -1] = ob[:, :, 0]
            t = 0

            while True:
                # choose actions
                # action = model.act(current_ob, [0.1])
                action = model.act(current_ob, [0])
                ob, reward, done, _ = env.step(action)
                print("step %d: %s"%(t,action))

                next_ob = np.roll(next_ob, shift=-1, axis=3)
                next_ob[0, :, :, -1] = ob[:, :, 0]

                # print("d_target")
                # print("==========")
                d_value = sess.run(model.d_t_selected, feed_dict={model.x: current_ob,
                                                                     model.actions_t: action,
                                                                     model.rewards_t: [reward],
                                                                     model.x_target: next_ob,
                                                                     model.done_mask: [done]})


                # print(d_value[0])
                # print("==========")
                # visualisation for debugging process
                if args.visualise:
                    # realtime plot
                    realtime_plot(t, d_value[0])
                    # Limit x and y lists to 20 items
                    xt = xt[-20:]
                    yt = yt[-20:]
                    vis = cv2.resize(ob[:, :, 0], (500, 500))
                    # print(t)
                    cv2.imshow('img', vis)
                    cv2.waitKey(10)

                ep_reward += reward
                current_ob = np.roll(current_ob, shift=-1, axis=3)
                current_ob[0, :, :, -1] = ob[:, :, 0]
                t += 1


                if done:
                    env.reset()
                    print('ep ', i, ' reward: ', ep_reward)
                    file.write('ep ' + str(i) + ' reward: ' + str(ep_reward) + "\n")
                    eval_rewards += [ep_reward]
                    break



        print('eval finish')
        print(np.mean(eval_rewards))
        print(np.std(eval_rewards))

        file.write("mean " + str(np.mean(eval_rewards)) + "\n")
        file.write("std " + str(np.std(eval_rewards)) + "\n")
        file.close()

    env.close()
    print("========================================================")
    print("env closed!")
