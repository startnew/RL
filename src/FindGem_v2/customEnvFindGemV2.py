#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/30 9:48
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : customEnvFindGem.py
# @Software: PyCharm
# @desc    : "FindGem In customEnv "

# model Train code Reference from :https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


import os
import gym
import time
import math
import random
import imageio
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import namedtuple
from itertools import count
from PIL import Image, ImageDraw
import sys

try:
    from util.ImageVedioProcess import embed_mp4
except:
    sys.path.append("../../")
    from util.ImageVedioProcess import embed_mp4

####################################
# 安装了两个cuda 环境，默认的不是这个需要指定下
env_dist = os.environ
print(env_dist)
env_dist["PATH"] = r"D:\Program Files\CUDA_10_1_cudnn76\bin;" + env_dist["PATH"]
env_dist["CUDA_PATH"] = r"D:\Program Files\CUDA_10_1_cudnn76;" + env_dist["PATH"]
####################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch

print("torch.backends.cudnn.version()", torch.backends.cudnn.version())


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(64, interpolation=Image.CUBIC),
                    T.ToTensor()])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    # print("screen_height {}, screen_width {}".format(screen_height,screen_width))
    screen = screen[:, int(screen_height * 0):int(screen_height * 0.9)]
    view_width = int(screen_width * 0.6)

    # Strip off the edges, so that we have a square image centered on a cart
    # screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def select_action(state, eval=False):
    global steps_done
    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    if eval:
        eps_threshold = 0.001
    print("eps_threshold:{} ,steps_done:{}".format(eps_threshold, steps_done))
    steps_done += 1

    if sample > eps_threshold:
        print("select Model")
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            if eval:
                return target_net(state).max(1)[1].view(1, 1)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        print("select random")
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    print("loss:{}".format(loss.item()))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def create_policy_eval_video(iseval, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"

    imgs = []
    with imageio.get_writer(filename, fps=fps) as video:
        for i in range(num_episodes):
            time_step = env.reset()
            img_array = env.render(mode='rgb_array')
            img = Image.fromarray(img_array)
            drawimg = ImageDraw.Draw(img)
            drawimg.text((240, 80), "episodes:{} no.{}".format(i, 0), fill=(255, 0, 0), font=None)
            img_data = np.array(drawimg.im, dtype=np.uint8)
            img_data = np.resize(img_data, np.shape(img_array))

            print(np.shape(img_data), np.shape(img_array))
            video.append_data(img_data)
            imgs.append(img_data)

            last_screen = get_screen()
            current_screen = get_screen()
            state = (current_screen + last_screen) / 2
            max_step = 100
            for t in count():
                if t > max_step:
                    break

                # Select and perform an action
                action = select_action(state, eval=iseval)
                action_ = actions[action.item()]
                s_, reward, done, _ = env.step(action_)
                reward = torch.tensor([reward], device=device)

                # Observe new state
                last_screen = current_screen
                current_screen = get_screen()
                img_array = env.render(mode='rgb_array')
                img = Image.fromarray(img_array)
                drawimg = ImageDraw.Draw(img)
                drawimg.text((240, 80), "episodes:{} no.{}".format(i, t), fill=(255, 0, 0), font=None)
                img_data = np.array(drawimg.im, dtype=np.uint8)
                img_data = np.resize(img_data, np.shape(img_array))

                video.append_data(img_data)
                imgs.append(img_data)
                if not done:
                    next_state = current_screen * 0.8 + last_screen * 0.2  # - last_screen
                else:
                    print("Eval t:{} state:{} reward:{} done:{} action:{}".format(t, s_, reward.item(), done, action_))
                    next_state = None

                # Move to the next state
                state = next_state
                if done:
                    break
    filename_gif = filename + ".gif"
    imageio.mimsave(filename_gif, imgs, fps=fps)

    return embed_mp4(filename)


def parseArgs():
    parser = argparse.ArgumentParser(description='FindGem Version 1')
    parser.add_argument('--mode', type=str, default="train",
                        help='train or test 训练还是测试')
    parser.add_argument('--BATCH_SIZE', type=int, default=128,
                        help='Batch Size')
    parser.add_argument('--num_episodes', type=int, default=100000,
                        help='num_episodes of train ')
    parser.add_argument('--test_episodes', type=int, default=30,
                        help='num_episodes of test ')

    parser.add_argument('--fps', type=int, default=2,
                        help='num_episodes of test ')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parseArgs()
    BATCH_SIZE = args.BATCH_SIZE
    is_train = args.mode == "train"
    fps = args.fps

    num_episodes = args.num_episodes
    eval_num_episodes = args.test_episodes

    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200000
    TARGET_UPDATE = 10
    MODEL_PATH = "./model/target_net.model"

    is_ipython = 'inline' in matplotlib.get_backend()

    if is_ipython:
        from IPython import display
    plt.ion()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    env_name = "GridWorld-v2"
    env = gym.make(env_name).unwrapped
    print("metadata", env.metadata, env.rewards, list(env.states))
    st = time.time()
    actions = env.getAction()
    n_actions = len(actions)
    step_ind = 0
    env.reset()
    ############################################################################

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    if os.path.exists(MODEL_PATH):
        print("Loading Model:{}".format(MODEL_PATH))
        policy_net.load_state_dict(torch.load(MODEL_PATH))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    steps_done = 0
    episode_durations = []

    #######################################################################
    if is_train:
        eval_ = False
        for i_episode in tqdm(range(num_episodes)):
            if i_episode == num_episodes - 1:
                print("Eval Poli")
                eval_ = True
            # Initialize the environment and state
            bf_s_ = env.reset()
            print("First state:{}".format(bf_s_))
            env.render()
            last_screen = get_screen()
            current_screen = get_screen()
            state = current_screen
            max_step = 100
            for t in count():
                if t > max_step:
                    break
                # Select and perform an action
                action = select_action(state, eval=eval_)
                action_ = actions[action.item()]

                s_, reward, done, _ = env.step(action_)
                # 随着时间增加 reward 降低
                reward -= t * 0.0008
                n_s_ = s_
                strs = "--- t:{} before_state:{} now state:{} reward:{} done:{} action:{}".format(t, bf_s_, s_, reward,
                                                                                                  done,
                                                                                                  action_)
                bf_s_ = s_
                print(strs)
                reward = torch.tensor([reward], device=device)

                # Observe new state
                last_screen = current_screen
                current_screen = get_screen()
                #time.sleep(1)
                if not done:
                    next_state = current_screen
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                if not eval_:
                    # Perform one step of the optimization (on the target network)
                    optimize_model()
                if done:
                    print("t:{} state:{} reward:{} done:{} action:{}".format(t, s_, reward.item(), done, action_))
                    episode_durations.append(t + 1)
                    # plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print('Complete')
        torch.save(target_net.state_dict(), MODEL_PATH)

    env.render()
    html = create_policy_eval_video(iseval=True, filename="./result/result_polyDL", num_episodes=eval_num_episodes,
                                    fps=fps)
    with open("./result/result.html", "w", encoding="utf-8") as f:
        f.write(html)
    env.close()
    plt.ioff()
    plt.show()
