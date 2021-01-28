#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/01/05 10:02
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : grid_findGem_v1.py
# @Software: PyCharm
# @desc    : "env code in gym"

# from: https://blog.csdn.net/extremebingo/article/details/80867486
import logging
import random
import gym

'''
env变 rebot所处位置会变时
'''
logger = logging.getLogger(__name__)


class GridEnv2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.states = range(1, 16)  # 状态空间
        self.col = 4
        self.row = 4
        self.reset_ = False

        self.x = [150, 250, 350, 450] * 4
        self.y = [450] * 4 + [350] * 4 + [250] * 4 + [150] * 4

        self.terminate_states = dict()  # 终止状态为字典格式
        self.hole_ids = [11, 12]  # 编号 从1 开始
        self.gem_ids = [15]  # 编号 从1 开始
        self.stone_id = 6  # 编号 从1 开始
        for id_ in self.hole_ids + self.gem_ids:
            self.terminate_states[id_] = 1
        # n 上 s下  w左 e 右
        self.actions = ['n', 'e', 's', 'w']
        self.gen_t()
        print("生成出来的t:{}".format(self.t))
        self.gen_rewards()
        print("生成出来的rewards:{}".format(self.rewards))


        self.gamma = 0.8  # 折扣因子
        self.viewer = None
        self.state = None
        self.reset()


    def gen_t(self):

        pos_indexs = []  # 位置编号，其中位置为石柱的不计入编号，标识为-1 编号 从1 开始
        pos_index = 0
        self.t = dict()
        self.t_ = dict()
        for i in range(self.row):
            for j in range(self.col):
                loc_index = i * self.row + j
                if loc_index == self.stone_id:
                    pos_indexs.append(-1)

                else:
                    pos_index += 1
                    pos_indexs.append(pos_index)

                if i > 0:
                    loc_indexs_n = loc_index - self.col
                    self.t_["{}_{}".format(loc_index + 1, "n")] = loc_indexs_n
                if i < self.row - 1:
                    loc_indexs_s = loc_index + self.col
                    self.t_["{}_{}".format(loc_index + 1, "s")] = loc_indexs_s
                if j > 0:
                    loc_indexs_w = loc_index - 1
                    self.t_["{}_{}".format(loc_index + 1, "w")] = loc_indexs_w
                if j < self.col - 1:
                    loc_indexs_e = loc_index + 1
                    self.t_["{}_{}".format(loc_index + 1, "e")] = loc_indexs_e
        # 根据 石头的 位置修正转换矩阵
        for k, v in self.t_.items():
            v_ = v + 1

            n = int(k.split("_")[0])
            s = k.split("_")[-1]

            if n < self.stone_id:
                pass
            elif n == self.stone_id:
                continue
            elif n > self.stone_id:
                n -= 1
            if n in self.hole_ids:
                continue
            if n in self.gem_ids:
                continue
            k_n = "{}_{}".format(n, s)

            if v_ < self.stone_id:
                self.t[k_n] = v_
            elif v_ == self.stone_id:
                continue
            elif v_ > self.stone_id:
                self.t[k_n] = v_ - 1
        # print("状态转移:",self.t)

    def gen_rewards(self):
        self.rewards = dict();  # 回报的数据结构为字典
        for k, v in self.t.items():
            if v in self.hole_ids:
                self.rewards[k] = -1.0
            elif v in self.gem_ids:
                self.rewards[k] = 1.0

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self, s):
        self.state = s

    def step(self, action):
        # 系统当前状态
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}

        key = "%d_%s" % (state, action)  # 将状态和动作组成字典的键值

        # 状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]

        return next_state, r, is_terminal, {}

    def reset(self,sign_force_reset=True):
        if sign_force_reset: # 是否将所有环境均改变
            self.hole_ids = random.choices(self.states, k=2)  # 编号 从1 开始
            # [11, 12]
            res = list(set(self.states) - set(self.hole_ids))
            self.gem_ids = [random.choice(res)]  # 编号 从1 开始
            res = list(set(res) - set(self.gem_ids))
            self.stone_id = random.choice(res)
            res = list(set(res) - set([self.stone_id]))
            self.terminate_states = dict()
            for id_ in self.hole_ids + self.gem_ids:
                self.terminate_states[id_] = 1
            self.state = random.choice(res)  # self.states[int(random.random() * len(self.states))]
            self.gen_t()
            self.gen_rewards()
            self.reset_ = True
        else:
            pass
        return self.state

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 600

        if self.viewer is None:
            if not self.viewer:
                self.viewer = rendering.Viewer(screen_width, screen_height)
            # 创建网格世界
            self.line1 = rendering.Line((100, 100), (500, 100))
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (500, 300))
            self.line4 = rendering.Line((100, 400), (500, 400))
            self.line5 = rendering.Line((100, 500), (500, 500))
            self.line6 = rendering.Line((100, 100), (100, 500))
            self.line7 = rendering.Line((200, 100), (200, 500))
            self.line8 = rendering.Line((300, 100), (300, 500))
            self.line9 = rendering.Line((400, 100), (400, 500))
            self.line10 = rendering.Line((500, 100), (500, 500))

            # 创建石柱
            self.shizhu = rendering.make_circle(40)
            self.shizhutrans = rendering.Transform()
            self.shizhu.add_attr(self.shizhutrans)
            self.shizhu.set_color(0.8, 0.6, 0.4)

            # 创建第一个火坑
            self.fire1 = rendering.make_circle(40)
            self.fire1trans = rendering.Transform()  # (450, 250)
            self.fire1.add_attr(self.fire1trans)
            self.fire1.set_color(1, 0, 0)

            # 创建第二个火坑
            self.fire2 = rendering.make_circle(40)
            self.fire2trans = rendering.Transform()  # (150, 150)
            self.fire2.add_attr(self.fire2trans)
            self.fire2.set_color(1, 0, 0)

            # 创建宝石
            self.diamond = rendering.make_circle(40)
            self.diamondtrans = rendering.Transform()  # (450, 150)
            self.diamond.add_attr(self.diamondtrans)
            self.diamond.set_color(0, 0, 1)

            # 创建机器人
            self.robot = rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0, 1, 0)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.shizhu)
            self.viewer.add_geom(self.fire1)
            self.viewer.add_geom(self.fire2)
            self.viewer.add_geom(self.diamond)
            self.viewer.add_geom(self.robot)
            self.reset_ = True

        if self.reset_:
            shizhuinf = (self.x[self.stone_id - 1], self.y[self.stone_id - 1])
            self.shizhutrans.set_translation(shizhuinf[0], shizhuinf[1])

            if self.hole_ids[0] <= self.stone_id - 1:
                inf = (self.x[self.hole_ids[0] - 1], self.y[self.hole_ids[0] - 1])
            else:
                inf = (self.x[self.hole_ids[0]], self.y[self.hole_ids[0]])
            fire1_inf = inf
            self.fire1trans.set_translation(fire1_inf[0], fire1_inf[1])

            if self.hole_ids[1] <= self.stone_id - 1:
                inf = (self.x[self.hole_ids[1] - 1], self.y[self.hole_ids[1] - 1])
            else:
                inf = (self.x[self.hole_ids[1]], self.y[self.hole_ids[1]])
            fire2_inf = inf
            self.fire2trans.set_translation(fire2_inf[0], fire2_inf[1])

            if self.gem_ids[0] <= self.stone_id - 1:
                inf = (self.x[self.gem_ids[0] - 1], self.y[self.gem_ids[0] - 1])
            else:
                inf = (self.x[self.gem_ids[0]], self.y[self.gem_ids[0]])
            diamond_inf = inf

            self.diamondtrans.set_translation(diamond_inf[0], diamond_inf[1])

        self.reset_ = False

        if self.state is None:
            return None
        if self.state <= self.stone_id - 1:
            self.robotrans.set_translation(self.x[self.state - 1], self.y[self.state - 1])
        else:
            self.robotrans.set_translation(self.x[self.state], self.y[self.state])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

# if __name__ == "__main__":
# a = GridEnv1()
