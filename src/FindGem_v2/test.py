#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/6 14:51
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : test.py
# @Software: PyCharm
# @desc    : ""
import gym

env_name = "LunarLanderContinuous-v2"
env = gym.make(env_name)
env.reset()
env.render()