#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/30 9:44
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : ImageVedioProcess.py
# @Software: PyCharm
# @desc    : "Some util For video encode"

import base64
import imageio


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())
    # return IPython.display.HTML(tag)
    return tag

