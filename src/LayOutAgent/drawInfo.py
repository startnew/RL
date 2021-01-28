#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/26 15:53
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : drawInfo.py
# @Software: PyCharm
# @desc    : ""

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib import patches
from matplotlib.font_manager import FontProperties
import json
import sys
import math
from PIL import Image
import fnmatch
import matplotlib.pyplot as plt
import os as os
import numpy as np
from collections import defaultdict
import cv2
import sklearn.cluster as skc
import random
import colorsys

colors = [(0.0, 1.0, 0.0),
          (0.0, 0.0, 1.0),
          (1.0, 1.0, 0.0),
          (1.0, 0.0, 1.0),
          (0.0, 1.0, 1.0),
          (0.0, 1.0, 1.0),
          (0.0, 1.0, 0.0),
          (0.5, 0.5, 1),
          (0.5, 1.0, 0.5),
          (1.0, 1.0, 0.5),
          (1.0, 0.5, 0.5),
          (0.5, 1.0, 1.0),
          (0.5, 1.0, 0.0),
          (1.0, 0.5, 1.0),

          (0.5, 0.5, 0.4),
          (0.3, 1.0, 0.5),
          (0.5, 1.0, 0.7),
          (0.5, 0.7, 1.0),
          (0.5, 0.8, 1.0),
          (0.5, 0.9, 1.0),
          (0.6, 0.7, 1.0)
          ]
select_zones = ['39', '40', '49', '52', '53', '54']
colors_zones = {}
mode = [['48', '38'], ['48', '38', '51'], ['39', '44', '46'], ['39', '40', '49', '52', '54']]
zone_dict = [
    {"48": [[125, 255, 125]], "38": [[124, 123, 255]]},
    {"48": [[125, 255, 125]], "38": [[129, 125, 250]], "51": [[254, 124, 125]]},
    {"39": [[125, 127, 255]], "44": [[255, 210, 125]], "46": [[255, 127, 126]]},
    {"39": [[206, 140, 212]], "40": [[255, 192, 143]], "49": [[172, 251, 142]], "52": [[254, 127, 128]],
     "53": [[249, 131, 191]], "54": [[137, 216, 249]]},
]
class ShowOriJson():
    def __init__(self, room_strs, is_new=False):

        if isinstance(room_strs, str):
            self.jsondata = json.loads(room_strs)
        else:
            self.jsondata = room_strs

        self.is_new = is_new
        self.color_dict = {}
        self.color_dict["wall"] = "k"
        self.color_dict["door"] = "g"
        self.color_dict["windows"] = "c"
        self.color_dict["zone"] = "b"
        self.color_dict["wall_text"] = "k"
        self.color_dict["door_text"] = "k"
        self.color_dict["zone_text"] = "k"
        self.color_dict["obj_text"] = "k"
        self.color_dict["windows_text"] = "k"

    def initFig(self, figsize=(5, 5), ax=None, fig=None):
        if fig or ax:
            pass
        else:
            fig, ax = plt.subplots(figsize=figsize)
        import sys
        if sys.platform == "windows":
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        else:
            myfont = FontProperties(fname='/usr/share/fonts/*.ttc')
        plt.rcParams['axes.unicode_minus'] = False
        self.ax = ax
        self.fig = fig

    def drawWalls(self, show_scid=False, scale=1, move=0):
        verts, texts = self.refuse_wallpoints()
        verts1 = []

        for i, vert in enumerate(verts):
            # print('vert' ,vert)
            vert = [[(float(x[0]) + move) * scale,
                     (float(x[1]) + move) * scale] for x in vert]
            verts1.append([round(vert[0][0]), round(vert[0][1]), round(vert[2][0]), round(vert[2][1])])
            p = Polygon(vert,
                        facecolor=self.color_dict["wall"],
                        edgecolor=self.color_dict["wall"])
            if show_scid:
                self.ax.text(x=vert[0][0], y=vert[0][1], s=texts[i])
            self.ax.add_patch(p)
        self.ax.autoscale_view()
        # print(verts1)
        # print('scale_ori , move_ori' ,scale ,move)
        return verts1

    def drawDoors(self, show_scid=False, scale=1, move=0):
        verts, texts = self.refuse_doorpoints()
        for i, vert in enumerate(verts):
            vert = [[(float(x[0]) + move) * scale,
                     (float(x[1]) + move) * scale] for x in vert]
            p = Polygon(vert,
                        facecolor=self.color_dict["door"],
                        edgecolor=self.color_dict["door"])
            if show_scid:
                self.ax.text(x=vert[0][0],
                             y=vert[0][1],
                             s=texts[i],
                             color=self.color_dict["door_text"])
            self.ax.add_patch(p)

        self.ax.autoscale_view()

    def drawWindows(self, show_scid=False, scale=1, move=0):
        verts, texts, l, w = self.refuse_windowpoints()
        for i, vert in enumerate(verts):
            vert = [[(float(x[0]) + move) * scale,
                     (float(x[1]) + move) * scale] for x in vert]
            p = Polygon(vert,
                        facecolor=self.color_dict["windows"],
                        edgecolor=self.color_dict["windows"])
            if show_scid:
                self.ax.text(x=vert[0][0],
                             y=vert[0][1],
                             s=texts[i],
                             color=self.color_dict["windows_text"])
            self.ax.add_patch(p)
        self.ax.autoscale_view()

    def saveFig(self, figpath=''):
        self.fig.savefig(figpath)

    def stand_rule(self, PointList, imgsize):
        '''
        param:z [xmin,xmax,ymin,ymax]
        param:imgsize
        '''
        xlist = [point[0] for point in PointList]
        ylist = [point[1] for point in PointList]
        w = max(xlist + ylist) - min(xlist + ylist)  # + self.thres
        offset = 0 - min(xlist + ylist)
        scale = imgsize / w
        dx_max = max(xlist) - min(xlist)
        dy_max = max(ylist) - min(ylist)
        use_scale, use_offset = 128 / w, offset
        return scale, offset, use_scale, use_offset, dx_max, dy_max, min(xlist), min(ylist), max(xlist), max(ylist)

    def refuse_windowpoints(self):
        windows = []
        texts = []
        for window in self.jsondata["windows"]:
            vert = []
            for point in window['points']:
                x_list = [point['x'] for point in window['points']]
                y_list = [point['y'] for point in window['points']]
                x_max, x_min = max(x_list), min(x_list)
                y_max, y_min = max(y_list), min(y_list)
                # print('x_list',x_list)
                # print('y_list',y_list)
                l = x_max - x_min
                w = y_max - y_min
                if w >= 250 and (l > w or (l - w >= -5 and l - w < 0)):
                    for point in window['points']:
                        if abs(point['y'] - y_max) <= 10 and x_max > 0:
                            point['y'] = y_min + 200
                        if y_min < 0 and abs(point['y'] - y_min) <= 10:
                            point['y'] = y_max - 200
                if l >= 250 and (w > l or (l - w <= 5 and l - w >= 0)):
                    for point in window['points']:
                        if abs(point['x'] - x_max) <= 10 and x_max > 0:
                            point['x'] = x_min + 200
                        if x_min < 0 and abs(point['x'] - x_min) <= 10:
                            point['x'] = x_max - 200

                # print(point)
                vert.append([point["x"], point["y"]])

            windows.append(vert)
            if "scid" in window.keys():
                texts.append(window["scid"])
            else:
                texts.append(3)
            # texts.append(window["scid"])
        if len(windows) > 0:
            return windows, texts, l, w
        else:
            return windows, texts, 0, 0

    def refuse_wallpoints(self):
        verts = []
        texts = []
        for wall in self.jsondata["walls"]:
            vert = []
            for point in wall['wallPoints']:
                vert.append([point["x"], point["y"]])
            # 每个墙都是矩形
            xs = [float(x[0]) for x in vert]
            ys = [float(x[1]) for x in vert]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            vert = [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]]

            verts.append(vert)
            if "scid" in wall.keys():
                texts.append(wall["scid"])
            else:
                texts.append(1)

        return verts, texts

    def refuse_doorpoints(self):
        doors = []
        texts = []
        for door in self.jsondata["doors"]:
            vert = []
            for point in door['points']:
                vert.append([point["x"], point["y"]])

            doors.append(vert)
            if "scid" in door.keys():
                texts.append(door["scid"])
            else:
                texts.append(2)
        return doors, texts

    def drawFunctionZones(self, show=["zid", "cid"], debug=False, scale=1,
                          move=0, colormap={}, show_direction=True):
        '''
        可视化 room json 中的功能区
        '''
        if self.is_new == False:

            for zone in self.jsondata["functionZones"]:
                # if int(zone["id"]) in [89, 58, 56, 39, 54, 52, 40, 49]:
                #     # if int(zone["id"]) in [89,56,57,47,50,54,59,58,52]:
                #     # if int(zone["id"]) in [51,89,57,59,47,54,43]:#bedroom
                #     # if int(zone["id"]) in [89,58,57,47,50]:
                #     continue
                k = "bound"
                dx = zone[k]["dx"] * scale
                dy = zone[k]["dy"] * scale

                x = (zone["center"]["x"] + move) * scale - 0.5 * dx
                y = (zone["center"]["y"] + move) * scale - 0.5 * dy
                if colormap:
                    edgecolor = colormap.get(zone["id"], "y")
                    facecolor = colormap.get(zone["id"], "r")
                else:
                    edgecolor = "y",
                    facecolor = 'b'
                # print('dx,dy',dx,dy)
                # print(zone["id"], x, y, dx, dx)

                p = patches.Rectangle((x, y),
                                      dx,
                                      dy,
                                      alpha=0.5,
                                      edgecolor=edgecolor,
                                      facecolor=facecolor)
                # print(p)
                # rot = grid_degree(float(zone["rotate"]["zAxis"]))
                self.ax.add_patch(p)
                rot = None
                # print("rot:{},".format(rot, (zone["rotate"]["zAxis"]) / (2 * np.pi) * 360))
                x_ = x + 0.5 * dx
                y_ = y + 0.5 * dy
                if rot == 0:
                    points = [[x_ - 0.5 * dx, y_ - 0.5 * dy], [x_, y_ + 0.5 * dy], [x_ + 0.5 * dx, y_ - 0.5 * dy]]
                elif rot == 1:
                    points = [[x_ + 0.5 * dx, y_ - 0.5 * dy], [x_ - 0.5 * dx, y_], [x_ + 0.5 * dx, y_ + 0.5 * dy]]
                elif rot == 2:
                    points = [[x_ - 0.5 * dx, y_ + 0.5 * dy], [x_, y_ - 0.5 * dy], [x_ + 0.5 * dx, y_ + 0.5 * dy]]
                elif rot == 3:
                    points = [[x_ - 0.5 * dx, y_ - 0.5 * dy], [x_ + 0.5 * dx, y_], [x_ - 0.5 * dx, y_ + 0.5 * dy]]

                # 用于表示方向
                show_direction = show_direction
                if show_direction:
                    # p1 = patches.Polygon(points, facecolor=(1,1,0), edgecolor=(1,1,0), linewidth=2, alpha=1)
                    p1 = patches.Polygon(points, facecolor='#000000', edgecolor='#000000', linewidth=2, alpha=1)
                    self.ax.add_patch(p1)

                if "zid" in show:
                    self.ax.text(x=x + 0.3 * dx, y=y + 0.3 * dy,
                                 s=str(zone["id"]),
                                 color="r")

        else:

            for zone in self.jsondata["functionZones_new"]:
                # if int(zone["id"]) in [89, 58, 56, 39, 54, 52, 40]:
                #     # if int(zone["id"]) in [89,56,57,47,50,54,59,58,52]:
                #     # if int(zone["id"]) in [51,89,57,59,47,54,43]:#bedroom
                #     # if int(zone["id"]) in [89,58,57,47,50]:
                #     continue
                if zone.get('model_id'):
                    idx = zone["model_id"]
                    select_zones = [zone["model_id"] for zone in self.jsondata["functionZones_new"]]
                    for i, zid in enumerate(select_zones):
                        colors_zones[zid] = colors[i]
                else:
                    idx = zone["id"]
                k = "bound"
                dx = zone[k]["dx"] * scale
                dy = zone[k]["dy"] * scale

                x = (zone["center"]["x"] + move) * scale - 0.5 * dx
                y = (zone["center"]["y"] + move) * scale - 0.5 * dy
                if colormap:
                    edgecolor = colormap[idx]
                    facecolor = colormap[idx]
                else:
                    edgecolor = "y",
                    facecolor = 'b'
                # print('dx,dy',dx,dy)
                # print(idx, x, y, dx, dx)

                p = patches.Rectangle((x, y),
                                      dx,
                                      dy,
                                      alpha=0.5,
                                      edgecolor=edgecolor,
                                      facecolor=facecolor)
                # print(p)
                # rot = grid_degree(float(zone["rotate"]["zAxis"]))
                self.ax.add_patch(p)
                rot = None
                # print("rot:{},".format(rot, (zone["rotate"]["zAxis"]) / (2 * np.pi) * 360))
                x_ = x + 0.5 * dx
                y_ = y + 0.5 * dy
                if rot == 0:
                    points = [[x_ - 0.5 * dx, y_ - 0.5 * dy], [x_, y_ + 0.5 * dy], [x_ + 0.5 * dx, y_ - 0.5 * dy]]
                elif rot == 1:
                    points = [[x_ + 0.5 * dx, y_ - 0.5 * dy], [x_ - 0.5 * dx, y_], [x_ + 0.5 * dx, y_ + 0.5 * dy]]
                elif rot == 2:
                    points = [[x_ - 0.5 * dx, y_ + 0.5 * dy], [x_, y_ - 0.5 * dy], [x_ + 0.5 * dx, y_ + 0.5 * dy]]
                elif rot == 3:
                    points = [[x_ - 0.5 * dx, y_ - 0.5 * dy], [x_ + 0.5 * dx, y_], [x_ - 0.5 * dx, y_ + 0.5 * dy]]

                # 用于表示方向
                show_direction = show_direction
                if show_direction:
                    # p1 = patches.Polygon(points, facecolor=(1,1,0), edgecolor=(1,1,0), linewidth=2, alpha=1)
                    p1 = patches.Polygon(points, facecolor='#000000', edgecolor='#000000', linewidth=2, alpha=1)
                    self.ax.add_patch(p1)

                if "zid" in show:
                    self.ax.text(x=x + 0.3 * dx, y=y + 0.3 * dy,
                                 s=str(idx),
                                 color="r")

    def show(self, title=""):
        self.ax.set_title(title)
        self.fig.show()

    def add_grid(self, st=-4000, ed=3000, split=1000):
        x = np.arange(st, ed, split)
        y = x
        w = self.jsondata['roomBound']['dx']
        h = self.jsondata['roomBound']['dy']

        self.ax.set_xticks(x)
        self.ax.set_yticks(y)
        # self.ax.set(xlabel='x', ylabel='y', title='room id 1')
        # self.fig.set_size_inches(h=5, w=5)
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.set_size_inches(h=10, w=10)
        self.ax.grid()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        # self.ax.set_title('h :{},w: {}'.format(h, w), fontsize=10)
        # self.ax.axis('off')


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def fig2data(fig):
    """
    转换matplotlib 的 figure 到 一个 RGBA 的 4D numpy 并返回
    :param fig:  a matplotlib figure
    :return: a numpy  array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    plt.close(fig)
    # canvas.tostring_argb give pixmap in ARGB model. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
