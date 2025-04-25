import os
import copy
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mping
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.ticker import NullFormatter
from ast import literal_eval as make_tuple


def show_01_df_placed(_01_df_placed, canvas_info, connection, save, fig_name):
    # old version
    print('this is old version function, please use: `draw_placement()` in `./tool/draw/draw_placement.py`')
    draw_placement(df_placed=_01_df_placed, canvas_info=canvas_info, figure_size=8, font_size=6, connection=connection, answer=None,
                   save_file=save, save_file_path=fig_name)
    pass


def draw_placement(df_placed, canvas_info: dict, title='', figure_size=8, font_size=6, connection=None, answer=None,
                   save_file=False, save_file_path=None, show=False):
    """
    # draw from 01_df_placed.csv (pandas dataframe)
    # now using this function

    """

    figsize = (figure_size, figure_size)

    x = df_placed['x'].values
    y = df_placed['y'].values
    w = df_placed['width'].values
    h = df_placed['height'].values
    id_ = df_placed['node_id'].values
    node_type_ = df_placed['node_type'].values

    x_shift = 0
    y_shift = 0

    TYPE_MACRO = 0
    TYPE_CLUSTER = 1
    TYPE_IO_PORT = 2
    TYPE_BLOCKAGE = 3
    TYPE_ANCHOR = 4
    color_box = {
        TYPE_MACRO: {'fc': (0, 0, 1, .2), 'ec': (0, 0, 1, 0.6)},
        TYPE_CLUSTER: {'fc': (1, 1, 0, .5), 'ec': (1, 1, 0, 0.6)},
        TYPE_IO_PORT: {'fc': 'none', 'ec': (0, 1, 1, 1)},
        TYPE_BLOCKAGE: {'fc': (.0, .0, .0, .8), 'ec': 'none'},
        TYPE_ANCHOR: {'fc': 'none', 'ec': (0, 1, 1, 1)},
    }

    fig, ax = plt.subplots(1, figsize=figsize, dpi=120)

    x_max_ = canvas_info['board info']['canvas_width']
    y_max_ = canvas_info['board info']['canvas_height']
    x_grid_size = canvas_info['board info']['grid_width']
    y_grid_size = canvas_info['board info']['grid_height']
    plt.xlim(- 10, x_max_ + 10)
    plt.ylim(- 10, y_max_ + 10)

    plt.xticks(np.arange(0, x_max_ + 10, x_grid_size), fontsize=font_size)
    plt.yticks(np.arange(0, y_max_ + 10, y_grid_size), fontsize=font_size)

    # plt.gca().invert_yaxis()
    collect = []
    for i in range(len(x)):

        # 0=macros
        if node_type_[i] == TYPE_MACRO:
            if answer is not None:
                color = 'darkorange' if x[i] == answer[i][0] and y[i] == answer[i][1] else 'blue'
            else:
                color = 'blue'
            rect = Rectangle((x[i] - w[i] / 2 - x_shift, y[i] - h[i] / 2 - y_shift), w[i], h[i],
                             facecolor=color_box[TYPE_MACRO]['fc'], edgecolor=color_box[TYPE_MACRO]['ec'])
            ax.add_patch(rect)
            ax.annotate(id_[i], (x[i] - x_shift, y[i] - y_shift), color='#484891',
                        weight='bold', fontsize=font_size, ha='center', va='center')
            collect.append((x[i] - x_shift, y[i] - y_shift, int(id_[i])))

        # 1=std clusters
        if node_type_[i] == TYPE_CLUSTER:
            color = 'royalblue'
            rect = Rectangle((x[i] - w[i] / 2 - x_shift, y[i] - h[i] / 2 - y_shift), w[i], h[i],
                             facecolor=color_box[TYPE_CLUSTER]['fc'], edgecolor=color_box[TYPE_CLUSTER]['ec'])
            ax.add_patch(rect)
            ax.annotate(id_[i], (x[i] - x_shift, y[i] - y_shift), color='g',
                        weight='bold', fontsize=font_size * 0.5, ha='center', va='center')
            collect.append((x[i] - x_shift, y[i] - y_shift, int(id_[i])))

        # 2=I/O ports

        # 3=blockages
        if node_type_[i] == TYPE_BLOCKAGE:
            rect = Rectangle((x[i] - w[i] / 2 - x_shift, y[i] - h[i] / 2 - y_shift), w[i], h[i],
                             facecolor=color_box[TYPE_BLOCKAGE]['fc'], edgecolor=color_box[TYPE_BLOCKAGE]['ec'])
            ax.add_patch(rect)
            ax.annotate('B', (x[i] - x_shift, y[i] - y_shift), color='w',
                        weight='bold', fontsize=font_size, ha='center', va='center')

        # 4=anchor points
        if node_type_[i] == TYPE_ANCHOR:
            ax.plot(x[i] + x_shift, y[i] + y_shift, 'ro', color='orange', markersize=font_size * 1.2)
            ax.annotate(id_[i], (x[i] - x_shift, y[i] - y_shift), color='black',
                        fontsize=font_size * 0.8, ha='center', va='center')

    for net in connection.values.tolist():
        head, tail, _ = make_tuple(net[0])
        if head < tail:
            continue
        if node_type_[head] == TYPE_ANCHOR or node_type_[tail] == TYPE_ANCHOR:
            # link with anchor points
            plt.plot([x[head], x[tail]], [y[head], y[tail]], c='#FF9224', alpha=.8, linewidth=1)
        elif node_type_[head] == TYPE_IO_PORT or node_type_[tail] == TYPE_IO_PORT:
            # link with OI port
            plt.plot([x[head], x[tail]], [y[head], y[tail]], c='#B766AD', alpha=.8, linewidth=1)
        else:
            # link between nodes
            plt.plot([x[head], x[tail]], [y[head], y[tail]], c='#2894FF', alpha=.8, linewidth=1)

    plt.grid(True)
    plt.title(title)
    if save_file:
        if save_file_path is not None:
            plt.savefig(save_file_path)

        else:
            now = datetime.now()
            date_time = now.strftime('%Y%m%d_%H%M%S')
            plt.savefig('01df_result_' + date_time + '.png')
    if show:
        plt.show()
    plt.close()


def draw_placement_old(input_data, outdir, CANVAS_GRID_WIDTH, CANVAS_GRID_HEIGHT, title=''):
    """
    draw function from Hau & Michael
    """

    nodes = list(filter(lambda node: node.node_type != 2, copy.deepcopy(input_data.nodes)))
    io_nodes = list(filter(lambda node: node.node_type ==
                           2, copy.deepcopy(input_data.nodes)))

    print('len(nodes): ', len(nodes))
    print('len(io_nodes): ', len(io_nodes))

    # Note: write placement png
    fig, ax = plt.subplots(1, figsize=(32, 32))
    plt.xlim(0, CANVAS_GRID_WIDTH)
    plt.ylim(0, CANVAS_GRID_HEIGHT)
    plt.gca().invert_yaxis()

    x_offset = CANVAS_GRID_WIDTH / 128
    y_offset = CANVAS_GRID_HEIGHT / 128

    for io_node in io_nodes:
        x = io_node.x
        y = io_node.y
        if x < y:
            if x / CANVAS_GRID_WIDTH + y / CANVAS_GRID_HEIGHT > 1:
                path = Path([[x, y], [x - x_offset, y + 4 * y_offset],
                             [x + x_offset, y + 4 * y_offset], [x, y]])
            else:
                path = Path([[x, y], [x - 4 * x_offset, y - y_offset],
                             [x - 4 * x_offset, y + y_offset], [x, y]])
        else:
            if x / CANVAS_GRID_WIDTH + y / CANVAS_GRID_HEIGHT > 1:
                path = Path([[x, y], [x + 4 * x_offset, y - y_offset],
                             [x + 4 * x_offset, y + y_offset], [x, y]])
            else:
                path = Path([[x, y], [x - x_offset, y - 4 * y_offset],
                             [x + x_offset, y - 4 * y_offset], [x, y]])
        patch = patches.PathPatch(
            path,
            clip_on=False,
            facecolor='none',
            edgecolor=(0, 1, 1, 1))
        ax.add_patch(patch)

    for node in nodes + input_data.blockages:
        if node.x == -1:
            continue
        ec = (1, 0, 0, 1)
        fc = (1, 0, 0, .2)
        if node.node_type == 1:
            ec = (0, 0, 1, 1)
            fc = (0, 0, 1, .2)
        elif node.node_type == 3:
            ec = 'none'
            fc = (.0, .0, .0, .8)

        patch = patches.Rectangle(
            (node.x - node.width / 2,
                node.y - node.height / 2),
            node.width,
            node.height,
            linewidth=1,
            edgecolor=ec,
            facecolor=fc)

        ax.add_patch(patch)

    plt.title(title)
    plt.savefig(os.path.join(outdir, 'placement.png'))
    plt.close()
