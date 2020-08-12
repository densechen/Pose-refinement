'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:45:51
'''
import numpy as np
from visdom import Visdom

PORT = 8097

_WINDOW_CASH = {}
_ENV_CASH = {}


def _vis(env="main"):
    if env not in _ENV_CASH:
        _ENV_CASH[env] = Visdom(env=env, port=PORT)
    return _ENV_CASH[env]


def visualize_losses(loss: dict, title: str, env="main", iteration=0):
    legend = list()
    scalars = list()
    for k, v in loss.items():
        legend.append(k)
        scalars.append(v)
    options = dict(
        width=400,
        height=400,
        xlabel="Iterations",
        ylabel="loss",
        title=title,
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
        legend=legend,
    )
    if title in _WINDOW_CASH:
        _vis(env).line(Y=[scalars],
                       X=[iteration],
                       win=_WINDOW_CASH[title],
                       update="append",
                       opts=options)
    else:
        _WINDOW_CASH[title] = _vis(env).line(Y=[scalars],
                                             X=[iteration],
                                             opts=options)


def visualize_scatters(scatters: list,
                       legend: list,
                       title: str,
                       env="main",
                       w=640,
                       h=640):
    """scatters: list contains np.array.
        legend: list contains str.
    """
    target = np.zeros(sum([s.shape[0] for s in scatters]), dtype=int)

    acc = 0
    for i, s in enumerate(scatters):
        target[acc:acc + s.shape[0]] = i + 1
        acc += s.shape[0]
    scatters = np.concatenate(scatters, axis=0)
    _WINDOW_CASH[title] = _vis(env).scatter(X=scatters,
                                            Y=target,
                                            win=_WINDOW_CASH.get(title),
                                            opts=dict(title=title,
                                                      width=w,
                                                      height=h,
                                                      markersize=3,
                                                      legend=legend))
