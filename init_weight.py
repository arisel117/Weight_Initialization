# -*- coding: utf-8 -*-
# 참고: https://alltommysworks.com/%EA%B0%80%EC%A4%91%EC%B9%98-%EC%B4%88%EA%B8%B0%ED%99%94


import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def affine(x, w):
    return np.dot(w, x)


def reset_params(shape, init_func):
    fan_in, fan_out = shape

    if isinstance(init_func, float):
        return init_func * np.random.randn(*shape)
    elif isinstance(init_func, str):
        if init_func.lower() == 'lecun':
            stddev = np.sqrt(1 / (fan_in))
        elif init_func.lower() == 'xavier':
            stddev = np.sqrt(2 / (fan_in + fan_out))
        elif init_func.lower() == 'he':
            stddev = np.sqrt(2 / (fan_in))
        else:
            raise Exception("init_func ['lecun', 'xavier', 'he', float] 만 사용 가능합니다.")
    else:
        raise Exception("init_func ['lecun', 'xavier', 'he', float] 만 사용 가능합니다.")

    return np.random.normal(0, stddev, shape)


def forward_layer(x, init_func, active_func='sigmoid'):
    fan_out, fan_in = x.shape

    weight = reset_params((fan_in, fan_out), init_func)

    x = affine(x, weight)
    if active_func.lower() == 'sigmoid':
        return sigmoid(x)
    elif active_func.lower() == 'relu':
        return relu(x)
    else:
        raise Exception("active_func은 ['sigmoid', 'relu'] 만 사용 가능합니다.")


def draw_histogram(x, init_func, active_func):
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    fig.suptitle(f'init={init_func}, active={active_func}')

    # 초기 입력 레이어
    y = forward_layer(x, init_func, active_func)
    axes[0].hist(y.flatten(), bins=100, alpha=1)
    axes[0].set_title("Input")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 8000)

    # 이후 레이어 100층까지
    current_axis = 0
    for i in range(1, 101):
        y = forward_layer(y, init_func, active_func)
        if i % 20 == 0:
            current_axis += 1
            axes[current_axis].hist(y.flatten(), bins=100, alpha=1)
            axes[current_axis].set_title(f"Hidden {i} Layer")
            axes[current_axis].set_xlim(0, 1)
            axes[current_axis].set_ylim(0, 8000)

    plt.show()
