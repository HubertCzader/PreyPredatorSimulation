import random

import numpy as np


def sequence3(input1, input2, input3):
    if isinstance(input1, np.bool_):
        input1 = bool(input1)
    if isinstance(input2, np.bool_):
        input2 = bool(input2)
    if isinstance(input2, np.bool_):
        input3 = bool(input3)
    for input in [input1, input2, input3]:
        if input is False:
            return False
        elif input is True:
            continue
        else:
            return input
    return True


def sequence2(input1, input2):
    if isinstance(input1, np.bool_):
        input1 = bool(input1)
    if isinstance(input2, np.bool_):
        input2 = bool(input2)
    for input in [input1, input2]:
        if input is False:
            return False
        elif input is True:
            continue
        else:
            return input
    return True


def selector2(input1, input2):
    if isinstance(input1, np.bool_):
        input1 = bool(input1)
    if isinstance(input2, np.bool_):
        input2 = bool(input2)
    for input in [input1, input2]:
        if input is False or input is True:
            continue
        else:
            return input
    return False


def selector3(input1, input2, input3):
    if isinstance(input1, np.bool_):
        input1 = bool(input1)
    if isinstance(input2, np.bool_):
        input2 = bool(input2)
    if isinstance(input3, np.bool_):
        input3 = bool(input3)
    for input in [input1, input2, input3]:
        if input is False or input is True:
            continue
        else:
            return input
    return False


def selector4(input1, input2, input3, input4):
    if isinstance(input1, np.bool_):
        input1 = bool(input1)
    if isinstance(input2, np.bool_):
        input2 = bool(input2)
    if isinstance(input3, np.bool_):
        input3 = bool(input3)
    if isinstance(input4, np.bool_):
        input4 = bool(input4)
    for input in [input1, input2, input3, input4]:
        if input is False or input is True:
            continue
        else:
            return input
    return False


def randomSelector2(input1, input2):
    return random.choice([input1, input2])
