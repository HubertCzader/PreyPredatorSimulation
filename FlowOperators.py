def sequence3(input1, input2, input3):
    for input in [input1, input2, input3]:
        if input is False:
            return False
        elif input is True:
            continue
        else:
            return input
    return True


def sequence2(input1, input2):
    for input in [input1, input2]:
        if input is False:
            return False
        elif input is True:
            continue
        else:
            return input
    return True


def selector2(input1, input2):
    for input in [input1, input2]:
        if input is False or input is True:
            continue
        else:
            return input
    return False


def selector3(input1, input2, input3):
    for input in [input1, input2, input3]:
        if input is False or input is True:
            continue
        else:
            return input
    return False
