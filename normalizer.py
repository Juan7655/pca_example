import types

from pandas import DataFrame


def normalizer(fun: types.FunctionType):
    def wrapped(data: DataFrame):
        data = data.copy(deep=True)
        for col in data:
            data[col] = fun(data[col])
        return data

    return wrapped


def Map(fun: types.FunctionType, data: DataFrame):
    @normalizer
    def apply():
        return fun(data)
    return apply
