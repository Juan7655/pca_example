def normalizer(fun):
    def wrapped(data):
        data = data.copy(deep=True)
        for col in data:
            data[col] = fun(data[col])
        return data
    return wrapped


@normalizer
def recenter(data_col):
    return data_col - data_col.mean()


@normalizer
def stretch(data_col):
    return data_col / data_col.std()


@normalizer
def stretch_and_recenter(data_col):
    return (data_col - data_col.mean()) / data_col.std()


@normalizer
def identity(data_col):
    return data_col
