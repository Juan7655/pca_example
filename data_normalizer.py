def normalizer(fun):
    def wrapped(data):
        data = data.copy(deep=True)
        for col in data:
            data[col] = fun(data[col])
        return data

    return wrapped


def create_normalizer(fun):
    @normalizer
    def apply(data_col):
        return fun(data_col)
    return apply