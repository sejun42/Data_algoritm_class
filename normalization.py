import numpy as np

def layer_norm(x, eps=1e-8):
    mean_fm = x.mean(axis=(1, 2, 3), keepdims=True)
    var_fm = x.var(axis=(1, 2, 3), keepdims=True)
    return(x-mean_fm) / np.sqrt(var_fm + eps)


def batch_norm(x, eps=1e-8):
    pass
    

def instance_norm(x, eps=1e-8):
    pass


def group_norm(x, n_groups, eps=1e-8):
    pass

