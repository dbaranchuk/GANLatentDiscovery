import torch
from scipy.stats import truncnorm


def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)
    else:
        return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)


def truncated_noise(size, truncation=1.0):
  values = truncnorm.rvs(-2, 2, size=size)
  return truncation * values


def one_hot(dims, value, indx):
    vec = torch.zeros(dims)
    vec[indx] = value
    return vec


def run_in_background(func: callable, *args, **kwargs):
    """ run f(*args, **kwargs) in background and return Future for its outputs """
    future = Future()

    def _run():
        try:
            future.set_result(func(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)

    Thread(target=_run).start()
    return future