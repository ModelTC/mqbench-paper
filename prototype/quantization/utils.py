import spring.linklink as link
import torch


def sync_tensor(tensor):
    if tensor.is_cuda is True:
        world_size = link.get_world_size()
        link.allreduce(tensor / world_size)


def is_per_channel(qshcme):
    return qshcme == torch.per_channel_affine or qshcme == torch.per_channel_symmetric


def is_symmetric(qscheme):
    return qscheme == torch.per_channel_symmetric or qscheme == torch.per_tensor_symmetric


def pot_quantization(tensor: torch.Tensor):
    log2t = torch.log2(tensor)
    log2t = (torch.round(log2t)-log2t).detach() + log2t
    return 2 ** log2t


def grad_scale(t, scale):
    return (t - (t*scale)).detach() + (t*scale)

