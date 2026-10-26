import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def broadcast_from_rank0(module):
    # module.parameters() lists a tied weight only once
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=0)


class NaiveDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        broadcast_from_rank0(module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad)
            p.grad /= world_size


class FlatDDP(NaiveDDP):
    def finish_gradient_synchronization(self):
        params = [p for p in self.module.parameters() if p.grad is not None]
        grads = [p.grad for p in params]
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat /= dist.get_world_size()
        for p, g in zip(params, _unflatten_dense_tensors(flat, grads)):
            p.grad.copy_(g)


class OverlapDDP(NaiveDDP):
    def __init__(self, module):
        super().__init__(module)
        self.handles = []
        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.sync_grad)

    def sync_grad(self, p):
        # runs during backward as soon as p.grad is ready
        p.grad /= dist.get_world_size()
        self.handles.append(dist.all_reduce(p.grad, async_op=True))

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
