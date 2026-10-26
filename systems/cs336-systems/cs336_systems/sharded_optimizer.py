import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.local_optimizer = None
        self.owner = {}
        self.shard_sizes = [0] * self.world_size
        # this calls self.add_param_group for every group
        super().__init__(params, kwargs)

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        my_params = []
        for p in group["params"]:
            # give each param to the rank that has the fewest elements so far
            owner = self.shard_sizes.index(min(self.shard_sizes))
            self.shard_sizes[owner] += p.numel()
            self.owner[p] = owner
            if owner == self.rank:
                my_params.append(p)
        if not my_params:
            return
        my_group = dict(group)
        my_group["params"] = my_params
        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls([my_group], **self.kwargs)
        else:
            self.local_optimizer.add_param_group(my_group)

    def step(self, closure=None, **kwargs):
        loss = None
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure, **kwargs)
        # every rank sends the params it updated to everyone else
        for group in self.param_groups:
            for p in group["params"]:
                dist.broadcast(p.data, src=self.owner[p])
        return loss
