import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding, Linear


class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, fsdp, i):
        w = fsdp.get_weight(i)
        x = x.to(w.dtype)
        out = x @ w.t()
        fsdp.after_forward(i)
        ctx.save_for_backward(x)
        ctx.fsdp = fsdp
        ctx.i = i
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        fsdp, i = ctx.fsdp, ctx.i
        # the full weight was freed after forward, so gather it again
        w = fsdp.get_weight(i)
        fsdp.prefetch_backward(i)
        grad_out = grad_out.to(w.dtype)
        grad_x = grad_out @ w
        grad_w = grad_out.reshape(-1, w.shape[0]).t() @ x.reshape(-1, w.shape[1])
        fsdp.free(i)
        fsdp.reduce_grad(i, grad_w)
        # the shard's grad is set later in finish_gradient_synchronization
        return grad_x, None, None, None


class EmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, shard, fsdp, i):
        w = fsdp.get_weight(i)
        out = w[ids]
        fsdp.after_forward(i)
        ctx.save_for_backward(ids)
        ctx.fsdp = fsdp
        ctx.i = i
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (ids,) = ctx.saved_tensors
        fsdp, i = ctx.fsdp, ctx.i
        # no weight needed here, just add each output grad into its row
        grad_w = torch.zeros(fsdp.shapes[i], dtype=torch.float32, device=grad_out.device)
        grad_w.index_add_(0, ids.reshape(-1), grad_out.reshape(-1, grad_w.shape[1]).float())
        fsdp.reduce_grad(i, grad_w)
        return None, None, None, None


class FSDP(nn.Module):
    def __init__(self, module, compute_dtype=None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        for t in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(t.data, src=0)

        # shard Linear and Embedding weights, keep small params like norms replicated
        self.layers = [m for m in module.modules() if isinstance(m, (Linear, Embedding))]
        self.shapes = []
        for i, layer in enumerate(self.layers):
            w = layer.weight
            self.shapes.append(w.shape)
            n = (w.numel() + self.world_size - 1) // self.world_size
            flat = w.data.flatten()
            flat = torch.cat([flat, flat.new_zeros(n * self.world_size - flat.numel())])
            w.data = flat[self.rank * n : (self.rank + 1) * n].clone()
            layer.forward = self.make_forward(layer, i)

        self.gathered = [None] * len(self.layers)
        # guess the call order for the first step, then use the real one
        self.order = list(range(len(self.layers)))
        self.called = []
        self.handles = []
        self.shard_grads = []
        sharded = {id(layer.weight) for layer in self.layers}
        for p in module.parameters():
            if p.requires_grad and id(p) not in sharded:
                p.register_post_accumulate_grad_hook(self.all_reduce_grad)

    def make_forward(self, layer, i):
        fn = LinearFn if isinstance(layer, Linear) else EmbeddingFn

        def forward(x):
            return fn.apply(x, layer.weight, self, i)

        return forward

    def forward(self, *inputs, **kwargs):
        self.called = []
        for i in self.order[:2]:
            self.start_gather(i)
        out = self.module(*inputs, **kwargs)
        self.order = self.called
        return out

    def start_gather(self, i):
        if self.gathered[i] is not None:
            return
        shard = self.layers[i].weight.data
        if self.compute_dtype is not None:
            shard = shard.to(self.compute_dtype)
        full = torch.empty(shard.numel() * self.world_size, dtype=shard.dtype, device=shard.device)
        handle = dist.all_gather_into_tensor(full, shard, async_op=True)
        self.gathered[i] = (full, handle, shard)

    def get_weight(self, i):
        self.start_gather(i)
        full, handle, _ = self.gathered[i]
        handle.wait()
        return full[: self.shapes[i].numel()].view(self.shapes[i])

    def free(self, i):
        self.gathered[i] = None

    def after_forward(self, i):
        self.called.append(i)
        self.free(i)
        # the layer two places ahead starts gathering only after this one is done
        pos = self.order.index(i) + 2
        if pos < len(self.order):
            self.start_gather(self.order[pos])

    def prefetch_backward(self, i):
        # backward goes in reverse call order, and embeddings need no weight in backward
        pos = self.order.index(i) - 1
        if pos >= 0 and isinstance(self.layers[self.order[pos]], Linear):
            self.start_gather(self.order[pos])

    def reduce_grad(self, i, grad):
        shard = self.layers[i].weight
        flat = grad.to(shard.dtype).flatten()
        flat = torch.cat([flat, flat.new_zeros(shard.numel() * self.world_size - flat.numel())])
        flat /= self.world_size
        out = torch.empty_like(shard.data)
        self.handles.append(dist.reduce_scatter_tensor(out, flat, async_op=True))
        self.shard_grads.append((shard, out))

    def all_reduce_grad(self, p):
        p.grad /= self.world_size
        self.handles.append(dist.all_reduce(p.grad, async_op=True))

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        for p, grad in self.shard_grads:
            p.grad = grad
        self.handles.clear()
        self.shard_grads.clear()
        # drop any prefetched weight that was not used
        for g in self.gathered:
            if g is not None:
                g[1].wait()
        self.gathered = [None] * len(self.layers)

    def gather_full_params(self):
        index = {id(layer.weight): i for i, layer in enumerate(self.layers)}
        params = {}
        for name, p in self.module.named_parameters():
            if id(p) not in index:
                params[name] = p.data
                continue
            shape = self.shapes[index[id(p)]]
            full = torch.empty(p.numel() * self.world_size, dtype=p.dtype, device=p.device)
            dist.all_gather_into_tensor(full, p.data)
            params[name] = full[: shape.numel()].view(shape)
        return params
