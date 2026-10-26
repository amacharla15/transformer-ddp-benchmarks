import math

import torch
import triton
import triton.language as tl


def flash_backward(Q, K, V, O, dO, L, is_causal):
    # recompute P from Q, K and L instead of storing it in forward
    scale = 1 / math.sqrt(Q.shape[-1])
    S = (Q @ K.transpose(-1, -2)).float() * scale
    if is_causal:
        q_idx = torch.arange(Q.shape[-2], device=Q.device)
        k_idx = torch.arange(K.shape[-2], device=Q.device)
        S = torch.where(q_idx[:, None] >= k_idx[None, :], S, S - 1e6)
    P = torch.exp(S - L[..., None])
    D = (O.float() * dO.float()).sum(-1)
    dP = (dO @ V.transpose(-1, -2)).float()
    dS = P * (dP - D[..., None])
    dV = P.to(V.dtype).transpose(-1, -2) @ dO
    dQ = (dS.to(Q.dtype) @ K) * scale
    dK = (dS.to(K.dtype).transpose(-1, -2) @ Q) * scale
    return dQ, dK, dV


compiled_flash_backward = torch.compile(flash_backward)


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, n_q, d = Q.shape
        n_k = K.shape[1]
        q_tile, k_tile = 16, 16
        scale = 1 / math.sqrt(d)
        O = torch.empty_like(Q)
        L = torch.empty(batch, n_q, device=Q.device, dtype=torch.float32)

        for i in range(0, n_q, q_tile):
            q = Q[:, i : i + q_tile].float()
            o = torch.zeros(batch, q.shape[1], d, device=Q.device)
            l = torch.zeros(batch, q.shape[1], device=Q.device)
            m = torch.full((batch, q.shape[1]), float("-inf"), device=Q.device)
            for j in range(0, n_k, k_tile):
                k = K[:, j : j + k_tile].float()
                v = V[:, j : j + k_tile].float()
                s = q @ k.transpose(-1, -2) * scale
                if is_causal:
                    q_idx = torch.arange(i, i + q.shape[1], device=Q.device)
                    k_idx = torch.arange(j, j + k.shape[1], device=Q.device)
                    s = torch.where(q_idx[:, None] >= k_idx[None, :], s, s - 1e6)
                # online softmax: rescale what we have so far to the new running max
                m_new = torch.maximum(m, s.amax(-1))
                p = torch.exp(s - m_new[..., None])
                alpha = torch.exp(m - m_new)
                l = alpha * l + p.sum(-1)
                o = alpha[..., None] * o + p @ v
                m = m_new
            O[:, i : i + q_tile] = (o / l[..., None]).to(Q.dtype)
            L[:, i : i + q_tile] = m + torch.log(l)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(Q_block_ptr)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    n_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        # key tiles fully above the diagonal are all masked, skip them
        n_tiles = tl.cdiv(tl.minimum(N_KEYS, (query_tile_index + 1) * Q_TILE_SIZE), K_TILE_SIZE)

    for j in range(n_tiles):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        s = tl.dot(q, tl.trans(k)) * scale
        if is_causal:
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            s = tl.where(q_idx[:, None] >= k_idx[None, :], s, s - 1e6)
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = alpha * l + tl.sum(p, axis=1)
        o = o * alpha[:, None]
        o = tl.dot(p.to(v.dtype), v, acc=o)
        m = m_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / l[:, None]
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, m + tl.log(l))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, n_q, d = Q.shape
        n_k = K.shape[1]
        # sizes are powers of 2 and at least 16, so tiles always divide evenly
        q_tile = min(64, n_q)
        k_tile = min(64, n_k)
        O = torch.empty_like(Q)
        L = torch.empty(batch, n_q, device=Q.device, dtype=torch.float32)
        grid = (triton.cdiv(n_q, q_tile), batch)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_q, n_k,
            1 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=q_tile,
            K_TILE_SIZE=k_tile,
            is_causal=is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = compiled_flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
