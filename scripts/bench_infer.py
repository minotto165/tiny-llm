"""Transformer 推論のベンチマーク（KV キャッシュあり / なし）。

    uv run --with torch python scripts/bench_infer.py --model testdata/tfm_v3.bin \
        --dir testdata/discord/model --n 500
"""

import argparse
import math
import time

import torch

from infer_torch import BPE, build_model, load_weights


def build(model_path):
    cfg, _, ws = load_weights(model_path)
    m = build_model(cfg, ws)
    return m, cfg["ctx"], cfg["heads"]


@torch.no_grad()
def forward_full(m, ids):
    return m(ids)


def split_heads(x, heads):
    b, t, d = x.shape
    return x.view(b, t, heads, d // heads).transpose(1, 2)


@torch.no_grad()
def step_cached(m, id, past, heads):
    """1 トークンだけ処理する。past は層ごとの (k, v) で shape (1,H,T,dh)。"""
    d = m.d
    pos = 0 if not past else past[0][0].shape[2]
    x = m.Wte[id].view(1, 1, d) + m.Wpe[pos : pos + 1].view(1, 1, d)
    new_past = []
    for i, b in enumerate(m.blocks):
        a = torch.nn.functional.layer_norm(x, (d,), b.ln1g[0], b.ln1b[0], 1e-5)
        q = split_heads(a @ b.Wq + b.Bq, heads)
        k = split_heads(a @ b.Wk + b.Bk, heads)
        v = split_heads(a @ b.Wv + b.Bv, heads)
        if past:
            k = torch.cat([past[i][0], k], dim=2)
            v = torch.cat([past[i][1], v], dim=2)
        new_past.append((k, v))
        s = (q @ k.transpose(2, 3)) / math.sqrt(d // heads)
        p = torch.softmax(s, dim=-1)
        o = (p @ v).transpose(1, 2).reshape(1, 1, d)
        x = x + o @ b.Wo + b.Bo
        a = torch.nn.functional.layer_norm(x, (d,), b.ln2g[0], b.ln2b[0], 1e-5)
        h = a @ b.W1 + b.B1
        h = 0.5 * h * (1 + torch.tanh(math.sqrt(2 / math.pi) * (h + 0.044715 * h**3)))
        x = x + h @ b.W2 + b.B2
    a = torch.nn.functional.layer_norm(x, (d,), m.lnfg[0], m.lnfb[0], 1e-5)
    return a @ m.Wout + m.Bout, new_past


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prompt", default="おはよう")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    m, ctx, heads = build(args.model)
    tok = BPE(f"{args.dir}/dic.txt", f"{args.dir}/merges.txt")

    ids = tok.encode(args.prompt)
    prompt_len = len(ids)

    # キャッシュあり / なしでロジットが一致するか（最初の 1 トークン）
    full = forward_full(m, torch.tensor([ids]))
    got, past = step_cached(m, torch.tensor([ids[-1]]), None, heads)
    # プロンプト全体を通す
    past = []
    for t in ids:
        got, past = step_cached(m, torch.tensor([t]), past, heads)
    diff = (got.view(-1) - full[0, -1]).abs().max().item()
    print(f"parity: max diff {diff:.2e}")

    # 1) キャッシュなし（毎回 ctx 全体を再計算）
    t0 = time.time()
    seq = list(ids)
    with torch.no_grad():
        for _ in range(args.n):
            toks = torch.tensor([seq[-ctx:]])
            logits = m(toks)[0, -1]
            seq.append(int(torch.argmax(logits)))
    t_full = time.time() - t0

    # 2) キャッシュあり
    past = []
    for t in ids:
        _, past = step_cached(m, torch.tensor([t]), past, heads)
    t0 = time.time()
    seq = list(ids)
    last = torch.tensor([ids[-1]])
    with torch.no_grad():
        for _ in range(args.n):
            logits, past = step_cached(m, last, past, heads)
            if past[0][0].shape[2] >= ctx:
                past = None  # 窓の張り直し（Go と同じ）
                for t in seq[-ctx // 2 :]:
                    logits, past = step_cached(m, torch.tensor([t]), past, heads)
            last = torch.tensor([int(torch.argmax(logits))])
            seq.append(int(last))
    t_cache = time.time() - t0

    print(f"prompt {prompt_len} tokens, generate {args.n}")
    print(f"no-cache : {t_full:6.2f}s  {1000*t_full/args.n:6.2f} ms/token")
    print(f"kv-cache : {t_cache:6.2f}s  {1000*t_cache/args.n:6.2f} ms/token")


if __name__ == "__main__":
    main()
