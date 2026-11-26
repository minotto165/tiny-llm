#!/usr/bin/env python3
"""tiny-llm の Transformer を PyTorch (GPU) で学習し、Go 用の JPTF1 に書き出す。

使い方:
    # 1. Go 側で ID 列を作る (dic.txt / merges.txt でトークン化)
    go run . ids -in testdata/lit.txt -out testdata/ids.bin

    # 2. GPU で学習して書き出し
    python3 scripts/train_torch.py --ids testdata/ids.bin --out testdata/tfm.bin \\
        --dim 128 --layers 3 --heads 4 --ffn 512 --ctx 128 --steps 4000 --batch 32

    # 3. Go で生成
    go run . gen -model testdata/tfm.bin -prompt "その日の午後、" -n 100

重みの並びは internal/nn の Model.All と同一。最後に --parity で
Go と同じ参照実装 (scripts/tfm_ref.py) と logits を照合する。
"""
import argparse
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

MAGIC = b"JPTF1\n"


class Block(torch.nn.Module):
    def __init__(self, d, f, scale, p=0.0):
        super().__init__()
        self.drop = torch.nn.Dropout(p)
        self.ln1g = torch.nn.Parameter(torch.ones(1, d))
        self.ln1b = torch.nn.Parameter(torch.zeros(1, d))
        self.Wq = torch.nn.Parameter(torch.randn(d, d) * 0.02)
        self.Wk = torch.nn.Parameter(torch.randn(d, d) * 0.02)
        self.Wv = torch.nn.Parameter(torch.randn(d, d) * 0.02)
        self.Wo = torch.nn.Parameter(torch.randn(d, d) * scale)
        self.Bq = torch.nn.Parameter(torch.zeros(1, d))
        self.Bk = torch.nn.Parameter(torch.zeros(1, d))
        self.Bv = torch.nn.Parameter(torch.zeros(1, d))
        self.Bo = torch.nn.Parameter(torch.zeros(1, d))
        self.ln2g = torch.nn.Parameter(torch.ones(1, d))
        self.ln2b = torch.nn.Parameter(torch.zeros(1, d))
        self.W1 = torch.nn.Parameter(torch.randn(d, f) * 0.02)
        self.B1 = torch.nn.Parameter(torch.zeros(1, f))
        self.W2 = torch.nn.Parameter(torch.randn(f, d) * scale)
        self.B2 = torch.nn.Parameter(torch.zeros(1, d))


class TF(torch.nn.Module):
    def __init__(self, vocab, ctx, d, heads, layers, ffn, p=0.0):
        super().__init__()
        self.vocab, self.ctx, self.d, self.heads, self.layers, self.ffn = vocab, ctx, d, heads, layers, ffn
        scale = 0.02 / math.sqrt(2 * layers)
        self.Wte = torch.nn.Parameter(torch.randn(vocab, d) * 0.02)
        self.Wpe = torch.nn.Parameter(torch.randn(ctx, d) * 0.01)
        self.blocks = torch.nn.ModuleList([Block(d, ffn, scale, p) for _ in range(layers)])
        self.lnfg = torch.nn.Parameter(torch.ones(1, d))
        self.lnfb = torch.nn.Parameter(torch.zeros(1, d))
        self.Wout = torch.nn.Parameter(torch.randn(d, vocab) * 0.02)
        self.Bout = torch.nn.Parameter(torch.zeros(1, vocab))

    def forward(self, ids):
        t = ids.shape[1]
        x = self.Wte[ids] + self.Wpe[:t]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=ids.device), 1)
        dh = self.d // self.heads
        for b in self.blocks:
            a = F.layer_norm(x, (self.d,), b.ln1g[0], b.ln1b[0], 1e-5)
            q, k, v = a @ b.Wq + b.Bq, a @ b.Wk + b.Bk, a @ b.Wv + b.Bv
            # (B, T, H, dh) -> (B, H, T, dh)
            sh = lambda z: z.view(z.shape[0], t, self.heads, dh).transpose(1, 2)
            q, k, v = sh(q), sh(k), sh(v)
            s = q @ k.transpose(-2, -1) / math.sqrt(dh)
            s = s.masked_fill(mask, float("-inf"))
            o = (s.softmax(dim=-1) @ v).transpose(1, 2).reshape(ids.shape[0], t, self.d)
            x = x + b.drop(o @ b.Wo + b.Bo)

            a = F.layer_norm(x, (self.d,), b.ln2g[0], b.ln2b[0], 1e-5)
            pre = a @ b.W1 + b.B1
            act = 0.5 * pre * (1 + torch.tanh(math.sqrt(2 / math.pi) * (pre + 0.044715 * pre**3)))
            x = x + b.drop(act @ b.W2 + b.B2)
        x = F.layer_norm(x, (self.d,), self.lnfg[0], self.lnfb[0], 1e-5)
        return x @ self.Wout + self.Bout


def param_order(model):
    yield model.Wte
    yield model.Wpe
    for b in model.blocks:
        yield from (b.ln1g, b.ln1b, b.Wq, b.Wk, b.Wv, b.Wo, b.Bq, b.Bk, b.Bv, b.Bo,
                    b.ln2g, b.ln2b, b.W1, b.B1, b.W2, b.B2)
    yield model.lnfg
    yield model.lnfb
    yield model.Wout
    yield model.Bout


def fnv64(paths):
    h = 0xCBF29CE484222325
    for p in paths:
        with open(p, "rb") as fh:
            for byte in fh.read():
                h ^= byte
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def save(model, out, tokhash):
    with open(out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<6i", model.vocab, model.ctx, model.d, model.heads, model.layers, model.ffn))
        fh.write(struct.pack("<Q", tokhash))
        ps = list(param_order(model))
        fh.write(struct.pack("<i", len(ps)))
        for p in ps:
            fh.write(p.detach().double().cpu().numpy().astype("<f8").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="testdata/ids.bin", help="go run . ids の出力")
    ap.add_argument("--out", default="testdata/tfm.bin")
    ap.add_argument("--vocab", type=int, default=0, help="語彙数 (省略時は dic.txt から)")
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.0, help="0.1 程度を推奨")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=0, help="この step 数まで線形に lr を上げる")
    ap.add_argument("--cosine", action="store_true", help="warmup 後を余弦減衰させる")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--parity", action="store_true", help="書き出し後に参照実装と照合する")
    ap.add_argument("--rows", type=int, default=0,
                    help="ID列をこの長さの行に区切ってサンプルする（セクション窓用。0=ランダム位置から切り出す）")
    ap.add_argument("--holdout-rows", type=float, default=0.0,
                    help="末尾のこの割合の行を学習から外し、最後に検証lossを出す（--rows と併用）")
    ap.add_argument("--dir", default="",
                    help="dic.txt / merges.txt のあるディレクトリ (省略時は ../testdata)")
    args = ap.parse_args()

    here0 = os.path.dirname(os.path.abspath(__file__))
    tokdir = args.dir or os.path.join(here0, "..", "testdata")
    if args.vocab == 0:
        import json
        with open(os.path.join(tokdir, "dic.txt"), encoding="utf-8") as fh:
            args.vocab = len(json.load(fh))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with open(args.ids, "rb") as fh:
        (n,) = struct.unpack("<i", fh.read(4))
        ids = np.frombuffer(fh.read(4 * n), dtype="<i4")
    print(f"ids: {len(ids):,} tokens", file=sys.stderr)

    model = TF(args.vocab, args.ctx, args.dim, args.heads, args.layers, args.ffn, args.dropout).to(args.device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model: {nparam/1e6:.2f}M params, device {args.device}", file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    t0 = time.time()
    data = torch.from_numpy(ids.astype(np.int64))
    win = args.rows if args.rows else args.ctx
    nrows = 0
    nrows_train = 0
    if args.rows:
        nrows = len(ids) // args.rows
        nrows_train = nrows
        if args.holdout_rows > 0:
            nrows_train = max(1, int(nrows * (1.0 - args.holdout_rows)))
        print(f"rows: {nrows:,} 行 × {args.rows} トークン "
              f"(学習 {nrows_train:,} / 検証 {nrows - nrows_train:,})", file=sys.stderr)
    for step in range(1, args.steps + 1):
        if args.warmup > 0 and step <= args.warmup:
            lr = args.lr * step / args.warmup
        elif args.cosine:
            prog = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        else:
            lr = args.lr
        for g in opt.param_groups:
            g["lr"] = lr
        if args.rows:
            rs = rng.integers(0, nrows_train, size=args.batch)
            idx = rs[:, None] * args.rows + np.arange(args.rows)[None, :]
        else:
            starts = rng.integers(0, len(ids) - args.ctx - 1, size=args.batch)
            idx = torch.from_numpy(starts[:, None] + np.arange(args.ctx + 1)[None, :])
        chunk = data[idx].to(args.device)
        logits = model(chunk[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, args.vocab), chunk[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            per = (time.time() - t0) / step
            seen = step * args.batch * win
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  f"({per*1000:.0f} ms/step, {seen/len(ids):.2f} epoch, 残り {(args.steps-step)*per/60:.1f} 分)",
                  file=sys.stderr)

    here = os.path.dirname(os.path.abspath(__file__))
    if args.rows and args.holdout_rows > 0 and nrows > nrows_train:
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for r0 in range(nrows_train, nrows, 64):
                rs = np.arange(r0, min(r0 + 64, nrows))
                vidx = rs[:, None] * args.rows + np.arange(args.rows)[None, :]
                vchunk = data[vidx].to(args.device)
                vlogits = model(vchunk[:, :-1])
                vloss = F.cross_entropy(vlogits.reshape(-1, args.vocab),
                                        vchunk[:, 1:].reshape(-1), reduction="sum")
                tot += float(vloss.item())
                cnt += int(vchunk[:, 1:].numel())
        model.train()
        print(f"val loss: {tot/cnt:.4f} (検証 {nrows-nrows_train:,} 行)", file=sys.stderr)
    tokhash = fnv64([os.path.join(tokdir, "dic.txt"),
                     os.path.join(tokdir, "merges.txt")])
    save(model, args.out, tokhash)
    print(f"saved: {args.out}", file=sys.stderr)

    if args.parity:
        model.eval()
        sys.path.insert(0, here)
        import tfm_ref
        cfg, _, w = tfm_ref.load(args.out)
        probe = list(range(1, min(args.ctx, 8) + 1))
        ref = tfm_ref.forward(cfg, w, probe)
        model.double()
        with torch.no_grad():
            got = model(torch.tensor([probe], dtype=torch.long).to(args.device)).double().cpu().numpy()[0]
        diff = float(np.abs(ref - got).max())
        print(f"parity: max diff {diff:.3g}", file=sys.stderr)
        if diff > 1e-4:
            raise SystemExit("参照実装と一致しません")


if __name__ == "__main__":
    main()
