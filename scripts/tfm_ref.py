#!/usr/bin/env python3
"""JPTF1 (Go の Transformer) 重みを numpy で読み、logits を計算する参照実装。

Go 実装との一致確認 (parity) 用。学習済みモデルを読み込んで
`--emit-parity` に参照 logits を書き出すと、Go 側のテストが照合する。

    python3 scripts/tfm_ref.py --model testdata/tfm.bin --ids 1,23,456
    python3 scripts/tfm_ref.py --model testdata/tfm.bin --ids-file /tmp/ids.txt --emit-parity testdata/parity.bin
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = b"JPTF1\n"


def shapes(vocab, ctx, dim, heads, layers, ffn):
    """Go の Model.All と同じ順序で (名前, 形) を返す。"""
    d, f = dim, ffn
    out = [("Wte", (vocab, d)), ("Wpe", (ctx, d))]
    for l in range(layers):
        out += [
            (f"L{l}.ln1g", (1, d)), (f"L{l}.ln1b", (1, d)),
            (f"L{l}.Wq", (d, d)), (f"L{l}.Wk", (d, d)),
            (f"L{l}.Wv", (d, d)), (f"L{l}.Wo", (d, d)),
            (f"L{l}.Bq", (1, d)), (f"L{l}.Bk", (1, d)),
            (f"L{l}.Bv", (1, d)), (f"L{l}.Bo", (1, d)),
            (f"L{l}.ln2g", (1, d)), (f"L{l}.ln2b", (1, d)),
            (f"L{l}.W1", (d, f)), (f"L{l}.B1", (1, f)),
            (f"L{l}.W2", (f, d)), (f"L{l}.B2", (1, d)),
        ]
    out += [("lnfg", (1, d)), ("lnfb", (1, d)), ("Wout", (d, vocab)), ("Bout", (1, vocab))]
    return out


def load(path):
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise SystemExit(f"{path}: JPTF1 ではありません")
        vocab, ctx, dim, heads, layers, ffn = struct.unpack("<6i", fh.read(24))
        (tokhash,) = struct.unpack("<Q", fh.read(8))
        (n,) = struct.unpack("<i", fh.read(4))
        w = {}
        for name, shape in shapes(vocab, ctx, dim, heads, layers, ffn):
            cnt = int(np.prod(shape))
            w[name] = np.frombuffer(fh.read(8 * cnt), dtype="<f8").reshape(shape)
    cfg = dict(vocab=vocab, ctx=ctx, dim=dim, heads=heads, layers=layers, ffn=ffn)
    return cfg, tokhash, w


def ln(x, g, b, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def softmax_masked(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def forward(cfg, w, ids):
    t = len(ids)
    d, h = cfg["dim"], cfg["heads"]
    dh = d // h
    x = w["Wte"][np.asarray(ids)] + w["Wpe"][:t]
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    for l in range(cfg["layers"]):
        p = lambda name: w[f"L{l}.{name}"]
        a = ln(x, p("ln1g"), p("ln1b"))
        q = a @ p("Wq") + p("Bq")
        k = a @ p("Wk") + p("Bk")
        v = a @ p("Wv") + p("Bv")
        o = np.zeros((t, d))
        for hh in range(h):
            sl = slice(hh * dh, (hh + 1) * dh)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(dh)
            s[mask] = -np.inf
            o[:, sl] = softmax_masked(s) @ v[:, sl]
        x = x + o @ p("Wo") + p("Bo")

        a = ln(x, p("ln2g"), p("ln2b"))
        pre = a @ p("W1") + p("B1")
        act = 0.5 * pre * (1 + np.tanh(np.sqrt(2 / np.pi) * (pre + 0.044715 * pre**3)))
        x = x + act @ p("W2") + p("B2")
    x = ln(x, w["lnfg"], w["lnfb"])
    return x @ w["Wout"] + w["Bout"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", help="カンマ区切りのトークン ID")
    ap.add_argument("--ids-file", help="1 行 1 ID のテキスト")
    ap.add_argument("--emit-parity", help="参照 logits の書き出し先 (Go テスト用)")
    args = ap.parse_args()

    cfg, _, w = load(args.model)
    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    elif args.ids_file:
        ids = [int(x) for x in open(args.ids_file).read().split()]
    else:
        raise SystemExit("--ids か --ids-file が必要です")
    if len(ids) > cfg["ctx"]:
        raise SystemExit(f"ids が長すぎます (ctx={cfg['ctx']})")

    logits = forward(cfg, w, ids)
    if args.emit_parity:
        with open(args.emit_parity, "wb") as fh:
            fh.write(struct.pack("<i", len(ids)))
            fh.write(np.asarray(ids, dtype="<i4").tobytes())
            fh.write(np.asarray(logits, dtype="<f8").tobytes())
        print(f"wrote {args.emit_parity}: {len(ids)} ids x {cfg['vocab']} logits", file=sys.stderr)
    else:
        np.set_printoptions(precision=4, suppress=True, linewidth=160)
        print(logits[0][:10])


if __name__ == "__main__":
    main()
