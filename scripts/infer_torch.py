#!/usr/bin/env python3
"""JPTF1 (tiny-llm の Transformer) を PyTorch で読み、生成する。

BPE の分割も Python 側で行うので、Go のバイナリ無しで推論できる。

  python scripts/infer_torch.py --model testdata/tfm_all.bin \
      --dir ~/Downloads/discord_extract/corpus_raw/model -p "草" -n 60 --temp 1.0

  # 分割だけ確認する
  python scripts/infer_torch.py --dir <dir> --dump-ids "吾輩は猫である"
"""
import argparse
import json
import math
import os
import struct
import sys

MAGIC = b"JPTF1\n"


class BPE:
    """Go の tokenizer.Model.Encode と同じ規則（全体で最小ランクのペアから併合）。"""

    def __init__(self, dic_path, merges_path):
        with open(dic_path) as fh:
            self.vocab = json.load(fh)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.unk = 1
        # merges.txt は "A B NEW" の id 3 つ組（tokenizer.WriteMerges の出力）
        self.rank = {}
        self.merged = {}
        with open(merges_path) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                a, b, new = (int(x) for x in line.split())
                self.rank[(a, b)] = i
                self.merged[(a, b)] = new

    def encode(self, text):
        ids = [self.index.get(ch, self.unk) for ch in text]
        while len(ids) > 1:
            best, best_rank = -1, None
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best < 0:
                break
            ids[best:best + 2] = [self.merged[(ids[best], ids[best + 1])]]
        return ids


def shapes(vocab, ctx, dim, heads, layers, ffn):
    out = [(vocab, dim), (ctx, dim)]
    for _ in range(layers):
        out += [(1, dim), (1, dim), (dim, dim), (dim, dim), (dim, dim), (dim, dim),
                (1, dim), (1, dim), (1, dim), (1, dim),
                (1, dim), (1, dim), (dim, ffn), (1, ffn), (ffn, dim), (1, dim)]
    out += [(1, dim), (1, dim), (dim, vocab), (1, vocab)]
    return out


def load_weights(path):
    import numpy as np
    with open(path, "rb") as fh:
        if fh.read(6) != MAGIC:
            raise SystemExit(f"{path}: JPTF1 ではありません")
        vocab, ctx, dim, heads, layers, ffn = struct.unpack("<6i", fh.read(24))
        tokhash, = struct.unpack("<Q", fh.read(8))
        n, = struct.unpack("<i", fh.read(4))
        shp = shapes(vocab, ctx, dim, heads, layers, ffn)
        if n != len(shp):
            raise SystemExit(f"param 数が合いません: {n} != {len(shp)}")
        ws = []
        for s in shp:
            cnt = 1
            for d in s:
                cnt *= d
            ws.append(np.frombuffer(fh.read(8 * cnt), dtype="<f8").reshape(s).copy())
    return dict(vocab=vocab, ctx=ctx, dim=dim, heads=heads, layers=layers, ffn=ffn), tokhash, ws


def build_model(cfg, ws):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_torch import TF, param_order
    m = TF(cfg["vocab"], cfg["ctx"], cfg["dim"], cfg["heads"], cfg["layers"], cfg["ffn"], 0.0)
    with torch.no_grad():
        # Module.parameters() は登録順（直下のパラメータが先）なので param_order を使う
        for p, w in zip(param_order(m), ws):
            p.copy_(torch.from_numpy(w))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="testdata/tfm.bin")
    ap.add_argument("--dir", default="testdata", help="dic.txt / merges.txt のあるディレクトリ")
    ap.add_argument("-p", "--prompt", default="")
    ap.add_argument("-n", "--n", type=int, default=60)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dump-ids", default="", help="この文字列を分割して id を表示するだけ")
    args = ap.parse_args()

    bpe = BPE(os.path.join(args.dir, "dic.txt"), os.path.join(args.dir, "merges.txt"))
    if args.dump_ids:
        print(" ".join(str(i) for i in bpe.encode(args.dump_ids)))
        print("|".join(bpe.vocab[i] if i < len(bpe.vocab) else "<oob>" for i in bpe.encode(args.dump_ids)))
        return

    import torch
    cfg, _, ws = load_weights(args.model)
    if cfg["vocab"] != len(bpe.vocab):
        raise SystemExit(f"語彙数が合いません: model {cfg['vocab']} != dic {len(bpe.vocab)}")
    model = build_model(cfg, ws).to(args.device)

    ids = bpe.encode(args.prompt)
    if not ids:
        ids = [4]
    g = torch.Generator().manual_seed(args.seed)
    out = []
    with torch.no_grad():
        for _ in range(args.n):
            ctx = ids[-cfg["ctx"]:]
            x = torch.tensor([ctx], dtype=torch.long, device=args.device)
            logits = model(x)[0, -1].float().cpu()
            if args.temp <= 0:
                nxt = int(torch.argmax(logits))
            else:
                logits = logits / args.temp
                if args.topk > 0:
                    kth = torch.topk(logits, args.topk).values[-1]
                    logits[logits < kth] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=g))
            if nxt < len(bpe.vocab) and bpe.vocab[nxt].startswith("<"):
                break
            ids.append(nxt)
            out.append(bpe.vocab[nxt] if nxt < len(bpe.vocab) else "")
            print(out[-1], end="", flush=True)
    print()


if __name__ == "__main__":
    main()
