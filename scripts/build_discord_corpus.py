#!/usr/bin/env python3
"""Discord 全発言から 2 種類のコーパスを作る。

  all.jsonl   … 全発言（話者ID付き）の対話データ
  mine.jsonl  … 自分（ME）の発言だけを、直前の文脈付きで並べたもの
  train/valid/test_{all,mine}.txt … 言語モデル用のプレーンテキスト

使い方: python3 scripts/build_discord_corpus.py [--gap 分] [--me ユーザーID]

入力: testdata/discord/raw/all_part*.json (Discord API の生 JSON)
出力: testdata/discord/corpus/
"""
import argparse
import glob
import hashlib
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "testdata", "discord")
ME = "1192260281984893018"

URL_RE = re.compile(r"https?://\S+")
CUSTOM_EMOJI_RE = re.compile(r"<a?:([a-zA-Z0-9_]+):\d+>")
MENTION_RE = re.compile(r"<@!?(\d+)>")
CHANNEL_RE = re.compile(r"<#(\d+)>")
ROLE_RE = re.compile(r"<@&(\d+)>")
CODEBLOCK_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
MD_RE = re.compile(r"(\*\*|__|\*|_|~~|\|\|)")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def normalize(text, keep_urls=False, keep_mentions=False, names=None):
    if not text:
        return ""
    t = text.replace("\r\n", "\n")
    t = CODEBLOCK_RE.sub(lambda m: m.group(0).strip("`").strip(), t)
    t = INLINE_CODE_RE.sub(r"\1", t)
    t = MD_RE.sub("", t)
    t = "\n".join(l.lstrip("> ").rstrip() for l in t.split("\n"))
    t = CUSTOM_EMOJI_RE.sub(r":\1:", t)
    if names:
        t = MENTION_RE.sub(lambda m: "@" + names.get(m.group(1), m.group(0)), t)
    elif not keep_mentions:
        t = MENTION_RE.sub("@user", t)
        t = CHANNEL_RE.sub("#channel", t)
        t = ROLE_RE.sub("@role", t)
    if not keep_urls:
        t = URL_RE.sub("<url>", t)
    t = CTRL_RE.sub("", t)
    t = unicodedata.normalize("NFKC", t)
    t = "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def load(paths):
    seen, out = set(), []
    for p in sorted(paths):
        try:
            data = json.load(open(p))
        except Exception as e:
            print("skip", p, e)
            continue
        for m in data:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            out.append(m)
    out.sort(key=lambda m: (m["ts"], int(m["id"])))
    return out


def dedupe(rows):
    """連投の同一発言と、長文の使い回しだけを畳む（短文の繰り返しは残す）。"""
    seen_long, kept = set(), []
    prev = None
    for m in rows:
        key = re.sub(r"[\s\W_]+", "", m["text"]).lower()
        if prev and prev["author_id"] == m["author_id"] and prev["channel"] == m["channel"] \
                and re.sub(r"[\s\W_]+", "", prev["text"]).lower() == key:
            prev["repeat"] = prev.get("repeat", 1) + 1
            continue
        if len(key) >= 60:
            if key in seen_long:
                continue
            seen_long.add(key)
        kept.append(m)
        prev = m
    return kept


def sessions(rows, gap):
    from datetime import datetime
    cur, cur_key = [], None
    for m in rows:
        ts = datetime.fromisoformat(m["ts"].replace("Z", "+00:00"))
        key = (m["guild"], m["channel"], m["thread"])
        if cur and (key != cur_key or (ts - cur[-1]["_ts"]).total_seconds() > gap * 60):
            yield cur
            cur = []
        cur_key = key
        m["_ts"] = ts
        cur.append(m)
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--me", default=ME)
    ap.add_argument("--glob", default=os.path.join(DATA, "raw", "all_part*.json"))
    ap.add_argument("--out", default=os.path.join(DATA, "corpus"))
    ap.add_argument("--keep-urls", action="store_true", help="URL を <url> に畳まずそのまま残す")
    ap.add_argument("--no-bots", action="store_true", help="bot の発言を除外する")
    ap.add_argument("--keep-mentions", action="store_true", help="メンションを @user に畳まずそのまま残す")
    ap.add_argument("--drop-urls", action="store_true", help="URL を本文から削除する")
    ap.add_argument("--drop-ascii-noise", action="store_true", help="日本語を含まない英数字ノイズや bot コマンド行を落とす")
    ap.add_argument("--url-domain", action="store_true", help="URL をドメインだけに置換する (x.com など)")
    ap.add_argument("--cap-repeat", type=int, default=0, help="同一ch・同一人・同一文の連投を N 回までにする")
    ap.add_argument("--url-half", action="store_true", help="URL を含む発言を半分に間引く")
    ap.add_argument("--short-half", action="store_true", help="短い発言を半分に間引く")
    ap.add_argument("--short-chars", type=int, default=5, help="「短い」の閾値")
    ap.add_argument("--names", default=os.path.join(DATA, "raw", "names.json"),
                    help="{id: 表示名} の JSON。あれば <@id> を @名前 に置換")
    args = ap.parse_args()
    names_map = json.load(open(args.names)) if args.names and os.path.exists(args.names) else None

    os.makedirs(args.out, exist_ok=True)
    rows = load(glob.glob(args.glob))
    print("raw", len(rows))
    if args.no_bots:
        before = len(rows)
        rows = [m for m in rows if not m.get("bot")]
        print("bots removed", before - len(rows))
    for m in rows:
        m["text"] = normalize(m["text"], args.keep_urls, args.keep_mentions, names_map)
    rows = [m for m in rows if m["text"]]
    print("after normalize", len(rows))
    if args.url_domain:
        from urllib.parse import urlparse
        def dom(m):
            m = dict(m)
            def sub(mm):
                u = mm.group(0)
                try:
                    host = urlparse(u).netloc.lower()
                except Exception:
                    return u
                if host.startswith("www."):
                    host = host[4:]
                return host
            m["text"] = re.sub(r"https?://\S+", sub, m["text"])
            return m
        rows = [dom(m) for m in rows]
        print("after url-domain", len(rows))
    if args.drop_urls:
        def strip_url(m):
            m = dict(m)
            m["text"] = re.sub(r"\s*https?://\S+", "", m["text"]).strip()
            return m
        rows = [strip_url(m) for m in rows]
        rows = [m for m in rows if m["text"]]
        print("after drop-urls", len(rows))
    if args.cap_repeat:
        last, kept, dropped = {}, [], 0
        for m in rows:
            k = (m["channel"], m["author_id"], m["text"])
            n = last.get(k, 0)
            if n >= args.cap_repeat:
                dropped += 1
                continue
            last[k] = n + 1
            kept.append(m)
        rows = kept
        print(f"after cap-repeat(<= {args.cap_repeat})", len(rows), f"dropped {dropped}")
    if args.drop_ascii_noise:
        JP = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
        CMD = re.compile(r"^\s*(m!|!|/|\.)\S")
        def keep(m):
            t = m["text"]
            if JP.search(t):
                return True
            if CMD.match(t):
                return False
            if re.fullmatch(r"[wｗ]+", t, re.I):
                return True
            if not re.search(r"[0-9A-Za-z]", t):
                return True
            if re.fullmatch(r"[!?！？…。、~〜ー\-]+", t):
                return True
            if len(t) <= 2:
                return True
            return False
        before = len(rows)
        rows = [m for m in rows if keep(m)]
        print("after drop-ascii-noise", len(rows), "dropped", before - len(rows))
    if args.url_half or args.short_half:
        import hashlib
        def keep(m, tag):
            h = hashlib.md5((tag + m["id"]).encode()).digest()[0]
            return h % 2 == 0
        before = len(rows)
        if args.url_half:
            rows = [m for m in rows if "http" not in m["text"] or keep(m, "u")]
        if args.short_half:
            rows = [m for m in rows if len(m["text"]) > args.short_chars or keep(m, "s")]
        print("after half-sampling", len(rows), f"({before} -> {len(rows)})")
    rows = dedupe(rows)
    print("after dedupe", len(rows))

    convs, mine = [], []
    for sid, s in enumerate(sessions(rows, args.gap)):
        cid = "c%05d" % sid
        turns = [{"s": m["author_id"], "t": m["text"], "ts": m["ts"]} for m in s]
        convs.append({"conv_id": cid, "channel": s[0]["channel"], "thread": s[0]["thread"],
                      "guild": s[0]["guild"], "turns": turns})
        for i, m in enumerate(s):
            if m["author_id"] == args.me:
                prev = s[max(0, i - 3):i]
                mine.append({"id": m["id"], "ts": m["ts"], "channel": m["channel"],
                             "thread": m["thread"], "text": m["text"],
                             "context": [{"s": p["author_id"], "t": p["text"]} for p in prev]})

    with open(os.path.join(args.out, "all.jsonl"), "w") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "mine.jsonl"), "w") as f:
        for m in mine:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def dump(conv_subset, name):
        txt = "\n\n".join("\n".join(t["t"] for t in c["turns"]) for c in conv_subset)
        open(os.path.join(args.out, name), "w").write(txt)
        return len(txt)

    n = len(convs)
    i90, i95 = int(n * 0.9), int(n * 0.95)
    sizes = {
        "train_all.txt": dump(convs[:i90], "train_all.txt"),
        "valid_all.txt": dump(convs[i90:i95], "valid_all.txt"),
        "test_all.txt": dump(convs[i95:], "test_all.txt"),
    }
    mtxt = "\n".join(m["text"] for m in mine)
    open(os.path.join(args.out, "mine.txt"), "w").write(mtxt)
    sizes["mine.txt"] = len(mtxt)

    speakers = {}
    for c in convs:
        for t in c["turns"]:
            speakers[t["s"]] = speakers.get(t["s"], 0) + 1
    report = {
        "messages": len(rows),
        "conversations": n,
        "mine": len(mine),
        "speakers": len(speakers),
        "top_speakers": sorted(speakers.items(), key=lambda x: -x[1])[:10],
        "sizes": sizes,
    }
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
