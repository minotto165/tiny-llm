"""
!!! VIBE-CODED !!!

日本語自然言語処理用コーパス収集スクリプト
========================================
ソース:
  1. 青空文庫 (aozora.gr.jp) - パブリックドメイン近代文学 25作品
  2. Wikipedia日本語版 API   - 厳選リスト + 全記事空間から約1200記事

使い方:
  python3 collect_corpus.py

出力:
  testdata/input.txt        -- 結合済みコーパス (トークナイザの入力)
  testdata/corpus_stats.txt -- 収集統計

出力先はこのスクリプトのあるディレクトリ基準なので、どこから実行してもよい。
Wikipedia は prop=revisions で wikitext を 50 記事まとめて取得し、
スクリプト内でプレーンテキストに変換する（prop=extracts の全文は
1 リクエスト 1 記事の制約があるため使わない）。
依存ライブラリ: 標準ライブラリのみ (Python 3.6+)
"""

import urllib.request
import urllib.parse
import urllib.error
import html
import random
import csv
import zipfile
import io
import re
import os
import sys
import json
import time
import datetime

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(ROOT, "testdata", "input.txt")
STATS_FILE  = os.path.join(ROOT, "testdata", "corpus_stats.txt")
# 取得済みテキストのキャッシュ。途中で止めても次の実行で続きから集める。
CACHE_FILE  = os.path.join(ROOT, "testdata", ".collect_cache.jsonl")

# 青空文庫: リクエスト間隔（マナー）
AOZORA_SLEEP = 1.0
# Wikipedia API: リクエスト間隔 (レート制限に当たらない程度に空ける)
WIKI_SLEEP   = 1.0
# Wikipedia: 1 リクエストで取る記事数 (API の上限 50)
WIKI_BATCH_LIMIT = 50
# Wikipedia: 目標記事数 (厳選リスト + 全記事空間からの追加分)
WIKI_TARGET  = 10000

# 青空文庫: 作品目録から追加で拾う目標作品数と最小文字数
AOZORA_TARGET    = 150
AOZORA_MIN_CHARS = 15000
AOZORA_INDEX_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"

# ─────────────────────────────────────────────
# 青空文庫 作品リスト
#   (zip_url, タイトル表示名)
# ─────────────────────────────────────────────
AOZORA_WORKS = [
    # 夏目漱石
    ("https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip",  "吾輩は猫である（漱石）"),
    ("https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip",  "坊っちゃん（漱石）"),
    ("https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip",   "こころ（漱石）"),
    ("https://www.aozora.gr.jp/cards/000148/files/794_ruby_4237.zip",  "三四郎（漱石）"),
    ("https://www.aozora.gr.jp/cards/000148/files/1746_ruby_18324.zip",  "それから（漱石）"),
    # 芥川龍之介
    ("https://www.aozora.gr.jp/cards/000879/files/127_ruby_150.zip",   "羅生門（芥川）"),
    ("https://www.aozora.gr.jp/cards/000879/files/42_ruby_154.zip",   "鼻（芥川）"),
    ("https://www.aozora.gr.jp/cards/000879/files/55_ruby_1843.zip",   "芋粥（芥川）"),
    ("https://www.aozora.gr.jp/cards/000879/files/179_ruby_168.zip", "藪の中（芥川）"),
    ("https://www.aozora.gr.jp/cards/000879/files/45761_ruby_38235.zip",  "河童（芥川）"),
    ("https://www.aozora.gr.jp/cards/000879/files/60_ruby_821.zip", "地獄変（芥川）"),
    # 森鴎外
    ("https://www.aozora.gr.jp/cards/000129/files/58126_ruby_73643.zip",  "舞姫（鴎外）"),
    ("https://www.aozora.gr.jp/cards/000129/files/691_ruby_15351.zip","高瀬舟（鴎外）"),
    ("https://www.aozora.gr.jp/cards/000129/files/689_ruby_23256.zip",  "山椒大夫（鴎外）"),
    # 太宰治
    ("https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip", "走れメロス（太宰）"),
    ("https://www.aozora.gr.jp/cards/000035/files/1565_ruby_8220.zip",  "斜陽（太宰）"),
    ("https://www.aozora.gr.jp/cards/000035/files/301_ruby_5915.zip",  "人間失格（太宰）"),
    # 宮沢賢治
    ("https://www.aozora.gr.jp/cards/000081/files/456_ruby_145.zip",  "銀河鉄道の夜（賢治）"),
    ("https://www.aozora.gr.jp/cards/000081/files/462_ruby_716.zip",  "風の又三郎（賢治）"),
    ("https://www.aozora.gr.jp/cards/000081/files/43737_ruby_19028.zip","注文の多い料理店（賢治）"),
    # 樋口一葉
    ("https://www.aozora.gr.jp/cards/000064/files/389_ruby_15296.zip",  "たけくらべ（一葉）"),
    ("https://www.aozora.gr.jp/cards/000064/files/387_ruby_15292.zip",  "にごりえ（一葉）"),
    # 島崎藤村
    ("https://www.aozora.gr.jp/cards/000158/files/1502_ruby_24534.zip", "破戒（藤村）"),
    # 志賀直哉
    ("https://www.aozora.gr.jp/cards/000083/files/1388_ruby_17236.zip", "暗夜行路（志賀）"),
    # 幸田露伴
    ("https://www.aozora.gr.jp/cards/000051/files/43289_ruby_16762.zip",  "五重塔（露伴）"),
]

# ─────────────────────────────────────────────
# Wikipedia: 取得するカテゴリ・記事タイトルリスト
# ─────────────────────────────────────────────
WIKI_TITLES = [
    # 日本史
    "江戸時代", "明治維新", "大正時代", "昭和時代", "平安時代", "鎌倉時代",
    "室町時代", "戦国時代_(日本)", "江戸幕府", "徳川家康", "豊臣秀吉",
    "織田信長", "坂本龍馬", "西郷隆盛", "伊藤博文", "福沢諭吉",
    "聖徳太子", "源頼朝", "足利尊氏", "北条時宗", "上杉謙信", "武田信玄",
    # 地理・都市
    "東京都", "大阪府", "京都府", "北海道", "沖縄県", "富士山",
    "琵琶湖", "瀬戸内海", "日本アルプス", "本州", "四国", "九州",
    "東海道", "中山道", "奥州街道", "長崎街道",
    # 文化・芸術
    "能楽", "歌舞伎", "文楽", "茶道", "華道", "書道", "俳句",
    "和歌", "短歌", "連歌", "狂言", "落語", "講談", "浪曲",
    "日本画", "浮世絵", "蒔絵", "陶磁器", "漆器",
    "日本建築", "神社建築", "寺院建築", "城郭",
    # 科学・技術
    "量子力学", "相対性理論", "進化論", "DNA", "光合成",
    "人工知能", "機械学習", "深層学習", "自然言語処理",
    "インターネット", "半導体", "新幹線", "リニアモーターカー",
    "宇宙開発", "国際宇宙ステーション", "ブラックホール",
    # 文学・思想
    "夏目漱石", "芥川龍之介", "太宰治", "川端康成", "三島由紀夫",
    "松尾芭蕉", "与謝野晶子", "正岡子規", "石川啄木",
    "儒教", "仏教", "神道", "禅", "武士道",
    # 社会・政治
    "日本国憲法", "民主主義", "国会_(日本)", "内閣総理大臣",
    "日本の経済", "高度経済成長", "バブル景気", "少子高齢化",
    "社会保障", "環境問題", "地球温暖化", "再生可能エネルギー",
    # 自然・生物
    "桜", "竹", "松", "梅", "菊", "蓮",
    "タヌキ", "キツネ", "ニホンザル", "ツル", "カラス",
    "日本海", "太平洋", "地震", "台風", "津波", "火山",
    # スポーツ・娯楽
    "相撲", "柔道", "剣道", "空手", "弓道", "合気道",
    "野球", "サッカー", "バスケットボール",
    "将棋", "囲碁", "花札", "かるた",
    # 食文化
    "日本料理", "寿司", "天ぷら", "そば", "うどん", "ラーメン",
    "味噌", "醤油", "豆腐", "納豆", "日本酒", "緑茶",
    # 現代文化
    "アニメ", "漫画", "ゲーム", "ラノベ", "ボカロ",
    "東京ディズニーリゾート", "秋葉原", "渋谷", "新宿",
]


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)

def fetch_bytes(url, timeout=30, retries=5):
    """429/5xx は Retry-After を見て待ち、指数バックオフで再試行する。"""
    headers = {"User-Agent": "JapaneseCorpusCollector/1.0 (educational use)"}
    delay = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = delay
            retry_after = e.headers.get("Retry-After") if e.headers else None
            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
            wait = min(wait, 30.0)
            log(f"    HTTP {e.code}: {wait:.0f} 秒待って再試行 ({attempt + 1}/{retries - 1})")
            time.sleep(wait)
            delay = min(delay * 2, 60)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            log(f"    {e}: {delay:.0f} 秒待って再試行 ({attempt + 1}/{retries - 1})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")

def fetch_text(url, encoding="utf-8", timeout=30):
    return fetch_bytes(url, timeout).decode(encoding, errors="replace")

# 日本語 (ひらがな・カタカナ・漢字・々〆) を含む行だけを残すための判定。
CJK_RE = re.compile(r"[\u3041-\u309f\u30a1-\u30fa\u4e00-\u9fff々〆ヵヶ]")

# clean_wikitext が落としきれなかった見出し語（過去のコーパスの再清掃用）。
WIKI_HEADING_WORDS = set("""脚注 出典 注釈 参考文献 関連項目 関連文献 外部リンク 概要 歴史 沿革 特徴 来歴 経歴 生涯 人物
登場人物 あらすじ ストーリー 解説 評価 受賞歴 作品 小説 漫画 映画 アニメ テレビ ラジオ ゲーム 舞台 音楽 主題歌
シングル アルバム ディスコグラフィ ディスコグラフィー 収録曲 収録内容 出演 出演者 キャスト スタッフ メンバー
パーソナリティ ゲスト 書籍 単行本 書誌情報 仕様 性能 諸元 要目 設計 開発 運用 運用国 採用国 艦歴 機体 武装
エンジン 分類 備考 その他 ギャラリー 不具合 主な機能 放送時間 放送局 各話リスト エピソード タイアップ
大会概要 試合結果 イベント バリエーション 各型 派生型 概説 背景 名称 呼称 交通 地理 教育 文化
脚注・出典 注記 註釈 関連作品 関連人物 出典・脚注 中略 （中略）""".split())


def strip_heading_lines(text):
    """見出し語だけの行を取り除く。"""
    return "\n".join(l for l in text.split("\n") if l.strip() not in WIKI_HEADING_WORDS)


def clean_corpus(text):
    """収集後のコーパスを行単位で清掃する。

    - 区切り行 (= の連続)、収集ヘッダ (#)、作品見出し (■) を落とす
    - 空行を落とす
    - ルビの残り (｜) を落とす
    - 日本語を含まない行 (URL・記号・数字だけの行) を落とす
    - URL を含む行 (Wikipedia の出典・外部リンク) を落とす
    """
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0] in "#=":
            continue
        if line.startswith("■"):
            continue
        line = line.replace("｜", "").strip()
        if not line or not CJK_RE.search(line):
            continue
        if "://" in line:
            continue
        out.append(line)
    return "\n".join(out)


# ─────────────────────────────────────────────
# 青空文庫
# ─────────────────────────────────────────────
def fetch_aozora(zip_url):
    data = fetch_bytes(zip_url)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        txt_files = [n for n in z.namelist() if n.endswith(".txt")]
        if not txt_files:
            return ""
        raw = z.read(txt_files[0])
        return raw.decode("shift_jis", errors="replace")

def clean_aozora(text):
    # ヘッダ除去（区切り線まで）
    text = re.sub(r'^.*?-{50,}', '', text, count=1, flags=re.DOTALL)
    # 「テキスト中に現れる記号について」の説明ブロック（全作品に共通の定型文）
    text = re.sub(r'【テキスト中に現れる記号について】.*?-{50,}', '', text, count=1, flags=re.DOTALL)
    # フッタ除去
    text = re.sub(r'底本：.*$', '', text, flags=re.DOTALL)
    # ルビ《》除去
    text = re.sub(r'《[^》]*》', '', text)
    # ［＃注記］除去
    text = re.sub(r'［＃[^］]*］', '', text)
    # ルビ開始記号
    text = text.replace('|', '')
    # ※行除去
    text = re.sub(r'※[^\n]*\n', '', text)
    # 連続空行圧縮
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def load_cache():
    """キャッシュ済みのテキストを読む。"""
    items = []
    if not os.path.exists(CACHE_FILE):
        return items
    with open(CACHE_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def append_cache(kind, key, text):
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "key": key, "text": text}, ensure_ascii=False) + "\n")


def aozora_index_works():
    """青空文庫の作品目録から、新字新仮名の日本文学作品を選ぶ。

    目録は全作品・全人物の CSV を zip で固めたもの。翻訳物を避けるため
    分類番号が日本文学 (NDC 91x) のものだけを使い、同じ著者から取りすぎ
    ないように 1 人 2 作品までにする。
    """
    data = fetch_bytes(AOZORA_INDEX_URL)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        raw = z.read(name).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    known = {url for url, _ in AOZORA_WORKS}
    cand = [
        r for r in rows
        if r.get("文字遣い種別") == "新字新仮名"
        and r.get("役割フラグ") == "著者"
        and r.get("分類番号", "").startswith("NDC 91")
        and r.get("テキストファイルURL")
        and r["テキストファイルURL"] not in known
    ]
    random.Random(0).shuffle(cand)
    picked, per_author = [], {}
    for r in cand:
        if len(picked) >= AOZORA_TARGET - len(AOZORA_WORKS):
            break
        pid = r.get("人物ID", "")
        if per_author.get(pid, 0) >= 2:
            continue
        per_author[pid] = per_author.get(pid, 0) + 1
        picked.append((r["テキストファイルURL"],
                       f"{r['作品名']}（{r.get('姓', '')}{r.get('名', '')}）"))
    log(f"青空文庫 目録: {len(rows):,} 行 → 追加候補 {len(picked)} 作品")
    return picked


def collect_aozora():
    log("\n【青空文庫】収集開始")
    log(f"{'作品名':<30} {'文字数':>10} {'状態'}")
    log("-" * 55)

    try:
        works = AOZORA_WORKS + aozora_index_works()
    except Exception as e:
        log(f"目録の取得に失敗、厳選リストのみ使う: {e}")
        works = AOZORA_WORKS

    cached = {c["key"]: c["text"] for c in load_cache() if c["kind"] == "aozora"}
    results = []
    for url, title in works:
        if url in cached:
            text = cached[url]
            results.append({"title": title, "text": text, "chars": len(text), "ok": True})
            log(f"{title:<30} {len(text):>10,}  = キャッシュ")
            continue
        try:
            raw = fetch_aozora(url)
            text = clean_aozora(raw)
            n = len(text)
            if n < AOZORA_MIN_CHARS and (url, title) not in AOZORA_WORKS:
                log(f"{title:<30} {n:>10,}  - 短いので省略")
                time.sleep(AOZORA_SLEEP)
                continue
            append_cache("aozora", url, text)
            results.append({"title": title, "text": text, "chars": n, "ok": True})
            log(f"{title:<30} {n:>10,}  ✓")
        except Exception as e:
            results.append({"title": title, "text": "", "chars": 0, "ok": False, "error": str(e)})
            log(f"{title:<30} {'---':>10}  ✗  {e}")
        time.sleep(AOZORA_SLEEP)

    ok = [r for r in results if r["ok"]]
    log(f"\n青空文庫: {len(ok)}/{len(works)} 作品取得, "
        f"合計 {sum(r['chars'] for r in ok):,} 文字")
    return results


# ─────────────────────────────────────────────
# Wikipedia日本語版
# ─────────────────────────────────────────────
WIKI_API = "https://ja.wikipedia.org/w/api.php"

# 全記事空間を横断するときの開始位置（タイトル先頭文字）。
WIKI_SEEDS = list("あかさたなはまやらわアカサタナハマヤラワ") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# 記事として不適切なタイトル。
SKIP_TITLE_RE = re.compile(
    r"^(一覧|年表|Category:|Wikipedia:|ファイル:|Portal:|Template:|Help:|プロジェクト:)"
)

def fetch_wiki_batch(titles=None, gapfrom=None, gapcontinue=None):
    """wikitext をまとめて取得する。

    prop=extracts の全文は 1 リクエスト 1 記事だが、prop=revisions なら
    titles に 50 件、generator=allpages なら 1 回で 50 記事ぶんの
    wikitext が返る。
    """
    params = {
        "action":        "query",
        "format":        "json",
        "formatversion": "2",
        "prop":          "revisions",
        "rvprop":        "content",
        "rvslots":       "main",
    }
    if titles:
        params["titles"] = "|".join(titles)
        params["redirects"] = "1"
    else:
        params["generator"]    = "allpages"
        params["gapnamespace"] = "0"
        params["gaplimit"]     = str(WIKI_BATCH_LIMIT)
        params["gapfrom"]      = gapfrom
        params["gapfilterredir"] = "nonredirects"
        if gapcontinue:
            params["gapcontinue"] = gapcontinue

    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    data = json.loads(fetch_text(url))
    if "error" in data:
        raise RuntimeError(f"API error: {data['error'].get('info', data['error'])}")

    pages = {}
    for page in data.get("query", {}).get("pages", []):
        if page.get("missing"):
            continue
        revs = page.get("revisions") or []
        if not revs:
            continue
        content = ((revs[0].get("slots") or {}).get("main") or {}).get("content")
        if content:
            pages[page["title"]] = content
    nxt = (data.get("continue") or {}).get("gapcontinue")
    return pages, nxt

def clean_wikitext(text):
    """wikitext を素朴にプレーンテキストへ落とす（toy 用の割り切り）。"""
    # コメント・参照・HTML タグ
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.S)
    text = re.sub(r'<[^<>]{1,200}>', '', text)
    # 表
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.S)
    # テンプレート（入れ子を数回剥がす）
    for _ in range(8):
        new = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if new == text:
            break
        text = new
    # ファイル・カテゴリ（入れ子やキャプションを数回剥がす）
    for _ in range(5):
        new = re.sub(r'\[\[(ファイル|File|画像|Image|Category|カテゴリ):[^\n]*?\]\]', '', text, flags=re.I)
        if new == text:
            break
        text = new
    # 内部リンク・外部リンク
    text = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]*\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    # リンク剥がしの残骸と、テンプレート除去で残った読点の連続を片付ける
    text = re.sub(r'\[\[|\]\]', '', text)
    text = re.sub(r'[、，,]{2,}', '、', text)
    text = re.sub(r'[。．]{2,}', '。', text)
    text = re.sub(r'（\s*[、,]+\s*', '（', text)
    text = re.sub(r'\s*[、,]+\s*）', '）', text)
    text = re.sub(r'（\s*）|\(\s*\)', '', text)
    # 見出しは本文ではないので行ごと落とす（"脚注" などの見出し語が
    # 本文に混ざって生成結果に出るのを防ぐ）。強調・箇条書き記号も落とす。
    text = re.sub(r'^=+\s*.*?\s*=+\s*$', '', text, flags=re.M)
    text = text.replace("'''", '').replace("''", '')
    text = re.sub(r'^[#*:;]+\s*', '', text, flags=re.M)
    text = html.unescape(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def clean_wiki(text):
    # セクション見出し（== ... ==）を削除
    text = re.sub(r'={2,}[^=]+={2,}', '', text)
    # テンプレート・表の残骸行を落とす
    lines = []
    for line in text.split('\n'):
        if ('{{' in line or '}}' in line or '[[' in line or ']]' in line
                or line.lstrip().startswith(('|', '!'))):
            continue
        lines.append(line)
    text = '\n'.join(lines)
    # 連続空行圧縮
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def collect_wikipedia():
    log("\n【Wikipedia日本語版】収集開始")
    log(f"目標 {WIKI_TARGET} 記事 / {WIKI_BATCH_LIMIT} 記事perリクエスト")
    log("-" * 55)

    results = []
    seen = set()
    for c in load_cache():
        if c["kind"] == "wiki" and c["key"] not in seen:
            seen.add(c["key"])
            results.append({"title": c["key"], "text": c["text"], "chars": len(c["text"]), "ok": True})
    if results:
        log(f"  キャッシュから {len(results)} 記事")
        for seed in WIKI_SEEDS:
            pass

    def add(title, raw):
        if title in seen or len(results) >= WIKI_TARGET:
            return
        if SKIP_TITLE_RE.search(title):
            return
        head = raw[:300].lower()
        if '#redirect' in head or '{{曖昧さ回避' in raw or '{{aimai' in head:
            return
        text = clean_wiki(clean_wikitext(raw))
        if len(text) < 800:
            return
        seen.add(title)
        append_cache("wiki", title, text)
        results.append({"title": title, "text": text, "chars": len(text), "ok": True})

    # 1) 厳選リスト（従来の記事）
    for i in range(0, len(WIKI_TITLES), WIKI_BATCH_LIMIT):
        chunk = WIKI_TITLES[i:i + WIKI_BATCH_LIMIT]
        try:
            pages, _ = fetch_wiki_batch(titles=chunk)
        except Exception as e:
            log(f"  [厳選] ✗ {e}")
            time.sleep(WIKI_SLEEP)
            continue
        for title, raw in pages.items():
            add(title, raw)
        log(f"  厳選リスト {min(i + WIKI_BATCH_LIMIT, len(WIKI_TITLES)):>4}/{len(WIKI_TITLES)} "
            f"→ 累計 {len(results)} 記事")
        time.sleep(WIKI_SLEEP)

    # 2) 全記事空間を開始位置ごとに巡回して収集
    #    （1 つの開始位置を掘り続けると「あ」行だけに偏るのでラウンドロビン）
    conts = {seed: None for seed in WIKI_SEEDS}
    rnd = 0
    while len(results) < WIKI_TARGET:
        progressed = False
        for seed in WIKI_SEEDS:
            if len(results) >= WIKI_TARGET:
                break
            if conts[seed] is False:
                continue
            try:
                pages, nxt = fetch_wiki_batch(gapfrom=seed, gapcontinue=conts[seed])
            except Exception as e:
                log(f"  [{seed}] ✗ {e}")
                conts[seed] = False
                continue
            conts[seed] = nxt if nxt else False
            if not pages:
                continue
            before = len(results)
            for title, raw in pages.items():
                add(title, raw)
            progressed = progressed or len(results) > before
            log(f"  [{seed}] {len(pages):>2} 件取得 (+{len(results) - before:>2}) "
                f"→ 累計 {len(results)}/{WIKI_TARGET} 記事")
            time.sleep(WIKI_SLEEP)
        rnd += 1
        if not progressed:
            break

    ok = [r for r in results if r["ok"]]
    log(f"\nWikipedia: {len(ok)} 記事取得, "
        f"合計 {sum(r['chars'] for r in ok):,} 文字")
    return results


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    log("=" * 60)
    log("日本語コーパス収集スクリプト")
    log(f"開始: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ── 収集 ──
    aozora_results = collect_aozora()
    wiki_results   = collect_wikipedia()

    all_results = aozora_results + wiki_results
    ok_results  = [r for r in all_results if r["ok"]]
    raw_chars   = sum(r["chars"] for r in ok_results)

    # ── 後処理（清掃）──
    # 区切り行・URL 行・ルビ残り・空行を落とす。メタ情報は input.txt ではなく
    # corpus_stats.txt にだけ残し、学習データに混ぜない。
    for r in ok_results:
        r["text"] = clean_corpus(r["text"])
        r["chars"] = len(r["text"])
    total_chars = sum(r["chars"] for r in ok_results)

    # ── コーパスファイル出力 ──
    log(f"\n【出力】{OUTPUT_FILE}")
    corpus = strip_heading_lines(clean_corpus("\n\n".join(r["text"] for r in ok_results)))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(corpus + "\n")

    log(f"  → {raw_chars:,} 文字 → {len(corpus):,} 文字 (清掃で {raw_chars - len(corpus):,} 文字削減)")

    # ── 統計ファイル出力 ──
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write("収集統計レポート\n")
        f.write(f"生成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        f.write("【青空文庫】\n")
        for r in aozora_results:
            mark = "✓" if r["ok"] else "✗"
            err = f"  ({r.get('error', '')})" if not r["ok"] else ""
            f.write(f"  {mark} {r['title']:<30} {r['chars']:>10,} 文字{err}\n")

        f.write("\n【Wikipedia】\n")
        for r in wiki_results:
            mark = "✓" if r["ok"] else "✗"
            err = f"  ({r.get('error', '')})" if not r["ok"] else ""
            f.write(f"  {mark} {r['title']:<30} {r['chars']:>10,} 文字{err}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"清掃前: {raw_chars:,} 文字\n")
        f.write(f"合計: {total_chars:,} 文字 / {len(ok_results)} 件\n")

    # ── サマリ表示 ──
    log("\n" + "=" * 60)
    log("収集完了サマリ")
    log("=" * 60)
    aok = [r for r in aozora_results if r["ok"]]
    wok = [r for r in wiki_results   if r["ok"]]
    log(f"  青空文庫 : {len(aok):>3} 作品  {sum(r['chars'] for r in aok):>10,} 文字")
    log(f"  Wikipedia: {len(wok):>3} 記事  {sum(r['chars'] for r in wok):>10,} 文字")
    log(f"  合計     :          {total_chars:>10,} 文字 (清掃後)")
    log(f"\n出力ファイル:")
    log(f"  {OUTPUT_FILE}  ({os.path.getsize(OUTPUT_FILE):,} bytes)")
    log(f"  {STATS_FILE}")
    log(f"\n終了: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()