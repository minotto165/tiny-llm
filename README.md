# tiny-llm

日本語のトークナイザと、その上で動く小さな言語モデル。

```console
$ go run . tok                      # input.txt → output.txt, dic.txt, merges.txt
$ go run . train -order 3           # n-gram を学習して testdata/ngram.bin に保存
$ go run . gen -prompt "吾輩は"      # 保存済みモデルで続きを生成
$ go run . lm -order 3              # 末尾 5% を held-out にした BPB
$ go run . tfm -steps 2000          # Transformer を学習して testdata/tfm.bin に保存
$ python3 scripts/train_torch.py --ids testdata/ids.bin --out testdata/tfm.bin  # GPU で学習
$ go test ./...
$ python3 scripts/collect_corpus.py # 青空文庫と Wikipedia から input.txt を作る
$ python3 scripts/build_discord_corpus.py # Discord の生 JSON から testdata/discord/corpus を作る
```

`tok`（トークナイザ）→ `train`（n-gram のカウント）→ `gen`（生成）の順に使う。
`lm` は学習と評価を 1 コマンドで行うので、`train` は不要。

| コマンド | フラグ |
|---|---|
| `tok` | `-in` `-dir` (出力先) `-vocab` |
| `train` | `-in` `-order` `-out` |
| `gen` | `-model` `-dir` `-prompt` `-n` `-temp` `-seed` `-check` `-checkn` `-samples` |
| `lm` | `-in` `-order` `-eval` `-holdout` |
| `tfm` | `-in` `-out` `-ctx` `-dim` `-heads` `-layers` `-ffn` `-steps` `-batch` `-lr` `-seed` |
| `ids` | `-in` `-out` |

`output.txt` は `/` 区切り、`dic.txt` は語彙 (JSON 配列)、`merges.txt` は併合規則。`/` を除くと入力に戻る。
`ngram.bin` / `tfm.bin` は学習済みモデルで、`dic.txt`/`merges.txt` のハッシュを記録しており、
語彙が変わっていると `gen` が読み込みを拒否する。`gen` はモデル先頭のマジックで n-gram と
Transformer を自動判別する。GPU 学習は `ids` で ID 列を作り、`scripts/train_torch.py` に渡す。

## Discord コーパス

`testdata/discord/` に生データと生成物を置く（どちらも git 管理外）。

```
testdata/discord/raw/all_part*.json   Discord API から落とした生 JSON
testdata/discord/raw/names.json       ユーザーID → 表示名
testdata/discord/corpus/              生成された train/valid/test_all.txt
testdata/discord/model/               dic.txt, merges.txt, ids_all.bin
```

```
$ python3 scripts/build_discord_corpus.py \
    --keep-urls --no-bots --drop-urls --drop-ascii-noise --cap-repeat 10
$ go run . tok -in testdata/discord/corpus/train_all.txt -vocab 12000 -minchar 10 -dir testdata/discord/model
$ go run . ids -in testdata/discord/corpus/train_all.txt -dir testdata/discord/model -out testdata/discord/model/ids_all.bin
$ go run . lm  -in testdata/discord/corpus/train_all.txt -eval testdata/discord/corpus/test_all.txt -dir testdata/discord/model
```

`--drop-urls` は URL を本文から除去、`--drop-ascii-noise` は日本語を含まない英数字ノイズ
（`m!repeat`, `KNTRYK` など）を除去、`--cap-repeat N` は同一人物の同一文の連投を N 回までに畳む。
`--url-domain` は URL をドメインだけに残す。

## 丸写し率の計測

`gen` に `-check <参照コーパス>` を渡すと、生成が学習データをどれだけコピーしているかを測る。

```
$ go run . gen -model testdata/discord/model/tfm_v3_3000.bin -dir testdata/discord/model \
    -prompt "おはよう" -n 120 -samples 8 -check testdata/discord/corpus/train_all.txt
memcheck[gen]: 8 samples, 20-gram in corpus 216/2043 (10.6%), longest 97 chars (median 13)
  longest verbatim: "\n🐮am͜a͉nyon\nもう...僕をッ...終わらせてくれえ ..."
```

- `-checkn` 窓の長さ（既定 20 文字）
- `-samples` 生成するサンプル数（seed を 1 つずつずらす）
- 20-gram がコーパスに存在する割合が高いほど暗記している
