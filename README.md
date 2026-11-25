# tiny-llm

日本語のトークナイザと、その上で動く小さな言語モデル。

```console
$ go run . tok                      # input.txt → output.txt, dic.txt, merges.txt
$ go run . train -order 3           # n-gram を学習して testdata/ngram.bin に保存
$ go run . gen -prompt "吾輩は"      # 保存済みモデルで続きを生成
$ go run . lm -order 3              # 末尾 5% を held-out にした BPB
$ go test ./...
$ python3 scripts/collect_corpus.py # 青空文庫と Wikipedia から input.txt を作る
```

`tok`（トークナイザ）→ `train`（n-gram のカウント）→ `gen`（生成）の順に使う。
`lm` は学習と評価を 1 コマンドで行うので、`train` は不要。

| コマンド | フラグ |
|---|---|
| `tok` | `-in` `-vocab` |
| `train` | `-in` `-order` `-out` |
| `gen` | `-model` `-prompt` `-n` `-temp` `-seed` |
| `lm` | `-in` `-order` `-eval` `-holdout` |

`output.txt` は `/` 区切り、`dic.txt` は語彙 (JSON 配列)、`merges.txt` は併合規則。`/` を除くと入力に戻る。
`ngram.bin` は学習済みモデルで、`dic.txt`/`merges.txt` のハッシュを記録しており、
語彙が変わっていると `gen` が読み込みを拒否する。
