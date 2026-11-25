# tiny-llm

日本語のトークナイザと、その上で動く小さな言語モデル。

```console
$ go run . tok                 # testdata/input.txt → output.txt, dic.txt, merges.txt
$ go run . lm -order 3         # 末尾 5% を held-out にした BPB
$ go run . gen -n 80 -prompt "吾輩は"   # マルコフ連鎖で続きを生成
$ go test ./...
$ python3 scripts/collect_corpus.py    # 青空文庫と Wikipedia から input.txt を作る
```

サブコマンドは `tok`（トークナイザ学習）、`lm`（BPB 評価）、`gen`（生成）。
フラグは `go run . <command> -h` で確認できる。

| コマンド | フラグ |
|---|---|
| `tok` | `-in` `-vocab` |
| `lm` | `-in` `-order` `-eval` `-holdout` |
| `gen` | `-in` `-prompt` `-n` `-temp` `-order` `-seed` `-save` `-load` |

`gen -save model.bin` で n-gram を保存でき、次回は `gen -load model.bin` で
トークン化とカウントを省ける（3 分 → 1 秒）。

`output.txt` は `/` 区切り、`dic.txt` は語彙 (JSON 配列)、`merges.txt` は併合規則。`/` を除くと入力に戻る。
