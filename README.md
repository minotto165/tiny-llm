# tiny-llm

日本語のトークナイザと、その上で動く小さな言語モデル。

```console
$ go run ./cmd/jptok        # testdata/input.txt → output.txt と dic.txt
$ go test ./...
$ python3 scripts/collect_corpus.py   # 青空文庫と Wikipedia から input.txt を作る
```

`output.txt` は `/` 区切り、`dic.txt` は語彙 (JSON 配列)。`/` を除くと入力に戻る。

フラグ: `-in` `-out` `-dict` `-vocab` 32000 `-mincount` 2 `-batch` 2000
