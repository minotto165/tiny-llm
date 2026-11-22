# tiny-llm

日本語のトークナイザと、その上で動く小さな言語モデル。

```console
$ go run ./cmd/jptok        # testdata/input.txt → output.txt, dic.txt, merges.txt
$ go run ./cmd/jplm -order 2   # 末尾 5% を held-out にした BPB
$ go run ./cmd/jpgen -n 80 -prompt "吾輩は"   # マルコフ連鎖で続きを生成
$ go test ./...
$ python3 scripts/collect_corpus.py   # 青空文庫と Wikipedia から input.txt を作る
```

`output.txt` は `/` 区切り、`dic.txt` は語彙 (JSON 配列)、`merges.txt` は併合規則。`/` を除くと入力に戻る。

フラグ: `-in` `-out` `-dict` `-merges` `-vocab` 18000 `-mincount` 2 `-minchar` 50 `-batch` 200 `-holdout` 0
