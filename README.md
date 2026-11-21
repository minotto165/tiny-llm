# jp_tokenizer

日本語を BPE で分割するトークナイザ。

```console
$ go run .
```

`testdata/input.txt` を読んで `testdata/output.txt` (`/` 区切り) と `testdata/dic.txt` (語彙) に書く。
`/` を除くと入力に戻る。

フラグ: `-in` `-out` `-dict` `-vocab` 32000 `-mincount` 2 `-batch` 2000

```console
$ python3 collect_corpus.py   # 青空文庫と Wikipedia から testdata/input.txt を作る
$ go test ./...
```
