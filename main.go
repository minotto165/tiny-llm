// Command jpllm は tiny-llm のサブコマンドをまとめて起動する。
//
//	jpllm tok   トークナイザ (BPE) を学習する
//	jpllm train n-gram モデルを学習して保存する
//	jpllm lm   n-gram 言語モデルの BPB を評価する
//	jpllm gen  n-gram で続きを生成する
package main

import (
	"fmt"
	"os"

	"github.com/minotto165/tiny-llm/internal/cli"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "tok":
		cli.Tok(os.Args[2:])
	case "train":
		cli.Train(os.Args[2:])
	case "lm":
		cli.LM(os.Args[2:])
	case "gen":
		cli.Gen(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", os.Args[1])
		usage()
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage: jpllm <command> [flags]

commands:
  tok   corpus から BPE 語彙と併合規則を学習する
  train corpus から n-gram モデルを学習して保存する
  lm    n-gram マルコフ連鎖の held-out BPB を評価する
  gen   n-gram マルコフ連鎖で続きを生成する

examples:
  go run . tok -in testdata/input.txt
  go run . train -in testdata/input.txt -order 3
  go run . lm  -in testdata/input.txt -order 3
  go run . gen -model testdata/ngram.bin -prompt "その日の午後、"

各コマンドのフラグは "jpllm <command> -h" で確認できる。
`)
	os.Exit(2)
}
