// Command jptok はコーパスを BPE で分割し、分割結果と語彙を書き出す。
package main

import (
	"flag"
	"log"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/minotto165/tiny-llm/tokenizer"
)

func main() {
	inPath := flag.String("in", "testdata/input.txt", "input text file (UTF-8)")
	outPath := flag.String("out", "testdata/output.txt", "segmented output file")
	dictPath := flag.String("dict", "testdata/dic.txt", "vocabulary (JSON) output file")
	vocabSize := flag.Int("vocab", 32000, "target vocabulary size (merges stop here)")
	minCount := flag.Int("mincount", 2, "minimum pair frequency to merge")
	batch := flag.Int("batch", 2000, "merges applied per counting pass (speed/quality knob)")
	quiet := flag.Bool("quiet", false, "suppress progress logs")
	flag.Parse()

	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}

	t0 := time.Now()
	res, err := tokenizer.Train(src, tokenizer.Config{
		VocabSize: *vocabSize,
		MinCount:  *minCount,
		Batch:     *batch,
		Quiet:     *quiet,
	})
	if err != nil {
		log.Fatal(err)
	}
	buildDur := time.Since(t0)

	tokens := res.Tokens()
	// 可逆性の検査: トークンを連結すると入力に戻るか。
	if got := strings.Join(tokens, ""); got != string(src) {
		log.Fatalf("internal error: detokenization mismatch (%d vs %d bytes)", len(got), len(src))
	}

	if err := os.WriteFile(*outPath, []byte(strings.Join(tokens, "/")), 0o644); err != nil {
		log.Fatal(err)
	}
	if err := tokenizer.WriteVocab(*dictPath, res.Vocab); err != nil {
		log.Fatal(err)
	}

	if !*quiet {
		totalLen, oneChar := 0, 0
		for _, t := range tokens {
			if l := utf8.RuneCountInString(t); l == 1 {
				totalLen++
				oneChar++
			} else {
				totalLen += l
			}
		}
		if len(res.Vocab) > *vocabSize {
			log.Printf("warning: input has %d distinct runes, more than vocab %d", len(res.Vocab), *vocabSize)
		}
		log.Printf("corpus: %d runes", len(tokenizer.RuneOffsets(src))-1)
		log.Printf("tokens: %d, vocab: %d, avg len: %.2f, 1-char: %.1f%%",
			len(tokens), len(res.Vocab), float64(totalLen)/float64(len(tokens)),
			100*float64(oneChar)/float64(len(tokens)))
		log.Printf("build (all merges): %s, total: %s", buildDur, time.Since(t0))
	}
}
