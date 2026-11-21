// Command jptok はコーパスを BPE で分割し、分割結果・語彙・併合規則を書き出す。
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
	mergesPath := flag.String("merges", "testdata/merges.txt", "merge rules (priority order) output file")
	vocabSize := flag.Int("vocab", 32000, "target vocabulary size (merges stop here)")
	minCount := flag.Int("mincount", 2, "minimum pair frequency to merge")
	batch := flag.Int("batch", 2000, "merges applied per counting pass (speed/quality knob)")
	holdout := flag.Float64("holdout", 0, "fraction of the tail held out for tokens/char (0 = off)")
	quiet := flag.Bool("quiet", false, "suppress progress logs")
	flag.Parse()

	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}
	if *holdout < 0 || *holdout >= 1 {
		log.Fatalf("holdout must be in [0, 1): %v", *holdout)
	}

	// 末尾を held-out にして、先頭だけで学習する（語彙と併合規則の汎化を見る）。
	trainSrc, held := src, ""
	if *holdout > 0 {
		offs := tokenizer.RuneOffsets(src)
		cut := offs[int(float64(len(offs)-1)*(1-*holdout))]
		trainSrc, held = src[:cut], string(src[cut:])
	}

	t0 := time.Now()
	res, err := tokenizer.Train(trainSrc, tokenizer.Config{
		VocabSize: *vocabSize,
		MinCount:  *minCount,
		Batch:     *batch,
		Quiet:     *quiet,
	})
	if err != nil {
		log.Fatal(err)
	}
	buildDur := time.Since(t0)

	// 書き出す分割は、配布する語彙と併合規則 (Model) で振り直す。
	// 学習中の id 列は -batch の近似を含むので、Model.Encode を正とする。
	t1 := time.Now()
	model := res.Model()
	ids := model.Encode(string(trainSrc))
	tokens := make([]string, len(ids))
	for i, id := range ids {
		tokens[i] = model.Vocab[id]
	}
	encodeDur := time.Since(t1)
	// 可逆性の検査: トークンを連結すると入力に戻るか。
	if got := strings.Join(tokens, ""); got != string(trainSrc) {
		log.Fatalf("internal error: detokenization mismatch (%d vs %d bytes)", len(got), len(trainSrc))
	}

	if err := os.WriteFile(*outPath, []byte(strings.Join(tokens, "/")), 0o644); err != nil {
		log.Fatal(err)
	}
	if err := tokenizer.WriteVocab(*dictPath, res.Vocab); err != nil {
		log.Fatal(err)
	}
	if err := tokenizer.WriteMerges(*mergesPath, res.Merges); err != nil {
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
		log.Printf("corpus: %d runes, %d merges", len(tokenizer.RuneOffsets(trainSrc))-1, len(res.Merges))
		log.Printf("train: %d tokens, encode: %d tokens", len(res.IDs), len(tokens))
		log.Printf("avg len: %.2f, 1-char: %.1f%%",
			float64(totalLen)/float64(len(tokens)),
			100*float64(oneChar)/float64(len(tokens)))
		reportHeldOut(model, held)
		log.Printf("build: %s, encode: %s, total: %s", buildDur, encodeDur, time.Since(t0))
	}
}

// reportHeldOut は学習に使っていない末尾を分割し、圧縮率を報告する。
func reportHeldOut(m *tokenizer.Model, held string) {
	if held == "" {
		return
	}
	ids := m.Encode(held)
	chars := utf8.RuneCountInString(held)
	unk := 0
	for _, id := range ids {
		if id == tokenizer.UnkID {
			unk++
		}
	}
	log.Printf("held-out: %d chars, %d tokens, %.2f chars/token, unk %d",
		chars, len(ids), float64(chars)/float64(len(ids)), unk)
}
