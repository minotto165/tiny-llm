// tok はコーパスを BPE で分割し、分割結果・語彙・併合規則を testdata/ に書き出す。
package cli

import (
	"flag"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/minotto165/tiny-llm/tokenizer"
)

const (
	minCount   = 2
	batch      = 200
	minRareCut = 100
)

func Tok(args []string) {
	fs := flag.NewFlagSet("tok", flag.ExitOnError)
	inPath := fs.String("in", "testdata/input.txt", "コーパス (UTF-8)")
	vocabSize := fs.Int("vocab", 32000, "語彙数")
	dir := fs.String("dir", "testdata", "dic.txt / merges.txt / output.txt の出力先")
	minChar := fs.Int("minchar", 50, "これ未満の頻度の文字は <unk> にする")
	fs.Parse(args)

	outPath := filepath.Join(*dir, "output.txt")
	dictPath := filepath.Join(*dir, "dic.txt")
	mergesPath := filepath.Join(*dir, "merges.txt")
	if err := os.MkdirAll(*dir, 0o755); err != nil {
		log.Fatal(err)
	}

	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}

	t0 := time.Now()
	res, err := tokenizer.Train(src, tokenizer.Config{
		VocabSize:    *vocabSize,
		MinCount:     minCount,
		MinCharCount: *minChar,
		Batch:        batch,
	})
	if err != nil {
		log.Fatal(err)
	}
	buildDur := time.Since(t0)

	// 学習中の id 列は -batch の近似を含むので、配布する語彙と併合規則で振り直す。
	t1 := time.Now()
	model := res.Model()
	ids := model.Encode(string(src))
	tokens := make([]string, len(ids))
	for i, id := range ids {
		tokens[i] = model.Vocab[id]
	}
	encodeDur := time.Since(t1)

	// 可逆性の検査: <unk> 以外を連結すると、-minchar 未満の文字を除いた入力に戻るか。
	counts := make(map[rune]int)
	for _, r := range string(src) {
		counts[r]++
	}
	var expected, joined strings.Builder
	for _, r := range string(src) {
		if counts[r] >= *minChar {
			expected.WriteRune(r)
		}
	}
	for i, tok := range tokens {
		if ids[i] != tokenizer.UnkID {
			joined.WriteString(tok)
		}
	}
	if joined.String() != expected.String() {
		log.Fatalf("internal error: detokenization mismatch (%d vs %d bytes)", joined.Len(), expected.Len())
	}

	if err := os.WriteFile(outPath, []byte(strings.Join(tokens, "/")), 0o644); err != nil {
		log.Fatal(err)
	}
	if err := tokenizer.WriteVocab(dictPath, res.Vocab); err != nil {
		log.Fatal(err)
	}
	if err := tokenizer.WriteMerges(mergesPath, res.Merges); err != nil {
		log.Fatal(err)
	}

	totalLen, oneChar := 0, 0
	for _, t := range tokens {
		if l := utf8.RuneCountInString(t); l == 1 {
			totalLen++
			oneChar++
		} else {
			totalLen += l
		}
	}
	freq := make(map[int32]int, len(res.Vocab))
	for _, id := range ids {
		freq[id]++
	}
	rare := 0
	for i := range res.Vocab {
		if freq[int32(i)] < minRareCut {
			rare++
		}
	}
	log.Printf("corpus: %d runes, %d merges", len(tokenizer.RuneOffsets(src))-1, len(res.Merges))
	log.Printf("vocab: %d, rare (<%d): %d (%.1f%%)", len(res.Vocab), minRareCut, rare, 100*float64(rare)/float64(len(res.Vocab)))
	log.Printf("corpus unk: %d (%.3f%%)", freq[tokenizer.UnkID], 100*float64(freq[tokenizer.UnkID])/float64(len(ids)))
	log.Printf("avg len: %.2f, 1-char: %.1f%%",
		float64(totalLen)/float64(len(tokens)), 100*float64(oneChar)/float64(len(tokens)))
	log.Printf("build: %s, encode: %s, total: %s", buildDur.Round(time.Millisecond), encodeDur.Round(time.Millisecond), time.Since(t0).Round(time.Millisecond))
}
