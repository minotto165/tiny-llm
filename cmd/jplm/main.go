// jplm は n-gram マルコフ連鎖言語モデルの評価コマンド。
// 末尾 holdout を学習から外し、その BPB (bits per UTF-8 byte) を出す。
package main

import (
	"flag"
	"log"
	"math"
	"os"
	"time"

	"github.com/minotto165/tiny-llm/lm"
	"github.com/minotto165/tiny-llm/tokenizer"
)

func main() {
	inPath := flag.String("in", "testdata/input.txt", "corpus file")
	vocabSize := flag.Int("vocab", 18000, "tokenizer vocabulary size")
	minCount := flag.Int("mincount", 2, "minimum pair frequency to merge")
	minChar := flag.Int("minchar", 50, "minimum rune frequency for a base token")
	batch := flag.Int("batch", 200, "merges applied per counting pass")
	order := flag.Int("order", 2, "Markov order (1..6)")
	discountFlag := flag.Float64("discount", lm.DefaultDiscount, "Kneser-Ney discount D")
	hiKeep := flag.Int("hikeep", lm.DefaultHiKeep, "minimum count kept for 4+-grams")
	hiWeight := flag.Float64("hiweight", lm.DefaultHiWeight, "weight of the 4+-gram estimate")
	holdout := flag.Float64("holdout", 0.05, "tail fraction held out for BPB")
	evalPath := flag.String("eval", "", "separate evaluation file (trains on the whole -in file)")
	quiet := flag.Bool("quiet", false, "suppress progress logs")
	flag.Parse()

	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}

	// jptok -holdout と同じ位置で切る（語彙・分割を held-out から隔離する）。
	head, tail := src, ""
	if *evalPath != "" {
		ev, err := os.ReadFile(*evalPath)
		if err != nil {
			log.Fatal(err)
		}
		tail = string(ev)
	} else {
		if *holdout <= 0 || *holdout >= 1 {
			log.Fatalf("holdout must be in (0, 1): %v", *holdout)
		}
		offs := tokenizer.RuneOffsets(src)
		cut := offs[int(float64(len(offs)-1)*(1-*holdout))]
		head, tail = src[:cut], string(src[cut:])
	}

	t0 := time.Now()
	res, err := tokenizer.Train(head, tokenizer.Config{
		VocabSize:    *vocabSize,
		MinCount:     *minCount,
		MinCharCount: *minChar,
		Batch:        *batch,
		Quiet:        *quiet,
	})
	if err != nil {
		log.Fatal(err)
	}
	model := res.Model()
	headIDs := model.Encode(string(head))
	tailIDs := model.Encode(tail)
	tokDur := time.Since(t0)

	t1 := time.Now()
	m := lm.TrainFull(headIDs, len(res.Vocab), *order, *discountFlag, int32(*hiKeep), *hiWeight)
	bits, unk := 0.0, 0
	for i, id := range tailIDs {
		from := i - (*order - 1)
		if from < 0 {
			from = 0
		}
		p := m.Prob(tailIDs[from:i], id)
		bits -= math.Log2(p)
		if id == tokenizer.UnkID {
			unk++
		}
	}
	evalDur := time.Since(t1)

	bpb := bits / float64(len(tail))
	log.Printf("order %d: held-out %d bytes, %d tokens, unk %d (%.2f%%), BPB %.4f",
		*order, len(tail), len(tailIDs), unk, 100*float64(unk)/float64(len(tailIDs)), bpb)
	log.Printf("train: %d tokens, %d grams; tokenize %v, apply %v", len(headIDs), m.Grams(), tokDur.Round(time.Millisecond), evalDur.Round(time.Millisecond))
}
