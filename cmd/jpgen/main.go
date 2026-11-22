// jpgen は n-gram マルコフ連鎖で続きを生成する。
// 学習済みの dic.txt / merges.txt を使い、input.txt で n-gram を数える。
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"

	"github.com/minotto165/tiny-llm/lm"
	"github.com/minotto165/tiny-llm/tokenizer"
)

func main() {
	inPath := flag.String("in", "testdata/input.txt", "corpus file")
	dictPath := flag.String("dict", "testdata/dic.txt", "vocabulary (JSON array)")
	mergesPath := flag.String("merges", "testdata/merges.txt", "merge rules")
	order := flag.Int("order", 2, "Markov order (1..6)")
	n := flag.Int("n", 80, "number of tokens to generate")
	temp := flag.Float64("temp", 0.8, "sampling temperature (0 = greedy)")
	topK := flag.Int("topk", 40, "sample from the top K candidates (0 = all)")
	repeat := flag.Float64("repeat", 1.15, "penalty on recently used tokens (1 = off)")
	hiKeep := flag.Int("hikeep", lm.DefaultHiKeep, "minimum count kept for 4+-grams")
	hiWeight := flag.Float64("hiweight", lm.DefaultHiWeight, "weight of the 4+-gram estimate")
	seed := flag.Int64("seed", 1, "random seed")
	prompt := flag.String("prompt", "", "prompt text")
	flag.Parse()

	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}
	vocab, err := tokenizer.ReadVocab(*dictPath)
	if err != nil {
		log.Fatal(err)
	}
	merges, err := tokenizer.ReadMerges(*mergesPath)
	if err != nil {
		log.Fatal(err)
	}
	model := tokenizer.NewModel(vocab, merges)
	m := lm.TrainFull(model.Encode(string(src)), len(vocab), *order, lm.DefaultDiscount, int32(*hiKeep), *hiWeight)

	out := model.Encode(*prompt)
	probs := make([]float64, len(vocab))
	rng := rand.New(rand.NewSource(*seed))
	const repeatWindow = 32
	for i := 0; i < *n; i++ {
		from := len(out) - (*order - 1)
		if from < 0 {
			from = 0
		}
		ctx := out[from:]

		sum := 0.0
		for w := range probs {
			probs[w] = 0 // 特殊トークンは生成しない
			if w < len(tokenizer.SpecialTokens) {
				continue
			}
			p := m.Prob(ctx, int32(w))
			if *repeat > 1 {
				for j := len(out) - repeatWindow; j < len(out); j++ {
					if j >= 0 && int32(w) == out[j] {
						p /= *repeat
						break
					}
				}
			}
			if *temp > 0 && *temp != 1 {
				p = math.Pow(p, 1 / *temp)
			}
			probs[w] = p
			sum += p
		}
		if sum <= 0 {
			log.Fatal("no probability mass")
		}
		out = append(out, sample(probs, sum, *temp, *topK, rng))
	}
	fmt.Println(model.Decode(out))
}

// sample は probs (合計 sum) から 1 つ選ぶ。temp <= 0 なら最大確率のものを返す。
// topK > 0 なら上位 K 個だけを候補にする。
func sample(probs []float64, sum, temp float64, topK int, rng *rand.Rand) int32 {
	if temp <= 0 {
		best := int32(0)
		for i, p := range probs {
			if p > probs[best] {
				best = int32(i)
			}
		}
		return best
	}
	if topK > 0 && topK < len(probs) {
		// 上位 K 個の確率の位置を探して、その分だけで再正規化する。
		idx := make([]int, 0, len(probs))
		for i, p := range probs {
			if p > 0 {
				idx = append(idx, i)
			}
		}
		if len(idx) > topK {
			sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
			idx = idx[:topK]
			sum = 0
			for _, i := range idx {
				sum += probs[i]
			}
			// 候補外は 0 にして、下の線形探索をそのまま使う。
			keep := make([]bool, len(probs))
			for _, i := range idx {
				keep[i] = true
			}
			for i := range probs {
				if !keep[i] {
					probs[i] = 0
				}
			}
		}
	}
	r := rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}
