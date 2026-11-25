// gen は n-gram マルコフ連鎖で続きを生成する。
// testdata/dic.txt と merges.txt で分割し、-in のコーパスで n-gram を数える。
package cli

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

const (
	repeatPenalty = 1.15 // 直近 32 トークンの重複ペナルティ
	repeatWindow  = 32
	topK          = 0 // 0 = 全候補からサンプル
)

func Gen(args []string) {
	fs := flag.NewFlagSet("gen", flag.ExitOnError)
	inPath := fs.String("in", "testdata/input.txt", "コーパス")
	prompt := fs.String("prompt", "", "プロンプト")
	n := fs.Int("n", 80, "生成するトークン数")
	temp := fs.Float64("temp", 0.8, "温度 (0 = 貪欲)")
	order := fs.Int("order", 2, "マルコフ次数 (1..10)")
	seed := fs.Int64("seed", 1, "乱数シード")
	fs.Parse(args)

	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}
	vocab, err := tokenizer.ReadVocab(dictPath)
	if err != nil {
		log.Fatal(err)
	}
	merges, err := tokenizer.ReadMerges(mergesPath)
	if err != nil {
		log.Fatal(err)
	}
	model := tokenizer.NewModel(vocab, merges)
	m := lm.TrainFull(model.Encode(string(src)), len(vocab), *order, lm.DefaultDiscount,
		hiKeepFor(*order), hiWeightFor(*order))

	out := model.Encode(*prompt)
	probs := make([]float64, len(vocab))
	rng := rand.New(rand.NewSource(*seed))
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
			for j := len(out) - repeatWindow; j < len(out); j++ {
				if j >= 0 && int32(w) == out[j] {
					p /= repeatPenalty
					break
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
		out = append(out, sample(probs, sum, *temp, rng))
	}
	fmt.Println(model.Decode(out))
}

// sample は probs (合計 sum) から 1 つ選ぶ。temp <= 0 なら最大確率のものを返す。
func sample(probs []float64, sum, temp float64, rng *rand.Rand) int32 {
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
		// 上位 K 個だけで再正規化する。
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
			keep := make([]bool, len(probs))
			for _, i := range idx {
				keep[i] = true
				sum += probs[i]
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
