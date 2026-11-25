// gen は学習済み n-gram モデルで続きを生成する。
// モデルは `train` が保存したファイルを読み込み、分割は testdata/dic.txt を使う。
package cli

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"

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
	modelPath := fs.String("model", "testdata/ngram.bin", "学習済みモデル (train で作成)")
	prompt := fs.String("prompt", "", "プロンプト")
	n := fs.Int("n", 80, "生成するトークン数")
	temp := fs.Float64("temp", 0.8, "温度 (0 = 貪欲)")
	seed := fs.Int64("seed", 1, "乱数シード")
	fs.Parse(args)

	vocab, err := tokenizer.ReadVocab(dictPath)
	if err != nil {
		log.Fatal(err)
	}
	merges, err := tokenizer.ReadMerges(mergesPath)
	if err != nil {
		log.Fatal(err)
	}
	hash, err := tokenizerHash()
	if err != nil {
		log.Fatal(err)
	}
	model := tokenizer.NewModel(vocab, merges)

	t0 := time.Now()
	m, err := lm.LoadFile(*modelPath)
	if err != nil {
		log.Fatalf("%v (先に `go run . train` を実行してください)", err)
	}
	if m.Vocab() != len(vocab) {
		log.Fatalf("model vocab %d != dict vocab %d (train をやり直してください)", m.Vocab(), len(vocab))
	}
	if m.TokenizerHash() != hash {
		log.Fatalf("model の語彙が dic.txt/merges.txt と一致しません (train をやり直してください)")
	}
	log.Printf("model: %s (order %d, %d grams) in %s", *modelPath, m.Order(), m.Grams(), time.Since(t0).Round(time.Millisecond))

	out := model.Encode(*prompt)
	fmt.Print(*prompt)
	probs := make([]float64, len(vocab))
	rng := rand.New(rand.NewSource(*seed))
	for i := 0; i < *n; i++ {
		from := len(out) - (m.Order() - 1)
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
		id := sample(probs, sum, *temp, rng)
		out = append(out, id)
		fmt.Print(model.Vocab[id])
	}
	fmt.Println()
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
