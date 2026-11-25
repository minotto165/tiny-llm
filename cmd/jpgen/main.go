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
	"strings"

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
	balance := flag.Bool("balance", true, "括弧の対応を保つ（開いた括弧を閉じる方向にバイアス）")
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
		stack := openBrackets(model.Decode(out))

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
			if *balance {
				p *= bracketBias(stack, model.Vocab[w])
			}
			probs[w] = p
			sum += p
		}
		if sum <= 0 {
			log.Fatal("no probability mass")
		}
		out = append(out, sample(probs, sum, *temp, *topK, rng))
	}
	txt := model.Decode(out)
	if *balance {
		txt += closeBrackets(txt)
	}
	fmt.Println(txt)
}

// 括弧の対応を扱う。生成中に開いた括弧を閉じる方向へ確率を寄せ、
// 最後に閉じ忘れを補う。
var bracketPairs = map[rune]rune{'「': '」', '『': '』', '（': '）', '(': ')', '【': '】'}

// openBrackets は閉じていない括弧を開いた順に返す。
func openBrackets(text string) []rune {
	var stack []rune
	for _, r := range text {
		if _, ok := bracketPairs[r]; ok {
			stack = append(stack, r)
			continue
		}
		for i := len(stack) - 1; i >= 0; i-- {
			if bracketPairs[stack[i]] == r {
				stack = append(stack[:i], stack[i+1:]...)
				break
			}
		}
	}
	return stack
}

// bracketBias は、閉じていない括弧があるとき閉じ括弧を優遇し、開き括弧を抑える。
func bracketBias(stack []rune, token string) float64 {
	if len(stack) == 0 {
		return 1
	}
	want := bracketPairs[stack[len(stack)-1]]
	for _, r := range token {
		if r == want {
			return 3
		}
		if _, ok := bracketPairs[r]; ok {
			return 0.25
		}
	}
	return 1
}

// closeBrackets は閉じ忘れた括弧を末尾に補う。
func closeBrackets(text string) string {
	stack := openBrackets(text)
	var sb strings.Builder
	for i := len(stack) - 1; i >= 0; i-- {
		sb.WriteRune(bracketPairs[stack[i]])
	}
	return sb.String()
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
