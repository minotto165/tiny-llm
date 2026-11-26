// gentfm は Transformer モデルでの生成。gen から形式を判定して呼ばれる。
package cli

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/minotto165/tiny-llm/internal/nn"
	"github.com/minotto165/tiny-llm/tokenizer"
)

func genTfm(path, prompt string, n int, temp float64, seed int64, tok *tokenizer.Model, hash uint64, check string, checkn, samples int) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	m, tokHash, err := nn.Load(f)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	if m.Cfg.Vocab != len(tok.Vocab) {
		log.Fatalf("model vocab %d != dict vocab %d (tfm をやり直してください)", m.Cfg.Vocab, len(tok.Vocab))
	}
	if tokHash != hash {
		log.Fatalf("model の語彙が dic.txt/merges.txt と一致しません (tfm をやり直してください)")
	}
	log.Printf("model: %s (dim %d, %d layers, ctx %d)", path, m.Cfg.Dim, m.Cfg.Layers, m.Cfg.Ctx)

	if len(tok.Encode(strings.TrimPrefix(prompt, "<bos>"))) == 0 {
		log.Fatal("プロンプトを指定してください")
	}
	var all []string
	for k := 0; k < samples; k++ {
		if k > 0 {
			fmt.Println()
		}
		all = append(all, genTfmOne(m, tok, prompt, n, temp, seed+int64(k)))
	}
	if check != "" {
		mc, err := newMemCheck(check, checkn)
		if err != nil {
			log.Fatal(err)
		}
		multiReport(mc, all)
	}
}

// genTfmOne は 1 サンプル生成し、生成部分を返す。
func genTfmOne(m *nn.Model, tok *tokenizer.Model, prompt string, n int, temp float64, seed int64) string {
	// "<bos>" で始めると文書の先頭から生成する（<bos> は特殊トークンなので Encode では出てこない）
	out := tok.Encode(prompt)
	if rest, ok := strings.CutPrefix(prompt, "<bos>"); ok {
		out = append([]int32{tokenizer.BosID}, tok.Encode(rest)...)
	}
	fmt.Print(prompt)
	var gen strings.Builder
	rng := rand.New(rand.NewSource(seed))
	for i := 0; i < n; i++ {
		from := len(out) - m.Cfg.Ctx
		if from < 0 {
			from = 0
		}
		c := m.Forward(out[from:])
		t := c.T
		row := append([]float64(nil), c.Logits.D[(t-1)*c.Logits.C:t*c.Logits.C]...)
		recent := out
		if len(recent) > repWindow {
			recent = recent[len(recent)-repWindow:]
		}
		id := sampleLogits(row, temp, 40, len(tokenizer.SpecialTokens), recent, rng)
		out = append(out, id)
		gen.WriteString(tok.Vocab[id])
		fmt.Print(tok.Vocab[id])
	}
	fmt.Println()
	return gen.String()
}

const (
	repPenalty = 1.15 // 直近 repWindow トークンの重複ペナルティ
	repWindow  = 32
)

// sampleLogits はロジットから 1 トークンをサンプルする。recent の重複は減衰させる。
func sampleLogits(logits []float64, temp float64, k, skip int, recent []int32, rng *rand.Rand) int32 {
	penalize := func(i int) float64 {
		v := logits[i]
		for _, r := range recent {
			if int(r) == i {
				v -= math.Log(repPenalty)
			}
		}
		if temp > 0 {
			v /= temp
		}
		return v
	}
	if temp <= 0 {
		best := skip
		for i := skip; i < len(logits); i++ {
			if penalize(i) > penalize(best) {
				best = i
			}
		}
		return int32(best)
	}
	for i := range logits {
		logits[i] = penalize(i)
	}
	idx := make([]int, 0, len(logits)-skip)
	for i := skip; i < len(logits); i++ {
		idx = append(idx, i)
	}
	if k > 0 && k < len(idx) {
		sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
		idx = idx[:k]
	}
	max := math.Inf(-1)
	for _, i := range idx {
		if logits[i] > max {
			max = logits[i]
		}
	}
	var sum float64
	probs := make(map[int]float64, len(idx))
	for _, i := range idx {
		p := math.Exp(logits[i] - max)
		probs[i] = p
		sum += p
	}
	r := rng.Float64() * sum
	for _, i := range idx {
		r -= probs[i]
		if r <= 0 {
			return int32(i)
		}
	}
	return int32(idx[len(idx)-1])
}
