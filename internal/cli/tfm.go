// tfm はコーパスから小さな Transformer を学習して保存する。
// 学習はユーザーのマシンで回す前提で、進捗をログに出す。
package cli

import (
	"flag"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/minotto165/tiny-llm/internal/nn"
	"github.com/minotto165/tiny-llm/tokenizer"
)

func Tfm(args []string) {
	fs := flag.NewFlagSet("tfm", flag.ExitOnError)
	inPath := fs.String("in", "testdata/lit.txt", "コーパス")
	outPath := fs.String("out", "testdata/tfm.bin", "保存先")
	ctx := fs.Int("ctx", 128, "文脈長")
	dim := fs.Int("dim", 128, "埋め込み次元")
	heads := fs.Int("heads", 4, "ヘッド数")
	layers := fs.Int("layers", 3, "ブロック数")
	ffn := fs.Int("ffn", 512, "FFN の次元")
	steps := fs.Int("steps", 2000, "学習ステップ数")
	batch := fs.Int("batch", 8, "1 ステップあたりの系列数")
	lr := fs.Float64("lr", 3e-4, "学習率")
	seed := fs.Int64("seed", 1, "乱数シード")
	dir := fs.String("dir", "testdata", "dic.txt / merges.txt のあるディレクトリ")
	fs.Parse(args)
	dictPath, mergesPath := artifacts(*dir)

	vocab, err := tokenizer.ReadVocab(dictPath)
	if err != nil {
		log.Fatal(err)
	}
	merges, err := tokenizer.ReadMerges(mergesPath)
	if err != nil {
		log.Fatal(err)
	}
	hash, err := tokenizerHash(*dir)
	if err != nil {
		log.Fatal(err)
	}
	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}
	tok := tokenizer.NewModel(vocab, merges)
	t0 := time.Now()
	ids := tok.Encode(string(src))
	log.Printf("corpus: %d tokens (%s)", len(ids), time.Since(t0).Round(time.Millisecond))

	cfg := nn.Config{Vocab: len(vocab), Ctx: *ctx, Dim: *dim, Heads: *heads, Layers: *layers, FFN: *ffn, Seed: *seed}
	m := nn.NewModel(cfg)
	opt := nn.NewAdam(m.All, *lr)
	log.Printf("model: dim %d, %d layers, %d heads, ctx %d, %.2fM params",
		cfg.Dim, cfg.Layers, cfg.Heads, cfg.Ctx, float64(m.NumParams())/1e6)

	if len(ids) < cfg.Ctx+1 {
		log.Fatalf("corpus too short: %d tokens", len(ids))
	}
	rnd := rand.New(rand.NewSource(*seed))
	start := time.Now()
	first := 0.0
	for step := 1; step <= *steps; step++ {
		m.ZeroGrads()
		var loss float64
		for b := 0; b < *batch; b++ {
			s := rnd.Intn(len(ids) - cfg.Ctx - 1)
			c := m.Forward(ids[s : s+cfg.Ctx])
			loss += m.Backward(c, ids[s+1:s+cfg.Ctx+1])
		}
		loss /= float64(*batch)
		opt.Step(m.All, 1.0)
		if step == 1 {
			first = loss
		}
		if step%50 == 0 || step == 1 {
			per := time.Since(start) / time.Duration(step)
			left := per * time.Duration(*steps-step)
			log.Printf("step %d/%d loss %.4f (%.2f ms/step, 残り %s)", step, *steps, loss,
				float64(per.Microseconds())/1000, left.Round(time.Second))
		}
	}
	log.Printf("loss %.4f -> %.4f in %s", first, m.Loss(m.Forward(ids[:cfg.Ctx]), ids[1:cfg.Ctx+1]), time.Since(start).Round(time.Second))

	f, err := os.Create(*outPath)
	if err != nil {
		log.Fatal(err)
	}
	if err := m.Save(f, hash); err != nil {
		f.Close()
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
	log.Printf("saved: %s", *outPath)
}
