// lm は n-gram マルコフ連鎖言語モデルを学習し、held-out の BPB (bits per UTF-8 byte) を出す。
package cli

import (
	"flag"
	"log"
	"math"
	"os"
	"time"

	"github.com/minotto165/tiny-llm/lm"
	"github.com/minotto165/tiny-llm/tokenizer"
)

func LM(args []string) {
	fs := flag.NewFlagSet("lm", flag.ExitOnError)
	inPath := fs.String("in", "testdata/input.txt", "コーパス")
	order := fs.Int("order", 3, "マルコフ次数 (1..10)")
	evalPath := fs.String("eval", "", "別ファイルで評価する（-in 全体で学習）")
	modelPath := fs.String("model", "", "保存済み Transformer を評価する (tfm.bin)")
	limit := fs.Int("limit", 0, "先頭 N トークンだけ評価 (0=全部)")
	holdout := fs.Float64("holdout", 0.05, "末尾から評価に回す割合")
	minChar := fs.Int("minchar", 50, "これ未満の頻度の文字は <unk> にする")
	dir := fs.String("dir", "testdata", "dic.txt / merges.txt のあるディレクトリ")
	fs.Parse(args)

	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}

	// tok と同じ位置で切る（語彙・分割を held-out から隔離する）。
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

	if *modelPath != "" {
		evalTfm(*modelPath, tail, *limit, *dir)
		return
	}

	t0 := time.Now()
	res, err := tokenizer.Train(head, tokenizer.Config{
		VocabSize:    32000,
		MinCount:     minCount,
		MinCharCount: *minChar,
		Batch:        batch,
	})
	if err != nil {
		log.Fatal(err)
	}
	model := res.Model()
	headIDs := model.Encode(string(head))
	tailIDs := model.Encode(tail)
	tokDur := time.Since(t0)

	t1 := time.Now()
	m := lm.TrainFull(headIDs, len(res.Vocab), *order, lm.DefaultDiscount, hiKeepFor(*order), hiWeightFor(*order))
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
	log.Printf("train: %d tokens, %d grams; tokenize %v, apply %v",
		len(headIDs), m.Grams(), tokDur.Round(time.Millisecond), evalDur.Round(time.Millisecond))
}

// hiKeepFor / hiWeightFor は 4 次以上の n-gram の扱いを次数から決める。
// 既定 (2, 1.0) のままでは 4 次以上が全部 pruning されて次数を上げる意味がなく、
// hikeep 1 をそのまま重み 1 で使うと学習データの丸暗記に寄るため。
func hiKeepFor(order int) int32 {
	if order >= 4 {
		return 1
	}
	return lm.DefaultHiKeep
}

func hiWeightFor(order int) float64 {
	if order >= 4 {
		return 0.5
	}
	return lm.DefaultHiWeight
}
