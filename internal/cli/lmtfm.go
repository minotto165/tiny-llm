// lmtfm は保存済み Transformer の held-out BPB を評価する。
package cli

import (
	"log"
	"math"
	"os"
	"time"

	"github.com/minotto165/tiny-llm/internal/nn"
	"github.com/minotto165/tiny-llm/tokenizer"
)

// evalTfm は tail を Transformer で評価し、BPB (bits per UTF-8 byte) を出す。
func evalTfm(modelPath, tail string, limit int) {
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
	f, err := os.Open(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	m, tokHash, err := nn.Load(f)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	if tokHash != hash || m.Cfg.Vocab != len(vocab) {
		log.Fatalf("model の語彙が dic.txt/merges.txt と一致しません (tfm をやり直してください)")
	}

	tok := tokenizer.NewModel(vocab, merges)
	ids := tok.Encode(tail)

	t0 := time.Now()
	ctx := m.Cfg.Ctx
	numWin := (len(ids) - 1 + ctx - 1) / ctx
	stride := 1
	if limit > 0 && limit < len(ids) {
		stride = (numWin*ctx + limit - 1) / limit
		if stride < 1 {
			stride = 1
		}
	}
	var bits float64
	var n, unk int
	for wi := 0; wi < numWin; wi += stride {
		from := wi * ctx
		to := from + ctx
		if to > len(ids) {
			to = len(ids)
		}
		c := m.Forward(ids[from:to])
		for i := 0; i < to-from-1; i++ {
			row := append([]float64(nil), c.Logits.D[i*c.Logits.C:(i+1)*c.Logits.C]...)
			max := row[0]
			for _, v := range row {
				if v > max {
					max = v
				}
			}
			var sum float64
			for j := range row {
				row[j] = math.Exp(row[j] - max)
				sum += row[j]
			}
			id := ids[from+i+1]
			p := row[id] / sum
			if p < 1e-12 {
				p = 1e-12
			}
			bits -= math.Log2(p)
			if id == tokenizer.UnkID {
				unk++
			}
			n++
		}
	}
	bytes := int(float64(n) * float64(len(tail)) / float64(len(ids)))
	log.Printf("transformer: %d bytes, %d tokens, unk %d (%.2f%%), BPB %.4f",
		bytes, n, unk, 100*float64(unk)/float64(n), bits/float64(bytes))
	log.Printf("apply: %s", time.Since(t0).Round(time.Millisecond))
}
