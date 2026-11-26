// train はコーパスから n-gram マルコフ連鎖モデルを学習し、ファイルに保存する。
// 分割には testdata/dic.txt と merges.txt を使うので、先に `tok` を実行しておく。
package cli

import (
	"flag"
	"hash/fnv"
	"log"
	"os"
	"time"

	"github.com/minotto165/tiny-llm/lm"
	"github.com/minotto165/tiny-llm/tokenizer"
)

func Train(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	inPath := fs.String("in", "testdata/input.txt", "コーパス")
	order := fs.Int("order", 3, "マルコフ次数 (1..10)")
	outPath := fs.String("out", "testdata/ngram.bin", "保存先")
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

	model := tokenizer.NewModel(vocab, merges)
	t0 := time.Now()
	ids := model.Encode(string(src))
	tokDur := time.Since(t0)

	t1 := time.Now()
	m := lm.TrainFull(ids, len(vocab), *order, lm.DefaultDiscount, hiKeepFor(*order), hiWeightFor(*order))
	trainDur := time.Since(t1)
	m.SetTokenizerHash(hash)

	t2 := time.Now()
	if err := m.SaveFile(*outPath); err != nil {
		log.Fatal(err)
	}
	log.Printf("corpus: %d tokens, order %d, %d grams", len(ids), *order, m.Grams())
	log.Printf("tokenize: %s, train: %s, save: %s", tokDur.Round(time.Millisecond), trainDur.Round(time.Millisecond), time.Since(t2).Round(time.Millisecond))
	log.Printf("saved: %s", *outPath)
}

// tokenizerHash は dic.txt と merges.txt の内容から語彙の同一性を表すハッシュを作る。
func tokenizerHash(dir string) (uint64, error) {
	h := fnv.New64a()
	dictPath, mergesPath := artifacts(dir)
	for _, p := range []string{dictPath, mergesPath} {
		b, err := os.ReadFile(p)
		if err != nil {
			return 0, err
		}
		h.Write(b)
	}
	return h.Sum64(), nil
}
