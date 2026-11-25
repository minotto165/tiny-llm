// ids はコーパスをトークン ID の列に変換してバイナリで書き出す。
// PyTorch 側 (scripts/train_torch.py) が同じ ID 列を読めるようにするためのもの。
package cli

import (
	"bufio"
	"encoding/binary"
	"flag"
	"log"
	"os"
	"time"

	"github.com/minotto165/tiny-llm/tokenizer"
)

func Ids(args []string) {
	fs := flag.NewFlagSet("ids", flag.ExitOnError)
	inPath := fs.String("in", "testdata/lit.txt", "コーパス")
	outPath := fs.String("out", "testdata/ids.bin", "出力 (int32 little endian)")
	fs.Parse(args)

	vocab, err := tokenizer.ReadVocab(dictPath)
	if err != nil {
		log.Fatal(err)
	}
	merges, err := tokenizer.ReadMerges(mergesPath)
	if err != nil {
		log.Fatal(err)
	}
	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}
	t0 := time.Now()
	ids := tokenizer.NewModel(vocab, merges).Encode(string(src))

	f, err := os.Create(*outPath)
	if err != nil {
		log.Fatal(err)
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	if err := binary.Write(bw, binary.LittleEndian, int32(len(ids))); err != nil {
		log.Fatal(err)
	}
	if err := binary.Write(bw, binary.LittleEndian, ids); err != nil {
		log.Fatal(err)
	}
	if err := bw.Flush(); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
	log.Printf("%s: %d tokens (%s)", *outPath, len(ids), time.Since(t0).Round(time.Millisecond))
}
