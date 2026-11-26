// ids はコーパスをトークン ID の列に変換してバイナリで書き出す。
// PyTorch 側 (scripts/train_torch.py) が同じ ID 列を読めるようにするためのもの。
package cli

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"log"
	"os"
	"strings"
	"time"

	"github.com/minotto165/tiny-llm/tokenizer"
)

// doc は -jsonl で読む1文書。
type doc struct {
	Title string `json:"title"`
	Text  string `json:"text"`
}

// window は -windows で読む1窓（タイトル＋見出し＋本文）。
type window struct {
	Title   string `json:"title"`
	Heading string `json:"heading"`
	Body    string `json:"body"`
}

func Ids(args []string) {
	fs := flag.NewFlagSet("ids", flag.ExitOnError)
	inPath := fs.String("in", "testdata/lit.txt", "コーパス")
	outPath := fs.String("out", "testdata/ids.bin", "出力 (int32 little endian)")
	dir := fs.String("dir", "testdata", "dic.txt / merges.txt のあるディレクトリ")
	jsonlPath := fs.String("jsonl", "", "1行1文書の JSONL を読み、文書ごとに <bos>...<eos> で囲む")
	windowsPath := fs.String("windows", "", "1行1窓の JSONL {title,heading,body} を読み、固定長の行として書き出す")
	rowLen := fs.Int("rowlen", 128, "-windows のときの1行のトークン数")
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
	src, err := os.ReadFile(*inPath)
	if *jsonlPath != "" || *windowsPath != "" {
		src = nil
	} else if err != nil {
		log.Fatal(err)
	}
	t0 := time.Now()
	tok := tokenizer.NewModel(vocab, merges)
	var ids []int32
	if *windowsPath != "" {
		f, err := os.Open(*windowsPath)
		if err != nil {
			log.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<28)
		rows, skipped := 0, 0
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			var w window
			if err := json.Unmarshal([]byte(line), &w); err != nil {
				f.Close()
				log.Fatalf("%s: %v", *windowsPath, err)
			}
			row := []int32{tokenizer.BosID}
			row = append(row, tok.Encode(w.Title)...)
			if w.Heading != "" {
				row = append(row, tok.Encode("== "+w.Heading+" ==\n")...)
			}
			row = append(row, tok.Encode(w.Body)...)
			if len(row) < *rowLen/2 {
				skipped++
				continue
			}
			if len(row) > *rowLen {
				row = row[:*rowLen]
			}
			for len(row) < *rowLen {
				row = append(row, tokenizer.PadID)
			}
			ids = append(ids, row...)
			rows++
		}
		if err := sc.Err(); err != nil {
			f.Close()
			log.Fatal(err)
		}
		f.Close()
		log.Printf("window rows: %d (短すぎて除外 %d), rowlen %d", rows, skipped, *rowLen)
	} else if *jsonlPath != "" {
		f, err := os.Open(*jsonlPath)
		if err != nil {
			log.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<28)
		nDocs := 0
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			var d doc
			if err := json.Unmarshal([]byte(line), &d); err != nil {
				f.Close()
				log.Fatalf("%s: %v", *jsonlPath, err)
			}
			text := d.Text
			if d.Title != "" {
				text = d.Title + "\n" + text
			}
			ids = append(ids, tokenizer.BosID)
			ids = append(ids, tok.Encode(text)...)
			ids = append(ids, tokenizer.EosID)
			nDocs++
		}
		if err := sc.Err(); err != nil {
			f.Close()
			log.Fatal(err)
		}
		f.Close()
		log.Printf("documents: %d", nDocs)
	} else {
		ids = tok.Encode(string(src))
	}

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
