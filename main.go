package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// BPE (byte pair encoding) による日本語トークナイザ。
//
// 旧実装は「各位置で最大長10の部分文字列を strings.Count で数え、score=len^4*count の
// 最良を辞書化 → 最長一致で再分割」だった。数え直しが O(n^2) 超で 80KB に 23 秒かかり、
// 辞書化と再分割で基準がずれ、助詞をまたぐ糊付けが起きていた。
//
// 新実装は隣接ペアの頻度を上から順にマージする BPE。頻度の数え直しは 1 パスで済み、
// 併合は「すでに単位になったものの連結」だけなので、辞書と分割が構造的に一致する。
//
// 空白・改行はマージ禁止の境界として扱い、トークンとしてそのまま残す（可逆）。
func main() {
	inPath := flag.String("in", "testdata/input.txt", "input text file (UTF-8)")
	outPath := flag.String("out", "testdata/output.txt", "segmented output file")
	dictPath := flag.String("dict", "testdata/dic.txt", "vocabulary (JSON) output file")
	vocabSize := flag.Int("vocab", 32000, "target vocabulary size (merges stop here)")
	minCount := flag.Int("mincount", 2, "minimum pair frequency to merge")
	batch := flag.Int("batch", 2000, "merges applied per counting pass (speed/quality knob)")
	quiet := flag.Bool("quiet", false, "suppress progress logs")
	flag.Parse()

	src, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatal(err)
	}
	if !utf8.Valid(src) {
		log.Fatal("input is not valid UTF-8")
	}
	if *minCount < 1 {
		log.Fatal("mincount must be >= 1")
	}
	if *batch < 1 {
		log.Fatal("batch must be >= 1")
	}

	offs := runeOffsets(src)
	n := len(offs) - 1
	if n == 0 {
		log.Fatal("empty input")
	}

	t0 := time.Now()
	ids, vocab, _ := trainBPE(src, offs, int32(*minCount), *vocabSize, *batch, *quiet)
	buildDur := time.Since(t0)

	tokens := make([]string, len(ids))
	var totalLen int
	oneChar := 0
	for i, id := range ids {
		t := vocab[id]
		tokens[i] = t
		l := utf8.RuneCountInString(t)
		totalLen += l
		if l == 1 {
			oneChar++
		}
	}

	// 可逆性の検査: トークンを連結すると入力に戻るか。
	if got := strings.Join(tokens, ""); got != string(src) {
		log.Fatalf("internal error: detokenization mismatch (%d vs %d bytes)", len(got), len(src))
	}

	out := strings.Join(tokens, "/")
	if err := os.WriteFile(*outPath, []byte(out), 0o644); err != nil {
		log.Fatal(err)
	}
	if err := writeVocab(*dictPath, vocab); err != nil {
		log.Fatal(err)
	}

	if !*quiet {
		log.Printf("corpus: %d runes", n)
		log.Printf("tokens: %d, vocab: %d, avg len: %.2f, 1-char: %.1f%%",
			len(tokens), len(vocab), float64(totalLen)/float64(len(tokens)),
			100*float64(oneChar)/float64(len(tokens)))
		log.Printf("build (all merges): %s, total: %s", buildDur, time.Since(t0))
	}
}

// runeOffsets は各ルーン先頭のバイト位置と末尾を返す。
func runeOffsets(b []byte) []int {
	offs := make([]int, 0, len(b)+1)
	for i := 0; i < len(b); {
		offs = append(offs, i)
		_, w := utf8.DecodeRune(b[i:])
		i += w
	}
	offs = append(offs, len(b))
	return offs
}

func isBarrier(r rune) bool {
	return unicode.IsSpace(r) || unicode.IsControl(r)
}

// trainBPE は隣接ペアの頻度上位をまとめてマージする。
// 1 パスで batch 個のペアを併合するので、ペア数え直しは O(n * log(vocab) / batch) 回で済む。
func trainBPE(b []byte, offs []int, minCount int32, vocabSize, batch int, quiet bool) ([]int32, []string, []bool) {
	n := len(offs) - 1

	base := make([]string, 0, 8192)
	index := make(map[string]int32, 8192)
	ids := make([]int32, n)
	for i := 0; i < n; i++ {
		r := string(b[offs[i]:offs[i+1]])
		id, ok := index[r]
		if !ok {
			id = int32(len(base))
			index[r] = id
			base = append(base, r)
		}
		ids[i] = id
	}

	vocab := base
	capHint := vocabSize + 4096
	if capHint < len(vocab) {
		capHint = len(vocab)
	}
	barrier := make([]bool, len(vocab), capHint)
	for i, t := range vocab {
		barrier[i] = isBarrier([]rune(t)[0])
	}

	for pass := 0; len(vocab) < vocabSize; pass++ {
		// 1) 隣接ペアを 1 パスで数える。
		counts := make(map[int64]int32, len(ids)/2+1)
		for i := 0; i+1 < len(ids); i++ {
			a, c := ids[i], ids[i+1]
			if barrier[a] || barrier[c] {
				continue
			}
			counts[int64(a)<<32|int64(c)]++
		}

		// 2) 頻度上位 batch 個のペアを選ぶ。
		type cand struct {
			key int64
			n   int32
		}
		cands := make([]cand, 0, len(counts))
		for k, v := range counts {
			if v >= minCount {
				cands = append(cands, cand{k, v})
			}
		}
		if len(cands) == 0 {
			break
		}
		sort.Slice(cands, func(i, j int) bool {
			if cands[i].n != cands[j].n {
				return cands[i].n > cands[j].n
			}
			return cands[i].key < cands[j].key
		})
		if len(cands) > batch {
			cands = cands[:batch]
		}
		if room := vocabSize - len(vocab); len(cands) > room {
			cands = cands[:room]
		}

		// 3) 併合先の id を確定する。
		rank := make(map[int64]int32, len(cands))
		for _, c := range cands {
			a, bb := int32(c.key>>32), int32(c.key&0xffffffff)
			id := int32(len(vocab))
			vocab = append(vocab, vocab[a]+vocab[bb])
			barrier = append(barrier, false)
			rank[c.key] = id
		}

		// 4) 併合を 1 パスで適用する。左から見て、右隣のペアより優先度が高ければ併合。
		rankAt := func(i int) int32 {
			if i+1 >= len(ids) {
				return -1
			}
			a, c := ids[i], ids[i+1]
			if barrier[a] || barrier[c] {
				return -1
			}
			key := int64(a)<<32 | int64(c)
			if id, ok := rank[key]; ok {
				return id
			}
			return -1
		}
		out := make([]int32, 0, len(ids))
		for i := 0; i < len(ids); {
			if i+1 < len(ids) {
				l := rankAt(i)
				if l >= 0 {
					r := rankAt(i + 1)
					if r < 0 || l < r {
						out = append(out, l)
						i += 2
						continue
					}
				}
			}
			out = append(out, ids[i])
			i++
		}
		ids = out

		if !quiet && (pass%10 == 0 || len(vocab) >= vocabSize) {
			log.Printf("pass %d: vocab %d, symbols %d", pass, len(vocab), len(ids))
		}
	}
	return ids, vocab, barrier
}

func writeVocab(path string, vocab []string) error {
	data, err := json.MarshalIndent(vocab, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal vocabulary: %w", err)
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}
