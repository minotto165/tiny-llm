package tokenizer

import (
	"log"
	"sort"
)

// trainBPE は隣接ペアの頻度上位をまとめてマージする。
//
// 旧実装は「各位置で最大長 10 の部分文字列を strings.Count で数え、score=len^4*count の
// 最良を辞書化 → 最長一致で再分割」だった。数え直しが O(n^2) 超で 80KB に 23 秒かかり、
// 辞書化と再分割で基準がずれ、助詞をまたぐ糊付けが起きていた。
//
// こちらは 1 パスで batch 個のペアを併合するので、ペア数え直しは
// O(n * vocabSize / batch) 回で済む。併合は「すでに単位になったものの連結」だけなので、
// 辞書と分割が構造的に一致する。
func trainBPE(b []byte, offs []int, minCount int32, vocabSize, batch int, quiet bool) ([]int32, []string, []bool) {
	n := len(offs) - 1

	// 初期記号は 1 ルーンずつ。id は出現順に振るので、同じ入力からは同じ id 列になる。
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
		barrier[i] = IsBarrier([]rune(t)[0])
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

		// 2) 頻度上位 batch 個のペアを選ぶ。同点はキーで割って結果を決定的にする。
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
