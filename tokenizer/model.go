package tokenizer

import (
	"bufio"
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// 特殊トークンは id を 0 から固定で予約する。通常の語彙はこの後ろに並ぶので、
// 学習後に特殊トークンを足しても既存の id は変わらない。
const (
	PadID int32 = 0
	UnkID int32 = 1
	BosID int32 = 2
	EosID int32 = 3
)

// SpecialTokens は語彙の先頭に必ず置く特殊トークン。
var SpecialTokens = []string{"<pad>", "<unk>", "<bos>", "<eos>"}

// Merge は 2 トークン (A, B) を 1 トークン New に併合する規則。
// 学習結果では有効な順（優先度順）に並ぶ。
type Merge struct {
	A, B, New int32
}

// Model は学習済みの語彙と併合規則。新しいテキストの分割に使う。
type Model struct {
	Vocab  []string
	Merges []Merge

	index map[string]int32
	rank  map[int64]int32
}

// NewModel は語彙と併合規則から分割器を作る。
func NewModel(vocab []string, merges []Merge) *Model {
	m := &Model{
		Vocab:  vocab,
		Merges: merges,
		index:  make(map[string]int32, len(vocab)),
		rank:   make(map[int64]int32, len(merges)),
	}
	for i, t := range vocab {
		m.index[t] = int32(i)
	}
	for _, g := range merges {
		m.rank[pairKey(g.A, g.B)] = g.New
	}
	return m
}

// Model は学習結果から分割器を作る。
func (r Result) Model() *Model { return NewModel(r.Vocab, r.Merges) }

func pairKey(a, b int32) int64 { return int64(a)<<32 | int64(uint32(b)) }

// Encode は text をトークン id 列に分割する。語彙に無いルーンは <unk> になる。
//
// 優先度 (id) の小さいペアから順に、重ならない出現をすべて併合する。学習時の
// 併合順と同じ規則なので、併合規則をそのまま再現できる。
func (m *Model) Encode(text string) []int32 {
	ids := make([]int32, 0, len(text))
	for _, r := range text {
		if id, ok := m.index[string(r)]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, UnkID)
		}
	}
	n := len(ids)
	if n < 2 || len(m.rank) == 0 {
		return ids
	}

	// 双方向リストと優先度ヒープで、低い id のペアから併合する。
	prev := make([]int, n)
	next := make([]int, n)
	for i := range ids {
		prev[i] = i - 1
		next[i] = i + 1
	}
	next[n-1] = -1

	h := &pairHeap{}
	push := func(i int) {
		if i < 0 || next[i] < 0 {
			return
		}
		a, b := ids[i], ids[next[i]]
		if id, ok := m.rank[pairKey(a, b)]; ok {
			heap.Push(h, pairItem{rank: id, pos: i, a: a, b: b})
		}
	}
	for i := 0; i < n; i++ {
		push(i)
	}

	dead := make([]bool, n)
	for h.Len() > 0 {
		it := heap.Pop(h).(pairItem)
		if dead[it.pos] {
			continue
		}
		j := next[it.pos]
		if j < 0 || ids[it.pos] != it.a || ids[j] != it.b {
			continue // 併合で内容が変わった古い候補
		}
		ids[it.pos] = it.rank
		dead[j] = true
		nj := next[j]
		next[it.pos] = nj
		if nj >= 0 {
			prev[nj] = it.pos
		}
		push(prev[it.pos])
		push(it.pos)
	}

	out := make([]int32, 0, n)
	for i := 0; i < n; i++ {
		if !dead[i] {
			out = append(out, ids[i])
		}
	}
	return out
}

// Decode はトークン id 列をテキストに戻す。特殊トークンは出力しない。
func (m *Model) Decode(ids []int32) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < int32(len(SpecialTokens)) || int(id) >= len(m.Vocab) {
			continue
		}
		sb.WriteString(m.Vocab[id])
	}
	return sb.String()
}

type pairItem struct {
	rank int32
	pos  int
	a, b int32
}

type pairHeap []pairItem

func (h pairHeap) Len() int { return len(h) }
func (h pairHeap) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].pos < h[j].pos
}
func (h pairHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *pairHeap) Push(x any)   { *h = append(*h, x.(pairItem)) }
func (h *pairHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// WriteMerges は併合規則を優先度順の "a b new" 行として書き出す。
func WriteMerges(path string, merges []Merge) error {
	var sb strings.Builder
	for _, g := range merges {
		fmt.Fprintf(&sb, "%d %d %d\n", g.A, g.B, g.New)
	}
	return os.WriteFile(path, []byte(sb.String()), 0o644)
}

// ReadMerges は WriteMerges の出力を読み込む。
func ReadMerges(path string) ([]Merge, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var merges []Merge
	sc := bufio.NewScanner(f)
	for line := 1; sc.Scan(); line++ {
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 3 {
			return nil, fmt.Errorf("%s:%d: want 3 fields, got %d", path, line, len(fields))
		}
		var v [3]int64
		for i, s := range fields {
			x, err := strconv.ParseInt(s, 10, 32)
			if err != nil {
				return nil, fmt.Errorf("%s:%d: %w", path, line, err)
			}
			v[i] = x
		}
		merges = append(merges, Merge{A: int32(v[0]), B: int32(v[1]), New: int32(v[2])})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return merges, nil
}

// ReadVocab は WriteVocab の出力を読み込む。
func ReadVocab(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var vocab []string
	if err := json.Unmarshal(data, &vocab); err != nil {
		return nil, fmt.Errorf("parse vocabulary: %w", err)
	}
	return vocab, nil
}
