// Package lm はトークン列の n-gram マルコフ連鎖言語モデル。
//
// 学習は数えるだけで、確率は absolute discounting による線形補間で求める。
// 高次から低次へ順に混ぜるので、見たことのない文脈は自動的に低次に落ちる。
package lm

// discount は absolute discounting の割引率 D。
const discount = 0.75

// Model は n-gram のカウントと補間用の統計を持つ。
type Model struct {
	order int
	v     int

	uni   []int32
	total int64

	grams map[uint64]int32 // n-gram の出現数
	ctxs  map[uint64]int32 // (n-1)-gram 文脈の出現数
	n1    map[uint64]int32 // 文脈ごとの異なり継続数
}

// Train は ids から order 次のモデルを作る。order は 1..3。
func Train(ids []int32, v, order int) *Model {
	if order < 1 {
		order = 1
	}
	if order > 3 {
		order = 3
	}
	m := &Model{
		order: order,
		v:     v,
		uni:   make([]int32, v),
		grams: make(map[uint64]int32, len(ids)*2),
		ctxs:  make(map[uint64]int32, len(ids)),
		n1:    make(map[uint64]int32, len(ids)),
	}
	for _, id := range ids {
		m.uni[id]++
		m.total++
	}
	for n := 2; n <= order; n++ {
		for i := n - 1; i < len(ids); i++ {
			g := ids[i-n+1 : i+1]
			m.grams[pack(g...)]++
			m.ctxs[pack(g[:n-1]...)]++
		}
	}
	// 文脈ごとの異なり継続数。gram のキーから末尾 id を落として数える。
	for g := range m.grams {
		n := int(g >> 48)
		ctx := (uint64(n-1) << 48) | ((g & 0x0000ffffffffffff) >> 16)
		m.n1[ctx]++
	}
	return m
}

// Prob は文脈 ctx の後に w が来る確率を返す。ctx は order-1 個まで。
func (m *Model) Prob(ctx []int32, w int32) float64 {
	if len(ctx) > m.order-1 {
		ctx = ctx[len(ctx)-(m.order-1):]
	}
	p := m.uniProb(w)
	for k := 1; k <= len(ctx); k++ {
		sub := ctx[len(ctx)-k:]
		ck := pack(sub...)
		c := m.ctxs[ck]
		if c == 0 {
			continue
		}
		cw := m.grams[keyWith(sub, w)]
		lambda := discount * float64(m.n1[ck]) / float64(c)
		disc := (float64(cw) - discount) / float64(c)
		if disc < 0 {
			disc = 0
		}
		p = disc + lambda*p
	}
	return p
}

// uniProb は加法平滑化した unigram 確率。
func (m *Model) uniProb(w int32) float64 {
	return (float64(m.uni[w]) + 0.5) / (float64(m.total) + 0.5*float64(m.v))
}

// pack は最大 3 個の id を 64bit に詰める。先頭 16bit は個数、
// 残り 48bit に id を下位から並べる。id は 16bit に収まる (語彙 65535 まで) 前提。
func pack(ids ...int32) uint64 {
	k := uint64(len(ids)) << 48
	for i, id := range ids {
		k |= uint64(uint16(id)) << (16 * (len(ids) - 1 - i))
	}
	return k
}

// keyWith は文脈 ctx に w を続けた (len(ctx)+1)-gram のキー。
// append による ctx の裏配列の書き換えを避けるため、キーを直接組み立てる。
func keyWith(ctx []int32, w int32) uint64 {
	n := len(ctx) + 1
	k := uint64(n) << 48
	for i, id := range ctx {
		k |= uint64(uint16(id)) << (16 * (n - 1 - i))
	}
	return k | uint64(uint16(w))
}

// Grams は学習した n-gram の種類数。
func (m *Model) Grams() int { return len(m.grams) }
