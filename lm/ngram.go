// Package lm はトークン列の n-gram マルコフ連鎖言語モデル。
//
// 学習は数えるだけで、確率は Kneser-Ney 補間で求める。低次の分布には
// 生の出現数ではなく「その語の左に何種類の語が来たか」(継続数) を使う。
// 高次から低次へ順に混ぜるので、見たことのない文脈は自動的に低次に落ちる。
package lm

// DefaultDiscount は Kneser-Ney の割引率 D の既定値。
const DefaultDiscount = 0.85

// DefaultHiKeep / DefaultHiWeight は 4 次以上の既定パラメータ。
const (
	DefaultHiKeep   = 2
	DefaultHiWeight = 1.0
)

// Model は n-gram のカウントと補間用の統計を持つ。
type Model struct {
	order    int
	v        int
	discount float64

	uni   []int32 // 生の unigram 出現数 (Grams 用に保持)
	total int64

	grams map[uint64]int32 // 2 次以上の n-gram 出現数
	ctxs  map[uint64]int32 // (n-1)-gram 文脈の出現数
	n1    map[uint64]int32 // 文脈ごとの異なり継続数

	// 低次の継続数。cont1[w] は bigram (v,w) の異なり v 数。
	cont1 []int32
	n1uni int64

	// 3 次のときの bigram 継続数。cont2[(v,w)] は trigram (u,v,w) の異なり u 数。
	cont2    map[uint64]int32
	cont2ctx map[uint64]int32 // v から始まる異なり (u,w) 数
	cont2n1  map[uint64]int32 // v の後ろに来る異なり w 数

	hiKeep   int32   // 4 次以上で残す最小出現数
	hiWeight float64 // 4 次以上の推定を混ぜる重み (1 = そのまま)

	// 4 次以上は Witten-Bell の逃げ確率で積み上げる。添字は次数。
	hiCount []map[uint64]int32 // L-gram の出現数 (L >= 4)
	hiTot   []map[uint64]int32 // (L-1)-gram 文脈の総出現数
	hiDist  []map[uint64]int32 // (L-1)-gram 文脈の異なり継続数
}

// Train は ids から order 次のモデルを作る。order は 1..6。
func Train(ids []int32, v, order int) *Model {
	return TrainFull(ids, v, order, DefaultDiscount, DefaultHiKeep, DefaultHiWeight)
}

// TrainWithDiscount は割引率 D を指定してモデルを作る。
func TrainWithDiscount(ids []int32, v, order int, d float64) *Model {
	return TrainFull(ids, v, order, d, DefaultHiKeep, DefaultHiWeight)
}

// TrainFull は 4 次以上のパラメータも指定してモデルを作る。
func TrainFull(ids []int32, v, order int, d float64, hiKeep int32, hiWeight float64) *Model {
	if order < 1 {
		order = 1
	}
	if order > 6 {
		order = 6
	}
	m := &Model{
		order:    order,
		v:        v,
		discount: d,
		hiKeep:   hiKeep,
		hiWeight: hiWeight,
		uni:      make([]int32, v),
		grams:    make(map[uint64]int32, len(ids)*2),
		ctxs:     make(map[uint64]int32, len(ids)),
		n1:       make(map[uint64]int32, len(ids)),
		cont1:    make([]int32, v),
	}
	for _, id := range ids {
		m.uni[id]++
		m.total++
	}
	knOrder := order
	if knOrder > 3 {
		knOrder = 3
	}
	for n := 2; n <= knOrder; n++ {
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
	// unigram の継続数: bigram (v,w) の異なり v 数。
	for g := range m.grams {
		if g>>48 != 2 {
			continue
		}
		w := int32(g & 0xffff)
		m.cont1[w]++
		m.n1uni++
	}
	if knOrder >= 3 {
		m.cont2 = make(map[uint64]int32, len(m.grams)/2)
		m.cont2ctx = make(map[uint64]int32, len(m.grams)/8)
		m.cont2n1 = make(map[uint64]int32, len(m.grams)/8)
		for g := range m.grams {
			if g>>48 != 3 {
				continue
			}
			// (u,v,w) -> (v,w)
			k := (uint64(2) << 48) | (((g >> 16) & 0xffff) << 16) | (g & 0xffff)
			m.cont2[k]++
		}
		for k, c := range m.cont2 {
			vKey := (uint64(1) << 48) | ((k >> 16) & 0xffff)
			m.cont2ctx[vKey] += c
			m.cont2n1[vKey]++
		}
	}
	if order >= 4 {
		m.trainHigher(ids)
	}
	return m
}

// trainHigher は 4 次以上のカウントを作る。単独出現の n-gram は落として
// メモリを抑える (落とした分は逃げ確率として低次に流れる)。
func (m *Model) trainHigher(ids []int32) {
	m.hiCount = make([]map[uint64]int32, m.order+1)
	m.hiTot = make([]map[uint64]int32, m.order+1)
	m.hiDist = make([]map[uint64]int32, m.order+1)
	keep := m.hiKeep // これ未満の n-gram は捨てる
	for n := 4; n <= m.order; n++ {
		counts := make(map[uint64]int32, len(ids)/2)
		for i := n - 1; i < len(ids); i++ {
			counts[hashIDs(ids[i-n+1:i+1])]++
		}
		for k, c := range counts {
			if c < keep {
				delete(counts, k)
			}
		}
		// 残した n-gram についてだけ文脈の統計を作る。
		tot := make(map[uint64]int32, len(counts))
		dist := make(map[uint64]int32, len(counts))
		for i := n - 1; i < len(ids); i++ {
			g := ids[i-n+1 : i+1]
			if counts[hashIDs(g)] == 0 {
				continue
			}
			ck := hashIDs(g[:n-1])
			tot[ck]++
			dist[ck]++
		}
		m.hiCount[n], m.hiTot[n], m.hiDist[n] = counts, tot, dist
	}
}

// hashIDs は id 列を 64bit に畳む (個数も混ぜるので次数をまたいでも衝突しない)。
func hashIDs(ids []int32) uint64 {
	h := uint64(len(ids)) * 0x9e3779b97f4a7c15
	for _, id := range ids {
		h ^= uint64(uint32(id))
		h *= 0x100000001b3
		h ^= h >> 29
	}
	return h
}

// Prob は文脈 ctx の後に w が来る確率を返す。ctx は order-1 個まで。
// 3 次までは Kneser-Ney、4 次以上は Witten-Bell の逃げ確率で積み上げる。
func (m *Model) Prob(ctx []int32, w int32) float64 {
	if len(ctx) > m.order-1 {
		ctx = ctx[len(ctx)-(m.order-1):]
	}
	p := m.knProb(ctx, w)
	for n := 4; n <= m.order && n-1 <= len(ctx); n++ {
		sub := ctx[len(ctx)-(n-1):]
		h := hashIDs(sub)
		c := m.hiTot[n][h]
		if c == 0 {
			continue
		}
		t := m.hiDist[n][h]
		cw := m.hiCount[n][hashIDs(append(append([]int32{}, sub...), w))]
		k := float64(c) + float64(t)
		hp := float64(cw)/k + float64(t)/k*p
		p += m.hiWeight * (hp - p)
	}
	return p
}

// knProb は 3 次までの Kneser-Ney 補間。
func (m *Model) knProb(ctx []int32, w int32) float64 {
	if len(ctx) > 2 {
		ctx = ctx[len(ctx)-2:]
	}
	p := m.baseProb(w)
	for k := 1; k <= len(ctx); k++ {
		sub := ctx[len(ctx)-k:]
		ck := pack(sub...)
		c := m.ctxs[ck]
		if c == 0 {
			continue
		}
		var cw, n1 int32
		if len(sub) == 1 && m.cont2 != nil {
			// 直下の低次が unigram 継続数のときは、bigram も継続数で数える。
			cw = m.cont2[pack(sub[0], w)]
			n1 = m.cont2n1[ck]
			c = m.cont2ctx[ck]
			if c == 0 {
				continue
			}
		} else {
			cw = m.grams[keyWith(sub, w)]
			n1 = m.n1[ck]
		}
		disc := (float64(cw) - m.discount) / float64(c)
		if disc < 0 {
			disc = 0
		}
		p = disc + m.discount*float64(n1)/float64(c)*p
	}
	return p
}

// baseProb は継続数による unigram 確率。未出現語にも微小な確率を残す。
func (m *Model) baseProb(w int32) float64 {
	return (float64(m.cont1[w]) + 0.5) / (float64(m.n1uni) + 0.5*float64(m.v))
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
