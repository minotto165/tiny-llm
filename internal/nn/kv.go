package nn

import "math"

// KVCache は推論時に各層の K, V を保持する。
// Forward は毎回文脈全体を計算し直すが、Step は新しい 1 トークンだけを処理する。
type KVCache struct {
	cfg Config
	n   int
	k   []Mat // 層ごとの Ctx×Dim
	v   []Mat
	ids []int32 // 文脈長を超えたときに窓を張り直すための履歴
}

// NewKVCache は空のキャッシュを作る。
func NewKVCache(m *Model) *KVCache {
	c := &KVCache{cfg: m.Cfg}
	d := m.Cfg.Dim
	c.k = make([]Mat, m.Cfg.Layers)
	c.v = make([]Mat, m.Cfg.Layers)
	for l := range c.k {
		c.k[l] = NewMat(m.Cfg.Ctx, d)
		c.v[l] = NewMat(m.Cfg.Ctx, d)
	}
	return c
}

// Reset はキャッシュを空にする。
func (c *KVCache) Reset() {
	c.n = 0
	c.ids = c.ids[:0]
}

// Len はキャッシュに入っているトークン数を返す。
func (c *KVCache) Len() int { return c.n }

// History はキャッシュが現在見ているトークン列を返す。
func (c *KVCache) History() []int32 { return c.ids }

// Step は 1 トークンを処理して、その位置のロジット（語彙長）を返す。
// 文脈長に達したら末尾 Ctx/2 トークンだけを残して窓を張り直す
// （位置埋め込みが絶対位置なので、全長を保ったスライドはできない）。
func (m *Model) Step(c *KVCache, id int32) []float64 {
	cfg := m.Cfg
	if c.n >= cfg.Ctx {
		keep := cfg.Ctx / 2
		if keep < 1 {
			keep = 1
		}
		hist := append([]int32{}, c.ids[len(c.ids)-keep:]...)
		c.Reset()
		for _, prev := range hist {
			m.Step(c, prev)
		}
	}
	logits := m.step(c, id)
	c.ids = append(c.ids, id)
	c.n++
	return logits
}

func (m *Model) step(c *KVCache, id int32) []float64 {
	cfg := m.Cfg
	d, h := cfg.Dim, cfg.Heads
	dh := cfg.headDim()
	ctx := cfg.Ctx
	pos := c.n

	x := NewMat(1, d)
	for j := 0; j < d; j++ {
		x.D[j] = m.Wte.W.D[int(id)*d+j] + m.Wpe.W.D[pos*d+j]
	}

	scale := 1 / math.Sqrt(float64(dh))
	row := make([]float64, ctx)
	o := NewMat(1, d)
	for l := 0; l < cfg.Layers; l++ {
		b := m.Blocks[l]
		a1, _ := layernorm(x, b.Ln1g.W, b.Ln1b.W)
		q := linear(a1, b.Wq.W, b.Bq.W)
		k := linear(a1, b.Wk.W, b.Bk.W)
		v := linear(a1, b.Wv.W, b.Bv.W)
		for j := 0; j < d; j++ {
			c.k[l].D[pos*d+j] = k.D[j]
			c.v[l].D[pos*d+j] = v.D[j]
		}

		for hh := 0; hh < h; hh++ {
			base := hh * dh
			for j := 0; j <= pos; j++ {
				var dot float64
				for kk := 0; kk < dh; kk++ {
					dot += q.D[base+kk] * c.k[l].D[j*d+base+kk]
				}
				row[j] = dot * scale
			}
			softmaxRow(row[:pos+1])
			for kk := 0; kk < dh; kk++ {
				var acc float64
				for j := 0; j <= pos; j++ {
					acc += row[j] * c.v[l].D[j*d+base+kk]
				}
				o.D[base+kk] = acc
			}
		}
		x = addMat(x, linear(o, b.Wo.W, b.Bo.W))

		a2, _ := layernorm(x, b.Ln2g.W, b.Ln2b.W)
		pre := linear(a2, b.W1.W, b.B1.W)
		for i := range pre.D {
			pre.D[i] = gelu(pre.D[i])
		}
		x = addMat(x, linear(pre, b.W2.W, b.B2.W))
	}

	xout, _ := layernorm(x, m.LnFg.W, m.LnFb.W)
	return linear(xout, m.Wout.W, m.Bout.W).D
}
