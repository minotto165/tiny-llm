package nn

import (
	"math"
	"math/rand"
)

// Config はモデルの形。Dim は Heads で割り切れる必要がある。
type Config struct {
	Vocab  int
	Ctx    int
	Dim    int
	Heads  int
	Layers int
	FFN    int
	Seed   int64
}

func (c Config) headDim() int { return c.Dim / c.Heads }

// Param は重みとその勾配をまとめたもの。
type Param struct {
	W Mat
	G Mat
}

func newParam(r, c int) *Param { return &Param{W: NewMat(r, c), G: NewMat(r, c)} }

// Block は Pre-LN の Transformer ブロック 1 段。
type Block struct {
	Ln1g, Ln1b *Param
	Wq, Wk     *Param
	Wv, Wo     *Param
	Bq, Bk     *Param
	Bv, Bo     *Param
	Ln2g, Ln2b *Param
	W1, B1     *Param
	W2, B2     *Param
}

// Model は埋め込み + N 段のブロック + 出力層。
type Model struct {
	Cfg    Config
	Wte    *Param
	Wpe    *Param
	Blocks []*Block
	LnFg   *Param
	LnFb   *Param
	Wout   *Param
	Bout   *Param
	All    []*Param
}

// NewModel は重みを乱数で初期化する。
func NewModel(cfg Config) *Model {
	if cfg.Dim%cfg.Heads != 0 {
		panic("nn: Dim must be divisible by Heads")
	}
	rnd := rand.New(rand.NewSource(cfg.Seed))
	m := &Model{Cfg: cfg}
	add := func(r, c int, std float64) *Param {
		p := newParam(r, c)
		for i := range p.W.D {
			p.W.D[i] = rnd.NormFloat64() * std
		}
		m.All = append(m.All, p)
		return p
	}
	ones := func(r, c int) *Param {
		p := newParam(r, c)
		for i := range p.W.D {
			p.W.D[i] = 1
		}
		m.All = append(m.All, p)
		return p
	}
	d, f := cfg.Dim, cfg.FFN
	m.Wte = add(cfg.Vocab, d, 0.02)
	m.Wpe = add(cfg.Ctx, d, 0.01)
	for l := 0; l < cfg.Layers; l++ {
		b := &Block{}
		b.Ln1g, b.Ln1b = ones(1, d), add(1, d, 0)
		b.Wq, b.Wk = add(d, d, 0.02), add(d, d, 0.02)
		b.Wv, b.Wo = add(d, d, 0.02), add(d, d, 0.02/math.Sqrt(float64(2*cfg.Layers)))
		b.Bq, b.Bk = add(1, d, 0), add(1, d, 0)
		b.Bv, b.Bo = add(1, d, 0), add(1, d, 0)
		b.Ln2g, b.Ln2b = ones(1, d), add(1, d, 0)
		b.W1, b.B1 = add(d, f, 0.02), add(1, f, 0)
		b.W2, b.B2 = add(f, d, 0.02/math.Sqrt(float64(2*cfg.Layers))), add(1, d, 0)
		m.Blocks = append(m.Blocks, b)
	}
	m.LnFg, m.LnFb = ones(1, d), add(1, d, 0)
	m.Wout = add(d, cfg.Vocab, 0.02)
	m.Bout = add(1, cfg.Vocab, 0)
	return m
}

// NumParams はパラメータ数を返す。
func (m *Model) NumParams() int {
	n := 0
	for _, p := range m.All {
		n += len(p.W.D)
	}
	return n
}

// ZeroGrads は全パラメータの勾配を 0 にする。
func (m *Model) ZeroGrads() {
	for _, p := range m.All {
		p.G.Zero()
	}
}

type blockCache struct {
	ln1 lnCache
	a1  Mat // ln1 の出力
	q   Mat
	k   Mat
	v   Mat
	att []Mat // ヘッドごとの T×T 注意確率
	o   Mat   // ヘッド結合後
	ln2 lnCache
	a2  Mat
	pre Mat // W1 の出力 (gelu 前)
	h   Mat // gelu 後
}

// Cache は逆伝播に必要な中間値。
type Cache struct {
	IDs    []int32
	T      int
	emb    Mat
	blocks []blockCache
	lnf    lnCache
	xin    Mat
	xout   Mat
	Logits Mat
}

// Forward は ids 全体 (長さ <= Ctx) を処理してロジットを返す。
func (m *Model) Forward(ids []int32) *Cache {
	cfg := m.Cfg
	t, d, h := len(ids), cfg.Dim, cfg.Heads
	dh := cfg.headDim()
	c := &Cache{IDs: ids, T: t}
	x := NewMat(t, d)
	for i := 0; i < t; i++ {
		src := m.Wte.W.D[int(ids[i])*d:]
		for j := 0; j < d; j++ {
			x.D[i*d+j] = src[j] + m.Wpe.W.D[i*d+j]
		}
	}
	c.emb = x

	for l := 0; l < cfg.Layers; l++ {
		b := m.Blocks[l]
		bc := blockCache{}
		bc.a1, bc.ln1 = layernorm(x, b.Ln1g.W, b.Ln1b.W)
		bc.q = linear(bc.a1, b.Wq.W, b.Bq.W)
		bc.k = linear(bc.a1, b.Wk.W, b.Bk.W)
		bc.v = linear(bc.a1, b.Wv.W, b.Bv.W)
		bc.att = make([]Mat, h)
		bc.o = NewMat(t, d)
		scale := 1 / math.Sqrt(float64(dh))
		for hh := 0; hh < h; hh++ {
			base := hh * dh
			p := NewMat(t, t)
			for i := 0; i < t; i++ {
				for j := 0; j <= i; j++ {
					var dot float64
					for k := 0; k < dh; k++ {
						dot += bc.q.At(i, base+k) * bc.k.At(j, base+k)
					}
					p.Set(i, j, dot*scale)
				}
				row := p.D[i*p.C : i*p.C+i+1]
				softmaxRow(row)
				for k := 0; k < dh; k++ {
					var acc float64
					for j := 0; j <= i; j++ {
						acc += p.At(i, j) * bc.v.At(j, base+k)
					}
					bc.o.Set(i, base+k, acc)
				}
			}
			bc.att[hh] = p
		}
		attOut := linear(bc.o, b.Wo.W, b.Bo.W)
		x = addMat(x, attOut)

		bc.a2, bc.ln2 = layernorm(x, b.Ln2g.W, b.Ln2b.W)
		bc.pre = linear(bc.a2, b.W1.W, b.B1.W)
		bc.h = NewMat(t, cfg.FFN)
		for i := range bc.h.D {
			bc.h.D[i] = gelu(bc.pre.D[i])
		}
		x = addMat(x, linear(bc.h, b.W2.W, b.B2.W))
		c.blocks = append(c.blocks, bc)
	}

	c.xin = x
	c.xout, c.lnf = layernorm(x, m.LnFg.W, m.LnFb.W)
	c.Logits = linear(c.xout, m.Wout.W, m.Bout.W)
	return c
}

// Loss は平均交差エントロピーを返す (勾配は触らない)。
func (m *Model) Loss(c *Cache, targets []int32) float64 {
	var loss float64
	for i, id := range targets {
		row := append([]float64(nil), c.Logits.D[i*c.Logits.C:(i+1)*c.Logits.C]...)
		softmaxRow(row)
		p := row[id]
		if p < 1e-12 {
			p = 1e-12
		}
		loss -= math.Log(p)
	}
	return loss / float64(len(targets))
}

// Backward は loss の勾配を各 Param.G に加算し、loss を返す。
func (m *Model) Backward(c *Cache, targets []int32) float64 {
	cfg := m.Cfg
	t, d, h := c.T, cfg.Dim, cfg.Heads
	dh := cfg.headDim()

	// softmax + 交差エントロピー
	dlogits := NewMat(t, cfg.Vocab)
	var loss float64
	for i, id := range targets {
		row := append([]float64(nil), c.Logits.D[i*c.Logits.C:(i+1)*c.Logits.C]...)
		softmaxRow(row)
		p := row[id]
		if p < 1e-12 {
			p = 1e-12
		}
		loss -= math.Log(p)
		copy(dlogits.D[i*dlogits.C:(i+1)*dlogits.C], row)
		dlogits.D[i*dlogits.C+int(id)] -= 1
	}
	for i := range dlogits.D {
		dlogits.D[i] /= float64(t)
	}
	loss /= float64(t)

	dx := linearBackward(c.xout, m.Wout.W, dlogits, &m.Wout.G, &m.Bout.G)
	dx = layernormBackward(c.lnf, m.LnFg.W, dx, &m.LnFg.G, &m.LnFb.G)

	for l := len(c.blocks) - 1; l >= 0; l-- {
		b := m.Blocks[l]
		bc := &c.blocks[l]

		// FFN (x3 = x2 + f)
		dF := dx
		dH := linearBackward(bc.h, b.W2.W, dF, &b.W2.G, &b.B2.G)
		for i := range dH.D {
			dH.D[i] *= geluGrad(bc.pre.D[i])
		}
		dA2 := linearBackward(bc.a2, b.W1.W, dH, &b.W1.G, &b.B1.G)
		dX2 := layernormBackward(bc.ln2, b.Ln2g.W, dA2, &b.Ln2g.G, &b.Ln2b.G)
		addInto(&dX2, dx)

		// Attention
		dO := linearBackward(bc.o, b.Wo.W, dX2, &b.Wo.G, &b.Bo.G)
		dQ, dK, dV := NewMat(t, d), NewMat(t, d), NewMat(t, d)
		scale := 1 / math.Sqrt(float64(dh))
		for hh := 0; hh < h; hh++ {
			base := hh * dh
			p := bc.att[hh]
			dP := NewMat(t, t)
			for i := 0; i < t; i++ {
				for j := 0; j <= i; j++ {
					var dot float64
					for k := 0; k < dh; k++ {
						dot += dO.At(i, base+k) * bc.v.At(j, base+k)
					}
					dP.Set(i, j, dot)
				}
			}
			for i := 0; i < t; i++ {
				var sum float64
				for j := 0; j <= i; j++ {
					sum += p.At(i, j) * dP.At(i, j)
				}
				for j := 0; j <= i; j++ {
					dp := p.At(i, j) * (dP.At(i, j) - sum)
					for k := 0; k < dh; k++ {
						dQ.Set(i, base+k, dQ.At(i, base+k)+dp*bc.k.At(j, base+k)*scale)
						dK.Set(j, base+k, dK.At(j, base+k)+dp*bc.q.At(i, base+k)*scale)
						dV.Set(j, base+k, dV.At(j, base+k)+p.At(i, j)*dO.At(i, base+k))
					}
				}
			}
		}
		dA1 := linearBackward(bc.a1, b.Wq.W, dQ, &b.Wq.G, &b.Bq.G)
		addInto(&dA1, linearBackward(bc.a1, b.Wk.W, dK, &b.Wk.G, &b.Bk.G))
		addInto(&dA1, linearBackward(bc.a1, b.Wv.W, dV, &b.Wv.G, &b.Bv.G))
		dX1 := layernormBackward(bc.ln1, b.Ln1g.W, dA1, &b.Ln1g.G, &b.Ln1b.G)
		addInto(&dX1, dX2)
		dx = dX1
	}

	for i := 0; i < t; i++ {
		for j := 0; j < d; j++ {
			v := dx.At(i, j)
			m.Wte.G.D[int(c.IDs[i])*d+j] += v
			m.Wpe.G.D[i*d+j] += v
		}
	}
	return loss
}

// Adam は Adam オプティマイザ。
type Adam struct {
	LR, B1, B2, Eps float64
	step            int
	m, v            [][]float64
}

// NewAdam はパラメータ数に合わせて状態を確保する。
func NewAdam(params []*Param, lr float64) *Adam {
	a := &Adam{LR: lr, B1: 0.9, B2: 0.95, Eps: 1e-8, m: make([][]float64, len(params)), v: make([][]float64, len(params))}
	for i, p := range params {
		a.m[i] = make([]float64, len(p.W.D))
		a.v[i] = make([]float64, len(p.W.D))
	}
	return a
}

// Step は勾配を全体ノルムでクリップしてから 1 ステップ更新する。
func (a *Adam) Step(params []*Param, maxNorm float64) {
	var norm float64
	for _, p := range params {
		for _, g := range p.G.D {
			norm += g * g
		}
	}
	norm = math.Sqrt(norm)
	clip := 1.0
	if maxNorm > 0 && norm > maxNorm {
		clip = maxNorm / (norm + 1e-12)
	}
	a.step++
	bc1 := 1 - math.Pow(a.B1, float64(a.step))
	bc2 := 1 - math.Pow(a.B2, float64(a.step))
	for i, p := range params {
		mi, vi := a.m[i], a.v[i]
		for j := range p.W.D {
			g := p.G.D[j] * clip
			mi[j] = a.B1*mi[j] + (1-a.B1)*g
			vi[j] = a.B2*vi[j] + (1-a.B2)*g*g
			mh := mi[j] / bc1
			vh := vi[j] / bc2
			p.W.D[j] -= a.LR * mh / (math.Sqrt(vh) + a.Eps)
		}
	}
}
