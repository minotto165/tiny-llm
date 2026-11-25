package nn

import (
	"math"
	"math/rand"
	"testing"
)

func smallModel() (*Model, []int32, []int32) {
	cfg := Config{Vocab: 17, Ctx: 6, Dim: 8, Heads: 2, Layers: 1, FFN: 16, Seed: 1}
	m := NewModel(cfg)
	rnd := rand.New(rand.NewSource(2))
	ids := make([]int32, cfg.Ctx)
	tgt := make([]int32, cfg.Ctx)
	for i := range ids {
		ids[i] = int32(rnd.Intn(cfg.Vocab))
		tgt[i] = int32(rnd.Intn(cfg.Vocab))
	}
	return m, ids, tgt
}

// TestNumericGradient は誤差逆伝播を数値微分と照合する。
// 手書き backprop のバグはここでしか確実に捕まえられない。
func TestNumericGradient(t *testing.T) {
	m, ids, tgt := smallModel()
	m.ZeroGrads()
	c := m.Forward(ids)
	loss := m.Backward(c, tgt)

	const eps = 1e-5
	for pi, p := range m.All {
		// パラメータごとに先頭・中間・末尾の 3 点を検査する。
		for _, idx := range []int{0, len(p.W.D) / 2, len(p.W.D) - 1} {
			orig := p.W.D[idx]
			p.W.D[idx] = orig + eps
			lp := m.Loss(m.Forward(ids), tgt)
			p.W.D[idx] = orig - eps
			lm := m.Loss(m.Forward(ids), tgt)
			p.W.D[idx] = orig

			num := (lp - lm) / (2 * eps)
			ana := p.G.D[idx]
			if math.Abs(num-ana) > 1e-6+1e-5*math.Abs(num) {
				t.Fatalf("param %d idx %d: numeric %.8g, analytic %.8g (loss %.6f)", pi, idx, num, ana, loss)
			}
		}
	}
}

// TestCausal は未来のトークンが過去のロジットに影響しないことを確認する。
func TestCausal(t *testing.T) {
	m, ids, _ := smallModel()
	a := m.Forward(ids)
	ids2 := append([]int32(nil), ids...)
	ids2[len(ids2)-1] = (ids2[len(ids2)-1] + 3) % int32(m.Cfg.Vocab)
	b := m.Forward(ids2)
	for i := 0; i < len(ids)-1; i++ {
		for j := 0; j < m.Cfg.Vocab; j++ {
			if math.Abs(a.Logits.At(i, j)-b.Logits.At(i, j)) > 1e-12 {
				t.Fatalf("position %d leaked future information", i)
			}
		}
	}
}

// TestLearn は小さな系列を丸暗記できるかを見る (最適化の配線確認)。
func TestLearn(t *testing.T) {
	cfg := Config{Vocab: 8, Ctx: 4, Dim: 16, Heads: 2, Layers: 1, FFN: 32, Seed: 3}
	m := NewModel(cfg)
	opt := NewAdam(m.All, 3e-3)

	ids := []int32{1, 2, 3, 4}
	tgt := []int32{2, 3, 4, 5}
	first := 0.0
	for step := 0; step < 200; step++ {
		m.ZeroGrads()
		c := m.Forward(ids)
		loss := m.Backward(c, tgt)
		opt.Step(m.All, 1.0)
		if step == 0 {
			first = loss
		}
		if step == 199 {
			if loss > first*0.2 {
				t.Fatalf("loss did not drop: first %.4f last %.4f", first, loss)
			}
			t.Logf("loss %.4f -> %.4f", first, loss)
		}
	}
}
