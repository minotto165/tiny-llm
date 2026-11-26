package nn

import (
	"math"
	"testing"
)

// Step のロジットが Forward の同じ位置のロジットと一致するか。
func TestKVMatchesForward(t *testing.T) {
	cfg := Config{Vocab: 50, Ctx: 8, Dim: 8, Heads: 2, Layers: 2, FFN: 16, Seed: 3}
	m := NewModel(cfg)
	ids := make([]int32, cfg.Ctx)
	for i := range ids {
		ids[i] = int32((i*7 + 3) % cfg.Vocab)
	}

	c := NewKVCache(m)
	for pos := 0; pos < len(ids); pos++ {
		got := m.Step(c, ids[pos])

		full := m.Forward(ids[:pos+1])
		want := full.Logits.D[pos*cfg.Vocab : (pos+1)*cfg.Vocab]
		for v := range want {
			if d := math.Abs(got[v] - want[v]); d > 1e-9 {
				t.Fatalf("pos %d, vocab %d: step=%v forward=%v (diff %v)", pos, v, got[v], want[v], d)
			}
		}
	}
	if c.Len() != len(ids) {
		t.Fatalf("cache len = %d, want %d", c.Len(), len(ids))
	}
}

// 文脈長を超えても、そのときキャッシュが見ているトークン列の Forward と一致するか。
func TestKVSlide(t *testing.T) {
	cfg := Config{Vocab: 50, Ctx: 8, Dim: 8, Heads: 2, Layers: 2, FFN: 16, Seed: 4}
	m := NewModel(cfg)
	c := NewKVCache(m)

	for i := 0; i < 40; i++ {
		id := int32((i * 13) % cfg.Vocab)
		got := m.Step(c, id)

		hist := c.History()
		full := m.Forward(hist)
		want := full.Logits.D[(len(hist)-1)*cfg.Vocab:]
		for v := range want {
			if d := math.Abs(got[v] - want[v]); d > 1e-9 {
				t.Fatalf("step %d, vocab %d: step=%v forward=%v (diff %v)", i, v, got[v], want[v], d)
			}
		}
		if c.Len() > cfg.Ctx {
			t.Fatalf("step %d: cache len = %d, want <= %d", i, c.Len(), cfg.Ctx)
		}
	}
}
