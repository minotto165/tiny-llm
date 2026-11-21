package lm

import (
	"math"
	"testing"
)

func trainToy(order int) *Model {
	// 0=<pad> 1=<unk> 2=<bos> 3=<eos> を想定した小さな id 列。
	ids := []int32{10, 11, 12, 10, 11, 13, 10, 11, 12, 10, 12, 11, 10, 11, 12}
	return Train(ids, 16, order)
}

func TestBigramSumsToOne(t *testing.T) {
	m := trainToy(2)
	for _, ctx := range [][]int32{{10}, {11}, {12}} {
		sum := 0.0
		for w := int32(0); w < 16; w++ {
			sum += m.Prob(ctx, w)
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Fatalf("ctx %v: sum=%v", ctx, sum)
		}
	}
}

func TestSeenBigramBeatsUnseen(t *testing.T) {
	m := trainToy(2)
	seen := m.Prob([]int32{10}, 12)
	unseen := m.Prob([]int32{10}, 14)
	if !(seen > unseen) {
		t.Fatalf("seen=%v unseen=%v", seen, unseen)
	}
}

func TestTrigramFallsBack(t *testing.T) {
	m := trainToy(3)
	if p := m.Prob([]int32{99, 98}, 12); p <= 0 {
		t.Fatalf("backoff prob = %v", p)
	}
	if p := m.Prob(nil, 10); p <= 0 {
		t.Fatalf("unigram prob = %v", p)
	}
}

func TestTrigramSumsToOne(t *testing.T) {
	m := trainToy(3)
	sum := 0.0
	for w := int32(0); w < 16; w++ {
		sum += m.Prob([]int32{10, 11}, w)
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("sum=%v", sum)
	}
}
