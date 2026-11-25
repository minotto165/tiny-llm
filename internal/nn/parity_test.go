package nn

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// TestParityWithReference は scripts/tfm_ref.py が書き出した logits と比較する。
// 参照ファイルが無い環境ではスキップする (CI では走らない)。
//
//	python3 scripts/tfm_ref.py --model testdata/tfm.bin --ids 1,2,3 --emit-parity testdata/parity.bin
//	go test ./internal/nn -run Parity
func TestParityWithReference(t *testing.T) {
	modelPath := "../../testdata/tfm.bin"
	parityPath := "../../testdata/parity.bin"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("testdata/tfm.bin がありません")
	}
	if _, err := os.Stat(parityPath); err != nil {
		t.Skip("testdata/parity.bin がありません")
	}

	f, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	m, _, err := Load(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}

	pf, err := os.Open(parityPath)
	if err != nil {
		t.Fatal(err)
	}
	defer pf.Close()
	var n int32
	if err := binary.Read(pf, binary.LittleEndian, &n); err != nil {
		t.Fatal(err)
	}
	ids := make([]int32, n)
	if err := binary.Read(pf, binary.LittleEndian, &ids); err != nil {
		t.Fatal(err)
	}
	ref := make([]float64, int(n)*m.Cfg.Vocab)
	if err := binary.Read(pf, binary.LittleEndian, &ref); err != nil {
		t.Fatal(err)
	}

	c := m.Forward(ids)
	var maxDiff float64
	for i := range ref {
		d := math.Abs(ref[i] - c.Logits.D[i])
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-6 {
		t.Fatalf("参照実装と一致しません: 最大差 %.3g", maxDiff)
	}
	t.Logf("max diff %.3g (%d ids x %d vocab)", maxDiff, n, m.Cfg.Vocab)
}
