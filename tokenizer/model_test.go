package tokenizer

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestSpecialTokensReserved(t *testing.T) {
	res := train(t, strings.Repeat("あいうえお", 10), 20, 8)
	for i, want := range SpecialTokens {
		if got := res.Vocab[i]; got != want {
			t.Fatalf("vocab[%d] = %q, want %q", i, got, want)
		}
	}
	for _, id := range res.IDs {
		if id < int32(len(SpecialTokens)) {
			t.Fatalf("training id %d collides with a special token", id)
		}
	}
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	src := strings.Repeat("吾輩は猫である。名前はまだ無い。\n", 20)
	res := train(t, src, 128, 16)
	m := res.Model()

	text := "名前はまだ無い。"
	ids := m.Encode(text)
	if len(ids) >= len([]rune(text)) {
		t.Fatalf("encode did not use merges: %d tokens for %d runes", len(ids), len([]rune(text)))
	}
	if got := m.Decode(ids); got != text {
		t.Fatalf("round trip: got %q, want %q", got, text)
	}
}

func TestEncodeStrictMatchesTraining(t *testing.T) {
	src := strings.Repeat("日本語のトークナイザを作る。日本語の分割は難しい。", 30)
	// batch 1 は厳密な BPE なので、encode は学習時の id 列と一致するはず。
	res := train(t, src, 200, 1)
	m := res.Model()
	got := m.Encode(src)
	if len(got) != len(res.IDs) {
		t.Fatalf("encode produced %d tokens, training %d", len(got), len(res.IDs))
	}
	for i := range got {
		if got[i] != res.IDs[i] {
			t.Fatalf("token %d: encode %d, training %d", i, got[i], res.IDs[i])
		}
	}
}

func TestUnknownRune(t *testing.T) {
	res := train(t, strings.Repeat("あいうえお", 20), 32, 4)
	m := res.Model()
	ids := m.Encode("あXい")
	if n := countID(ids, UnkID); n != 1 {
		t.Fatalf("got %d unk tokens, want 1", n)
	}
	if got := m.Decode(ids); got != "あい" {
		t.Fatalf("decode: got %q, want %q", got, "あい")
	}
}

func countID(ids []int32, want int32) int {
	n := 0
	for _, id := range ids {
		if id == want {
			n++
		}
	}
	return n
}

func TestMergesFileRoundTrip(t *testing.T) {
	src := strings.Repeat("春はあけぼの。やうやう白くなりゆく山ぎは。", 20)
	res := train(t, src, 200, 16)
	dir := t.TempDir()
	dictPath := filepath.Join(dir, "dic.txt")
	mergesPath := filepath.Join(dir, "merges.txt")
	if err := WriteVocab(dictPath, res.Vocab); err != nil {
		t.Fatal(err)
	}
	if err := WriteMerges(mergesPath, res.Merges); err != nil {
		t.Fatal(err)
	}

	vocab, err := ReadVocab(dictPath)
	if err != nil {
		t.Fatal(err)
	}
	merges, err := ReadMerges(mergesPath)
	if err != nil {
		t.Fatal(err)
	}
	loaded := NewModel(vocab, merges)

	text := "春はあけぼの。"
	want := res.Model().Encode(text)
	got := loaded.Encode(text)
	if len(got) != len(want) {
		t.Fatalf("loaded model: %d tokens, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d: loaded %d, want %d", i, got[i], want[i])
		}
	}
	if s := loaded.Decode(got); s != text {
		t.Fatalf("loaded decode: got %q, want %q", s, text)
	}
}
