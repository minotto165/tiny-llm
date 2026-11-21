package tokenizer

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func train(t *testing.T, src string, vocabSize, batch int) Result {
	t.Helper()
	res, err := Train([]byte(src), Config{VocabSize: vocabSize, MinCount: 2, Batch: batch, Quiet: true})
	if err != nil {
		t.Fatalf("Train: %v", err)
	}
	return res
}

func TestReconstruction(t *testing.T) {
	src := "吾輩は猫である。名前はまだ無い。\nどこで生れたかとんと見当がつかぬ。\n"
	if got := train(t, src, 64, 16).Text(); got != src {
		t.Fatalf("detokenization mismatch:\n got %q\nwant %q", got, src)
	}
}

func TestNoMergeAcrossWhitespace(t *testing.T) {
	src := strings.Repeat("あ い\nう え\n", 40)
	res := train(t, src, 64, 8)
	for _, tok := range res.Tokens() {
		if utf8.RuneCountInString(tok) > 1 {
			for _, r := range tok {
				if r == ' ' || r == '\n' {
					t.Fatalf("token %q merged across a whitespace boundary", tok)
				}
			}
		}
	}
	if got := res.Text(); got != src {
		t.Fatalf("detokenization mismatch")
	}
}

func TestDeterministic(t *testing.T) {
	src := strings.Repeat("日本語のトークナイザを作る。日本語の分割は難しい。", 50)
	a := train(t, src, 128, 16).Text()
	b := train(t, src, 128, 16).Text()
	if a != b {
		t.Fatal("training twice produced different segmentations")
	}
}

func TestVocabularySizeCap(t *testing.T) {
	src := strings.Repeat("あいうえおかきくけこ", 200)
	res := train(t, src, 20, 4)
	if len(res.Vocab) > 20 {
		t.Fatalf("vocab grew to %d, want <= 20", len(res.Vocab))
	}
}

func TestEmptyInput(t *testing.T) {
	if _, err := Train(nil, Config{VocabSize: 8, MinCount: 1, Batch: 1, Quiet: true}); err == nil {
		t.Fatal("empty input should be an error")
	}
}

func TestRuneOffsets(t *testing.T) {
	src := "aあ🙂b"
	offs := RuneOffsets([]byte(src))
	if len(offs) != utf8.RuneCountInString(src)+1 {
		t.Fatalf("got %d offsets", len(offs))
	}
	for i := 0; i+1 < len(offs); i++ {
		if offs[i] >= offs[i+1] {
			t.Fatalf("offsets not increasing at %d", i)
		}
	}
	if got := RuneOffsets(nil)[0]; got != 0 {
		t.Fatal("empty input should yield a single offset")
	}
}
