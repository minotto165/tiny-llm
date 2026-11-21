package main

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func train(t *testing.T, src string, vocabSize, batch int) ([]string, []string) {
	t.Helper()
	b := []byte(src)
	offs := runeOffsets(b)
	ids, vocab, _ := trainBPE(b, offs, 2, vocabSize, batch, true)
	tokens := make([]string, len(ids))
	for i, id := range ids {
		tokens[i] = vocab[id]
	}
	return tokens, vocab
}

func TestReconstruction(t *testing.T) {
	src := "吾輩は猫である。名前はまだ無い。\nどこで生れたかとんと見当がつかぬ。\n"
	tokens, _ := train(t, src, 64, 16)
	if got := strings.Join(tokens, ""); got != src {
		t.Fatalf("detokenization mismatch:\n got %q\nwant %q", got, src)
	}
}

func TestNoMergeAcrossWhitespace(t *testing.T) {
	src := strings.Repeat("あ い\nう え\n", 40)
	tokens, _ := train(t, src, 64, 8)
	for _, tok := range tokens {
		if utf8.RuneCountInString(tok) > 1 {
			for _, r := range tok {
				if r == ' ' || r == '\n' {
					t.Fatalf("token %q merged across a whitespace boundary", tok)
				}
			}
		}
	}
	if got := strings.Join(tokens, ""); got != src {
		t.Fatalf("detokenization mismatch")
	}
}

func TestDeterministic(t *testing.T) {
	src := strings.Repeat("日本語のトークナイザを作る。日本語の分割は難しい。", 50)
	a, _ := train(t, src, 128, 16)
	b, _ := train(t, src, 128, 16)
	if strings.Join(a, "/") != strings.Join(b, "/") {
		t.Fatal("training twice produced different segmentations")
	}
}

func TestVocabularySizeCap(t *testing.T) {
	src := strings.Repeat("あいうえおかきくけこ", 200)
	_, vocab := train(t, src, 20, 4)
	if len(vocab) > 20 {
		t.Fatalf("vocab grew to %d, want <= 20", len(vocab))
	}
}

func TestRuneOffsets(t *testing.T) {
	src := "aあ🙂b"
	offs := runeOffsets([]byte(src))
	if len(offs) != utf8.RuneCountInString(src)+1 {
		t.Fatalf("got %d offsets", len(offs))
	}
	for i := 0; i+1 < len(offs); i++ {
		if offs[i] >= offs[i+1] {
			t.Fatalf("offsets not increasing at %d", i)
		}
	}
	if got := runeOffsets([]byte(""))[0]; got != 0 {
		t.Fatal("empty input should yield a single offset")
	}
}
