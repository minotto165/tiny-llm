// Package tokenizer は日本語テキストを BPE (byte pair encoding) で分割する。
//
// 空白・改行・制御文字は併合禁止の境界として扱い、トークンとしてそのまま残す。
// したがって分割結果を連結すると必ず入力に戻る。
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Config は学習の設定。
type Config struct {
	VocabSize int  // 語彙数の上限。ここに達したら併合を止める
	MinCount  int  // 併合するペアの最低頻度
	Batch     int  // 1 回の計数で併合するペア数（速度と厳密さのつまみ）
	Quiet     bool // 進捗ログを抑止する
}

// Result は学習の結果。IDs は入力の分割で、Vocab[IDs[i]] が i 番目のトークン。
// Merges は併合規則を優先度順に並べたもので、新しいテキストの分割に使う。
type Result struct {
	IDs    []int32
	Vocab  []string
	Merges []Merge
}

// Tokens は分割を文字列の列に戻す。
func (r Result) Tokens() []string {
	tokens := make([]string, len(r.IDs))
	for i, id := range r.IDs {
		tokens[i] = r.Vocab[id]
	}
	return tokens
}

// Text は分割を元のテキストに戻す。
func (r Result) Text() string {
	return strings.Join(r.Tokens(), "")
}

// Train は src を BPE で分割し、トークン id 列と語彙を返す。
func Train(src []byte, cfg Config) (Result, error) {
	if !utf8.Valid(src) {
		return Result{}, fmt.Errorf("input is not valid UTF-8")
	}
	if cfg.MinCount < 1 {
		return Result{}, fmt.Errorf("mincount must be >= 1")
	}
	if cfg.Batch < 1 {
		return Result{}, fmt.Errorf("batch must be >= 1")
	}
	offs := RuneOffsets(src)
	if len(offs) <= 1 {
		return Result{}, fmt.Errorf("empty input")
	}
	ids, vocab, merges := trainBPE(src, offs, int32(cfg.MinCount), cfg.VocabSize, cfg.Batch, cfg.Quiet)
	return Result{IDs: ids, Vocab: vocab, Merges: merges}, nil
}

// RuneOffsets は各ルーン先頭のバイト位置と、末尾の位置を返す。
func RuneOffsets(b []byte) []int {
	offs := make([]int, 0, len(b)+1)
	for i := 0; i < len(b); {
		offs = append(offs, i)
		_, w := utf8.DecodeRune(b[i:])
		i += w
	}
	offs = append(offs, len(b))
	return offs
}

// IsBarrier はそのルーンが併合の境界になるかを返す。
func IsBarrier(r rune) bool {
	return unicode.IsSpace(r) || unicode.IsControl(r)
}

// WriteVocab は語彙を JSON 配列として書き出す（添字がトークン id）。
func WriteVocab(path string, vocab []string) error {
	var sb strings.Builder
	enc := json.NewEncoder(&sb)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(vocab); err != nil {
		return fmt.Errorf("marshal vocabulary: %w", err)
	}
	return os.WriteFile(path, []byte(sb.String()), 0o644)
}
