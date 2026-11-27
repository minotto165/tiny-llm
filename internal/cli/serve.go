// serve は学習済み Transformer を OpenAI 互換の HTTP API で提供する。
//
//	GET  /v1/models
//	POST /v1/chat/completions   (stream 対応)
//	POST /v1/completions
//	GET  /health
//
// 推論は KV キャッシュ (nn.KVCache) で 1 トークンずつ進める。
// モデルの重みは読み取り専用なので、リクエストごとにキャッシュを作れば並行して扱える。
package cli

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/minotto165/tiny-llm/internal/nn"
	"github.com/minotto165/tiny-llm/tokenizer"
)

// minJapaneseRatio はプロンプトとして受け付ける日本語の最小割合。
const minJapaneseRatio = 0.1

type serveEngine struct {
	m          *nn.Model
	tok        *tokenizer.Model
	name       string
	ctx        int
	def        sampling
	keepSystem bool
	logPrompt  bool
	keepAsst   bool
	history    int
	jpOnly     bool
	allowInstr bool
	stripMeta  bool
	defTokens  int
	maxTokens  int
}

// Serve は OpenAI 互換サーバーを起動する。
func Serve(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	modelPath := fs.String("model", "", "Transformer モデル (gen と同じ JPTF1 ファイル)")
	dir := fs.String("dir", "testdata", "dic.txt / merges.txt のあるディレクトリ")
	addr := fs.String("addr", "127.0.0.1:8100", "待ち受けアドレス")
	name := fs.String("name", "lib-v3", "OpenAI の model id として返す名前")
	temp := fs.Float64("temp", 0.8, "温度 (0 = 貪欲)")
	topk := fs.Int("topk", 40, "上位 K 個からサンプル (0 = 全候補)")
	reppen := fs.Float64("reppen", 1.15, "重複ペナルティ (1 = 無効)")
	repwin := fs.Int("repwin", 32, "重複ペナルティを見るトークン数")
	keepSystem := fs.Bool("keep-system", false, "system / developer ロールのメッセージもプロンプトに含める")
	keepAssistant := fs.Bool("keep-assistant", false, "assistant ロール（自分の過去出力）もプロンプトに含める")
	keepEnglish := fs.Bool("keep-english", false, "日本語を含まないメッセージもプロンプトに含める")
	allowInstruction := fs.Bool("allow-instruction", false, "英語の指示文（要約・翻訳など）もそのまま生成に回す")
	quoteOnly := fs.Bool("quote-only", false, "引用ブロックの除去だけを行い、他のフィルタを全部切る")
	history := fs.Int("history", 1, "プロンプトに含める直近のユーザーメッセージ数 (1 = 最新のみ)")
	defaultTokens := fs.Int("default-tokens", 160, "クライアントが max_tokens を指定しないときの生成数")
	maxTokens := fs.Int("max-tokens", 2048, "1リクエストで生成するトークン数の上限")
	logPrompt := fs.Bool("log-prompt", false, "受け取ったプロンプトの先頭をログに出す（切り分け用）")
	fs.Parse(args)

	if *modelPath == "" {
		log.Fatal("serve: -model を指定してください")
	}

	dictPath, mergesPath := artifacts(*dir)
	vocab, err := tokenizer.ReadVocab(dictPath)
	if err != nil {
		log.Fatal(err)
	}
	merges, err := tokenizer.ReadMerges(mergesPath)
	if err != nil {
		log.Fatal(err)
	}
	hash, err := tokenizerHash(*dir)
	if err != nil {
		log.Fatal(err)
	}
	tok := tokenizer.NewModel(vocab, merges)

	f, err := os.Open(*modelPath)
	if err != nil {
		log.Fatal(err)
	}
	m, tokHash, err := nn.Load(f)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	if m.Cfg.Vocab != len(tok.Vocab) {
		log.Fatalf("model vocab %d != dict vocab %d (tfm をやり直してください)", m.Cfg.Vocab, len(tok.Vocab))
	}
	if tokHash != hash {
		log.Fatal("model の語彙が dic.txt/merges.txt と一致しません (tfm をやり直してください)")
	}

	e := &serveEngine{
		m:          m,
		tok:        tok,
		name:       *name,
		ctx:        m.Cfg.Ctx,
		def:        sampling{temp: *temp, topK: *topk, repPen: *reppen, repWin: *repwin},
		keepSystem: *keepSystem,
		logPrompt:  *logPrompt,
		keepAsst:   *keepAssistant,
		history:    *history,
		jpOnly:     !*keepEnglish,
		allowInstr: *allowInstruction,
		stripMeta:  !*quoteOnly,
		defTokens:  *defaultTokens,
		maxTokens:  *maxTokens,
	}
	if *quoteOnly {
		e.keepSystem = true
		e.keepAsst = true
		e.jpOnly = false
		e.allowInstr = true
		e.history = 0 // 全メッセージを使う
	}
	log.Printf("model: %s (dim %d, %d layers, ctx %d)", *modelPath, m.Cfg.Dim, m.Cfg.Layers, m.Cfg.Ctx)
	log.Printf("listening: http://%s/v1  model=%s", *addr, e.name)

	srv := &http.Server{
		Addr:              *addr,
		Handler:           e.routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Fatal(srv.ListenAndServe())
}

// effectiveTokens は要求された max_tokens をサーバーの既定値・上限に当てはめる。
// 0 以下（未指定）は -default-tokens、上限は -max-tokens。
func (e *serveEngine) effectiveTokens(requested int) int {
	n := requested
	if n <= 0 {
		n = e.defTokens
	}
	if e.maxTokens > 0 && n > e.maxTokens {
		n = e.maxTokens
	}
	if n <= 0 {
		n = 1
	}
	return n
}

// generate はプロンプトの続きを n トークン生成し、トークンごとに emit する。
// 生成したトークン数を返す。
func (e *serveEngine) generate(prompt string, n int, sp sampling, seed int64, emit func(string)) int {
	ids := e.tok.Encode(prompt)
	if len(ids) == 0 {
		return 0
	}
	if len(ids) > e.ctx-1 {
		ids = ids[len(ids)-(e.ctx-1):]
	}

	cache := nn.NewKVCache(e.m)
	var row []float64
	for _, id := range ids {
		row = e.m.Step(cache, id)
	}

	hist := append([]int32{}, ids...)
	rng := rand.New(rand.NewSource(seed))
	count := 0
	for i := 0; i < n; i++ {
		recent := hist
		if sp.repWin > 0 && len(recent) > sp.repWin {
			recent = recent[len(recent)-sp.repWin:]
		}
		id := sampleLogits(row, sp, len(tokenizer.SpecialTokens), recent, rng)
		row = e.m.Step(cache, id)
		hist = append(hist, id)
		if s := e.tok.Vocab[id]; s != "" {
			emit(s)
			count++
		}
	}
	return count
}

func (e *serveEngine) routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{
			"object": "list",
			"data": []any{map[string]any{
				"id": e.name, "object": "model", "created": 0, "owned_by": "tiny-llm",
			}},
		})
	})

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{"status": "ok", "ctx": e.ctx, "model": e.name})
	})

	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
			MaxTokens     int     `json:"max_tokens"`
			Temperature   float64 `json:"temperature"`
			TopP          float64 `json:"top_p"`
			Stream        bool    `json:"stream"`
			Seed          int64   `json:"seed"`
			StreamOptions struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, 400, map[string]any{"error": "invalid json"})
			return
		}
		if len(body.Messages) == 0 {
			writeJSON(w, 400, map[string]any{"error": "messages is required"})
			return
		}
		// system / developer は役割ラベル付きの対話で学習していないため混ぜない。
		// -keep-system を付けたときだけ含める。
		parts := make([]string, 0, len(body.Messages))
		for _, msg := range body.Messages {
			if !e.keepSystem && isSystemRole(msg.Role) {
				continue
			}
			if !e.keepAsst && isAssistantRole(msg.Role) {
				continue
			}
			if s := e.cleanMessage(rawText(msg.Content)); s != "" {
				// 日本語コーパスで学習したモデルなので、日本語を含まない
				// メッセージ（英語の指示・実行環境の説明など）は捨てる。
				if e.jpOnly && japaneseRatio(s) < minJapaneseRatio {
					continue
				}
				parts = append(parts, s)
			}
		}
		if len(parts) == 0 {
			if e.jpOnly {
				log.Printf("reject 日本語なし msgs=%d (先頭: %q)", len(body.Messages), firstText(body.Messages))
				writeJSON(w, 400, map[string]any{
					"error": "プロンプトが空です (引用だけのメッセージ、または日本語が含まれていません)",
				})
				return
			}
			// 全部落ちてしまった場合は、そのままの本文から生成する
			for _, msg := range body.Messages {
				if s := strings.TrimSpace(stripQuoted(rawText(msg.Content))); s != "" {
					parts = append(parts, s)
				}
			}
		}
		limit := e.history
		if limit > 0 && len(parts) > limit {
			parts = parts[len(parts)-limit:]
		}
		prompt := strings.Join(parts, "\n")

		if !e.allowInstr && looksLikeInstruction(prompt) {
			head := []rune(prompt)
			if len(head) > 40 {
				head = head[:40]
			}
			log.Printf("reject 指示文 msgs=%d (先頭: %q)", len(body.Messages), string(head))
			writeJSON(w, 400, map[string]any{
				"error": "指示文と判定しました (このモデルは日本語の続き生成のみ。" +
					"要約・翻訳などの指示には応答できません)",
			})
			return
		}
		sp := e.def
		if body.Temperature > 0 {
			sp.temp = body.Temperature
		}
		seed := body.Seed
		if seed == 0 {
			// クライアントが指定しなければ毎回違うサンプルにする（ログの seed を送れば再現できる）
			seed = time.Now().UnixNano()
		}
		n := e.effectiveTokens(body.MaxTokens)
		if e.logPrompt {
			rs := []rune(prompt)
			head := rs
			if len(head) > 400 {
				head = head[:400]
			}
			tail := rs
			if len(tail) > 80 {
				tail = tail[len(tail)-80:]
			}
			ua := r.UserAgent()
			if len(ua) > 60 {
				ua = ua[:60]
			}
			client := firstHeader(r, "X-Title", "X-Client-Name", "X-Client", "HTTP-Referer")
			log.Printf("chat 要求max_tokens=%d → 使用n=%d temp=%.2f seed=%d msgs=%d prompt=%d文字 ua=%q client=%q\n  全文=%q\n  末尾=%q",
				body.MaxTokens, n, sp.temp, seed, len(body.Messages), len(rs), ua, client, string(head), string(tail))
		}
		e.complete(w, r, prompt, n, sp, seed, body.Stream, true, body.StreamOptions.IncludeUsage)
	})

	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Prompt      string  `json:"prompt"`
			MaxTokens   int     `json:"max_tokens"`
			Temperature float64 `json:"temperature"`
			Stream      bool    `json:"stream"`
			Seed        int64   `json:"seed"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, 400, map[string]any{"error": "invalid json"})
			return
		}
		sp := e.def
		if body.Temperature > 0 {
			sp.temp = body.Temperature
		}
		seed := body.Seed
		if seed == 0 {
			seed = time.Now().UnixNano()
		}
		n := e.effectiveTokens(body.MaxTokens)
		e.complete(w, r, body.Prompt, n, sp, seed, body.Stream, false, true)
	})

	return mux
}

// complete は生成を実行し、stream なら SSE、そうでなければ 1 個の JSON で返す。
func (e *serveEngine) complete(w http.ResponseWriter, r *http.Request, prompt string, n int, sp sampling, seed int64, stream, chat, includeUsage bool) {
	start := time.Now()
	id := fmt.Sprintf("cmpl-%d", start.UnixNano())
	promptTokens := len(e.tok.Encode(prompt))

	if !stream {
		var sb strings.Builder
		count := e.generate(prompt, n, sp, seed, func(s string) { sb.WriteString(s) })
		elapsed := time.Since(start)
		usage := map[string]any{
			"prompt_tokens": promptTokens, "completion_tokens": count,
			"total_tokens": promptTokens + count,
		}
		log.Printf("gen %d tok (prompt %d) in %s (%.0f tok/s)",
			count, promptTokens, elapsed.Round(time.Millisecond), tokRate(count, elapsed))

		if chat {
			writeJSON(w, 200, map[string]any{
				"id": id, "object": "chat.completion", "created": start.Unix(), "model": e.name,
				"choices": []any{map[string]any{
					"index":         0,
					"message":       map[string]any{"role": "assistant", "content": sb.String()},
					"finish_reason": "stop",
				}},
				"usage": usage,
			})
			return
		}
		writeJSON(w, 200, map[string]any{
			"id": id, "object": "text_completion", "created": start.Unix(), "model": e.name,
			"choices": []any{map[string]any{"index": 0, "text": sb.String(), "finish_reason": "stop"}},
			"usage":   usage,
		})
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, 500, map[string]any{"error": "streaming unsupported"})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(200)

	send := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}
	delta := func(payload map[string]any) map[string]any {
		return map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": start.Unix(), "model": e.name,
			"choices": []any{map[string]any{"index": 0, "delta": payload, "finish_reason": nil}},
		}
	}
	send(delta(map[string]any{"role": "assistant"}))
	count := e.generate(prompt, n, sp, seed, func(s string) { send(delta(map[string]any{"content": s})) })
	send(map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": start.Unix(), "model": e.name,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}},
	})
	if includeUsage {
		send(map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": start.Unix(), "model": e.name,
			"choices": []any{},
			"usage": map[string]any{
				"prompt_tokens": promptTokens, "completion_tokens": count,
				"total_tokens": promptTokens + count,
			},
		})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	elapsed := time.Since(start)
	log.Printf("stream %d tok (prompt %d) in %s (%.0f tok/s)",
		count, promptTokens, elapsed.Round(time.Millisecond), tokRate(count, elapsed))
}

var (
	// <acp tokens="4" ...> のような構造タグ
	tagRe = regexp.MustCompile(`</?[a-z][a-z0-9-]*\b[^>]{0,300}>`)
	// タグを剥がしたあとに ID だけが残るメッセージ (例: m00001) はメタデータ
	bareIDRe = regexp.MustCompile(`^[A-Za-z0-9_-]{1,16}$`)
)

// firstHeader は指定ヘッダのうち最初に見つかった値を返す（クライアント識別用）。
func firstHeader(r *http.Request, names ...string) string {
	for _, n := range names {
		if v := r.Header.Get(n); v != "" {
			if len(v) > 40 {
				v = v[:40]
			}
			return v
		}
	}
	return ""
}

// firstText は最初のメッセージ本文の先頭を返す（拒否ログ用）。
func firstText(msgs []struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}) string {
	if len(msgs) == 0 {
		return ""
	}
	head := []rune(rawText(msgs[0].Content))
	if len(head) > 40 {
		head = head[:40]
	}
	return string(head)
}

// quoteRe は Discord の返信を引用したブロックの先頭。
// picoclaw は "\[quoted message from 名前\]: <引用本文>\n\n<ユーザーの本文>" の形で送ってくる。
var quoteRe = regexp.MustCompile(`(?s)\A\s*\[quoted message from [^\]]{0,80}\]:`)

// quoteEndRe は picoclaw のフォーク版が付ける終了マーカー。
var quoteEndRe = regexp.MustCompile(`\n\[/quoted\]`)

// stripQuoted は引用ブロックだけを取り除く。
// 引用以外の本文には一切手を触れない（マーカーが無ければそのまま返す）。
func stripQuoted(s string) string {
	if !quoteRe.MatchString(s) {
		return s
	}
	// 引用本文とユーザー本文の区切りは空行（picoclaw は refContent + "\n\n" + 本文 の順）。
	// 最初の空行で切るので、ユーザー本文の段落はそのまま残る。
	// 終了マーカーがあれば、そこまでが引用（本文中の空行に影響されない）
	if m := quoteEndRe.FindStringIndex(s); m != nil {
		return strings.TrimPrefix(s[m[1]:], "\n")
	}
	// 旧形式: 区切りは空行しかない
	if i := strings.Index(s, "\n\n"); i >= 0 {
		return s[i+2:]
	}
	return "" // 引用しか含まれていない
}

// japaneseRatio は空白を除いた文字のうち日本語（ひらがな・カタカナ・漢字）の割合を返す。
// 英文の指示文に日本語が数文字だけ混ざる場合（例: セッション名生成の JSON）を
// 「日本語のメッセージ」と誤認しないための判定に使う。
func japaneseRatio(s string) float64 {
	ja, total := 0, 0
	for _, r := range s {
		if unicode.IsSpace(r) {
			continue
		}
		total++
		if unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Han) {
			ja++
		}
	}
	if total == 0 {
		return 0
	}
	return float64(ja) / float64(total)
}

// looksLikeInstruction は「日本語の続きを書かせる」のではなく
// 「要約しろ」「翻訳しろ」のような英語の指示文かどうかを判定する。
// 1行目が ASCII 主体で3語以上なら指示文とみなす。
// このモデルは指示に従えないので、指示文を渡すと英語の崩れた続きが返るだけになる。
func looksLikeInstruction(s string) bool {
	line := s
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		line = s[:i]
	}
	line = strings.TrimSpace(line)
	if line == "" {
		return false
	}
	if len(strings.Fields(line)) < 3 {
		return false
	}
	ascii := 0
	total := 0
	for _, r := range line {
		total++
		if r < 128 {
			ascii++
		}
	}
	if total == 0 {
		return false
	}
	return float64(ascii)/float64(total) > 0.8
}

// cleanMessage は プロンプトに混ぜたくない飾りを落とす。
// ACP などのクライアントは <acp tokens="4" ...>m00001</acp> のような
// メタデータ行をユーザー発話として送ってくる。英語タグが入ると
// 「続きの文章」が英語に流れるので、タグとIDだけの行は捨てる。
func (e *serveEngine) cleanMessage(s string) string {
	s = stripQuoted(s)
	if !e.stripMeta {
		return strings.TrimSpace(s)
	}
	s = tagRe.ReplaceAllString(s, "")
	kept := make([]string, 0, 8)
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimSpace(line)
		// ID だけの行はメタデータとして落とす。空行は本文の一部なので残す。
		if bareIDRe.MatchString(trimmed) {
			continue
		}
		kept = append(kept, strings.TrimRight(line, " \t"))
	}
	return strings.Trim(strings.Join(kept, "\n"), "\n")
}

// isAssistantRole は自分の過去出力かどうかを返す。
// このモデルは会話履歴を持たないので、過去の生成（崩れたものも含む）を
// プロンプトに戻すと、崩れが自己増殖する。
func isAssistantRole(role string) bool {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "assistant", "model", "bot":
		return true
	}
	return false
}

// isSystemRole は文脈に混ぜない役割かどうかを返す。
// このモデルは役割ラベル付きの会話で学習していないため、クライアントが付ける
// system 指示は「続きの文章」として生成に混ざるだけで、指示としては働かない。
func isSystemRole(role string) bool {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system", "developer":
		return true
	}
	return false
}

// rawText は content が文字列でもマルチパート配列でも本文を取り出す。
func rawText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			sb.WriteString(p.Text)
		}
		return sb.String()
	}
	return string(raw)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.Encode(v)
}

func tokRate(n int, d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(n) / d.Seconds()
}
