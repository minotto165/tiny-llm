package cli

import (
	"fmt"
	"os"
	"sort"
)

// memCheck は生成テキストが参照コーパスをどれだけ丸写ししているかを測る。
// n 文字の窓がコーパスに存在する割合と、最長の一致長を出す。
type memCheck struct {
	n     int
	corp  []rune
	exist map[string]struct{}
}

func newMemCheck(path string, n int) (*memCheck, error) {
	if n <= 0 {
		n = 20
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	corp := []rune(string(b))
	c := &memCheck{n: n, corp: corp, exist: make(map[string]struct{}, len(corp))}
	for i := 0; i+n <= len(corp); i++ {
		c.exist[string(corp[i:i+n])] = struct{}{}
	}
	return c, nil
}

// ratio は n 文字窓のうちコーパスに存在するものの割合を返す。
func (c *memCheck) ratio(text []rune) (hit, total int) {
	for i := 0; i+c.n <= len(text); i++ {
		total++
		if _, ok := c.exist[string(text[i:i+c.n])]; ok {
			hit++
		}
	}
	return hit, total
}

// longest は text がコーパスと一致する最長の部分文字列を返す。
func (c *memCheck) longest(text []rune) string {
	lo, hi := 0, len(text)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		found := ""
		for i := 0; i+mid <= len(text); i++ {
			w := string(text[i : i+mid])
			if containsRunes(c.corp, []rune(w)) {
				found = w
				break
			}
		}
		if found != "" {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	if lo == 0 {
		return ""
	}
	for i := 0; i+lo <= len(text); i++ {
		w := text[i : i+lo]
		if containsRunes(c.corp, w) {
			return string(w)
		}
	}
	return ""
}

// containsRunes は素朴な部分一致。窓は短いので十分速い。
func containsRunes(hay, needle []rune) bool {
	if len(needle) == 0 {
		return true
	}
	if len(needle) > len(hay) {
		return false
	}
	first := needle[0]
	for i := 0; i+len(needle) <= len(hay); i++ {
		if hay[i] != first {
			continue
		}
		ok := true
		for j := 1; j < len(needle); j++ {
			if hay[i+j] != needle[j] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// Report は 1 サンプル分の計測結果を stderr に出す。
func (c *memCheck) Report(label, generated string) {
	text := []rune(generated)
	hit, total := c.ratio(text)
	best := c.longest(text)
	pct := 0.0
	if total > 0 {
		pct = float64(hit) / float64(total) * 100
	}
	fmt.Fprintf(os.Stderr, "memcheck[%s]: %d chars, %d-gram in corpus %d/%d (%.1f%%), longest %d chars\n",
		label, len(text), c.n, hit, total, pct, len([]rune(best)))
	if best != "" && len([]rune(best)) >= c.n {
		fmt.Fprintf(os.Stderr, "  verbatim: %q\n", best)
	}
}

// multiReport は複数サンプルをまとめて集計する。
func multiReport(c *memCheck, samples []string) {
	if len(samples) == 0 {
		return
	}
	var hits, totals int
	var longest []int
	var worst string
	for i, s := range samples {
		h, t := c.ratio([]rune(s))
		hits += h
		totals += t
		b := c.longest([]rune(s))
		n := len([]rune(b))
		longest = append(longest, n)
		if n > len([]rune(worst)) {
			worst = b
		}
		_ = i
	}
	sort.Ints(longest)
	pct := 0.0
	if totals > 0 {
		pct = float64(hits) / float64(totals) * 100
	}
	fmt.Fprintf(os.Stderr, "memcheck: %d samples, %d-gram in corpus %d/%d (%.1f%%), longest %d chars (median %d)\n",
		len(samples), c.n, hits, totals, pct, longest[len(longest)-1], longest[len(longest)/2])
	if worst != "" {
		fmt.Fprintf(os.Stderr, "  longest verbatim: %q\n", worst)
	}
}
