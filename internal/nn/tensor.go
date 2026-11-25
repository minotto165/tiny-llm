// Package nn は Transformer の最小実装 (forward/backward/Adam) を提供する。
// 外部ライブラリは使わず、行列は flat な []float64 で持つ。
package nn

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

// Mat は行優先の 2 次元行列。
type Mat struct {
	R, C int
	D    []float64
}

// NewMat は r×c のゼロ行列を作る。
func NewMat(r, c int) Mat {
	return Mat{R: r, C: c, D: make([]float64, r*c)}
}

// At は (i, j) の要素を返す。
func (m *Mat) At(i, j int) float64 { return m.D[i*m.C+j] }

// Set は (i, j) に値を書き込む。
func (m *Mat) Set(i, j int, v float64) { m.D[i*m.C+j] = v }

// Zero は全要素を 0 にする。
func (m *Mat) Zero() {
	for i := range m.D {
		m.D[i] = 0
	}
}

// parallelFor は 0..n-1 を複数の goroutine で処理する。
// 1 要素あたりの仕事が小さいときは分割しない (オーバーヘッドの方が大きい)。
func parallelFor(n, minWork int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || n < workers || n*minWork < 200000 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var next int64 = -1
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				i := int(atomic.AddInt64(&next, 1))
				if i >= n {
					return
				}
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// Mul は a·b を返す。
func Mul(a, b Mat) Mat {
	out := NewMat(a.R, b.C)
	parallelFor(a.R, a.C*b.C, func(i int) {
		orow := out.D[i*out.C : (i+1)*out.C]
		for k := 0; k < a.C; k++ {
			v := a.D[i*a.C+k]
			if v == 0 {
				continue
			}
			brow := b.D[k*b.C : (k+1)*b.C]
			for j := range brow {
				orow[j] += v * brow[j]
			}
		}
	})
	return out
}

func addMat(a, b Mat) Mat {
	out := NewMat(a.R, a.C)
	for i := range out.D {
		out.D[i] = a.D[i] + b.D[i]
	}
	return out
}

func addInto(dst *Mat, src Mat) {
	for i := range dst.D {
		dst.D[i] += src.D[i]
	}
}

// linear は y = x·w + b を計算する。x は T×in、w は in×out、b は 1×out。
func linear(x, w, b Mat) Mat {
	y := Mul(x, w)
	for t := 0; t < y.R; t++ {
		row := y.D[t*y.C : (t+1)*y.C]
		for j := range row {
			row[j] += b.D[j]
		}
	}
	return y
}

// linearBackward は dL/dx を返し、dL/dw と dL/db を accum する。
func linearBackward(x, w, dy Mat, dw, db *Mat) Mat {
	// dw += xᵀ·dy （行 i は独立なので並列化できる）
	parallelFor(x.C, x.R*dy.C, func(i int) {
		wrow := dw.D[i*dw.C : (i+1)*dw.C]
		for t := 0; t < x.R; t++ {
			v := x.D[t*x.C+i]
			if v == 0 {
				continue
			}
			drow := dy.D[t*dy.C : (t+1)*dy.C]
			for j := range drow {
				wrow[j] += v * drow[j]
			}
		}
	})
	// db += Σ dy
	for t := 0; t < dy.R; t++ {
		drow := dy.D[t*dy.C : (t+1)*dy.C]
		for j := range drow {
			db.D[j] += drow[j]
		}
	}
	// dx = dy·wᵀ （w は in×out なので、行 i が入力 i の重み）
	dx := NewMat(x.R, x.C)
	parallelFor(dy.R, x.C*dy.C, func(t int) {
		dyrow := dy.D[t*dy.C : (t+1)*dy.C]
		drow := dx.D[t*dx.C : (t+1)*dx.C]
		for i := 0; i < x.C; i++ {
			wrow := w.D[i*w.C : (i+1)*w.C]
			var acc float64
			for j, v := range dyrow {
				acc += v * wrow[j]
			}
			drow[i] = acc
		}
	})
	return dx
}

type lnCache struct {
	x, xhat   Mat
	mean, std []float64
}

const lnEps = 1e-5

// layernorm は行ごとに正規化する。g と b は 1×d。
func layernorm(x Mat, g, b Mat) (Mat, lnCache) {
	t, d := x.R, x.C
	out := NewMat(t, d)
	c := lnCache{x: x, xhat: NewMat(t, d), mean: make([]float64, t), std: make([]float64, t)}
	for i := 0; i < t; i++ {
		var mean float64
		for j := 0; j < d; j++ {
			mean += x.At(i, j)
		}
		mean /= float64(d)
		var v float64
		for j := 0; j < d; j++ {
			dx := x.At(i, j) - mean
			v += dx * dx
		}
		v /= float64(d)
		std := 1 / math.Sqrt(v+lnEps)
		c.mean[i], c.std[i] = mean, std
		for j := 0; j < d; j++ {
			xh := (x.At(i, j) - mean) * std
			c.xhat.Set(i, j, xh)
			out.Set(i, j, xh*g.D[j]+b.D[j])
		}
	}
	return out, c
}

// layernormBackward は dL/dx を返し、g と b の勾配を accum する。
func layernormBackward(c lnCache, g Mat, dy Mat, dg, db *Mat) Mat {
	t, d := dy.R, dy.C
	dx := NewMat(t, d)
	for i := 0; i < t; i++ {
		var sum1, sum2 float64
		for j := 0; j < d; j++ {
			dyg := dy.At(i, j) * g.D[j]
			dg.D[j] += dy.At(i, j) * c.xhat.At(i, j)
			db.D[j] += dy.At(i, j)
			sum1 += dyg
			sum2 += dyg * c.xhat.At(i, j)
			dx.Set(i, j, dyg)
		}
		for j := 0; j < d; j++ {
			dx.Set(i, j, c.std[i]/float64(d)*(float64(d)*dx.At(i, j)-sum1-c.xhat.At(i, j)*sum2))
		}
	}
	return dx
}

// gelu は tanh 近似。
func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
}

func geluGrad(x float64) float64 {
	c := math.Sqrt(2 / math.Pi)
	t := math.Tanh(c * (x + 0.044715*x*x*x))
	return 0.5*(1+t) + 0.5*x*(1-t*t)*c*(1+3*0.044715*x*x)
}

// softmaxRow は x をその場で softmax にし、最大値も返す。
func softmaxRow(x []float64) float64 {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for i, v := range x {
		e := math.Exp(v - max)
		x[i] = e
		sum += e
	}
	for i := range x {
		x[i] /= sum
	}
	return max
}
