package lm

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// Save は学習済みモデルを w に書き出す。形式は独自のバイナリで、
// バージョン番号を含むので将来の変更は読み込み側で検出できる。
func (m *Model) Save(w io.Writer) error {
	bw := bufio.NewWriterSize(w, 1<<20)
	if _, err := bw.WriteString("JPLM2\n"); err != nil {
		return err
	}
	hdr := []any{
		int32(m.order), int32(m.v), m.discount, m.hiKeep, m.hiWeight, m.total, m.n1uni,
	}
	for _, v := range hdr {
		if err := binary.Write(bw, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	if err := writeInt32s(bw, m.uni); err != nil {
		return err
	}
	if err := writeInt32s(bw, m.cont1); err != nil {
		return err
	}
	for _, mp := range []map[uint64]int32{m.grams, m.ctxs, m.n1, m.cont2, m.cont2ctx, m.cont2n1} {
		if err := writeMap(bw, mp); err != nil {
			return err
		}
	}
	for _, level := range [][]map[uint64]int32{m.hiCount, m.hiTot, m.hiDist} {
		if err := binary.Write(bw, binary.LittleEndian, int32(len(level))); err != nil {
			return err
		}
		for _, mp := range level {
			if err := writeMap(bw, mp); err != nil {
				return err
			}
		}
	}
	return bw.Flush()
}

// Load は Save で書いたモデルを r から読む。
func Load(r io.Reader) (*Model, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	magic := make([]byte, 6)
	if _, err := io.ReadFull(br, magic); err != nil {
		return nil, err
	}
	if string(magic) != "JPLM2\n" {
		return nil, errors.New("lm: not a model file")
	}
	m := &Model{}
	var order, v int32
	for _, p := range []any{&order, &v, &m.discount, &m.hiKeep, &m.hiWeight, &m.total, &m.n1uni} {
		if err := binary.Read(br, binary.LittleEndian, p); err != nil {
			return nil, err
		}
	}
	m.order, m.v = int(order), int(v)
	var err error
	if m.uni, err = readInt32s(br); err != nil {
		return nil, err
	}
	if m.cont1, err = readInt32s(br); err != nil {
		return nil, err
	}
	maps := []*map[uint64]int32{&m.grams, &m.ctxs, &m.n1, &m.cont2, &m.cont2ctx, &m.cont2n1}
	for _, mp := range maps {
		if *mp, err = readMap(br); err != nil {
			return nil, err
		}
	}
	levels := []*[]map[uint64]int32{&m.hiCount, &m.hiTot, &m.hiDist}
	for _, lv := range levels {
		var n int32
		if err := binary.Read(br, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		*lv = make([]map[uint64]int32, n)
		for i := range *lv {
			if (*lv)[i], err = readMap(br); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// SaveFile はモデルを path に書き出す。
func (m *Model) SaveFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := m.Save(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// LoadFile は path からモデルを読む。
func LoadFile(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(f)
}

// Order はモデルの次数を返す。
func (m *Model) Order() int { return m.order }

// Vocab は学習時の語彙サイズを返す。
func (m *Model) Vocab() int { return m.v }

func writeInt32s(w io.Writer, a []int32) error {
	if err := binary.Write(w, binary.LittleEndian, int32(len(a))); err != nil {
		return err
	}
	if len(a) == 0 {
		return nil
	}
	return binary.Write(w, binary.LittleEndian, a)
}

func readInt32s(r io.Reader) ([]int32, error) {
	var n int32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if n < 0 {
		return nil, errors.New("lm: negative length")
	}
	a := make([]int32, n)
	if n == 0 {
		return a, nil
	}
	if err := binary.Read(r, binary.LittleEndian, a); err != nil {
		return nil, err
	}
	return a, nil
}

func writeMap(w io.Writer, m map[uint64]int32) error {
	if err := binary.Write(w, binary.LittleEndian, int64(len(m))); err != nil {
		return err
	}
	for k, v := range m {
		if err := binary.Write(w, binary.LittleEndian, k); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return nil
}

func readMap(r io.Reader) (map[uint64]int32, error) {
	var n int64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if n < 0 || n > math.MaxInt32 {
		return nil, fmt.Errorf("lm: bad map size %d", n)
	}
	m := make(map[uint64]int32, n)
	for i := int64(0); i < n; i++ {
		var k uint64
		var v int32
		if err := binary.Read(r, binary.LittleEndian, &k); err != nil {
			return nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		m[k] = v
	}
	return m, nil
}
