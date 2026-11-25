package nn

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
)

// magic は保存形式の識別子。n-gram の "JPLM" と区別するために使う。
const magic = "JPTF1\n"

// Save は重みをバイナリで書き出す。tokHash は語彙の同一性チェック用。
func (m *Model) Save(w io.Writer, tokHash uint64) error {
	bw := bufio.NewWriterSize(w, 1<<20)
	if _, err := bw.WriteString(magic); err != nil {
		return err
	}
	c := m.Cfg
	for _, v := range []int32{int32(c.Vocab), int32(c.Ctx), int32(c.Dim), int32(c.Heads), int32(c.Layers), int32(c.FFN)} {
		if err := binary.Write(bw, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	if err := binary.Write(bw, binary.LittleEndian, tokHash); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, int32(len(m.All))); err != nil {
		return err
	}
	for _, p := range m.All {
		if err := binary.Write(bw, binary.LittleEndian, p.W.D); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// Load は Save が書いたモデルを読み込む。
func Load(r io.Reader) (*Model, uint64, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	head := make([]byte, len(magic))
	if _, err := io.ReadFull(br, head); err != nil {
		return nil, 0, err
	}
	if string(head) != magic {
		return nil, 0, errors.New("nn: 未知のモデル形式")
	}
	var vals [6]int32
	if err := binary.Read(br, binary.LittleEndian, &vals); err != nil {
		return nil, 0, err
	}
	var tokHash uint64
	if err := binary.Read(br, binary.LittleEndian, &tokHash); err != nil {
		return nil, 0, err
	}
	cfg := Config{Vocab: int(vals[0]), Ctx: int(vals[1]), Dim: int(vals[2]), Heads: int(vals[3]), Layers: int(vals[4]), FFN: int(vals[5])}
	m := NewModel(cfg)
	var n int32
	if err := binary.Read(br, binary.LittleEndian, &n); err != nil {
		return nil, 0, err
	}
	if int(n) != len(m.All) {
		return nil, 0, errors.New("nn: パラメータ数が設定と一致しません")
	}
	for _, p := range m.All {
		if err := binary.Read(br, binary.LittleEndian, p.W.D); err != nil {
			return nil, 0, err
		}
	}
	return m, tokHash, nil
}

// IsTransformer はファイル先頭が Transformer 形式かどうかを返す。
func IsTransformer(head []byte) bool {
	return len(head) >= len(magic) && string(head[:len(magic)]) == magic
}
