package cli

import "path/filepath"

// artifacts は -dir で指定されたディレクトリ内の dic.txt と merges.txt のパスを返す。
func artifacts(dir string) (dict, merges string) {
	return filepath.Join(dir, "dic.txt"), filepath.Join(dir, "merges.txt")
}
