Ordered defines an encoding of a sequence of basic typed values into []byte and back.
The encoding is “ordered” in the sense that bytes.Compare on the byte slices matches cmp.Compare on the original values.
The order-preserving encoding is particularly useful for preparing keys in an ordered key/value storage such as
Bigtable, Bolt, CockroachDB, LevelDB, Pebble, RocksDB, or Spanner.

This package is inspired by [github.com/google/orderedcode](https://github.com/google/orderedcode)
but uses a different, self-describing encoding.

See the [package documentation](https://pkg.go.dev/rsc.io/ordered) for details and examples.
