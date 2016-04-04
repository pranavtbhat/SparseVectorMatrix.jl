# SparseVectorMatrix
This packages provides an alternative implementation of SparseMatrices that maintains a
vector of SparseVectors. This implementation is best used when all matrix access operate
on a single column. Graph adjacency lists is one such use case.


## Usage

```julia
using SparseVectorMatrix

# Random Generation
a = spmrand(100, 100, 0.1)

# Getindex
a[:, 1]                      # Returns an entire column quickly
a[1, :]                      # Returns an entire row, but slowly.

# SetIndex
a[:, 1] = 1:100              # Assign an entire column quickly.
a[1, :] = 1:100              # Assign an entire row, by slowly.

#Concatenation
b = spmrand(100, 100, 0.1)
hcat(a, b)                   # Concatenates horizontally. Very fast.
vcat(a, b)                   # Concatenates vertically. Not as fast.
hvcat(a, b)                  # Grid Concatenation. Quite fast.
```

## What's supported?
- spmrand (Similar to sprand)
- getindex
- setindex
- hcat
- vcat
- hvcat
- A bunch of other basic methods like nnz, size, full, etc.
