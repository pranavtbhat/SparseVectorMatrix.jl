# SparseVectorMatrix
This packages provides an alternative implementation of SparseMatrices that maintains a
vector of SparseVectors. Such an implementation is best used when all matrix operations require access
to just one column each.


## Usage

```julia
using SparseVectorMatrix

# Random Generation
a = svmrand(100, 100, 0.1)

# Getindex
a[:, 1]                      # Returns an entire column quickly
a[1, :]                      # Returns an entire row, but slowly.

# SetIndex
a[:, 1] = 1:100              # Assign an entire column quickly.
a[1, :] = 1:100              # Assign an entire row, by slowly.

#Concatenation
b = svmrand(100, 100, 0.1)
hcat(a, b)                   # Concatenates horizontally. Very fast.
vcat(a, b)                   # Concatenates vertically. Not as fast.

arr = [svmrand(100, 100, 0.1) for i in 1:4]
hvcat((2,2), arr..)          # Grid Concatenation. Quite fast.
```

## What's supported?
- svmrand (Similar to sprand)
- getindex
- setindex
- hcat
- vcat
- hvcat
- A bunch of other basic methods like nnz, size, full, etc.

## Benchmarking
```julia
include("benchmarks/run.jl")
```
