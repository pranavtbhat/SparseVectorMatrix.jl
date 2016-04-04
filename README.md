# SparseVectorMatrix
This packages provides an alternative implementation of SparseMatrices that maintains a
vector of SparseVectors. This implementation is best used when all matrix access operate
on a single column. Graph adjacency lists is one such use case.


## Usage

```julia
a = spmrand(100, 100, 0.1)

a[:,1] # Returns an entire column quickly

a[1, :] # Returns an entire row, but slowly.

b = spmrand(100, 100, 0.1)

hcat(a, b) # Concatenates horizontally. Very fast.

vcat(a, b) # Concatenates vertically. Not as fast.
```

## What's supported?
- getindex
- hcat
- vcat
- spmrand (Similar to sprand)
