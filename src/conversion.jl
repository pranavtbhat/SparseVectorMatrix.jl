###
# This file contains conversions from vector-sparse matrices to other matrix types
###

import Base: full

export vsparse

# Full
full(A::SparseMatrixCD) = hcat(map(full, A.svlist)...)

# Sparse
function vsparse{Tv}(A::AbstractArray{Tv, 2})
    m, n = size(A)
    SparseMatrixCD{Tv, Int}(m, n, [sparse(A[:,c]) for c in 1 : size(A, 2)])
end

# Convert from SparseMatrixCSC to SparseMatrixCD
function SparseMatrixCD{Tv, Ti}(X::SparseMatrixCSC{Tv, Ti})
    m = X.m
    n = X.n
    svlist = Array{SparseVector{Tv, Ti}}(n)

    colptrX = X.colptr
    nzvalX = X.nzval
    rowvalX = X.rowval

    for col in 1 : n
        rr = colptrX[col] : (colptrX[col+1] - 1)
        svlist[col] = SparseVector{Tv, Ti}(m, rowvalX[rr], nzvalX[rr])
    end
    SparseMatrixCD{Tv, Ti}(m, n, svlist)
end
