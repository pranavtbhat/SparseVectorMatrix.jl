###
# This file contains conversions from vector-sparse matrices to other matrix types
###

import Base: full, SparseMatrixCSC

export vsparse

# Full
full(A::SparseMatrixCD) = hcat(map(full, A.svlist)...)

# Sparse
function vsparse{Tv}(A::AbstractArray{Tv, 2})
    m, n = size(A)
    svlist = Vector{SparseVector{Tv, Int}}(n)

    for c in 1 : n
        @inbounds svlist[c] = sparse(A[1+(c-1)*m : c*m])
    end
    SparseMatrixCD{Tv, Int}(m, n, svlist)
end

# Convert from SparseMatrixCSC to SparseMatrixCD
function SparseMatrixCD{Tv, Ti}(X::SparseMatrixCSC{Tv, Ti})
    m = X.m
    n = X.n
    svlist = Array{SparseVector{Tv, Ti}, 1}(n)

    colptrX = X.colptr
    nzvalX = X.nzval
    rowvalX = X.rowval

    for col in 1 : n
        rr = colptrX[col] : (colptrX[col+1] - 1)
        svlist[col] = SparseVector{Tv, Ti}(m, rowvalX[rr], nzvalX[rr])
    end
    SparseMatrixCD{Tv, Ti}(m, n, svlist)
end

function SparseMatrixCSC{Tv, Ti}(X::SparseMatrixCD{Tv,Ti})
    m = X.m
    n = X.n

    rowval = vcat(map(x->x.nzind, X.svlist)...)
    nzval = vcat(map(x->x.nzval, X.svlist)...)
    colptr = Array(Ti, n+1)

    colptr[1] = 1
    for c in 1 : n
        colptr[c+1] = colptr[c] + nnz(X.svlist[c])
    end
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end
