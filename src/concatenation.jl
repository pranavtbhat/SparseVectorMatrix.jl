###
# This file contains vector-sparse matrix concatenations
###

import Base: hcat, vcat, hvcat

# Horizontal Concatenation
function hcat(A::SparseMatrixCD...)
    m = A[1].m
    e = A[1]
    for i in 1 : length(A)
        if A[i].m != e.m
            error("All input matrices must have the same number of rows")
        end
    end

    Tv = promote_type(map(x->eltype(x.svlist[1].nzval), A)...)
    Ti = promote_type(map(x->eltype(x.svlist[1].nzind), A)...)

    n = mapreduce(x->x.n, +, 0, A)
    svlist = vcat(map(x->x.svlist, A)...)
    SparseMatrixCD{Tv, Ti}(m, n, svlist)
end

# Vertical Concatenation
function vcat(A::SparseMatrixCD...)
    m = mapreduce(x->x.m, +, 0, A)
    e = A[1]
    for i in 1 : length(A)
        if A[i].n != e.n
            error("All input matrices must have the same number of columns")
        end
    end

    Tv = promote_type(map(x->eltype(x.svlist[1].nzval), A)...)
    Ti = promote_type(map(x->eltype(x.svlist[1].nzind), A)...)

    n = A[1].n
    svlist = Vector{SparseVector{Tv, Ti}}(n)
    for c in 1 : n
        svlist[c] = vcat(map(x->x.svlist[c], A)...)
    end
    SparseMatrixCD{Tv, Ti}(m, n, svlist)
end

# Grid Concatenation
function hvcat(rows::Tuple{Vararg{Int}}, A::SparseMatrixCD...)
    nbr = length(rows)
    tmp_rows = Array(SparseMatrixCD, nbr)
    k = 0
    @inbounds for i = 1 : nbr
        tmp_rows[i] = hcat(A[(1 : rows[i]) + k]...)
        k += rows[i]
    end
    vcat(tmp_rows...)
end
