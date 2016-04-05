###
# This file contains GetIndex implementations for sparse-vector matrices
###

import Base: getindex

# Translations
getindex(A::SparseMatrixCD, I::Tuple{Integer,Integer}) = getindex(A, I[1], I[2])
getindex(A::SparseMatrixCD, ::Colon, ::Colon) = copy(A)
getindex(A::SparseMatrixCD, r, ::Colon) = getindex(A, r, 1 : size(A, 2))
getindex(A::SparseMatrixCD, ::Colon, c) = getindex(A, 1 : size(A, 1), c)

# Unit Indexing
function getindex(A::SparseMatrixCD, r::Integer, c::Integer)
    if !(1 <= r <= A.m && 1 <= c <= A.n)
        throw(BoundsError())
    end
    getindex(A.svlist[c], r) # The array reference that slows things down.
end

# Range Indexing
getindex(A::SparseMatrixCD, rr::Range, c::Integer) = getindex(A.svlist[c], rr)
getindex(A::SparseMatrixCD, r::Integer, cr::Range) = sparse([getindex(A.svlist[c], r) for c in cr])

function getindex{Tv, Ti}(A::SparseMatrixCD{Tv, Ti}, rr::Range, cr::Range)
    m = length(rr)
    n = length(cr)
    svlist = [getindex(A.svlist[c], rr) for c in cr]
    SparseMatrixCD{Tv, Ti}(m, n, svlist)
end

# Array Indexing
getindex(A::SparseMatrixCD, rv::AbstractVector, c::Integer) = getindex(A.svlist[c], rv)
getindex(A::SparseMatrixCD, r::Integer, cv::AbstractVector) = sparse([getindex(A.svlist[c], r) for c in cv])

function getindex{Tv, Ti}(A::SparseMatrixCD{Tv, Ti}, rv::AbstractVector, cv::AbstractVector)
    m = length(rv)
    n = length(cv)
    svlist = [getindex(A.svlist[c], rv) for c in cv]
    SparseMatrixCD{Tv, Ti}(m, n, svlist)
end
