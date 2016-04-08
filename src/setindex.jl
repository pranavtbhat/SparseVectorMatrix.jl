###
# This file contains SetIndex implementations for sparse-vector matrices
###

import Base: setindex!

# Translations
setindex!(A::AbstractSVM, v, I::Tuple{Integer,Integer}) = setindex!(A, v, I[1], I[2])
setindex!(A::AbstractSVM, x, ::Colon) = setindex!(A, x, 1:length(A))

setindex!(A::AbstractSVM, x, ::Colon, ::Colon) = setindex!(A, x, 1:size(A, 1), 1:size(A,2))
setindex!(A::AbstractSVM, x, ::Colon, j::Integer) = setindex!(A, x, 1:size(A, 1), j)
setindex!(A::AbstractSVM, x, i::Integer, ::Colon) = setindex!(A, x, i, 1:size(A, 2))

# Unit Indexing
function setindex!(A::SparseMatrixCD, v, r::Integer, c::Integer)
    if !(1 <= r <= A.m && 1 <= c <= A.n)
        throw(BoundsError())
    end
    setindex!(A.svlist[c], v, r)
end

function setindex!(A::SparseMatrixRD, v, r::Integer, c::Integer)
    if !(1 <= r <= A.m && 1 <= c <= A.n)
        throw(BoundsError())
    end
    setindex!(A.svlist[r], v, c)
end

# Range Indexing
setindex!(A::SparseMatrixCD, v::Vector, rr::Range, c::Integer) = setindex!(A.svlist[c], v, rr)
setindex!(A::SparseMatrixCD, v, rr::Range, c::Integer) = setindex!(A, [v for i in rr], rr, c)
function setindex!(A::SparseMatrixCD, v::Vector, r::Integer, cr::Range)
    if length(v) != length(cr)
        throw(DimensionMismatch(""))
    end
    count = 1
    for i in cr
        setindex!(A.svlist[i], v[count], r)
        count += 1
    end
    v
end
setindex!(A::SparseMatrixRD, v, r::Integer, cr::Range) = setindex!(A, [v for i in cr], r, cr)

setindex!(A::SparseMatrixRD, v::Vector, r::Integer, cr::Range) = setindex!(A.svlist[r], v, cr)
setindex!(A::SparseMatrixRD, v, r::Integer, cr::Range) = setindex!(A, [v for i in rr], rr, c)
function setindex!(A::SparseMatrixRD, v::Vector, rr::Range, c::Integer)
    if length(v) != length(rr)
        throw(DimensionMismatch(""))
    end
    count = 1
    for i in rr
        setindex!(A.svlist[i], v[count], c)
        count += 1
    end
    v
end
setindex!(A::SparseMatrixRD, v, rr::Range, c::Integer) = setindex!(A, [v for i in rr], rr, c)

# Todo: Array indexing
