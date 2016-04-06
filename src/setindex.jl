###
# This file contains SetIndex implementations for sparse-vector matrices
###

import Base: setindex!

# Translations
setindex!(A::SparseMatrixCD, v, I::Tuple{Integer,Integer}) = setindex!(A, v, I[1], I[2])
setindex!(A::SparseMatrixCD, x, ::Colon) = setindex!(A, x, 1:length(A))

setindex!(A::SparseMatrixCD, x, ::Colon, ::Colon) = setindex!(A, x, 1:size(A, 1), 1:size(A,2))
setindex!(A::SparseMatrixCD, x, ::Colon, j::Integer) = setindex!(A, x, 1:size(A, 1), j)
setindex!(A::SparseMatrixCD, x, i::Integer, ::Colon) = setindex!(A, x, i, 1:size(A, 2))

# Unit Indexing

function setindex!(A::SparseMatrixCD, v, r::Integer, c::Integer)
    if !(1 <= r <= A.m && 1 <= c <= A.n)
        throw(BoundsError())
    end
    setindex!(A.svlist[c], v, r)
end

# Range Indexing
function setindex!(A::SparseMatrixCD, v, r::Integer, cr::Range)
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

setindex!(A::SparseMatrixCD, v, rr::Range, c::Integer) = setindex!(A.svlist[c], v, rr)

# Todo: Array indexing
