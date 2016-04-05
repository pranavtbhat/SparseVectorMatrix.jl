###
# This file contains the type definition and basic properties of the Column-Disjoint
# sparse matrix type. SparseMatrixCD is best used when the matrix operations themselves
# are distjoint.
###

import Base: size, nnz, show, copy
export SparseMatrixCD

type SparseMatrixCD{Tv, Ti}
    m::Int
    n::Int
    svlist::Vector{SparseVector{Tv, Ti}}

    function SparseMatrixCD(m::Integer, n::Integer, svlist)
        m < 0 && throw(ArgumentError("number of rows (m) must be ≥ 0, got $m"))
        n < 0 && throw(ArgumentError("number of columns (n) must be ≥ 0, got $n"))
        new(Int(m), Int(n), svlist)
    end
end

### Basic Properties

# Size
size(x::SparseMatrixCD) = (x.m, x.n)
size(x::SparseMatrixCD, i::Integer) = (1<= i <=2)? size(x)[i] : 1

# NNZ
nnz(x::SparseMatrixCD) = mapreduce(nnz, +, 0, x.svlist)

# Show
function show{Tv, Ti}(io::IO, x::SparseMatrixCD{Tv, Ti})
    write(io, "SparseVectorMatrix{$Tv, $Ti}($(x.m) X $(x.n)) with $(nnz(x)) entries:\n")
    for c in 1 : x.n
        sv = x.svlist[c]
        for r in 1 : nnz(sv)
            write(io, "\t[$(sv.nzind[r]), $c] = $(sv.nzval[r])\n")
        end
    end
end

# Copy
copy{Tv, Ti}(x::SparseMatrixCD{Tv, Ti}) = SparseMatrixCD{Tv, Ti}(x.m, x.n, deepcopy(x.svlist))
