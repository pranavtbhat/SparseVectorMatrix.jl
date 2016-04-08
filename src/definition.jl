import Base: size, nnz, show, copy, ==, isequal

export ColumnVector, RowVector, AbstractSVM, SparseMatrixCD, SparseMatrixRD

abstract SliceDimension

"""Type to indicate Column-Vector slicing"""
type ColumnVector <: SliceDimension
end

"""Type to indicate Row-Vector slicing"""
type RowVector <: SliceDimension
end


"""Abstract Sparse Vector Matrix type"""
abstract AbstractSVM{Tv,Ti}

###
# Basic Properties
###

# Size
size(x::AbstractSVM) = (x.m, x.n)
size(x::AbstractSVM, i::Integer) = (1<= i <=2)? size(x)[i] : 1

# NNZ
nnz(x::AbstractSVM) = mapreduce(nnz, +, 0, x.svlist)

# Show
function show{Tv,Ti}(io::IO, S::AbstractSVM{Tv,Ti})
    println(io, S.m, "x", S.n, " sparse matrix with ", nnz(S), " ", Tv, " entries:")
    printentries(io, S)
end


###
# COLUMN DISJOINT SPARSE MATRIX
###

"""Column Disjoint sparse matrix"""
type SparseMatrixCD{Tv,Ti} <: AbstractSVM{Tv,Ti}
    m::Int
    n::Int
    svlist::Vector{SparseVector{Tv,Ti}}

    function SparseMatrixCD(m::Integer, n::Integer, svlist)
        m < 0 && throw(ArgumentError("number of rows (m) must be ≥ 0, got $m"))
        n < 0 && throw(ArgumentError("number of columns (n) must be ≥ 0, got $n"))
        new(Int(m), Int(n), svlist)
    end
end

# Type-relaxed constructor
function SparseMatrixCD(m::Integer, n::Integer, svlist::Vector)
    Tv = eltype(svlist[1].nzval)
    Ti = eltype(svlist[1].nzind)
    for i = 2:length(svlist)
        Tv = promote_type(Tv, eltype(svlist[i].nzval))
        Ti = promote_type(Ti, eltype(svlist[i].nzind))
    end
    SparseMatrixCD{Tv,Ti}(m, n, svlist)
end

# Show
function printentries{Tv,Ti}(io::IO, x::SparseMatrixCD{Tv,Ti})
    count = 0
    for c in 1 : x.n
        sv = x.svlist[c]
        for i in 1 : nnz(sv)
            count >= 15 && return
            println(io, "\t[$(sv.nzind[i]), $c] = $(sv.nzval[i])")
            count += 1
        end
    end
end

# Copy
copy{Tv, Ti}(x::SparseMatrixCD{Tv, Ti}) = SparseMatrixCD{Tv, Ti}(x.m, x.n, deepcopy(x.svlist))

# Equality
(==)(x::SparseMatrixCD, y::SparseMatrixCD) = (x.m == y.m) && (x.n == y.n) && reduce(&, map(==, x.svlist, y.svlist))
isequal(x::SparseMatrixCD, y::SparseMatrixCD) = isequal(x.m, y.m) && isequal(x.n, y.n) && reduce(&, map(isequal, x.svlist, y.svlist))


###
# ROW DISJOINT SPARSE MATRIX
###

"""Row Disjoint sparse matrix"""
type SparseMatrixRD{Tv,Ti} <: AbstractSVM{Tv,Ti}
    m::Int
    n::Int
    svlist::Vector{SparseVector{Tv,Ti}}

    function SparseMatrixRD(m::Integer, n::Integer, svlist)
        m < 0 && throw(ArgumentError("number of rows (m) must be ≥ 0, got $m"))
        n < 0 && throw(ArgumentError("number of columns (n) must be ≥ 0, got $n"))
        new(Int(m), Int(n), svlist)
    end
end

# Type-relaxed constructor
function SparseMatrixRD(m::Integer, n::Integer, svlist::Vector)
    Tv = eltype(svlist[1].nzval)
    Ti = eltype(svlist[1].nzind)
    for i = 2:length(svlist)
        Tv = promote_type(Tv, eltype(svlist[i].nzval))
        Ti = promote_type(Ti, eltype(svlist[i].nzind))
    end
    SparseMatrixRD{Tv,Ti}(m, n, svlist)
end

# Show
function printentries{Tv,Ti}(io::IO, x::SparseMatrixRD{Tv,Ti})
    count = 0
    for r in 1 : x.m
        sv = x.svlist[r]
        for i in 1 : nnz(sv)
            count >= 15 && return
            write(io, "\t[$r, $(sv.nzind[i])] = $(sv.nzval[i])\n")
            count += 1
        end
    end
end

# Copy
copy{Tv, Ti}(x::SparseMatrixRD{Tv, Ti}) = SparseMatrixRD{Tv, Ti}(x.m, x.n, deepcopy(x.svlist))

# Equality
(==)(x::SparseMatrixRD, y::SparseMatrixRD) = (x.m == y.m) && (x.n == y.n) && reduce(&, map(==, x.svlist, y.svlist))
isequal(x::SparseMatrixRD, y::SparseMatrixRD) = isequal(x.m, y.m) && isequal(x.n, y.n) && reduce(&, map(isequal, x.svlist, y.svlist))
