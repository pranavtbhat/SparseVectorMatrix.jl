import Base: size, nnz, show, getindex, setindex!, full, hcat, vcat, hvcat

export SVMatrix


type SVMatrix{Tv, Ti}
    m::Int
    n::Int
    svlist::Vector{SparseVector{Tv, Ti}}

    function SVMatrix(m::Integer, n::Integer, svlist)
        m < 0 && throw(ArgumentError("number of rows (m) must be ≥ 0, got $m"))
        n < 0 && throw(ArgumentError("number of columns (n) must be ≥ 0, got $n"))
        new(Int(m), Int(n), svlist)
    end
end

#### Basic Properties

# Size
size(x::SVMatrix) = (x.m, x.n)
size(x::SVMatrix, i::Integer) = (1<= i <=2)? size(x)[UInt(i)] : 1

# NNZ
nnz(x::SVMatrix) = mapreduce(nnz, +, 0, x.svlist)

# Show
function show{Tv, Ti}(io::IO, x::SVMatrix{Tv, Ti})
    write(io, "SparseVectorMatrix{$Tv, $Ti}($(x.m) X $(x.n)) with $(nnz(x)) entries:\n")
    for c in 1 : x.n
        sv = x.svlist[c]
        for r in 1 : nnz(sv)
            write(io, "\t[$(sv.nzind[r]), $c] = $(sv[r])\n")
        end
    end
end

# Copy
copy{Tv, Ti}(x::SVMatrix{Tv, Ti}) = SVMatrix{Tv, Ti}(x.m, x.n, deepcopy(x.svlist))

### GetIndex

# Translations
getindex(A::SVMatrix, I::Tuple{Integer,Integer}) = getindex(A, I[1], I[2])
getindex(A::SVMatrix, ::Colon, ::Colon) = copy(A)
getindex(A::SVMatrix, r, ::Colon) = getindex(A, r, 1 : size(A, 2))
getindex(A::SVMatrix, ::Colon, c) = getindex(A, 1 : size(A, 1), c)

# Unit Indexing
function getindex(A::SVMatrix, r::Integer, c::Integer)
    if !(1 <= r <= A.m && 1 <= c <= A.n)
        throw(BoundsError())
    end
    getindex(A.svlist[c], r) # The array reference that slows things down.
end

# Range Indexing
getindex(A::SVMatrix, rr::Range, c::Integer) = getindex(A.svlist[c], rr)
getindex(A::SVMatrix, r::Integer, cr::Range) = sparse([getindex(A.svlist[c], r) for c in cr])

function getindex{Tv, Ti}(A::SVMatrix{Tv, Ti}, rr::Range, cr::Range)
    m = length(rr)
    n = length(cr)
    svlist = [getindex(A.svlist[c], rr) for c in cr]
    SVMatrix{Tv, Ti}(m, n, svlist)
end

# Array Indexing
getindex(A::SVMatrix, rv::AbstractVector, c::Integer) = getindex(A.svlist[c], rv)
getindex(A::SVMatrix, r::Integer, cv::AbstractVector) = sparse([getindex(A.svlist[c], r) for c in cv])

function getindex{Tv, Ti}(A::SVMatrix{Tv, Ti}, rv::AbstractVector, cv::AbstractVector)
    m = length(rv)
    n = length(cv)
    svlist = [getindex(A.svlist[c], rv) for c in cv]
    SVMatrix{Tv, Ti}(m, n, svlist)
end


### SetIndex

# Translations
setindex!(A::SVMatrix, v, I::Tuple{Integer,Integer}) = setindex!(A, v, I[1], I[2])
setindex!(A::SVMatrix, x, ::Colon) = setindex!(A, x, 1:length(A))

setindex!(A::SVMatrix, x, ::Colon, ::Colon) = setindex!(A, x, 1:size(A, 1), 1:size(A,2))
setindex!(A::SVMatrix, x, ::Colon, j::Integer) = setindex!(A, x, 1:size(A, 1), j)
setindex!(A::SVMatrix, x, i::Integer, ::Colon) = setindex!(A, x, i, 1:size(A, 2))

# Unit Indexing

function setindex!(A::SVMatrix, v, r::Integer, c::Integer)
    if !(1 <= r <= A.m && 1 <= c <= A.n)
        throw(BoundsError())
    end
    setindex!(A.svlist[c], v, r)
end

# Range Indexing
function setindex!(A::SVMatrix, v, r::Integer, cr::Range)
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

setindex!(A::SVMatrix, v, rr::Range, c::Integer) = setindex!(A.svlist[c], v, rr)

# Todo: Array indexing

### Conversion
full(A::SVMatrix) = hcat(map(full, A.svlist)...)


### Concatenation

# Horizontal Concatenation
function hcat(A::SVMatrix...)
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
    SVMatrix{Tv, Ti}(m, n, svlist)
end

# Vertical Concatenation
function vcat(A::SVMatrix...)
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
    SVMatrix{Tv, Ti}(m, n, svlist)
end

# Grid Concatenation
function hvcat(rows::Tuple{Vararg{Int}}, A::SVMatrix...)
    nbr = length(rows)
    tmp_rows = Array(SVMatrix, nbr)
    k = 0
    @inbounds for i = 1 : nbr
        tmp_rows[i] = hcat(A[(1 : rows[i]) + k]...)
        k += rows[i]
    end
    vcat(tmp_rows...)
end
