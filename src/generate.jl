import Base: spzeros, speye, sprand, sprandbool

###
# SPZEROS
###

""" Generate an empty mxn sparse vector matrix """
function spzeros(Tv::Type, ::ColumnVector, m::Integer, n::Integer)
    SparseMatrixCD{Tv, Int}(m, n, [spzeros(Tv, m) for c in 1 : n])
end

function spzeros(Tv::Type, ::RowVector, m::Integer, n::Integer)
    SparseMatrixRD{Tv, Int}(m, n, [spzeros(Tv, m) for c in 1 : n])
end

spzeros(T::SliceDimension, m::Integer, n::Integer) = svmzeros(Float64, T, m, n)


###
# SPRAND
###

""" Generate a mxn sparse vector matrix with random floating point entries """
function sprand(::ColumnVector, m::Integer, n::Integer, d::Float64)
    SparseMatrixCD(m, n, [sprand(m, d) for c in 1 : n])
end

function sprand(::RowVector, m::Integer, n::Integer, d::Float64)
    SparseMatrixRD(m, n, [sprand(m, d) for c in 1 : n])
end


###
# SPRANDBOOL
###

""" Generate a mxn sparse vector matrix with boolean entries """
function sprandbool(::ColumnVector, m::Integer, n::Integer, d::Float64)
    SparseMatrixCD{Bool, Int}(m, n, [sprandbool(m, d) for c in 1 : n])
end

function sprandbool(::ColumnVector, m::Integer, n::Integer, d::Float64)
    SparseMatrixCD{Bool, Int}(m, n, [sprandbool(m, d) for c in 1 : n])
end

###
# SPEYE
###
""" Generate an identity mxn vector matrix with floating point entries """
function speye(Tv::Type, sd::SliceDimension, m::Integer, n::Integer)
    svm = spzeros(Tv, sd, m, n)
    for i in 1 : min(m, n)
        setindex!(svm.svlist[i], one(Tv), i)
    end
    svm
end

speye(sd::SliceDimension, m::Integer, n::Integer) = speye(Float64, sd, m, n)
