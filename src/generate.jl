export svmzeros, svmrand, svmrandbool, svmeye

# Generate an empty mxn sparse vector matrix
function svmzeros(Tv::Type, m::Integer, n::Integer)
    SparseMatrixCD{Tv, Int}(m, n, [spzeros(Tv, m) for c in 1 : n])
end
svmzeros(m::Integer, n::Integer) = svmzeros(Float64, m, n)

# Generate a mxn sparse vector matrix with floating point entries
function svmrand(m::Integer, n::Integer, d::Float64)
    SparseMatrixCD{Float64, Int}(m, n, [sprand(m, d) for c in 1 : n])
end

# Generate a mxn sparse vector matrix with boolean entries
function svmrandbool(m::Integer, n::Integer, d::Float64)
    SparseMatrixCD{Bool, Int}(m, n, [sprandbool(m, d) for c in 1 : n])
end

# Generate an identity mxn vector matrix with floating point entries
function svmeye(Tv::Type, m::Integer, n::Integer)
    svm = svmzeros(Tv, m, n)
    for i in 1 : min(m, n)
        setindex!(svm.svlist[i], one(Tv), i)
    end
    svm
end
svmeye(m::Integer, n::Integer) = svmeye(Float64, m, n)
