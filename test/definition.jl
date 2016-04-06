m = 5
n = 10

svlist_int = SparseVector{Int, Int}[spzeros(Int, m) for i in 1 : n]
svlist_float = SparseVector{Float64, Int}[spzeros(Float64, m) for i in 1 : n]

# Constructors
svm_int = SparseMatrixCD{Int, Int}(m, n, svlist_int)
svm_float = SparseMatrixCD{Float64, Int}(m, n, svlist_float)

# Size
@test size(svm_int) == (m, n)
@test size(svm_int, 1) == m
@test size(svm_int, 2) == n

# NNZ
@test nnz(svm_int) == mapreduce(nnz, +, 0, svlist_int)

# Copy
@test copy(svm_int) == svm_int
