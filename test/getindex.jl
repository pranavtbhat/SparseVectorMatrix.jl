m = 5
n = 10

svlist = SparseVector{Int, Int}[spzeros(Int, m) for i in 1 : n]
svm = SparseMatrixCD{Int, Int}(m, n, svlist)
