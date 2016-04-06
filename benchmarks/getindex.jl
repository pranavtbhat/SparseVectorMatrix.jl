sm = sprand(scale, scale, 0.1)
svm = svmrand(scale, scale, 0.1)

println("Unit Indexing on SparseMatrixCSC")
for i in 1:num_iters
    @time sm[rand(1:scale), rand(1:scale)];
end

println("Unit Indexing on SparseVectorMatrix")
for i in 1:num_iters
    @time svm[rand(1:scale), rand(1:scale)];
end

println("Column Indexing on SparseMatrixCSC")
for i in 1:num_iters
    @time sm[:, rand(1:scale)];
end

println("Column Indexing on SparseVectorMatrix")
for i in 1:num_iters
    @time svm[:, rand(1:scale)];
end

println("Row Indexing on SparseMatrixCSC")
for i in 1:num_iters
    @time sm[rand(1:scale), :];
end

println("Row Indexing on SparseVectorMatrix")
for i in 1:num_iters
    @time svm[rand(1:scale), :];
end
