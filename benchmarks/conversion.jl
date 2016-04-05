sm = sprand(scale, scale, 0.1)
svm = svmrand(scale, scale, 0.1)
arr = full(sprand(scale, scale, 0.1))


println("Conversion to Full array from SparseMatrixCSC")
for i in 1:num_iters
    @time full(sm);
end

println("Conversion to Full array from SparseMatrixCD")
for i in 1:num_iters
    @time full(svm);
end

println("Conversion from Full array to SparseMatrixCSC")
for i in 1:num_iters
    @time sparse(arr);
end

println("Conversion from Full array to SparseMatrixCD")
for i in 1:num_iters
    @time vsparse(arr);
end

println("Conversion from SparseMatrixCSC to SparseMatrixCD")
for i in 1:num_iters
    @time SparseMatrixCD(sm);
end
