sm_arr  = [sprand(scale, scale, 0.1) for i in 1:9]
svm_arr = [svmrand(scale, scale, 0.1) for i in 1:9]

println("SparseMatrixCSC Concatenation:")
for i in 1:num_iters
    @time hvcat((3, 3, 3), sm_arr...);
end

println("SparseVectorMatrix Concatenation")
for i in 1:num_iters
    @time hvcat((3, 3, 3), svm_arr...);
end
