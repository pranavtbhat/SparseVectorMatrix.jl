sm_arr  = [sprand(scale, scale, 0.1) for i in 1:num_iters]
svm_arr = [svmrand(scale, scale, 0.1) for i in 1:num_iters]

println("SparseMatrixCSC Concatenation:")
for i in 1:num_iters
    @time vcat(sm_arr...);
end

println("SparseVectorMatrix Concatenation")
for i in 1:num_iters
    @time vcat(svm_arr...);
end
