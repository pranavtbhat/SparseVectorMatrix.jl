export spmrand

function spmrand(m::Integer, n::Integer, d::Float64)
    SparseMatrixCD{Float64, Int}(m, n, [sprand(m, d) for c in 1:n])
end
