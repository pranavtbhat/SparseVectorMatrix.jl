export spmrand

function spmrand(m::Integer, n::Integer, d::Float64)
    SVMatrix(m, n, SparseVector[sprand(m, d) for c in 1:n])
end
