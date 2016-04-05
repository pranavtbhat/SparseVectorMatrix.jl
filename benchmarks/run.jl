using SparseVectorMatrix

const num_iters = 10
const scale = 10^3

println("Running benchmarks for GetIndex")
    include("getindex.jl")
println("----------------------------------------------------------------------------")

println("Running benchmarks for SetIndex")
    include("setindex.jl")
println("----------------------------------------------------------------------------")

println("Running benchmarks for Horizontal Concatenation")
    include("hcat.jl")
println("----------------------------------------------------------------------------")

println("Running benchmarks for Vertical Concatenation")
    include("vcat.jl")
println("----------------------------------------------------------------------------")

println("Running benchmarks for Grid Concatenation")
    include("hvcat.jl")
println("----------------------------------------------------------------------------")

println("Running benchmarks for Matrix Conversions")
    include("conversion.jl")
println("----------------------------------------------------------------------------")
