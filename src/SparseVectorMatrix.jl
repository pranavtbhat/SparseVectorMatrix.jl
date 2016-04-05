module SparseVectorMatrix

# SparseMatrixCD definition
include("definition.jl")

# GetIndex
include("getindex.jl")

# SetIndex
include("setindex.jl")

# Conversions
include("conversion.jl")

# Concatenations
include("concatenation.jl")

# Random Generation
include("generate.jl")

end # module
