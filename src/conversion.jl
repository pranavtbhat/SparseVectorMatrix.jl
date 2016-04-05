###
# This file contains conversions from vector-sparse matrices to other matrix types
###

import Base: full

# Full
full(A::SparseMatrixCD) = hcat(map(full, A.svlist)...)
