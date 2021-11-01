# DiffusionMap.jl
module DiffusionMap

# dependency
using LinearAlgebra, Arpack, KrylovKit, SparseArrays

export
# kernels.jl
AbstractKernel, InverseDistanceKernel,
GaussianKernel, CosineKernel, CustomKernel,
FlowTensionKernel,

# laplacians.jl
AbstractLaplacian, RowNormalizedLaplacian,
Adjacency, NormalizedAdjacency, 
SymmetricLaplacian, RegularLaplacian, 

# eigensolvers.jl
AbstractEigenSolver, FullEigen,
ArpackEigen, KrylovEigen,

# diffusionTypes.jl

Diffusionmap,

# diffusionCalculation.jl
get_laplacian, get_adjacency, thresholding!, solve


include("kernels.jl")
include("laplacians.jl")
include("eigensolvers.jl")
include("diffusionTypes.jl")
include("diffusionCalculation.jl")
# include("utils.jl")

end # module
