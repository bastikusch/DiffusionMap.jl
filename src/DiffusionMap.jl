# DiffusionMap.jl
module DiffusionMap

# dependency
using LinearAlgebra, Statistics, StatsBase, Arpack, KrylovKit, SparseArrays

export
# kernels.jl
AbstractKernel, InverseDistanceKernel,
GaussianKernel, CosineKernel, CustomKernel, similarity,

# diffusionTypes.jl
AbstractLaplacianMethod, NormalizedGraphLaplacian,
NormalizedAdjacencyLaplacian, AbstractEigenSolver,
FullEigen, ArpackEigen, KrylovEigen,
DiffusionMap, DiffusionProblem, eigenvals, 
eigenvecs,

# diffusionCalculation.jl
solve


include("kernels.jl")
include("diffusionTypes.jl")
include("diffusionCalculation.jl")
# include("utils.jl")

end # module
