# DiffusionMap.jl
module DiffusionMap

# dependency
using LinearAlgebra, Statistics, StatsBase, Arpack, KrylovKit, SparseArrays

export
# kernels.jl
AbstractKernel, InverseDistance,
Gaussian, Cosine, similarity,

# diffusionTypes.jl
AbstractLaplacianMethod, GraphLaplacian,
CoifmanLaplacian, AbstractEigenSolver,
FullEigen, ArpackEigen, KrylovEigen,
DiffusionMap, DiffusionProblem,

# diffusionCalculation.jl
solve


include("kernels.jl")
include("diffusionTypes.jl")
include("diffusionCalculation.jl")
# include("utils.jl")

end # module
