# DiffusionMap.jl
module DiffusionMap

using Plots, LinearAlgebra, Statistics, SparseArrays

export
# kernels.jl
Kernel, Diffusionmap,
InverseDistance, Gaussian,
Linear, Polynomial, Spearman,
Correlation, LaplaceKernel,
similarity,

# diffusion.jl
thresholding!, getAdjacency,
simpleLaplacian, normalizedLaplacian,
createDiffusionmap,

# utils.jl
standardize!, visualize

include("kernels.jl")
include("diffusion.jl")
include("utils.jl")


end # module
