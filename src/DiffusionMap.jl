# DiffusionMap.jl
module DiffusionMap

# dependency
using Plots, LinearAlgebra, Statistics, StatsBase

export
# kernels.jl
Kernel, DiffusionMap, DiffusionProblem,
InverseDistance, Gaussian, KLKernel,
Correlation, InformationCorrelation, similarity,

# diffusion.jl
thresholding!, calculateAdjacency,
calculateLaplacian, calculateDiffusionMap,

# utils.jl
standardize!, visualize,
multiVisualize, addTitles!

include("kernels.jl")
include("diffusion.jl")
include("utils.jl")

end # module
