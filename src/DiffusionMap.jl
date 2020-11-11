# DiffusionMap.jl
module DiffusionMap

# dependency
using Plots, LinearAlgebra, Statistics

export
# kernels.jl
Kernel, DiffusionMap, DiffusionProblem,
InverseDistance, Gaussian, Linear,
Polynomial, LaplaceKernel, similarity,

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
