# DiffusionMap.jl
module DiffusionMap

# dependency
using Plots, LinearAlgebra, Statistics, StatsBase, InformationMeasures

export
# kernels.jl
AbstractKernel, DiffusionMap, DiffusionProblem,
InverseDistance, Gaussian, MutualInformation,
Correlation, similarity, InformationCorrelation,

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
