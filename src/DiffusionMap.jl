# DiffusionMap.jl
module DiffusionMap

# dependency
using Plots, LinearAlgebra, Statistics, StatsBase, InformationMeasures

export
# kernels.jl
Kernel, DiffusionMap, DiffusionProblem,
InverseDistance, Gaussian, MutualInformation,
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
