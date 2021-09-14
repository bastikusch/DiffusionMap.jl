# DiffusionMap

Add package by typing "]" in REPL, next type: add https://github.com/bastikusch/DiffusionMap.jl

```julia
add https://github.com/bastikusch/DiffusionMap.jl
```

using StatsBase, Plots
using DiffusionMap

# put any data matrix you want here
# standardize(ZScoreTransform, data, dims=2) standardizes data along columns
data = standardize(ZScoreTransform, randn(100,20), dims=2)

# long form
nextNeighbors = 5
kernel = InverseDistanceKernel()
laplaceMethod = NormalizedGraphLaplacian() # alternatively NormalizedAdjacencyLaplacian()
eigenMethod = FullEigen() # alternatively KrylovEigen(numberOfEigenVectors), or ArpackEigen(numberOfEigenVectors)

dp = DiffusionProblem(data, kernel, laplaceMethod, nextNeighbors)
dm = solve(dp, eigenMethod)

dp2 = DiffusionProblem(data, (x,y) -> x*y, nextNeighbors)
dm2 = solve(dp2, eigenMethod)

#short form
dm = solve(DiffusionProblem(data, InverseDistance(), GraphLaplacian(), 5), FullEigen())


# scatter plot
scatter(dm.ϕ[2], dm.ϕ[3], marker_z=data[:,1], label="")
