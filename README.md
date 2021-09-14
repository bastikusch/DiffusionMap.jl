# DiffusionMap

Add package by typing "]" in REPL, then write

```julia
] add https://github.com/bastikusch/DiffusionMap.jl
```

## Example

```julia
using StatsBase, Plots, DiffusionMap
```

For a given dataset in matrix form (here standardized along the columns)
```julia
data = standardize(ZScoreTransform, randn(1000,20), dims=2)
```
one can perform the diffusion mapping method in two steps. First, formulate the specified `DiffusionProblem` by choosing a kernel, a representation for the laplace method and the number of nextneighbors used in the algorithm.

```julia
nextNeighbors = 5
kernel = InverseDistanceKernel() # alternatively GaussianKernel(alpha)
laplaceMethod = NormalizedGraphLaplacian() # alternatively NormalizedAdjacencyLaplacian()

dp = DiffusionProblem(data, kernel, laplaceMethod, nextNeighbors)
```

As an alternative to the given kernel classes, one can use a function with two variables as the kernel. The functions inputs should be two vectors and give back a scalar. Be careful to satisfy the kernel properties (positive semi-definiteness, symmetry,...). The example here is the inner product of two vectors.
```julia
kernel_function = (x,y) -> x'*y
dp_func = DiffusionProblem(data, kernel_function, laplaceMethod, nextNeighbors)
```

Secondly, specify the method for the eigen decomposition (full eigen decompostion with `FullEigen()`, partial decomposition of the n first eigen vectors with either `ArpackEigen(n)`, or `KrylovEigen(n)`) and solve the `DiffusionProblem`.

```julia
eigenMethod = FullEigen()
dm = solve(dp, eigenMethod)
```
The fields of the resulting `Diffusionmap` struct are a vector of eigenvalues, accessible by

```julia
eval = eigenvals(dm)
```
as well as the vector of eigenvectors,
```julia
evec = eigenvecs(dm)
```

Scatter plotting eigenvectors and coloring them by the data points value in a given underlying property (here each mapped data point´s first dimension) can be done as follows.
```julia
scatter(ev[2], ev[3], marker_z=data[:,1], label="")
```
