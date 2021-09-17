# DiffusionMap

## Introduction

Add package by opening the Julia package manager with "]" then type in REPL

```julia
add https://github.com/bastikusch/DiffusionMap.jl
```

This package is a concise implementation of the diffusion mapping method. It takes a given dataset in matrix form (here a standardized along the columns)
```julia
data = standardize(ZScoreTransform, randn(1000,20), dims=2)
```
and frames the diffusion map problem as follows.

```julia
dp = DiffusionProblem(data, kernel=InverseDistanceKernel(), laplace_type=RowNormalizedLaplacian(), threshold=size(data,1))
```
As a kernel for computing the adjacency matrix the following ones can be used

kernel | Description
------------ | -------------
InverseDistanceKernel() | (default) Computes the similarity of two vectors as the inverse of their euclidean distance.
GaussianKernel(\alpha) | Computes the similarity of two vectors with the gaussian kernel formula and parameter \alpha.
CustomKernel(func::Function) | Computes the similarity of two vectors by using a custom function that has two vectors as inputs and returns a scalar.

Supported types of Laplacian matrices are

kernel | Description
------------ | -------------
InverseDistanceKernel() | (default) Computes the similarity of two vectors as the inverse of their euclidean distance.
GaussianKernel(\alpha) | Computes the similarity of two vectors with the gaussian kernel formula and parameter \alpha.
CustomKernel(func::Function) | Computes the similarity of two vectors by using a custom function that has two vectors as inputs and returns a scalar.

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
