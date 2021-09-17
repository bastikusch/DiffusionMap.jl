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
GaussianKernel(<img src="https://render.githubusercontent.com/render/math?math=\alpha">.) | Computes the similarity of two vectors with the gaussian kernel formula and parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha">.
CustomKernel(func::Function) | Computes the similarity of two vectors by using a custom function that has two vectors as inputs and returns a scalar.

Supported types of Laplacian matrices are

laplace_type | Description
------------ | -------------
RegularLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D-A">
RowNormalizedLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1}*(D-A)">
SymmetricLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1/2}*(D-A)*D^{-1/2}">
Adjacency() | <img src="https://render.githubusercontent.com/render/math?math=L=A">
NormalizedAdjacency | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1}*A">

As a threshold, any integer betwen 0 and n (n being the number of data rows) can be chosen, the default is n.
Having framed the diffusion map problem, one can solve it with

Secondly, specify the method for the eigen decomposition (full eigen decompostion with `FullEigen()`, partial decomposition of the n first eigen vectors with either `ArpackEigen(n)`, or `KrylovEigen(n)`) and solve the `DiffusionProblem`.

```julia
eigenMethod = FullEigen()
dm = solve(dp, eigenMethod)
```
kernel | Description
------------ | -------------
RegularLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D-A">
RowNormalizedLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1}*(D-A)">
SymmetricLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1/2}*(D-A)*D^{-1/2}">
Adjacency() | <img src="https://render.githubusercontent.com/render/math?math=L=A">
NormalizedAdjacency | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1}*A">


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
