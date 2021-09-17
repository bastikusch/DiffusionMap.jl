# DiffusionMap

## Introduction

Add package by opening the Julia package manager with "]" then type in REPL

```julia
add https://github.com/bastikusch/DiffusionMap.jl
```

This package is a concise implementation of the diffusion mapping method. It takes a given dataset in matrix form
```julia
data = rand(100,10)
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
RowNormalizedLaplacian() | (default)<img src="https://render.githubusercontent.com/render/math?math=L=D^{-1}*(D-A)">
SymmetricLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1/2}*(D-A)*D^{-1/2}">
Adjacency() | <img src="https://render.githubusercontent.com/render/math?math=L=A">
NormalizedAdjacency | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1}*A">

As a threshold, any integer betwen 0 and n (n being the number of data rows) can be chosen, the default is n

Having framed the diffusion map problem, one can perform the eigen decomposition of the Laplacian with

```julia
evals, evecs = solve(dm, eigensolver=FullEigen())
```
eigensolver | Description
------------ | -------------
FullEigen() | Uses the method `eigen(L)` from the package `LinearAlgebra.jl`
ArpackEigen(n_first) | Uses the method `eigs(L)` from the package `Arpack.jl` to get the n first eigenvectors
KrylovEigen(n_first) | Uses the method `eigsolve(L)` from the package `KrylovKit.jl` to get the n first eigenvectors

## Example

```julia
using StatsBase, Plots, DiffusionMap

# put any data matrix you want here
data = standardize(ZScoreTransform, rand(1000,20), dims=2)

# Frame diffusion problem
dm = Diffusionmap(data, kernel = InverseDistanceKernel(), laplace_type = RowNormalizedLaplacian(), threshold=5)

# Perform eigen decomposition
evals, evecs = solve(dm, eigensolver=FullEigen());

# Scatter plot eigen vectors, cloured by one arbitrary data column
scatter(evecs[2], evecs[3], marker_z=data[:,1], label="")

```
