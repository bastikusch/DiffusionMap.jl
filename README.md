# DiffusionMap.jl

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
dp = DiffusionProblem(data; kernel, laplace_type, threshold)
```
### Kernel
As a kernel for computing the adjacency matrix the following ones can be used

kernel | Description | Default
------------ | ------------- | -------------
InverseDistanceKernel() | Computes the similarity of two vectors as the inverse of their euclidean distance. | :heavy_check_mark:
GaussianKernel(<img src="https://render.githubusercontent.com/render/math?math=\alpha">) | Computes the similarity of two vectors with the gaussian kernel formula and parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha">. | :x:
CustomKernel(func::Function) | Computes the similarity of two vectors by using a custom function that has two vectors as inputs and returns a scalar. | :x:

### Laplacian types
Supported types of Laplacian matrices are

laplace_type | Description | Default
------------ | ------------- | -------------
RegularLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D-A"> | :heavy_check_mark:
RowNormalizedLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1}*(D-A)"> | :x:
SymmetricLaplacian() | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1/2}*(D-A)*D^{-1/2}"> | :x:
Adjacency() | <img src="https://render.githubusercontent.com/render/math?math=L=A"> | :x:
NormalizedAdjacency() | <img src="https://render.githubusercontent.com/render/math?math=L=D^{-1}*A"> | :x:

### Next neighbor threshold
As a next neighbor threshold, any integer betwen 0 and n (n being the number of data rows) can be chosen (default is n). Determines the number of next neighbors to be kept in the adjacency matrix, which controls the amount of locality within the data set.

### solver method
Having framed the diffusion map problem, one can perform the eigen decomposition of the Laplacian with the 'solve()'-method.

```julia
evals, evecs = solve(dm, eigensolver=FullEigen())
```
It returns the eigen values and the eigen vecors, both in vector format and sorted for their impact and the underlying embedding. This means that graph laplacians are sorted by their smallest real part and adjacency type laplacians (Adjacency() and NormalizedAdjacency()) by their largest real part. Both types have the constant eigen vector eigen value pair as their first vector entry, so analyses should keep in mind to start with vector entries 2, as this is the first relevant eigen vector, given that the local network is connected.

### Eigen solver
Possible eigen solvers contain full decomposition, as well as faster partial decompositions.
eigensolver | Description | Default
------------ | ------------- | -------------
FullEigen()| Uses the method `eigen(L)` from the package `LinearAlgebra.jl` | :heavy_check_mark:
ArpackEigen(n_first) | Uses the method `eigs(L)` from the package `Arpack.jl` to get the n first eigenvectors | :x:
KrylovEigen(n_first) | Uses the method `eigsolve(L)` from the package `KrylovKit.jl` to get the n first eigenvectors | :x:

## Example

```julia
using StatsBase, Plots, DiffusionMap

# put any data matrix you want here
data = standardize(ZScoreTransform, rand(1000,20), dims=2)

# Frame diffusion problem
k = InverseDistanceKernel() # kernel
laplace = RowNormalizedLaplacian() # laplace_type
nn = 5 # next neighbors
dm = Diffusionmap(data, kernel = k, laplace_type = laplace, threshold=nn)

# Perform eigen decomposition
evals, evecs = solve(dm, eigensolver=FullEigen());

# Scatter plot eigen vectors, cloured by one arbitrary data column
scatter(evecs[2], evecs[3], marker_z=data[:,1], label="")

```
![Alt text](diffusionmap_example.png?raw=true)
