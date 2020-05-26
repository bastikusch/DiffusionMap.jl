# DiffusionMap.jl
module DiffusionMap

using Plots, LinearAlgebra, Statistics, SparseArrays

export Kernel, InverseDistance, Gaussian, Linear, Polynomial, Correlation, LaplaceKernel, Spearman, Diffusionmap
export standardize!, similarity, thresholding!, simpleLaplacian, normaliedLaplacian, createDiffusionmap, eigenVisualize, nonNormalizedLaplacian

## types

abstract type Kernel end

struct InverseDistance <: Kernel end

struct Gaussian <: Kernel
    σ::Float64
end

Gaussian() = Gaussian(1.0)

struct Linear <: Kernel end

struct Spearman <: Kernel end

struct Polynomial <: Kernel
    r::Float64
    n::Int
end

Polynomial() = Polynomial(0.0, 1)

struct Correlation <: Kernel end

struct LaplaceKernel <: Kernel
    α::Float64
end

LaplaceKernel() = LaplaceKernel(1.0)

struct Diffusionmap
    data::Matrix
    λ::Vector
    ϕ::Matrix
end

## data handling

function standardize!(data::Matrix)
    for i = 1:size(data, 2)
        data[:,i] = (data[:,i] .- mean(data[:,i])) / sqrt(var(data[:,i]))
    end
end

## functions

function similarity(k::InverseDistance, x::Vector, y::Vector)
    sim = 1 / norm(x .- y)^2
    return isinf(sim) ? 0 : sim
end

function similarity(k::Gaussian, x::Vector, y::Vector)
    sim = exp(-(norm(x .- y)^2) / (2 * k.σ ^2))
    return (x == y) ? 0 : sim
end

function similarity(k::LaplaceKernel, x::Vector, y::Vector)
    sim = exp(-k.α * (norm(x .- y)))
    return (x == y) ? 0 : sim
end

function similarity(k::Correlation, x::Vector, y::Vector)
    sim = abs(cor(x,y))
    return (x == y) ? 0 : sim
end

function similarity(k::Linear, x::Vector, y::Vector)
    sim = dot(x, y) / (norm(x) * norm(y))
    return (x == y) ? 0 : sim
end

function similarity(k::Polynomial, x::Vector, y::Vector)
    sim = (dot(x, y) / (norm(x) * norm(y)) + k.r)^k.n
    return (x == y) ? 0 : sim
end

function similarity(k::Spearman, x::Vector, y::Vector)
    sim = abs(cov(sortperm(x), sortperm(y)) / (std(sortperm(x)) * std(sortperm(y))))
    return (x == y) ? 0 : sim
end

function similarity(k::T, A::Matrix) where T <: Kernel
    l = size(A, 1)
    sim = zeros(l, l)
    for i = 1:l
        for j = i:l
            sim[i, j] = similarity(k, A[i,:], A[j,:])
        end
    end
    sim = sim + transpose(sim)
    return sim
end

function thresholding!(sim::Matrix, nextNeighbors::Int)
    le = size(sim,1)
    if (nextNeighbors != size(sim,1))
        for i = 1:le
            sortRow = sort(sim[i,:], rev = true)
            nth_Max = sortRow[nextNeighbors]
            sim[i,:] = map(x -> x < nth_Max ? 0 : x, sim[i,:])
        end
    end
    l = Symmetric(sim, :L)
    u = Symmetric(sim, :U)
    sim[:] = max.(l[:], u[:])
    reshape(sim, (le,le))
end

function normalizedLaplacian(A::Matrix)
    s = sum(A; dims=2)
    D = convert(SparseMatrixCSC, spdiagm(0 => s[:]))
    return Symmetric(I - D^(-1/2) * A * D^(-1/2))
end

function simpleLaplacian(A::Matrix)
    s = sum(A; dims=2)
    D = convert(SparseMatrixCSC, spdiagm(0 => s[:]))
    return D - A
end

function createDiffusionmap(k::T, data::Matrix, nextNeighbors=size(data,1), normalized=true) where T <: Kernel
    Adjacency = thresholding!(similarity(k, data), nextNeighbors)
    Laplace = normalized ? normalizedLaplacian(Adjacency) : simpleLaplacian(Adjacency)
    λ, ϕ = eigen(Laplace)
    if (abs(λ[1]) > 10^(-10))
        @warn "First eigen value of diffusion map not 0"
    end
    return Diffusionmap(data, λ, ϕ)
end

## visualisation

function eigenVisualize(difmap::Diffusionmap, markersize=4)
    ϕ = real.(difmap.ϕ)
    s1 = scatter(ϕ[:,2], ϕ[:,3], title="eigen vector 1 / 2", label="", ms=markersize)
    s2 = scatter(ϕ[:,2], ϕ[:,4], title="eigen vector 1 / 3", label="", ms=markersize)
    s3 = scatter(ϕ[:,2], ϕ[:,5], title="eigen vector 1 / 4", label="", ms=markersize)
    s4 = scatter(ϕ[:,2], ϕ[:,6], title="eigen vector 1 / 5", label="", ms=markersize)
    p = plot(s1,s2,s3,s4, layout = (2,2))
    return p
end

function eigenVisualize(difmap::Diffusionmap, color_z::Vector, markersize=4)
    ϕ = real.(difmap.ϕ)
    s1 = scatter(ϕ[:,2], ϕ[:,3], marker_z=color_z, title="eigen vector 1 / 2", label="", ms=markersize,color=:thermal)
    s2 = scatter(ϕ[:,2], ϕ[:,4], marker_z=color_z, title="eigen vector 1 / 3", label="", ms=markersize,color=:thermal)
    s3 = scatter(ϕ[:,2], ϕ[:,5], marker_z=color_z, title="eigen vector 1 / 4", label="", ms=markersize,color=:thermal)
    s4 = scatter(ϕ[:,2], ϕ[:,6], marker_z=color_z, title="eigen vector 1 / 5", label="", ms=markersize,color=:thermal)
    p = plot(s1,s2,s3,s4, layout = (2,2))
    return p
end

end # module
