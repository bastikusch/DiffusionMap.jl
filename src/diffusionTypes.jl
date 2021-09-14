## types for  diffusionmap calculation

abstract type AbstractLaplacianMethod end

struct NormalizedGraphLaplacian <: AbstractLaplacianMethod end

function Base.show(io::IO, ::MIME"text/plain", l::NormalizedGraphLaplacian)
    println(io, "NormalizedGraphLaplacian")  
end

struct NormalizedAdjacencyLaplacian <: AbstractLaplacianMethod end

function Base.show(io::IO, ::MIME"text/plain", l::NormalizedAdjacencyLaplacian)
    println(io, "NormalizedAdjacencyLaplacian")   
end

abstract type AbstractEigenSolver end

struct FullEigen <: AbstractEigenSolver end

function Base.show(io::IO, ::MIME"text/plain", l::FullEigen)
    println(io, "FullEigen")   
end

struct ArpackEigen <: AbstractEigenSolver
    n_first::Int
end

function Base.show(io::IO, ::MIME"text/plain", l::ArpackEigen)
    println(io, "ArpackEigen")   
end

struct KrylovEigen <: AbstractEigenSolver
    n_first::Int
end

function Base.show(io::IO, ::MIME"text/plain", l::KrylovEigen)
    println(io, "KrylovEigen")   
end

struct DiffusionProblem
    data::Matrix
    kernel::AbstractKernel
    laplaceMethod::AbstractLaplacianMethod
    threshold::Int64
end

DiffusionProblem(data, kernel) = DiffusionProblem(data, kernel, NormalizedGraphLaplacian(), 0)
DiffusionProblem(data, kernel, threshold) = DiffusionProblem(data, kernel, NormalizedGraphLaplacian(), threshold)

DiffusionProblem(data, func::Function) = DiffusionProblem(data, CustomKernel(func), 0)
DiffusionProblem(data, func::Function, threshold) = DiffusionProblem(data, CustomKernel(func), threshold)
DiffusionProblem(data, func::Function, laplaceMethod, threshold) = DiffusionProblem(data, CustomKernel(func), laplaceMethod, threshold)

function Base.show(io::IO, ::MIME"text/plain", dp::DiffusionProblem)
    println(io, "DiffusionProblem")
    println(io, "Kernel = $(typeof(dp.kernel))")
    println(io, "Laplace method = $(dp.laplaceMethod)")
    println(io, "Threshold = $(dp.threshold)")   
end

struct Diffusionmap
    λ::Vector
    ϕ::Vector
    laplaceMethod::AbstractLaplacianMethod
end

function eigenvecs(dm::Diffusionmap)
    return dm.ϕ
end

function eigenvals(dm::Diffusionmap)
    return dm.λ
end

function Base.show(io::IO, ::MIME"text/plain", dm::Diffusionmap)
    println(io, "Diffusionmap ($(length(dm.ϕ)))")
    println(io, "Type: $(dm.laplaceMethod)")
    dim = min(5,length(dm.λ))
    println(io, "First $(dim) Eigen Values:")
    for i in 1:dim
        println(io, dm.λ[i])
    end
end
