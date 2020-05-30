## type diffusion map

struct Diffusionmap
    data::Matrix
    λ::Vector
    ϕ::Matrix
end

## diffusion map methodolgy

function getAdjacency(k::T, data::Matrix) where T <: Kernel
    l = size(data, 1)
    A = zeros(l, l)
    for i = 1:l
        for j = i+1:l
            A[i, j] = (i==j) ? 0.0 : similarity(k, data[i,:], data[j,:])
        end
    end
    A = A + transpose(A)
    return A
end

function getAdjacency(k::T, data::Matrix, α::Float64) where T <: Kernel
    l = size(data, 1)
    A = zeros(l, l)
    for i = 1:l
        for j = i+1:l
            A[i, j] = (i==j) ? 0.0 : similarity(k, data[i,:], data[j,:])
        end
    end
    A = A + transpose(A)
    D = Diagonal(sum(A, dims=2)[:])
    A_α = inv(D)^α * A * inv(D)^α
    return A_α
end

function thresholding!(A::Matrix, nextNeighbors::Int)
    len = size(A,1)
    for i = 1:len
        cutOff = sort(A[i,:], rev = true)[nextNeighbors]
        A[i,:] = map(x -> x < cutOff ? 0 : x, A[i,:])
    end
    l = Symmetric(A, :L)
    u = Symmetric(A, :U)
    A[:] = max.(l[:], u[:])
    reshape(A, (len,len))
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

function createDiffusionmap(k::T, data::Matrix; α::Float64=0.0, normalized::Bool=true) where T <: Kernel
    Adjacency = (α == 0.0) ? getAdjacency(k, data) : getAdjacency(k, data, α)
    Laplace = normalized ? normalizedLaplacian(Adjacency) : simpleLaplacian(Adjacency)
    λ, ϕ = eigen(Laplace)
    if (abs(λ[1]) > 10^(-10))
        λ0 = λ[1]
        @warn "λ0 = $λ0"
    end
    return Diffusionmap(data, λ, ϕ)
end

function createDiffusionmap(k::T, data::Matrix, nextNeighbors::Int; α::Float64=0.0, normalized::Bool=true) where T <: Kernel
    initialAdjacency = (α == 0.0) ? getAdjacency(k, data) : getAdjacency(k, data, α)
    Adjacency = thresholding!(initialAdjacency, nextNeighbors)
    Laplace = normalized ? normalizedLaplacian(Adjacency) : simpleLaplacian(Adjacency)
    λ, ϕ = eigen(Laplace)
    if (abs(λ[1]) > 10^(-10))
        λ0 = λ[1]
        @warn "λ0 = $λ0"
    end
    return Diffusionmap(data, λ, ϕ)
end
