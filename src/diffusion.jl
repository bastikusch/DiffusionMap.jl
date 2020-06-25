## type diffusion map

struct Diffusionmap
    data::Matrix
    λ::Vector
    ϕ::Matrix
    kernel::Type
end

## diffusion map methodolgy

# function getAdjacency(k::T, data::Matrix, α::Float64=0.0) where T <: Kernel
#     l = size(data, 1)
#     A = zeros(l, l)
#     for i = 1:l
#         for j = i+1:l
#             A[i, j] = similarity(k, data[i,:], data[j,:])
#         end
#     end
#     A = A + transpose(A)
#     D = Diagonal(sum(A, dims=2)[:])
#     A_α = inv(D)^α * A * inv(D)^α
#     return A_α
# end
function getAdjacency(k::T, data::Matrix, α::Float64=0.0) where T <: Kernel
    l = size(data, 1)
    A = zeros(l, l)
    for i = 1:l
        for j = i+1:l
            A[i, j] = similarity(k, data[i,:], data[j,:])
        end
    end
    A = A + transpose(A)
    return A
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

function getLaplacian(A::Matrix, laplace::Symbol)
    D = Diagonal(sum(A, dims=2)[:])
    L = D - A
    mode = Dict(:Regular => L, :Normalized => inv(D) * L, :Symmetric => Symmetric(D^(-1/2) * L * D^(-1/2)))
    return mode[laplace]
end

function createDiffusionmap(k::T, data::Matrix; nextNeighbors::Int=0, α::Float64=0.0, laplace=:Normalized) where T <: Kernel
    initialAdjacency = getAdjacency(k, data, α)
    Adjacency = nextNeighbors > 0 ? thresholding!(initialAdjacency, nextNeighbors) : initialAdjacency
    Laplacian = getLaplacian(Adjacency, laplace)
    λ, ϕ = eigen(Laplacian)
    if (abs(λ[1]) > 10^(-10))
        λ0 = λ[1]
        @warn "λ0 = $λ0"
    end
    return Diffusionmap(data, λ, ϕ, typeof(k))
end
