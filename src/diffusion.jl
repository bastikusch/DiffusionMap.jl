## types for  diffusionmap calculation

struct DiffusionProblem
    data::Matrix
    kernel::Kernel
    threshold::Int64
    laplaceMathod::Symbol
end

DiffusionProblem(data, kernel) = DiffusionProblem(data, kernel, 0, :Normalized)
DiffusionProblem(data, kernel, threshold) = DiffusionProblem(data, kernel, threshold, :Normalized)

function Base.show(io::IO, ::MIME"text/plain", dp::DiffusionProblem)
    println(io, "DiffusionProblem")
    println(io, "Kernel = $(typeof(dp.kernel))")
    println(io, "Threshold = $(dp.threshold)")
    println(io, "Laplace method = $(dp.laplaceMathod)")    
end

struct Diffusionmap
    λ::Vector
    ϕ::Matrix
end

function Base.show(io::IO, ::MIME"text/plain", dm::Diffusionmap)
    println(io, "Diffusionmap $(size(dm.ϕ))")
    println(io, "First 10 Eigen Values:")
    for i in 1:10
        println(io, dm.λ[i])
    end
end

## diffusion map functions

# calculation Adjacency matrix
function calculateAdjacency(dP::DiffusionProblem)
    l = size(dP.data, 1)
    A = zeros(l, l)
    for i = 1:l
        for j = i+1:l
            A[i, j] = similarity(dP.kernel, dP.data[i,:], dP.data[j,:])
        end
    end
    A = A + transpose(A)
    return A
end

# thresholding to control local connectance
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

# calculate Laplacian matrix with dictonary controlled type of Laplacian (maybe add multiDisp with (matrix, Symbol))
function calculateLaplacian(dP::DiffusionProblem)
    A = calculateAdjacency(dP::DiffusionProblem)
    A = dP.threshold > 0 ? thresholding!(A, dP.threshold) : A
    D = Diagonal(sum(A, dims=2)[:])
    L = D - A
    Laplacian = Dict(:Regular => L, :Normalized => inv(D) * L, :Symmetric => Symmetric(D^(-1/2) * L * D^(-1/2)))
    return Laplacian[dP.laplaceMathod]
end

# eigen decomposition, short cut through all diffusion steps
function calculateDiffusionMap(dP::DiffusionProblem)
    Laplacian = calculateLaplacian(dP)
    λ, ϕ = eigen(Laplacian)
    abs(λ[1]) > 10^(-12) ? (@warn "First eigen value not 0: λ0 = $(λ[1])") : ()
    typeof(λ[2]) == Complex{Float64} ? (@warn "Complex eigenvalues") : ()
    return Diffusionmap(λ, ϕ)
end

