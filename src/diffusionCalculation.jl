
function get_adjacency(data::Matrix, kernel::T) where T <: AbstractKernel
    n = size(data)[1]
    A = zeros(n, n)
    Threads.@threads for j in 1:n
        @views for i in j+1:n
            A[i,j] = A[j,i] = similarity(kernel, data[i, :], data[j, :])
        end
    end
    return A
end

# function thresholding!(A::Matrix, threshold::Int)
#     if threshold > 0
#         n = size(A,1)
#         cutOff = sort(A,dims=1,rev = true)[threshold,:]
#         Threads.@threads for j in 1:n
#                 @views for i in j:n
#                         A[i,j] = A[j,i] = (A[i,j] < cutOff[i]) && (A[i,j] < cutOff[j]) ? 0.0 : A[i,j]
#                 end
#         end
#     end
# end

function thresholding!(A::Matrix, threshold::Int)
    if threshold > 0
        n = size(A,1)
        cutOff = rand(size(A,1))
        Threads.@threads for i in 1:n
            cutOff[i] = sort(A[:,i], rev=true)[threshold]
        end
        Threads.@threads for j in 1:n
            @views for i in j:n
                    A[i,j] = A[j,i] = (A[i,j] < cutOff[i]) && (A[i,j] < cutOff[j]) ? 0.0 : A[i,j]
            end
        end
    end
end

function get_laplacian(A::Matrix, method::NormalizedGraphLaplacian)
    D = Diagonal(sum(A, dims=2)[:])
    L = D - A
    return inv(D) * L
end


function get_laplacian(A::Matrix, method::NormalizedAdjacencyLaplacian)
    D = Diagonal(sum(A, dims=2)[:])
    return inv(D) * A
end

function eigen_sort(method::NormalizedGraphLaplacian)
    return :SR
end

function eigen_sort(method::NormalizedAdjacencyLaplacian)
    return :LR
end

# eigen decomposition, short cut through all diffusion steps
function solve(dP::DiffusionProblem, eigenSolver::T) where T <: AbstractEigenSolver
    A = get_adjacency(dP.data, dP.kernel)
    thresholding!(A, dP.threshold)
    L = get_laplacian(A, dP.laplaceMethod)
    sort_for = eigen_sort(dP.laplaceMethod)
    λ, ϕ = eigen_solve(L, eigenSolver, sort_for)
    return Diffusionmap(λ, ϕ, dP.laplaceMethod)
end

function eigen_solve(L::Matrix, eigenSolver::FullEigen, sort_for::Symbol)
    λ, ϕ = eigen(L)
    iter = Dict(:SR => 1:size(L,1), :LR => size(L,1):-1:1)
    ϕ_vec = [ϕ[:,i] for i in iter[sort_for]]
    return λ, ϕ_vec
end

function eigen_solve(L::Matrix, eigenSolver::ArpackEigen, sort_for::Symbol)
    λ, ϕ = eigs(sparse(L), nev=eigenSolver.n_first, which=sort_for)
    ϕ_vec = [ϕ[:,i] for i in 1:length(λ)]
    return real.(λ), real.(ϕ_vec)
end

function eigen_solve(L::Matrix, eigenSolver::KrylovEigen, sort_for::Symbol)
    λ, ϕ = eigsolve(sparse(L), eigenSolver.n_first, sort_for)
    return real.(λ), real.(ϕ)
end


