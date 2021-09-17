# diffusion map calculation

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

function thresholding!(A::Matrix, threshold::Int)
    if threshold < size(A,1)
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

function get_laplacian(A::Matrix, method::RowNormalizedLaplacian)
    D = Diagonal(sum(A, dims=2)[:])
    L = D - A
    return inv(D) * L
end

function get_laplacian(A::Matrix, method::NormalizedAdjacency)
    D = Diagonal(sum(A, dims=2)[:])
    return inv(D) * A
end

function get_laplacian(A::Matrix, method::Adjacency)
    return A
end

function get_laplacian(A::Matrix, method::SymmetricLaplacian)
    D = Diagonal(sum(A, dims=2)[:])
    L = D - A
    return D^(-1/2) * L * D^(-1/2)
end

function get_laplacian(A::Matrix, method::RegularLaplacian)
    D = Diagonal(sum(A, dims=2)[:])
    return D - A
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

# solve the whole Diffusionmap
function solve(dm::Diffusionmap, eigenSolver::T) where T <: AbstractEigenSolver
    A = get_adjacency(dm.data, dm.kernel)
    thresholding!(A, dm.threshold)
    L = get_laplacian(A, dm.laplace_type)
    sort_for = typeof(dm.laplace_type)==Adjacency || typeof(dm.laplaceMethod)==NormalizedAdjacency ? :LR : :SR
    λ, ϕ = eigen_solve(L, eigenSolver, sort_for)
    return λ, ϕ
end


