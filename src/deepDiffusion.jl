
## experimental deep Diffusion functions

# one step deep diffusion that gives back transpose of weighted data + diffusion map
function stepDiffusion(data::Matrix, k::T, nextNeighbors::Int=0, α::Float64=0.0) where T <: Kernel
    difmap = createDiffusionmap(k, data, nextNeighbors=nextNeighbors, α=α, laplace=:Symmetric)
    eigenWeight = difmap.λ[2] ./ difmap.λ
    eigenWeight[1] = 0.0
    newData = (transpose(difmap.ϕ) .* eigenWeight) * data
    return convert(Matrix, transpose(newData)), difmap
end

# chaining of one step dee diffusions
function deepDiffusion(k::T, data::Matrix; nextNeighbors::Tuple=(0,0), iter::Int=10, α::Float64=0.0) where T <: Kernel
    difmapArray1 = []
    difmapArray2 = []
    newData = copy(data)
    for i = 1:iter
        newData, difmap = stepDiffusion(newData, k, nextNeighbors[1], α)
        push!(difmapArray1, difmap)
        newData, difmap = stepDiffusion(newData, k, nextNeighbors[2], α)
        push!(difmapArray2, difmap)
    end
    return difmapArray1, difmapArray2
end

# evolution of eigenvalues for given array of diffusion maps
function eigenEvolution(difmapVector, nVec::Int)
    ev = zeros(size(difmapVector[1].λ,1),size(difmapVector,1))
    for i = 1:size(difmapVector,1)
        ev[:,i] = difmapVector[i].λ
    end
    p = plot(ev[1:nVec,:]', title="iteration / eigen values",legend = false)
    return p
end
