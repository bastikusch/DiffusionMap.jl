# DiffusionMap.jl
module DiffusionMap

using Distances, LinearAlgebra, Statistics

export standardize!, inverseDistance, thresholding, laplacian, quickDiffusionMap, eigenVisualisation

## types


## data handling
"""
 standardize!()
blabla
"""

function standardize!(data::Matrix)
    for i = 1:size(data, 2)
        data[:,i] = (data[:,i] .- mean(data[:,i])) / sqrt(var(data[:,i]))
    end
end

## kernels

function inverseDistance(data)
    l = size(data, 1)
    similarity = zeros(l, l)
    for i = 1:l
        for j = i:l
            similarity[i, j] = euclidean(data[i,:], data[j,:])
        end
    end
    similarity = similarity + transpose(similarity)
    similarity = 1 ./ similarity
    similarity[isinf.(similarity)] .= 0
    return similarity
end

## other functions

function thresholding(similarity, nextneighbors)
    l = size(similarity, 1)
    threshold = copy(similarity)
    for i = 1:l
        sortColumn = sort(similarity[:,i], rev = true)
        nth_Max = sortColumn[nextneighbors]
        threshold[:, i] = map(x -> x < nth_Max ? 0 : x, similarity[:, i])
    end
    lower = Symmetric(threshold, :L)
    upper = Symmetric(threshold, :U)
    threshold[:] = max.(lower[:], upper[:])
    return threshold
end

function laplacian(threshold)
    l = size(threshold, 1)
    Adjacency = copy(threshold)
    Laplace = Adjacency
    for i = 1:l
        Laplace[i,:] = Laplace[i,:] / sum(Laplace[i,:])
    end
    Laplace = Diagonal(ones(l)) - Laplace
    return Laplace
end

function quickDiffusionMap(data, nextneighbors)
    return eigen(laplacian(thresholding(inverseDistance(data), nextneighbors)))
end

function eigenVisualisation(ϕ, color_z, markersize=4)
    s1 = scatter(ϕ[:,1], ϕ[:,2], marker_z=color_z, title="EV 1/2", label="", ms=markersize,color=:thermal)
    s2 = scatter(ϕ[:,1], ϕ[:,3], marker_z=color_z, title="EV 1/3", label="", ms=markersize,color=:thermal)
    s3 = scatter(ϕ[:,1], ϕ[:,4], marker_z=color_z, title="EV 1/4", label="", ms=markersize,color=:thermal)
    s4 = scatter(ϕ[:,1], ϕ[:,5], marker_z=color_z, title="EV 1/5", label="", ms=markersize,color=:thermal)
    p = plot(s1,s2,s3,s4, layout = (2,2))
    return p
end

end # module
