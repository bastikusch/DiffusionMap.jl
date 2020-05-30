## data handling

function standardize!(data::Matrix)
    for i = 1:size(data, 2)
        data[:,i] = (data[:,i] .- mean(data[:,i])) / sqrt(var(data[:,i]))
    end
end

## visualisation

function visualize(difmap::Diffusionmap, markersize=4)
    ϕ = real.(difmap.ϕ)
    s1 = scatter(ϕ[:,2], ϕ[:,3], title="eigen vector 1 / 2", label="", ms=markersize)
    s2 = scatter(ϕ[:,2], ϕ[:,4], title="eigen vector 1 / 3", label="", ms=markersize)
    s3 = scatter(ϕ[:,2], ϕ[:,5], title="eigen vector 1 / 4", label="", ms=markersize)
    s4 = scatter(ϕ[:,2], ϕ[:,6], title="eigen vector 1 / 5", label="", ms=markersize)
    p = plot(s1,s2,s3,s4, layout = (2,2))
    return p
end

function visualize(difmap::Diffusionmap, color_z::Vector, markersize=4)
    ϕ = real.(difmap.ϕ)
    s1 = scatter(ϕ[:,2], ϕ[:,3], marker_z=color_z, title="eigen vector 1 / 2", label="", ms=markersize,color=:thermal)
    s2 = scatter(ϕ[:,2], ϕ[:,4], marker_z=color_z, title="eigen vector 1 / 3", label="", ms=markersize,color=:thermal)
    s3 = scatter(ϕ[:,2], ϕ[:,5], marker_z=color_z, title="eigen vector 1 / 4", label="", ms=markersize,color=:thermal)
    s4 = scatter(ϕ[:,2], ϕ[:,6], marker_z=color_z, title="eigen vector 1 / 5", label="", ms=markersize,color=:thermal)
    p = plot(s1,s2,s3,s4, layout = (2,2))
    return p
end
