
## utilization functions

# standardize data matrix across columns
function standardize!(data::Matrix)
    for i = 1:size(data, 2)
        data[:,i] = (data[:,i] .- mean(data[:,i])) / sqrt(var(data[:,i]))
    end
end

## visualisation stuff with dispatch for coloring

function visualize(dm::Diffusionmap; dims::Tuple=(2,3), color_z=[], markersize::Int64=5)
    ϕ = real.(dm.ϕ)
    if isempty(color_z)
        return scatter(ϕ[:,dims[1]], ϕ[:,dims[2]], label="", ms=markersize)
    end

    return scatter(ϕ[:,dims[1]], ϕ[:,dims[2]], marker_z=color_z, label="", ms=markersize, color=:thermal)
end

function multiVisualize(dm::Diffusionmap; maindim::Int64=2, subdims::UnitRange{Int64}=3:4, color_z::Vector=[],layout::Tuple=(2,1), markersize=5)
    ϕ = real.(dm.ϕ)
    plot_array = Any[]
    for i in 1:size(subdims,1)
      push!(plot_array,visualize(dm, dims=(maindim,subdims[i]), color_z=color_z, markersize=markersize))
    end
    p = plot(plot_array...,layout=layout)
    return p
end

function addTitles!(plot,subtitles::Vector{String})
    for i in 1:size(subtitles,1)
        plot.subplots[i].attr[:title] = subtitles[i]
    end
    display(plot)
end
