
## utilization functions

# standardize data matrix across columns
function standardize!(data::Matrix)
    for i = 1:size(data, 2)
        data[:,i] = (data[:,i] .- mean(data[:,i])) / sqrt(var(data[:,i]))
    end
end

## visualisation stuff with dispatch for coloring

function visualize(difmap::Diffusionmap, dims::Tuple, color_z, markersize)
    ϕ = real.(difmap.ϕ)
    isempty(color_z) ? (return visualize(difmap, dims, markersize)) : ()
    p = scatter(ϕ[:,dims[1]], ϕ[:,dims[2]], marker_z=color_z, label="", ms=markersize, color=:thermal)
    return p
end

function visualize(difmap::Diffusionmap, dims::Tuple, markersize)
    ϕ = real.(difmap.ϕ)
    p = scatter(ϕ[:,dims[1]], ϕ[:,dims[2]], label="", ms=markersize)
    return p
end

function visualize(difmap::Diffusionmap; dims::UnitRange{Int64}=2:3, color_z::Vector=[],layout::Tuple=(1,1), markersize=5)
    ϕ = real.(difmap.ϕ)
    plot_array = Any[]
    for i in 2:size(dims,1)
      push!(plot_array,visualize(difmap, (dims[1],dims[i]), color_z, markersize))
    end
    p = plot(plot_array...,layout=layout)
    return p
end

function addTitles!(plot::Plots.Plot{Plots.GRBackend},subtitles::Vector{String})
    for i in 1:size(subtitles,1)
        plot.subplots[i].attr[:title] = subtitles[i]
    end
    display(plot)
end
