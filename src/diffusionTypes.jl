## types for  diffusionmap calculation


struct Diffusionmap
    data::Matrix
    kernel::AbstractKernel
    laplace_type::AbstractLaplacian
    threshold::Int64
end

Diffusionmap(data; kernel::AbstractKernel=InverseDistanceKernel(), laplace_type::AbstractLaplacian=RowNormalizedLaplacian(), threshold::Int64=size(data,1)) = Diffusionmap(data, kernel, laplace_type, threshold)
Diffusionmap(data, func::Function; laplace_type::AbstractLaplacian=RowNormalizedLaplacian(), threshold::Int64=size(data,1)) = Diffusionmap(data, CustomKernel(func), laplace_type, threshold)


function Base.show(io::IO, ::MIME"text/plain", dp::Diffusionmap)
    println(io, "Diffusionmap")
    println(io, "Kernel = $(typeof(dp.kernel))")
    println(io, "Laplace type = $(dp.laplace_type)")
    println(io, "Threshold = $(dp.threshold)")   
end

