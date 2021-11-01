
## Kernel types

# abstract kernel supertype, all others inherit from Kernel
abstract type AbstractKernel end

# all subtypes characterize similarity calculations
struct InverseDistanceKernel <: AbstractKernel end

struct GaussianKernel <: AbstractKernel
    σ::Float64
end
GaussianKernel() = GaussianKernel(1.0)

struct CosineKernel <: AbstractKernel end

struct CustomKernel <: AbstractKernel
    func::Function
end

struct FlowTensionKernel <: AbstractKernel
    dist::Matrix
end

## similarity computation for each kernel

# multiple dispatch over kernel types
@inline function similarity(k::InverseDistanceKernel, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
    ret = zero(T)
    m = length(x)
    @inbounds @simd for k in 1:m
        ret += (x[k] - y[k])^2
    end
    return ret == 0.0 ? 0.0 : 1 / sqrt(ret)
end

@inline function similarity(k::GaussianKernel, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
    ret = zero(T)
    m = length(x)
    @inbounds @simd for k in 1:m
        ret += (x[k] - y[k])^2
    end
    return ret == 0.0 ? 0.0 : exp(-ret /(2 * k.σ^2))
end

function similarity(k::CosineKernel, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
    return (x'*y)/(norm(x)*norm(y))
end

function similarity(k::CustomKernel, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
    return k.func(x,y)
end

function similarity(k::FlowTensionKernel, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
    tension = x .- y
    indBig = findall(x -> x > 0.0, tension)
    indSmall = findall(x -> x < 0.0, tension)
    r = []
    for i in indBig
        for j in indSmall
            push!(r,k.dist[i,j] * (tension[i] - tension[j]))
        end
    end
    return 1/sqrt(sum(r.^2))
end
