
## Kernel types

# abstract kernel supertype, all others inherit from Kernel
abstract type AbstractKernel end

# all subtypes characterize similarity calculations
struct InverseDistance <: AbstractKernel end

struct Gaussian <: AbstractKernel
    σ::Float64
end
Gaussian() = Gaussian(1.0)

struct Cosine <: AbstractKernel end

## similarity computation for each kernel

# multiple dispatch over kernel types
@inline function similarity(k::InverseDistance, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
    ret = zero(T)
    m = length(x)
    @inbounds @simd for k in 1:m
        ret += (x[k] - y[k])^2
    end
    return ret == 0.0 ? 0.0 : 1 / sqrt(ret)
end

@inline function similarity(k::Gaussian, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
    ret = zero(T)
    m = length(x)
    @inbounds @simd for k in 1:m
        ret += (x[k] - y[k])^2
    end
    return ret == 0.0 ? 0.0 : exp(-ret /(2 * k.σ^2))
end

function similarity(k::Cosine, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T,N}
    return (x'*y)/(norm(x)*norm(y))
end
