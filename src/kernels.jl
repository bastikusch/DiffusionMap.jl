## Kernel types

abstract type Kernel end

struct InverseDistance <: Kernel end

struct Gaussian <: Kernel
    σ::Float64
end

Gaussian() = Gaussian(1.0)

struct Linear <: Kernel end

struct Spearman <: Kernel end

struct Polynomial <: Kernel
    r::Float64
    n::Int
end

Polynomial() = Polynomial(0.0, 1)

struct Correlation <: Kernel end

struct LaplaceKernel <: Kernel
    α::Float64
end

LaplaceKernel() = LaplaceKernel(1.0)


## similarity computation for each kernel

function similarity(k::InverseDistance, x::Vector, y::Vector)
    return x == y ? 0.0 : 1 / norm(x .- y)
end

function similarity(k::Gaussian, x::Vector, y::Vector)
    return exp(-(norm(x .- y)^2) / (2 * k.σ^2))
end

function similarity(k::LaplaceKernel, x::Vector, y::Vector)
    return exp(-k.α * (norm(x .- y)))
end

function similarity(k::Correlation, x::Vector, y::Vector)
    return abs(cor(x,y))
end

function similarity(k::Linear, x::Vector, y::Vector)
    return dot(x, y) / (norm(x) * norm(y))
end

function similarity(k::Polynomial, x::Vector, y::Vector)
    return (dot(x, y) / (norm(x) * norm(y)) + k.r)^k.n
end

function similarity(k::Spearman, x::Vector, y::Vector)
    return abs(cov(sortperm(x), sortperm(y)) / (std(sortperm(x)) * std(sortperm(y))))
end
