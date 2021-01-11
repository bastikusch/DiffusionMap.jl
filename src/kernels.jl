
## Kernel types

# abstract kernel supertype, all others inherit from Kernel
abstract type Kernel end

# all subtypes characterize similarity calculations
struct InverseDistance <: Kernel end

struct Gaussian <: Kernel
    σ::Float64
end
Gaussian() = Gaussian(1.0)

struct KLKernel <: Kernel
    α::Float64
end
KLKernel() = KLKernel(1.0)

struct Correlation <: Kernel end

struct InformationCorrelation <: Kernel
    α::Float64
end
InformationCorrelation() = InformationCorrelation(1.0)

## similarity computation for each kernel

# multiple dispatch over kernel types
function similarity(k::InverseDistance, x::Vector, y::Vector)
    return x==y ? 0 : 1 / norm(x .- y)
end

function similarity(k::Gaussian, x::Vector, y::Vector)
    return exp(-(norm(x .- y)^2) / (2 * k.σ^2))
end

function similarity(k::KLKernel, x::Vector, y::Vector)
    return renyientropy(x, k.α) + renyientropy(y, k.α) - renyientropy(vcat(x, y), k.α)
end

function similarity(k::Correlation, x::Vector, y::Vector)
    return abs(cor(x, y))
end

function similarity(k::InformationCorrelation, x::Vector, y::Vector)
    return abs(sign(cor(x, y)) * sqrt(1 - 2 ^ (-2 * (renyientropy(x, k.α) + renyientropy(y, k.α) - renyientropy(vcat(x, y), k.α)))))
end

