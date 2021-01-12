
## Kernel types

# abstract kernel supertype, all others inherit from Kernel
abstract type Kernel end

# all subtypes characterize similarity calculations
struct InverseDistance <: Kernel end

struct Gaussian <: Kernel
    σ::Float64
end
Gaussian() = Gaussian(1.0)

struct MutualInformation <: Kernel end

struct Correlation <: Kernel end

struct InformationCorrelation <: Kernel end

## similarity computation for each kernel

# multiple dispatch over kernel types
function similarity(k::InverseDistance, x::Vector, y::Vector)
    return x==y ? 0 : 1 / norm(x .- y)
end

function similarity(k::Gaussian, x::Vector, y::Vector)
    return exp(-(norm(x .- y)^2) / (2 * k.σ^2))
end

function similarity(k::MutualInformation, x::Vector, y::Vector)
    return get_mutual_information(x, y)
end

function similarity(k::Correlation, x::Vector, y::Vector)
    return abs(cor(x, y))
end

function similarity(k::InformationCorrelation, x::Vector, y::Vector)
    return abs(sign(cor(x, y)) * sqrt(1 - 2 ^ (-2 * (get_mutual_information(x, y)))))
end

