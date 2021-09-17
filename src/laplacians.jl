# laplacian types

abstract type AbstractLaplacian end

struct RowNormalizedLaplacian <: AbstractLaplacian end

function Base.show(io::IO, ::MIME"text/plain", l::RowNormalizedLaplacian)
    println(io, "RowNormalizedLaplacian")  
end

struct Adjacency <: AbstractLaplacian end

function Base.show(io::IO, ::MIME"text/plain", l::Adjacency)
    println(io, "Adjacency")   
end

struct NormalizedAdjacency <: AbstractLaplacian end

function Base.show(io::IO, ::MIME"text/plain", l::NormalizedAdjacency)
    println(io, "NormalizedAdjacency")   
end

struct SymmetricLaplacian <: AbstractLaplacian end

function Base.show(io::IO, ::MIME"text/plain", l::SymmetricLaplacian)
    println(io, "SymmetricLaplacian")   
end

struct RegularLaplacian <: AbstractLaplacian end

function Base.show(io::IO, ::MIME"text/plain", l::RegularLaplacian)
    println(io, "RegularLaplacian")   
end