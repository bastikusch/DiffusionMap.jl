# eigen solver types

abstract type AbstractEigenSolver end

struct FullEigen <: AbstractEigenSolver end

function Base.show(io::IO, ::MIME"text/plain", l::FullEigen)
    println(io, "FullEigen")   
end

struct ArpackEigen <: AbstractEigenSolver
    n_first::Int
end

function Base.show(io::IO, ::MIME"text/plain", l::ArpackEigen)
    println(io, "ArpackEigen")   
end

struct KrylovEigen <: AbstractEigenSolver
    n_first::Int
end

function Base.show(io::IO, ::MIME"text/plain", l::KrylovEigen)
    println(io, "KrylovEigen")   
end