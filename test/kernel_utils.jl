# utility wrappers to trigger KernelAbstractions kernels
struct MyBackend <: KA.Backend end

# These definitions are needed to be able to launch kernels on the fake backend
KA.isgpu(::MyBackend) = false
function KA.construct(::MyBackend, ::S, ::NDRange, xpu_name::XPUName) where {S <: KA._Size, NDRange <: KA._Size, XPUName}
    return KA.Kernel{KA.CPU, S, NDRange, XPUName}(KA.CPU(), xpu_name)
end
KA.synchronize(::MyBackend) = KA.synchronize(KA.CPU())

# Custom array wrapper
struct MyArray{T,N} <: AbstractArray{T,N}
    x::Array{T,N}
end

Base.size(x::MyArray) = size(x.x)
Base.getindex(x::MyArray, i::Int) = Base.getindex(x.x, i)
Base.getindex(x::MyArray, i::Int, j::Int) = Base.getindex(x.x, i, j)
Base.setindex!(x::MyArray, v, i::Int) = Base.setindex!(x.x, v, i::Int)
Base.setindex!(x::MyArray, v, i::Int, j::Int) = Base.setindex!(x.x, v, i::Int, j::Int)

# Additional KA functions to allocate on fake backend
KA.get_backend(::MyArray) = MyBackend()
KA.allocate(::MyBackend, T::Type, dims::Tuple) = MyArray(KA.allocate(KA.CPU(), T, dims))
