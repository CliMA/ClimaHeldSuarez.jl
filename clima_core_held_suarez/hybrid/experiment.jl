
using OrdinaryDiffEq
using DiffEqCallbacks
using JLD2


using ClimaCore.DataLayouts

include("baroclinic_wave_utilities.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = 4)
npoly = 4
z_max = FT(30e3)
z_elem = 10
t_end = FT(60 * 60 * 24 * 10)
dt = FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
# jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

# Additional values required for driver
upwinding_mode = :third_order

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry, ᶠlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
    held_suarez_cache(ᶜlocal_geometry),
)

function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    held_suarez_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry) =
    center_initial_condition(local_geometry, Val(:ρe))


include(joinpath(test_dir, "$test_file_name.jl"))

    const FT = Float64

    include("../common_spaces.jl")

    z_stretch = Meshes.Uniform()
    z_stretch_string = "uniform"


    function cubed_sphere_mesh(; radius, h_elem)
        domain = Domains.SphereDomain(radius)
        return Meshes.EquiangularCubedSphere(domain, h_elem)
    end
    horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = 4)

    t_start = FT(0)
    quadrature = Spaces.Quadratures.GLL{npoly + 1}()
    h_topology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), horizontal_mesh)
    h_space = Spaces.SpectralElementSpace2D(h_topology, quadrature)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_tags = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, z_stretch; nelems = z_elem)
    z_topology = Topologies.IntervalTopology(z_mesh)
    z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)



    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(face_space)
    Y = Fields.FieldVector(
        c = center_initial_condition(ᶜlocal_geometry),
        f = face_initial_condition(ᶠlocal_geometry),
    )


p = merge(
        default_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, upwinding_mode),
        additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt),
    )

    # Y: prognostic state vector
# p: "parameters"
# Y.c: centered state variables
# Y.f: interface state variables
dss_callback = FunctionCallingCallback(func_start = true) do Y, t, integrator
    p = integrator.p
    Spaces.weighted_dss!(Y.c, p.ghost_buffer.c)
    Spaces.weighted_dss!(Y.f, p.ghost_buffer.f)
end

problem = SplitODEProblem(
    ODEFunction(
        implicit_tendency!;
        tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
    ),
    remaining_tendency!,
    Y,
    (t_start, t_end),
    p,
)

integrator = OrdinaryDiffEq.init(
    problem,
    ode_algorithm(; alg_kwargs...);
    saveat = [],
    callback = CallbackSet(dss_callback),
    dt = dt,
    adaptive = false,
    progress_steps = 20,
)

walltime = @elapsed sol = OrdinaryDiffEq.solve!(integrator)