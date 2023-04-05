using OrdinaryDiffEq
using DiffEqCallbacks
using JLD2
using ClimaCore.DataLayouts
using ClimaCore: Geometry, Meshes, Spaces, Topologies, Fields

const FT = Float64

include("baroclinic_wave_utilities.jl")

const sponge = false

jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
test_implicit_solver = false
h_elem = 4
npoly = 4
z_max = FT(30e3)
z_elem = 10
day = FT(60 * 60 * 24)
t_end = day / 2
dt = FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
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

z_stretch = Meshes.Uniform()
z_stretch_string = "uniform"

t_start = FT(0)

# Horizontal space
domain = Domains.SphereDomain(R)
horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
quadrature = Spaces.Quadratures.GLL{npoly + 1}()
h_topology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), horizontal_mesh)
h_space = Spaces.SpectralElementSpace2D(h_topology, quadrature)

# Vertical space
z_bottom = Geometry.ZPoint(zero(z_max))
z_top = Geometry.ZPoint(z_max)
z_domain = Domains.IntervalDomain(z_bottom, z_top, boundary_tags = (:bottom, :top))
z_mesh = Meshes.IntervalMesh(z_domain, z_stretch; nelems = z_elem)
z_topology = Topologies.IntervalTopology(z_mesh)
z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)

# "Hybrid" space
center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

ᶜlocal_geometry = Fields.local_geometry_field(center_space)
ᶠlocal_geometry = Fields.local_geometry_field(face_space)

Y = Fields.FieldVector(c = center_initial_condition(ᶜlocal_geometry),
                       f = face_initial_condition(ᶠlocal_geometry))

p = merge(default_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, upwinding_mode),
          additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt))

if ode_algorithm <: Union{OrdinaryDiffEq.OrdinaryDiffEqImplicitAlgorithm,
                          OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm}

    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))

    W = SchurComplementW(Y, use_transform, jacobian_flags, test_implicit_solver)

    jac_kwargs = use_transform ?
                 (; jac_prototype = W, Wfact_t = Wfact!) :
                 (; jac_prototype = W, Wfact = Wfact!)

    alg_kwargs = (; linsolve = linsolve!)

    if ode_algorithm <: Union{OrdinaryDiffEq.OrdinaryDiffEqNewtonAlgorithm,
                              OrdinaryDiffEq.OrdinaryDiffEqNewtonAdaptiveAlgorithm}

        alg_kwargs = (; alg_kwargs..., nlsolve = NLNewton(; max_iter = max_newton_iters))
    end
else
    jac_kwargs = alg_kwargs = (;)
end

# Y: prognostic state vector
# p: "parameters"
# Y.c: centered state variables
# Y.f: interface state variables
dss_callback = FunctionCallingCallback(func_start = true) do Y, t, integrator
    p = integrator.p
    Spaces.weighted_dss!(Y.c, p.ghost_buffer.c)
    Spaces.weighted_dss!(Y.f, p.ghost_buffer.f)
end

# Explicit:
# prob = ODEProblem(rhs_explicit!, Y_init, (0.0, T), u)

# For implicit time-stepping we use SplitODEProblem
tgrad(dYdt, Y, parameters, time) = dYdt .= 0
implicit_tendency_func = ODEFunction(implicit_tendency!; jac_kwargs..., tgrad)
time_span = (t_start, t_end)
problem = SplitODEProblem(implicit_tendency_func, remaining_tendency!, Y, time_span, p)

integrator = OrdinaryDiffEq.init(problem,
                                 ode_algorithm(; alg_kwargs...);
                                 saveat = [],
                                 callback = CallbackSet(dss_callback),
                                 dt = dt,
                                 adaptive = false,
                                 progress_steps = 20)

walltime = @elapsed sol = OrdinaryDiffEq.solve!(integrator)

