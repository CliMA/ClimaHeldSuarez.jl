using LinearAlgebra: ×, norm, norm_sqr, dot

using ClimaCore: Operators, Fields

include("schur_complement_W.jl")
include("hyperdiffusion.jl")

# Constants required before `include("staggered_nonhydrostatic_model.jl")`
# const FT = ?    # floating-point type
# const p_0 = ?   # reference pressure
# const R_d = ?   # dry specific gas constant
# const κ = ?     # kappa
# const T_tri = ? # triple point temperature
# const grav = ?  # gravitational acceleration
# const Ω = ?     # planet's rotation rate (only required if space is spherical)
# const f = ?     # Coriolis frequency (only required if space is flat)

# To add additional terms to the explicit part of the tendency, define new
# methods for `additional_cache` and `additional_tendency!`.

const cp_d = R_d / κ     # heat capacity at constant pressure
const cv_d = cp_d - R_d  # heat capacity at constant volume
const γ = cp_d / cv_d    # heat capacity ratio

const divₕ = Operators.Divergence()
const wdivₕ = Operators.WeakDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜdivᵥ = Operators.DivergenceF2C(
    top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
)
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
)
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
)
const ᶜFC = Operators.FluxCorrectionC2C(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶠupwind_product1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind_product3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.ThirdOrderOneSided(),
    top = Operators.ThirdOrderOneSided(),
)

const ᶜinterp_stencil = Operators.Operator2Stencil(ᶜinterp)
const ᶠinterp_stencil = Operators.Operator2Stencil(ᶠinterp)
const ᶜdivᵥ_stencil = Operators.Operator2Stencil(ᶜdivᵥ)
const ᶠgradᵥ_stencil = Operators.Operator2Stencil(ᶠgradᵥ)

const C123 = Geometry.Covariant123Vector

pressure_ρθ(ρθ) = p_0 * (ρθ * R_d / p_0)^γ
pressure_ρe(ρe, K, Φ, ρ) = ρ * R_d * ((ρe / ρ - K - Φ) / cv_d + T_tri)
pressure_ρe_int(ρe_int, ρ) = R_d * (ρe_int / cv_d + ρ * T_tri)

get_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, dt, upwinding_mode) = merge(
    default_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, upwinding_mode),
    additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt),
)

function default_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, upwinding_mode)
    ᶜcoord = ᶜlocal_geometry.coordinates
    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        ᶜf = @. 2 * Ω * sind(ᶜcoord.lat)
    else
        ᶜf = map(_ -> f, ᶜlocal_geometry)
    end
    ᶜf = @. Geometry.Contravariant3Vector(Geometry.WVector(ᶜf))
    return (;
        ᶜuvw = similar(ᶜlocal_geometry, Geometry.Covariant123Vector{FT}),
        ᶜK = similar(ᶜlocal_geometry, FT),
        ᶜΦ = grav .* ᶜcoord.z,
        ᶜp = similar(ᶜlocal_geometry, FT),
        ᶜω³ = similar(ᶜlocal_geometry, Geometry.Contravariant3Vector{FT}),
        ᶠω¹² = similar(ᶠlocal_geometry, Geometry.Contravariant12Vector{FT}),
        ᶠu¹² = similar(ᶠlocal_geometry, Geometry.Contravariant12Vector{FT}),
        ᶠu³ = similar(ᶠlocal_geometry, Geometry.Contravariant3Vector{FT}),
        ᶜf,
        ∂ᶜK∂ᶠw_data = similar(
            ᶜlocal_geometry,
            Operators.StencilCoefs{-half, half, NTuple{2, FT}},
        ),
        ᶠupwind_product = upwinding_mode == :first_order ? ᶠupwind_product1 :
                          upwinding_mode == :third_order ? ᶠupwind_product3 :
                          nothing,
        ghost_buffer = (
            c = Spaces.create_dss_buffer(Y.c),
            f = Spaces.create_dss_buffer(Y.f),
            χ = Spaces.create_dss_buffer(Y.c.ρ), # for hyperdiffusion
            χw = Spaces.create_dss_buffer(Y.f.w.components.data.:1), # for hyperdiffusion
            χuₕ = Spaces.create_dss_buffer(Y.c.uₕ), # for hyperdiffusion
        ),
    )
end

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = (;)

function implicit_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp, ᶠupwind_product) = p

    # Used for automatically computing the Jacobian ∂Yₜ/∂Y. Currently requires
    # allocation because the cache is stored separately from Y, which means that
    # similar(Y, <:Dual) doesn't allocate an appropriate cache for computing Yₜ.
    if eltype(Y) <: Dual
        ᶜK = similar(ᶜρ)
        ᶜp = similar(ᶜρ)
    end

    @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2

    @. Yₜ.c.ρ = -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw))

    if :ρθ in propertynames(Y.c)
        ᶜρθ = Y.c.ρθ
        @. ᶜp = pressure_ρθ(ᶜρθ)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρθ = -(ᶜdivᵥ(ᶠinterp(ᶜρθ) * ᶠw))
        else
            @. Yₜ.c.ρθ =
                -(ᶜdivᵥ(ᶠinterp(Y.c.ρ) * ᶠupwind_product(ᶠw, ᶜρθ / Y.c.ρ)))
        end
    elseif :ρe in propertynames(Y.c)
        ᶜρe = Y.c.ρe
        @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρe = -(ᶜdivᵥ(ᶠinterp(ᶜρe + ᶜp) * ᶠw))
        else
            @. Yₜ.c.ρe = -(ᶜdivᵥ(
                ᶠinterp(Y.c.ρ) * ᶠupwind_product(ᶠw, (ᶜρe + ᶜp) / Y.c.ρ),
            ))
        end
    elseif :ρe_int in propertynames(Y.c)
        ᶜρe_int = Y.c.ρe_int
        @. ᶜp = pressure_ρe_int(ᶜρe_int, ᶜρ)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρe_int = -(
                ᶜdivᵥ(ᶠinterp(ᶜρe_int + ᶜp) * ᶠw) -
                ᶜinterp(dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw)))
            )
            # or, equivalently,
            # Yₜ.c.ρe_int = -(ᶜdivᵥ(ᶠinterp(ᶜρe_int) * ᶠw) + ᶜp * ᶜdivᵥ(ᶠw))
        else
            @. Yₜ.c.ρe_int = -(
                ᶜdivᵥ(
                    ᶠinterp(Y.c.ρ) *
                    ᶠupwind_product(ᶠw, (ᶜρe_int + ᶜp) / Y.c.ρ),
                ) -
                ᶜinterp(dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw)))
            )
        end
    end

    Yₜ.c.uₕ .= Ref(zero(eltype(Yₜ.c.uₕ)))

    @. Yₜ.f.w = -(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥ(ᶜK + ᶜΦ))

    # TODO: Add flux correction to the Jacobian
    # @. Yₜ.c.ρ += ᶜFC(ᶠw, ᶜρ)
    # if :ρθ in propertynames(Y.c)
    #     @. Yₜ.c.ρθ += ᶜFC(ᶠw, ᶜρθ)
    # elseif :ρe in propertynames(Y.c)
    #     @. Yₜ.c.ρe += ᶜFC(ᶠw, ᶜρe)
    # elseif :ρe_int in propertynames(Y.c)
    #     @. Yₜ.c.ρe_int += ᶜFC(ᶠw, ᶜρe_int)
    # end

    return Yₜ
end

function remaining_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    default_remaining_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    Spaces.weighted_dss_start!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_start!(Yₜ.f, p.ghost_buffer.f)
    Spaces.weighted_dss_internal!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_internal!(Yₜ.f, p.ghost_buffer.f)
    Spaces.weighted_dss_ghost!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_ghost!(Yₜ.f, p.ghost_buffer.f)
    return Yₜ
end

function default_remaining_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜp, ᶜω³, ᶠω¹², ᶠu¹², ᶠu³, ᶜf) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))
    @. ᶜK = norm_sqr(ᶜuvw) / 2

    # Mass conservation

    @. Yₜ.c.ρ -= divₕ(ᶜρ * ᶜuvw)
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))

    # Energy conservation

    ᶜρe = Y.c.ρe
    @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
    @. Yₜ.c.ρe -= divₕ((ᶜρe + ᶜp) * ᶜuvw)
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))

    # Momentum conservation

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
        @. ᶠω¹² = curlₕ(ᶠw)
    elseif point_type <: Geometry.Abstract2DPoint
        ᶜω³ .= Ref(zero(eltype(ᶜω³)))
        @. ᶠω¹² = Geometry.Contravariant12Vector(curlₕ(ᶠw))
    end
    @. ᶠω¹² += ᶠcurlᵥ(ᶜuₕ)

    # TODO: Modify to account for topography
    @. ᶠu¹² = Geometry.Contravariant12Vector(ᶠinterp(ᶜuₕ))
    @. ᶠu³ = Geometry.Contravariant3Vector(ᶠw)

    @. Yₜ.c.uₕ -=
        ᶜinterp(ᶠω¹² × ᶠu³) + (ᶜf + ᶜω³) × Geometry.Contravariant12Vector(ᶜuₕ)
    if point_type <: Geometry.Abstract3DPoint
        @. Yₜ.c.uₕ -= gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. Yₜ.c.uₕ -=
            Geometry.Covariant12Vector(gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
    end

    @. Yₜ.f.w -= ᶠω¹² × ᶠu¹²
end

additional_tendency!(Yₜ, Y, p, t) = nothing

function Wfact!(W, Y, p, dtγ, t)
    (; flags, dtγ_ref, ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄) = W
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp, ∂ᶜK∂ᶠw_data, ᶠupwind_product) = p

    dtγ_ref[] = dtγ

    ᶠw_data = ᶠw.components.data.:1

    εw = Ref(Geometry.Covariant3Vector(eps(FT)))
    to_scalar(vector) = vector.u₃

    @. ∂ᶜK∂ᶠw_data =
        ᶜinterp(ᶠw_data) *
        norm_sqr(one(ᶜinterp(ᶠw))) *
        ᶜinterp_stencil(one(ᶠw_data))

    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρ) * one(ᶠw)))


    ᶜρe = Y.c.ρe
    @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
    @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)

    if isnothing(ᶠupwind_product)
        @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
            -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρe + ᶜp) * one(ᶠw))) - compose(
                ᶜdivᵥ_stencil(ᶠw),
                compose(
                    ᶠinterp_stencil(one(ᶜp)),
                    -(ᶜρ * R_d / cv_d) * ∂ᶜK∂ᶠw_data,
                ),
            )
    else
        error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :no_∂ᶜp∂ᶜK when using ρe with \
                upwinding")
    end

    to_scalar_coefs(vector_coefs) =
        map(vector_coef -> vector_coef.u₃, vector_coefs)

    @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = to_scalar_coefs(
        -1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_stencil(R_d / cv_d * one(ᶜρe)),
    )

    # flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
    @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
        -1 / ᶠinterp(ᶜρ) *
        ᶠgradᵥ_stencil(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri)) +
        ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2 * ᶠinterp_stencil(one(ᶜρ)),
    )

    @. ∂ᶠ𝕄ₜ∂ᶠ𝕄 = to_scalar_coefs(
        compose(
            -1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_stencil(-(ᶜρ * R_d / cv_d)) +
            -1 * ᶠgradᵥ_stencil(one(ᶜK)),
            ∂ᶜK∂ᶠw_data,
        ),
    )

end
