hyperdiffusion_cache(
    ᶜlocal_geometry,
    ᶠlocal_geometry;
    κ₄ = FT(0),
    divergence_damping_factor = FT(1),
    use_tempest_mode = false,
) = merge(
    (;
        ᶜχ = similar(ᶜlocal_geometry, FT),
        ᶜχuₕ = similar(ᶜlocal_geometry, Geometry.Covariant12Vector{FT}),
        κ₄,
        divergence_damping_factor,
        use_tempest_mode,
    ),
    use_tempest_mode ? (; ᶠχw_data = similar(ᶠlocal_geometry, FT)) : (;),
)

function hyperdiffusion_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume that ᶜp has been updated
    (; ghost_buffer, κ₄, divergence_damping_factor, use_tempest_mode) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)


    @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe + ᶜp) / ᶜρ)) # ᶜχe
    Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
    @. Yₜ.c.ρe -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))

    # if point_type <: Geometry.Abstract3DPoint
        @. ᶜχuₕ =
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.Covariant12Vector(
                wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜuₕ))),
            )
        Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                Geometry.Covariant12Vector(
                    wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜχuₕ))),
                )
            )
    # elseif point_type <: Geometry.Abstract2DPoint
    #     @. ᶜχuₕ = Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜuₕ)))
    #     Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
    #     @. Yₜ.c.uₕ -=
    #         κ₄ *
    #         divergence_damping_factor *
    #         Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    # end
end
