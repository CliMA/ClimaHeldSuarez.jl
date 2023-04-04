module HeldSuarezModels

using ClimaCore
using ClimaComms

using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces, Fields

const Earth_radius = 6371e3 # km

struct CubedSphere end
struct LatitudeLongitude end

struct VerticallyStaggeredSpace{Desc, Cen, Face}
    center :: Cen
    face :: Face
    VerticallyStaggeredSpace{D}(center::C, face::F) where {D, C, F} =
        new{D, C, F}(center, face)
end

const CubedSphereShellSpace = VerticallyStaggeredSpace{<:CubedSphere}
const LatitudeLongitudeShellSpace = VerticallyStaggeredSpace{<:LatitudeLongitude}

Base.summary(space::CubedSphereShellSpace) = "CubedSphereShellSpace"
Base.summary(space::LatitudeLongitudeShellSpace) = "LatitudeLongitudeShellSpace"

function VerticallyStaggeredCubedSphereSpace(; z,
                                             z_stretching = Meshes.Uniform(),
                                             radius = Earth_radius,
                                             polynomial_degree = 3,
                                             horizontal_elements = 6,
                                             vertical_cells = 10,
                                             comms = ClimaComms.SingletonCommsContext())

    h_domain     = Domains.SphereDomain(radius)
    h_quadrature = Spaces.Quadratures.GLL{polynomial_degree + 1}()
    h_mesh       = Meshes.EquiangularCubedSphere(h_domain, horizontal_elements)
    h_topology   = Topologies.Topology2D(comms, h_mesh)
    h_space      = Spaces.SpectralElementSpace2D(h_topology, h_quadrature)

    z_bottom   = Geometry.ZPoint(first(z))
    z_top      = Geometry.ZPoint(last(z))
    z_domain   = Domains.IntervalDomain(z_bottom, z_top; boundary_tags = (:bottom, :top))
    z_mesh     = Meshes.IntervalMesh(z_domain, z_stretching; nelems = vertical_cells)
    z_topology = Topologies.IntervalTopology(z_mesh)
    z_space    = Spaces.CenterFiniteDifferenceSpace(z_topology)

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return VerticallyStaggeredSpace{CubedSphere}(center_space, face_space)
end

struct HeldSuarezModel{Space, State, Integ}
    space :: Space
    state :: State
    integrator :: Integ
end

function HeldSuarezModel(; space)
    integrator = nothing

    # geometry = (center = Fields.local_geometry_field(space.center),
    #             face = Fields.local_geometry_field(space.face))

    state = Fields.FieldVector(c = zeros(space.center),
                               f = zeros(space.face))

    return HeldSuarezModel(space, state, integrator)
end

Base.summary(model::HeldSuarezModel) = "HeldSuarezModel"

function Base.show(io::IO, model::HeldSuarezModel)
    print(io, summary(model), " in ", summary(model.space), '\n')
    print(io, "├─ space: ", summary(model.space), '\n')
    print(io, "├─ state: ", '\n')
    print(io, "└─ integrator: ")
end

end # module
