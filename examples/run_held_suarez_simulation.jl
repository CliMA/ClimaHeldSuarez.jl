using ClimaHeldSuarez.HeldSuarezModels: HeldSuarezModel, VerticallyStaggeredCubedSphereSpace

kilometers = 1e3
z = (0, 30kilometers)
space = VerticallyStaggeredCubedSphereSpace(; z)
model = HeldSuarezModel(; space)
