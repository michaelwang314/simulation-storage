using SoftSquishyMatter
using Random
cd(dirname(@__FILE__))

simulation = Simulation()
simulation.descriptor = "Active particles inside and outside an elastic ring"

R = 1e-6
γ_trans = 6 * pi * 8.9e-4 * R
D_trans = 1.38e-23 * 300 / γ_trans

particle_positions, simulation.L_x, simulation.L_y = triangular_lattice(s = 3e-6, M_x = 50, M_y = 29)

N_ring = 125
θ_ring = 2 * pi / N_ring
R_ring = R / sin(θ_ring / 2)
ring_positions = Array{Array{Float64, 1}, 1}()
for n = 0 : N_ring - 1
    push!(ring_positions, [simulation.L_x / 2, simulation.L_y / 2] + R_ring * [cos(n * θ_ring), sin(n * θ_ring)])
end

remove_overlaps!(particle_positions; fixed_positions = ring_positions, minimum_distance = 2^(1.0 / 6.0) * 2 * R, period_x = simulation.L_x, period_y = simulation.L_y)

for (x, y) in ring_positions
    push!(simulation.particles, Particle(ptype = :ring, x = x, y = y, R = R, γ_trans = γ_trans, D_trans = D_trans))
end
for (x, y) in particle_positions
    active_force = ActiveBrownian(γv = γ_trans * 15e-6, D_rot = 1.0)
    push!(simulation.particles, Particle(ptype = :active, x = x, y = y, R = R, γ_trans = γ_trans, D_trans = D_trans, active_force = active_force))
end
pgroup_active = group_by_type(simulation.particles; ptype = :active)
pgroup_ring = group_by_type(simulation.particles; ptype = :ring)
pgroup_all = simulation.particles

cell_list_active = CellList(particles = pgroup_active, L_x = simulation.L_x, L_y = simulation.L_y, cutoff = 2^(1.0 / 6.0) * 2 * R)
lj_aa = LennardJones(particles = pgroup_active, cell_list = cell_list_active, ϵ = 1.38e-23 * 300, multithreaded = true)
lj_ra = LennardJones(particles = pgroup_ring, cell_list = cell_list_active, ϵ = 1.38e-23 * 300, multithreaded = false, use_newton_3rd = true)

particle_pairs = Array{Tuple{Particle, Particle}, 1}()
for n = 1 : N_ring
    push!(particle_pairs, (pgroup_ring[n], pgroup_ring[mod(n, N_ring) + 1]))
end
hb = HarmonicBond(pairs = particle_pairs, k_bond = 1e-5, l_rest = 2 * R)

push!(simulation.cell_lists, cell_list_active)
append!(simulation.interactions, [lj_aa, lj_ra, hb])

simulation.dt = 0.0001
brownian = Brownian(particles = pgroup_all, dt = simulation.dt, multithreaded = true)
push!(simulation.integrators, brownian)

simulation.num_steps = trunc(Int64, 10 / simulation.dt)
simulation.save_interval = trunc(Int64, 1 / simulation.dt)
simulation.particles_to_save = pgroup_all

run_simulation!(simulation; save_as = "out/test.out")

simulation = load_simulation(file = "out/test.out")
visualize!(simulation; save_as = (:gif, "frames/test.gif"), fps = 1, particle_colors = Dict(:ring => "black", :active => "green"))
