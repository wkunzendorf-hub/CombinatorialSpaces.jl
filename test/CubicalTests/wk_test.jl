#= 
Using GR's Navier-Stokes code to create a rectangular cavity simulation
Using comments to narrate my thought process
=#

using Test
using CombinatorialSpaces
using LinearAlgebra
using CairoMakie
using Debugger

include("../../src/CubicalComplexes.jl")

# Defining the uniform grid primal mesh
l_xy = 1;
n_xy = 101;
s = uniform_grid(l_xy, n_xy);

# Defining the top velocity for the lid-driven cavity problem
top_v = 1 / (n_xy - 1);

# Initializing DEC Operators
Δ0 = laplacian(Val(0), s);

d0 = exterior_derivative(Val(0), s);
d1 = exterior_derivative(Val(1), s);

dual_d0 = dual_derivative(Val(0), s);
dual_d1 = dual_derivative(Val(1), s);

d_beta = 0.5 * abs.(dual_d1) * spdiagm(dual_d0 * ones(nquads(s)));

hdg_1 = hodge_star(Val(1), s);

inv_hdg_0 = inv_hodge_star(Val(0), s);
inv_hdg_1 = inv_hodge_star(Val(1), s);

δ1 = codifferential(Val(1), s);

# Creating initial conditions and variable properties
u_star_0 = zeros(ne(s));
Δt = 1e-4;
μ = 0.001;

function plot_zeroform(s::HasCubicalComplex, f)
  fig = Figure();
  ax = CairoMakie.Axis(fig[1, 1])
  msh = CairoMakie.mesh!(ax, s, color=f, colormap=:jet)
  Colorbar(fig[1, 2], msh)
  fig
end

# save("imgs/InitialCavity.png", plot_zeroform(s, u1))

# Implementing Navier Stokes using DEC operators
rhs_U_mat = (-1 / Δt) * I + μ * d0 * inv_hdg_0 * dual_d1 * hdg_1
rhs_Pd_mat = inv_hdg_1 * dual_d0
Wv(v) = spdiagm(v)
vΔ1 = -μ * d0 * inv_hdg_0 * d_beta
adv_u_star = 0.5 * abs.(d0) * inv_hdg_0 * dual_d1 * hdg_1
adv_v = 0.5 * abs.(d0) * inv_hdg_0 * d_beta

# Creating function to generate F vector
X = init_tensor_d(Val(0), s)
Y = init_tensor_d(Val(0), s)
v_ten = init_tensor(Val(1), s)

# Creating functions for u and v boundary conditions
function boundary_u(u_t)
  fh, fv = u_t
  fh[:, begin] .= 0.0
  fh[:, end] .= 0.0
  fv[begin, :] .= 0.0
  fv[end, :] .= 0.0

  return u_t
end

function boundary_v(v_t) 
  v_ten = deepcopy(v_t)
  fh, fv = v_ten
  fh[:, begin] .= 0.0
  fh[:, end] .= top_v
  fv[begin, :] .= 0.0 
  fv[end, :] .= 0.0 

  return v_ten
end

function generate_F(u_star)
  global X, Y, v_ten

  u = hdg_1 * u_star
  u_ten = tensorfy(s, deepcopy(u))

  u_ten = boundary_u(u_ten)

  sharp_dd!(X, Y, s, u_ten) 
  flat_dp!(v_ten, s, X, Y)

  v_ten = boundary_v(v_ten)

  v = detensorfy(Val(1), s, v_ten)
  u = detensorfy(Val(1), s, u_ten)

  u_star = inv_hdg_1 * u

  return (-1 / Δt) * u_star + vΔ1 * v + Wv(v) * (adv_u_star * u_star + adv_v * v)
end

# Generating the RHS matrix to be solved
rhs_top = hcat(rhs_U_mat, rhs_Pd_mat)
rhs_bottom = hcat(d1, spdiagm(zeros(nquads(s))))
rhs = vcat(rhs_top, rhs_bottom)

f_rhs = factorize(rhs)

# Generating empty U vector for iteration
U = zeros(ne(s) + nquads(s))

# Creating initial iteration variables
tₑ = 1.0
steps = ceil(Int64, tₑ / Δt)
Us = [u_star_0]

# Pressure vector organization
press_i = zeros(nquads(s));
press = deepcopy(press_i);
press_s = [press];


u_star = deepcopy(u_star_0)

# Initial Pre-Simulation Messages
println("Beginning Rectangular Cavity Simulation")
println("Viscosity: $μ, Time step: $Δt, Total time steps: $steps")

# Iterating for Navier Stokes over each time step
F₂ = zeros(nquads(s)); 
for step in 1:steps 
    F = vcat(generate_F(u_star), F₂) 
    U .= f_rhs \ F 

    u_star .= U[1:ne(s)]
    u_star_ten = tensorfy(s, deepcopy(u_star))
    u_star_ten = boundary_u(u_star_ten)
    u_star = detensorfy(Val(1), s, u_star_ten)

    press .= U[(ne(s) + 1):end] 

    # Checking to see if current U has NaN in it, which would indicate a numerical instability
    if any(isnan.(U))
        println("NaN detected in U at step $step. Ending simulation.")
        break
    end

    # Printing u_max every iteration to see issue
    println("Step: $step, u max: $(maximum(u_star)), u min: $(minimum(u_star))")

    if step % 1 == 0 # 50
        push!(press_s, deepcopy(press))
        push!(Us, deepcopy(u_star)) 
    end
end

# function plot_zeroform(s::HasCubicalComplex, f)
#   fig = Figure();
#   ax = CairoMakie.Axis(fig[1, 1])
#   msh = CairoMakie.mesh!(ax, s, color=f, colormap=:jet)
#   Colorbar(fig[1, 2], msh)
#   fig
# end

# s_dual = dual_mesh(s)

# plot_zeroform(s_dual, press_s[30])

# function plot_oneform(s::HasCubicalComplex, alpha)
#   dps = dual_points(s)
#   x = map(a -> a[1], dps)
#   y = map(a -> a[2], dps)

#   sharp_dd!(X, Y, s, tensorfy(s, alpha))

#   Xvec = detensorfy_d(Val(0), s, X)
#   Yvec = detensorfy_d(Val(0), s, Y)

#   color = sqrt.(Xvec.^2 + Yvec.^2)

#   fig = Figure(size=(1000, 1000));
#   ax = CairoMakie.Axis(fig[1,1])
#   arrows2d!(ax, x, y, Xvec, Yvec, color = color, normalize = true, lengthscale = 0.01)
#   wireframe!(ax, s; alpha = 0.1)
#   fig
# end

function plot_vars(s::HasCubicalComplex, u_star, press)
  dps = dual_points(s)
  x = map(a -> a[1], dps)
  y = map(a -> a[2], dps)

  sharp_dd!(X, Y, s, tensorfy(s, u_star))

  Xvec = detensorfy_d(Val(0), s, X)
  Yvec = detensorfy_d(Val(0), s, Y)

  color = sqrt.(Xvec.^2 + Yvec.^2)

  fig = Figure(size=(1000, 1000));
  ax = CairoMakie.Axis(fig[1,1])

  sd = dual_mesh(s)
  msh = CairoMakie.mesh!(ax, sd, color=press, colormap=:jet)
  arrows2d!(ax, x, y, Xvec, Yvec, color = color, normalize = true, lengthscale = 0.01)
  Colorbar(fig[1, 2], msh)

  fig
end

function plotcav(index)
  plot_vars(s, Us[index], press_s[index])
end


plotcav(20)

