from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#mesh
mesh = UnitSquareMesh(40, 40)
#space
V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)
#Initial setting for the problem.
#velocity field
#velocity = as_vector(( (-0.05*x - y+0.475 ) , ( x - 0.05*y-0.525)))
velocity = as_vector(( (0.5 - y ) , ( x - 0.5)))
u = Function(W).interpolate(velocity)

#initial condition for the atomsphere
bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
            conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
            0.0, 1.0), 0.0)
#+ bell + cone + slot_cyl
#initial condition for density equation
rho = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
rho_init = Function(V).assign(rho)

#solution list
rhos = []

#time period
T = 6*math.pi
dt = T/3600
dtc = Constant(dt)
rho_in = Constant(1.0)

drho_trial = TrialFunction(V)

phi = TestFunction(V)
a = phi*drho_trial*dx

#Set Kuzmin limiter
limiter = VertexBasedLimiter(V)


#elements
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

def both(vec):
    return vec('+') + vec('-')
#Courant number setting
##############CHANGED HERE!!!!!!!!!!!!!
DG1 = FunctionSpace(mesh, "DG", 1)
One = Function(DG1).assign(1.0)
v = TestFunction(DG1)
#c+
Courant_num_plus= Function(DG1)
Courant_num_form_plus = dt*(
    both(un*v)*(dS)
    + un*v*ds
)
Courant_denom_plus = Function(DG1)
Courant_plus = Function(DG1)


assemble(One*v*dx, tensor=Courant_denom_plus)
assemble(Courant_num_form_plus, tensor=Courant_num_plus)
Courant_plus.assign(Courant_num_plus/Courant_denom_plus)


#c-
Courant_num_minus = Function(DG1)
Courant_num_form_minus  = dt*(
    both(-un*v)*(dS)
    - un*v*ds
)
Courant_denom_minus  = Function(DG1)
Courant_minus  = Function(DG1)


assemble(One*v*dx, tensor=Courant_denom_minus )
assemble(Courant_num_form_minus , tensor=Courant_num_minus )
Courant_minus.assign(Courant_num_minus /Courant_denom_minus )

#Set for the second limiter.
beta = Function(DG1)
beta1 = Function(DG1)
beta2 = Function(DG1)

rho_bar = Function(DG1)
rho_hat_bar = Function(DG1)

rho1_bar = Function(DG1)
rho1_hat_bar = Function(DG1)

rho2_bar = Function(DG1)
rho2_hat_bar = Function(DG1)



#c_plus = Courant_plus * rho_hat / rho_hat_bar
#c_minus = Courant_minus * rho_hat / rho_hat_bar

#beta_expr_colin = Max(0, Min(1, (1 + c_minus - Courant_plus) / (c_plus - Courant_plus)))
#beta_expr_molin = Max(0, Min(1, (1 + Courant_minus - Courant_plus) / (c_minus - c_plus - Courant_minus + Courant_plus)))

#variational problems for density
L1_rho = dtc*(rho*dot(grad(phi),u)*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*rho_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*rho, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*rho('+') - un('-')*rho('-'))*dS)


rho1 = Function(V); rho2 = Function(V)
L2_rho = replace(L1_rho, {rho: rho1}); L3_rho = replace(L1_rho, {rho: rho2})

drho = Function(V)

params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1_rho = LinearVariationalProblem(a, L1_rho, drho)
solv1_rho = LinearVariationalSolver(prob1_rho, solver_parameters=params)
prob2_rho = LinearVariationalProblem(a, L2_rho, drho)
solv2_rho = LinearVariationalSolver(prob2_rho, solver_parameters=params)
prob3_rho = LinearVariationalProblem(a, L3_rho, drho)
solv3_rho = LinearVariationalSolver(prob3_rho, solver_parameters=params)




t = 0.0
step = 0
output_freq = 20
if step % output_freq == 0:
    rhos.append(rho.copy(deepcopy=True))
    print("t=", t)


#Apply the limiter to q and density first and find beta.
rho_bar.project(rho)
limiter.apply(rho)
rho_hat_bar.project(rho)
#here rho is rho_hat as the limiter is applied
#beta.assign(Max(0, Min(1, (1 + Courant_minus - Courant_plus*rho_hat_bar/rho_bar)
#/(Courant_plus - Courant_plus*rho_hat_bar/rho_bar))))
beta.assign(Max(0, Min(1, (1 + Courant_minus - Courant_plus)/(Courant_minus * rho / rho_hat_bar - Courant_plus * rho / rho_hat_bar - Courant_minus + Courant_plus))))
#apply the limiting scheme
rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))
print(rho.dat.data.max())
print(rho.dat.data.min())
i = 0
while t < T - 0.5*dt:
    #solve the density


    #first stage
    solv1_rho.solve()
    rho1.assign(rho + drho)

    #rho limiting scheme, beta1 found.
    rho1_bar.project(rho1)
    limiter.apply(rho1)
    rho1_hat_bar.project(rho1)
    beta1.assign(Max(0, Min(1, (1 + Courant_minus - Courant_plus)
    /(Courant_minus * rho1 / rho1_hat_bar - Courant_plus*rho1 / rho1_hat_bar - Courant_minus + Courant_plus))))
    #apply the limiting scheme
    rho1.project(rho1_hat_bar + beta1 * (rho1 - rho1_hat_bar))


    #second stage
    solv2_rho.solve()
    rho1.assign(rho1+drho)
    #limiter apply to rho1 another time.
    limiter.apply(rho1)
    rho1_hat_bar.project(rho1)
    beta1.assign(Max(0, Min(1, (1 + Courant_minus - Courant_plus)
    /(Courant_minus * rho1 / rho1_hat_bar - Courant_plus*rho1 / rho1_hat_bar - Courant_minus + Courant_plus))))
    #apply the limiting scheme
    rho1.project(rho1_hat_bar + beta1 * (rho1 - rho1_hat_bar))
    
    #Calculate for the second stage rho value.
    rho2.assign(0.75*rho + 0.25*(rho1))
    rho2_bar.project(rho2)
    limiter.apply(rho2)
    rho2_hat_bar.project(rho2)
    beta2.assign(Max(0, Min(1, (1 + Courant_minus - Courant_plus)
    /(Courant_minus * rho2 / rho2_hat_bar - Courant_plus*rho2 / rho2_hat_bar - Courant_minus + Courant_plus))))
    #apply the limiting scheme
    rho2.project(rho2_hat_bar + beta2 * (rho2 - rho2_hat_bar))


    #third stage
    solv3_rho.solve()
    rho2.assign(rho2+drho)

    #limiter apply to rho2 another time.
    limiter.apply(rho2)
    rho2_hat_bar.project(rho2)
    beta2.assign(Max(0, Min(1, (1 + Courant_minus - Courant_plus)
    /(Courant_minus * rho2 / rho2_hat_bar - Courant_plus*rho2 / rho2_hat_bar - Courant_minus + Courant_plus))))
    #apply the limiting scheme
    rho2.project(rho2_hat_bar + beta2 * (rho2 - rho2_hat_bar))

    #Calculate for the rho value.
    rho.assign((1.0/3.0)*rho + (2.0/3.0)*(rho2))
    rho_bar.project(rho)
    limiter.apply(rho)
    rho_hat_bar.project(rho)
    beta.assign(Max(0, Min(1, (1 + Courant_minus - Courant_plus)
    /(Courant_minus * rho / rho_hat_bar - Courant_plus*rho / rho_hat_bar - Courant_minus + Courant_plus))))
    #apply the limiting scheme
    rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))

    print(f'stage{i}',rho.dat.data.max())
    print(f'stage{i}',rho.dat.data.min())

    i +=1
    step += 1
    t += dt

    if step % output_freq == 0:
        rhos.append(rho.copy(deepcopy=True))
        print("t=", t)

L2_err_rho = sqrt(assemble((rho - rho_init)*(rho - rho_init)*dx))
L2_init_rho = sqrt(assemble(rho_init*rho_init*dx))
print(L2_err_rho/L2_init_rho)

nsp = 16
fn_plotter = FunctionPlotter(mesh, num_sample_points=nsp)

fig, axes = plt.subplots()
axes.set_aspect('equal')
colors = tripcolor(rho_init, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
fig.colorbar(colors)

def animate(q):
    colors.set_array(fn_plotter(q))


interval = 1e3 * output_freq * dt
animation_rho = FuncAnimation(fig, animate, frames=rhos, interval=interval)


try:
    animation_rho.save("DG_density_2limiters_long_test_simple_test.mp4", writer="ffmpeg")
except:
    print("Failed to write movie! Try installing `ffmpeg`.")