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
#velocity field
velocity = as_vector(( (-0.05*x - y+0.475 ) , ( x - 0.05*y-0.525)))
u = Function(W).interpolate(velocity)

#initial condition
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
T = 2*math.pi
dt = T/1200
dtc = Constant(dt)
rho_in = Constant(1.0)

drho_trial = TrialFunction(V)

phi = TestFunction(V)
a = phi*drho_trial*dx

#elements
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

def both(vec):
    return vec('+') + vec('-')
#Courant number setting

DG0 = FunctionSpace(mesh, "DG", 0)
One = Function(DG0).assign(1.0)
v = TestFunction(DG0)
Courant_num = Function(DG0, name="Courant numerator")
Courant_num_form = dt*(
    both(un*v)*(dS)
    + un*v*ds
)
Courant_denom = Function(DG0, name="Courant denominator")
assemble(One*v*dx, tensor=Courant_denom)
Courant = Function(DG0, name="Courant")


assemble(Courant_num_form, tensor=Courant_num)
Courant.assign(Courant_num/Courant_denom)
print(Courant)

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


#Set Kuzmin limiter
limiter = VertexBasedLimiter(V)

t = 0.0
step = 0
output_freq = 20
if step % output_freq == 0:
    rhos.append(rho.copy(deepcopy=True))
    print("t=", t)

#Apply the limiter to q and density first.
limiter.apply(rho)
print(rho.dat.data.max())

while t < T - 0.5*dt:
    #solve the density
    solv1_rho.solve()
    rho1.assign(rho + drho)
    limiter.apply(rho1)

    solv2_rho.solve()
    rho1.assign(rho1+drho)
    limiter.apply(rho1)
    rho2.assign(0.75*rho + 0.25*(rho1))
    limiter.apply(rho2)

    solv3_rho.solve()
    rho2.assign(rho2+drho)
    limiter.apply(rho2)
    rho.assign((1.0/3.0)*rho + (2.0/3.0)*(rho2))
    limiter.apply(rho)

    print(rho.dat.data.max())


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
    animation_rho.save("DG_rho_non-div_out.mp4", writer="ffmpeg")
except:
    print("Failed to write movie! Try installing `ffmpeg`.")