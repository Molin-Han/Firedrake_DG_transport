from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mesh = UnitSquareMesh(40, 40, quadrilateral=True)

V = FunctionSpace(mesh, "DQ", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)

velocity = as_vector(( (0.5 - y ) , ( x - 0.5) ))
u = Function(W).interpolate(velocity)


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
pho = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
pho_init = Function(V).assign(pho)

phos = []

T = 2*math.pi
dt = T/1200
dtc = Constant(dt)
pho_in = Constant(1.0)

dpho_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*dpho_trial*dx

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

L1 = dtc*(pho*(div(phi*u)-div(u))*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*pho_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*pho, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*pho('+') - un('-')*pho('-'))*dS)


pho1 = Function(V); pho2 = Function(V)
L2 = replace(L1, {pho: pho1}); L3 = replace(L1, {pho: pho2})

dpho = Function(V)

params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a, L1, dpho)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dpho)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, dpho)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)


#Set Kuzmin limiter
limiter = VertexBasedLimiter(V)

t = 0.0
step = 0
output_freq = 20


if step % output_freq == 0:
    phos.append(pho.copy(deepcopy=True))
    print("t=", t)

#Apply the limiter to q first.
limiter.apply(pho)
print(pho.dat.data.max())


#Main body

while t < T - 0.5*dt:
    solv1.solve()
    pho1.assign(pho + dpho)
    limiter.apply(pho1)

    solv2.solve()
    pho1.assign(pho1+dpho)
    limiter.apply(pho1)
    pho2.assign(0.75*pho + 0.25*(pho1))
    limiter.apply(pho2)

    solv3.solve()
    pho2.assign(pho2+dpho)
    limiter.apply(pho2)
    pho.assign((1.0/3.0)*pho + (2.0/3.0)*(pho2))
    limiter.apply(pho)


    print(pho.dat.data.max())
    step += 1
    t += dt

    if step % output_freq == 0:
        phos.append(pho.copy(deepcopy=True))
        print("t=", t)



L2_err = sqrt(assemble((pho - pho_init)*(pho - pho_init)*dx))
L2_init = sqrt(assemble(pho_init*pho_init*dx))
print(L2_err/L2_init)


#nsp = 16
#fn_plotter = FunctionPlotter(mesh, num_sample_points=nsp)

#fig, axes = plt.subplots()
#axes.set_aspect('equal')
#colors = tripcolor(q_init, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
#fig.colorbar(colors)

#def animate(q):
    #colors.set_array(fn_plotter(q))

#interval = 1e3 * output_freq * dt
#animation = FuncAnimation(fig, animate, frames=qs, interval=interval)
#try:
    #animation.save("DG_advection_oscillating1.mp4", writer="ffmpeg")
#except:
    #print("Failed to write movie! Try installing `ffmpeg`.")