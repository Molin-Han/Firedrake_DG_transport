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
velocity = as_vector(( (0.05*x - y+0.475 ) , ( x + 0.05*y-0.525)))
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
#initial condition for advection equation
q = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
q_init = Function(V).assign(q)


#solution list
rhos = []
qs = []

#time period
T = 2*math.pi
dt = T/1200
dtc = Constant(dt)
rho_in = Constant(1.0)

q_in = Constant(1.0)

drho_trial = TrialFunction(V)
dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*drho_trial*dx
b = phi*dq_trial*dx


#elements
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

#variational problems for density
L1_rho = dtc*(rho*dot(grad(phi),u)*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*rho_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*rho, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*rho('+') - un('-')*rho('-'))*dS)


rho1 = Function(V); rho2 = Function(V)
L2_rho = replace(L1_rho, {rho: rho1}); L3_rho = replace(L1_rho, {rho: rho2})

drho = Function(V)


#Flux Problem
# Surface Flux equation - build RT2 out of BDM1 and TDG1
Fluxes = FunctionSpace(mesh,"BDM",1)
Inners = FunctionSpace(mesh,"DRT",1)
W = MixedFunctionSpace((Fluxes,Inners))

wI = TestFunction(Inners)
assemble(inner(wI,u)*dx)

wF,wI = TestFunctions(W)
uF,uI = TrialFunctions(W)

aFs = (
    (inner(wF('+'),n('+'))*inner(uF('+'),n('+')) + 
     inner(wF('-'),n('-'))*inner(uF('-'),n('-')))*dS
    + inner(wI,uI)*dx
    )
LFs = (
    2.0*(inner(wF('+'),n('+'))*un('+')*rho('+') 
         + inner(wF('-'),n('-'))*un('-')*rho('-'))*dS
    + inner(wI,u)*rho*dx
    )

Fs = Function(W)

Fsproblem = LinearVariationalProblem(aFs, LFs, Fs)
Fssolver = LinearVariationalSolver(Fsproblem)
Fssolver.solve()
Fn  = 0.5*(dot(Fs, n) + abs(dot(Fs, n)))

#variational problem for q
L1_q = dtc*(q*dot(grad(phi),Fs)*dx
          - conditional(dot(Fs, n) < 0, phi*dot(Fs, n)*q_in, 0.0)*ds
          - conditional(dot(Fs, n) > 0, phi*dot(Fs, n)*q, 0.0)*ds
          - (phi('+') - phi('-'))*(Fn('+')*q('+') - Fn('-')*q('-'))*dS)

q1 = Function(V); q2 = Function(V)
L2_q = replace(L1_q, {q: q1}); L3_q = replace(L1_q, {q: q2})
dq = Function(V)




params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1_rho = LinearVariationalProblem(a, L1_rho, drho)
solv1_rho = LinearVariationalSolver(prob1_rho, solver_parameters=params)
prob2_rho = LinearVariationalProblem(a, L2_rho, drho)
solv2_rho = LinearVariationalSolver(prob2_rho, solver_parameters=params)
prob3_rho = LinearVariationalProblem(a, L3_rho, drho)
solv3_rho = LinearVariationalSolver(prob3_rho, solver_parameters=params)


prob1_q = LinearVariationalProblem(a, L1_q, dq)
solv1_q = LinearVariationalSolver(prob1_q, solver_parameters=params)
prob2_q = LinearVariationalProblem(a, L2_q, dq)
solv2_q = LinearVariationalSolver(prob2_q, solver_parameters=params)
prob3_q = LinearVariationalProblem(a, L3_q, dq)
solv3_q = LinearVariationalSolver(prob3_q, solver_parameters=params)




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
limiter.apply(q)
print(q.dat.data.max())




#Main body

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

    #solve the flux problem
    Fssolver.solve()

    solv1_q.solve()
    q1.assign(q + dq)
    limiter.apply(q1)

    solv2_q.solve()
    q1.assign(q1+dq)
    limiter.apply(q1)
    q2.assign(0.75*q + 0.25*(q1))
    limiter.apply(q2)

    solv3_q.solve()
    q2.assign(q2+dq)
    limiter.apply(q2)
    q.assign((1.0/3.0)*q + (2.0/3.0)*(q2))
    limiter.apply(q)



    step += 1
    t += dt

    if step % output_freq == 0:
        rhos.append(rho.copy(deepcopy=True))
        qs.append(q.copy(deepcopy=True))
        print("t=", t)



L2_err_rho = sqrt(assemble((rho - rho_init)*(rho - rho_init)*dx))
L2_init_rho = sqrt(assemble(rho_init*rho_init*dx))
print(L2_err_rho/L2_init_rho)

L2_err_q = sqrt(assemble((q - q_init)*(q - q_init)*dx))
L2_init_q = sqrt(assemble(q_init*q_init*dx))
print(L2_err_q/L2_init_q)


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
animation_q = FuncAnimation(fig, animate, frames=qs, interval=interval)
try:
    animation_rho.save("DG_rho_1.mp4", writer="ffmpeg")
    animation_q.save("DG_q_1.mp4", writer="ffmpeg")
except:
    print("Failed to write movie! Try installing `ffmpeg`.")