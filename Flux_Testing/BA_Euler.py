from importlib.metadata import metadata
from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#mesh
mesh = PeriodicUnitSquareMesh(40,40)
#mesh = UnitSquareMesh(40, 40)

#space
deg= 1
V = FunctionSpace(mesh, "DG", deg)
W = VectorFunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)
#Initial setting for the problem
#velocity field
#velocity = as_vector(( (-0.05*x - y + 0.475 ) , ( x - 0.05*y-0.525)))
velocity = as_vector(( (-sin(2*math.pi*x)*cos(2*math.pi*y) ) , ( cos(2*math.pi*x)*sin(math.pi*y))))
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
#initial condition for advection equation
q = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
q_init = Function(V).assign(q)


#solution list
rhos = []
qs = []

#Initial setting for time
#time period
T = 2 * math.pi /200
dt = 2* math.pi /1200
dtc = Constant(dt)
rho_in = Constant(1.0)

q_in = Constant(1.0)


#trial functions and test functions
drho_trial = TrialFunction(V)
dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*drho_trial*dx
b = phi*dq_trial*dx


#elements
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))



DG1 = FunctionSpace(mesh, "DG", deg)
One = Function(DG1).assign(1.0)
v = TestFunction(DG1)




#variational problems for density
L1_rho = dtc*(rho*dot(grad(phi),u)*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*rho_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*rho, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*rho('+') - un('-')*rho('-'))*dS(metadata={'quadrature_degree':4}))


drho = Function(V)


#Flux Problem
# Surface Flux equation - build RT2 out of BDM1 and TDG1

#Fluxes = FunctionSpace(mesh,"BDM",1)
Fluxes= FunctionSpace(mesh,"RT",2)
Inners = VectorFunctionSpace(mesh,"DG",0)
#Inners = FunctionSpace(mesh,"DRT",1)
#Trials = FunctionSpace(mesh,"RT",2)
W = MixedFunctionSpace((Fluxes,Inners))





wF,wI = TestFunctions(W)
uF,phi = TrialFunctions(W)


aFs = (
    0.5 * (inner(wF('+'),n('+'))*inner(uF('+'),n('+')) + 
     inner(wF('-'),n('-'))*inner(uF('-'),n('-')))*dS(metadata={'quadrature_degree':4})
    +inner(wF,n)*inner(uF,n) * ds
    + inner(wI,uF)*dx
    + inner(wF,phi)*dx
    )
LFs = (
    (inner(wF('+'),n('+'))*un('+')*rho('+') 
         + inner(wF('-'),n('-'))*un('-')*rho('-'))*dS(metadata={'quadrature_degree':4})
    + inner(wF,n)* un * rho * ds
    + inner(wF,n)* (1-un) * rho_in * ds
    + inner(wI,u)*rho*dx
    )

Fs = Function(W)
params = {'ksp_type': 'preonly', 'pc_type': 'lu','mat_type': 'aij','pc_factor_mat_solver_type':'mumps'}
Fsproblem = LinearVariationalProblem(aFs, LFs, Fs)
Fssolver = LinearVariationalSolver(Fsproblem,solver_parameters=params)
Fssolver.solve()
Fsf,phi = split(Fs)
Fnew = Fsf
#Fn = Function(DG1)
Fn=(0.5*(dot((Fnew), n) + abs(dot((Fnew), n))))

#variational problem for q
#L1_q = dtc*(q*dot(grad(phi),Fnew)*dx
          #- conditional(dot(Fnew, n) < 0, phi*dot(Fnew, n)*q_in, 0.0)*ds
          #- conditional(dot(Fnew, n) > 0, phi*dot(Fnew, n)*q, 0.0)*ds
          #- (phi('+') - phi('-'))*(Fn('+')*q('+') - Fn('-')*q('-'))*dS)

dq = Function(V)

# set solvers for rho and q.
params1 = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1_rho = LinearVariationalProblem(a, L1_rho, drho)
solv1_rho = LinearVariationalSolver(prob1_rho, solver_parameters=params1)

#prob1_q = LinearVariationalProblem(b, L1_q, dq)
#solv1_q = LinearVariationalSolver(prob1_q, solver_parameters=params1)


#Set Kuzmin limiter
limiter_rho = VertexBasedLimiter(V)
limiter_q = VertexBasedLimiter(V)


func = Function(DG1)
rho_prev = Function(DG1)


t = 0.0
step = 0
output_freq = 20

if step % output_freq == 0:
    rhos.append(rho.copy(deepcopy=True))
    qs.append(q.copy(deepcopy=True))
    print("t=", t)
print("rho_max=", rho.dat.data.max())
print("rho_min=", rho.dat.data.min())

print("q_max=", q.dat.data.max())
print("q_min=", q.dat.data.min())

#residual = assemble(pow(rho - rho_prev - dt * div(Fnew),2) *dx)
#print("residual=",residual)


f = File('flux1.pvd')

#Apply the limiter to q and density first and find beta, alpha.

while t < T - 0.5*dt:

    Fssolver.solve()
    solv1_rho.solve()
    rho_prev.assign(rho)

    rho.assign(rho + drho)



    func.project(drho + dt * div(Fnew))
    print("func_norm=",norm(func))
    f.write(func)


    residual = norm(func)
    print("residual=",residual)


    #solv1_q.solve()
    #q.assign(q + dq)



    print("rho_max=", rho.dat.data.max())
    print("rho_min=", rho.dat.data.min())

    print("q_max=", q.dat.data.max())
    print("q_min=", q.dat.data.min())



    #update the step and proceed to the next time step.
    step += 1
    t += dt

    if step % output_freq == 0:

        rhos.append(rho.copy(deepcopy=True))
        qs.append(q.copy(deepcopy=True))
        print("t=", t)



L2_err_rho = sqrt(assemble((rho - rho_init)*(rho - rho_init)*dx))
L2_init_rho = sqrt(assemble(rho_init*rho_init*dx))
print("error_rho =", L2_err_rho/L2_init_rho)

L2_err_q = sqrt(assemble((q - q_init)*(q - q_init)*dx))
L2_init_q = sqrt(assemble(q_init*q_init*dx))
print("error_q =", L2_err_q/L2_init_q)


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
    animation_rho.save("BA_noq_rho_1.mp4", writer="ffmpeg")
    animation_q.save("BA_noq_q_1.mp4", writer="ffmpeg")
except:
    print("Failed to write movie! Try installing `ffmpeg`.")