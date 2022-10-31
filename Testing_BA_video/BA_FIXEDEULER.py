from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#mesh
#for Flux to be solved correctly
mesh = PeriodicUnitSquareMesh(40,40)
#mesh = UnitSquareMesh(40, 40)

#space
#degree of space
deg = 1
V = FunctionSpace(mesh, "DG", deg)
W = VectorFunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)
#Initial setting for the problem
#velocity field
#velocity = as_vector(( (-0.05*x - y + 0.475 ) , ( x - 0.05*y-0.525)))
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
#initial condition for advection equation
q = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
q_init = Function(V).assign(q)


#solution list
rhos = []
qs = []

#Initial setting for time
#time period
T = 2*math.pi/40
dt = 2* math.pi /1200
dtc = Constant(dt)
rho_in = Constant(1.0)

q_in = Constant(1.0)

rho_new = Function(V)
#trial functions and test functions
drho_trial = TrialFunction(V)
dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*drho_trial*dx
b = phi*rho_new*dq_trial*dx


#elements
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))


#Courant number setting
def both(vec):
    return vec('+') + vec('-')


DG1 = FunctionSpace(mesh, "DG", deg)
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


#Set for the flux limiter for density.

rho1 = Function(V); rho2 = Function(V)


beta = Function(DG1)
beta1 = Function(DG1)
beta2 = Function(DG1)

rho_bar = Function(DG1)
rho_hat_bar = Function(DG1)

rho1_bar = Function(DG1)
rho1_hat_bar = Function(DG1)

rho2_bar = Function(DG1)
rho2_hat_bar = Function(DG1)


#set for the expression for beta
c_plus = Courant_plus * rho / rho_hat_bar
c_minus = Courant_minus * rho / rho_hat_bar

c1_plus = Courant_plus * rho1 / rho1_hat_bar
c1_minus = Courant_minus * rho1 / rho1_hat_bar

c2_plus = Courant_plus * rho2 / rho2_hat_bar
c2_minus = Courant_minus * rho2 / rho2_hat_bar

beta_expr = Max(0, Min(1, (1 + Courant_minus - Courant_plus)/(c_minus - c_plus - Courant_minus + Courant_plus)))

beta1_expr = Max(0, Min(1, (1 + Courant_minus - Courant_plus)/(c1_minus - c1_plus - Courant_minus + Courant_plus)))

beta2_expr = Max(0, Min(1, (1 + Courant_minus - Courant_plus)/(c2_minus - c2_plus - Courant_minus + Courant_plus)))







#variational problems for density
L1_rho = dtc*(rho*dot(grad(phi),u)*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*rho_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*rho, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*rho('+') - un('-')*rho('-'))*dS)



L2_rho = replace(L1_rho, {rho: rho1}); L3_rho = replace(L1_rho, {rho: rho2})

drho = Function(V)


#Flux Problem
# Surface Flux equation - build RT2 out of BDM1 and TDG1
Fluxes1 = FunctionSpace(mesh,"RT",2)
Inners1 = VectorFunctionSpace(mesh,"DG",0)
W1 = MixedFunctionSpace((Fluxes1,Inners1))


wF1,wI1 = TestFunctions(W1)
uF1,phi1_flux = TrialFunctions(W1)

aFs1 = (
    0.5 * (inner(wF1('+'),n('+'))*inner(uF1('+'),n('+')) + 
     inner(wF1('-'),n('-'))*inner(uF1('-'),n('-')))*dS(metadata={'quadrature_degree':4})
    +inner(wF1,n)*inner(uF1,n) * ds
    + inner(wI1,uF1)*dx
    + inner(wF1,phi1_flux)*dx
    )
LFs1 = (
    (inner(wF1('+'),n('+'))*un('+')*rho('+') 
         + inner(wF1('-'),n('-'))*un('-')*rho('-'))*dS(metadata={'quadrature_degree':4})
    + inner(wF1,n)* un * rho * ds
    + inner(wF1,n)* (1-un) * rho_in * ds
    + inner(wI1,u)*rho*dx
    )

Fs1 = Function(W1)
params_flux = {'ksp_type': 'preonly', 'pc_type': 'lu','mat_type': 'aij','pc_factor_mat_solver_type':'mumps'}
Fsproblem1 = LinearVariationalProblem(aFs1, LFs1, Fs1)
Fssolver1 = LinearVariationalSolver(Fsproblem1,solver_parameters=params_flux)
Fssolver1.solve()
Fsf1,phi1_flux = split(Fs1)
Fn1  = 0.5*(dot((Fsf1), n) + abs(dot((Fsf1), n)))



Fluxes2 = FunctionSpace(mesh,"RT",2)
Inners2 = VectorFunctionSpace(mesh,"DG",0)
W2 = MixedFunctionSpace((Fluxes2,Inners2))


wF2,wI2 = TestFunctions(W2)
uF2,phi2_flux = TrialFunctions(W2)


aFs2 = (
    0.5 * (inner(wF2('+'),n('+'))*inner(uF2('+'),n('+')) + 
     inner(wF2('-'),n('-'))*inner(uF2('-'),n('-')))*dS(metadata={'quadrature_degree':4})
    +inner(wF2,n)*inner(uF2,n) * ds
    + inner(wI2,uF2)*dx
    + inner(wF2,phi2_flux)*dx
    )
LFs2 = (
    (inner(wF2('+'),n('+'))*un('+')*rho('+') 
         + inner(wF2('-'),n('-'))*un('-')*rho('-'))*dS(metadata={'quadrature_degree':4})
    + inner(wF2,n)* un * rho * ds
    + inner(wF2,n)* (1-un) * rho_in * ds
    + inner(wI2,u)*rho*dx
    )

Fs2 = Function(W2)
Fsproblem2 = LinearVariationalProblem(aFs2, LFs2, Fs2)
Fssolver2 = LinearVariationalSolver(Fsproblem2,solver_parameters=params_flux)
Fssolver2.solve()
Fsf2,phi2_flux = split(Fs2)
Fn2  = 0.5*(dot((Fsf2), n) + abs(dot((Fsf2), n)))



Fluxes = FunctionSpace(mesh,"RT",2)
Inners = VectorFunctionSpace(mesh,"DG",0)
W = MixedFunctionSpace((Fluxes,Inners))


wF,wI = TestFunctions(W)
uF,phi_flux = TrialFunctions(W)

aFs = (
    0.5 * (inner(wF('+'),n('+'))*inner(uF('+'),n('+')) + 
     inner(wF('-'),n('-'))*inner(uF('-'),n('-')))*dS(metadata={'quadrature_degree':4})
    +inner(wF,n)*inner(uF,n) * ds
    + inner(wI,uF)*dx
    + inner(wF,phi_flux)*dx
    )
LFs = (
    (inner(wF('+'),n('+'))*un('+')*rho('+') 
         + inner(wF('-'),n('-'))*un('-')*rho('-'))*dS(metadata={'quadrature_degree':4})
    + inner(wF,n)* un * rho * ds
    + inner(wF,n)* (1-un) * rho_in * ds
    + inner(wI,u)*rho*dx
    )
Fs = Function(W)
Fsproblem = LinearVariationalProblem(aFs, LFs, Fs)
Fssolver = LinearVariationalSolver(Fsproblem,solver_parameters=params_flux)
Fssolver.solve()
Fsf,phi_flux= split(Fs)
Fn  = 0.5*(dot((Fsf), n) + abs(dot((Fsf), n)))





#set the flux limiter for tracer q.

q1 = Function(V); q2 = Function(V)


alpha = Function(DG1)
alpha1 = Function(DG1)
alpha2 = Function(DG1)

q_bar = Function(DG1)
q_hat_bar = Function(DG1)

q1_bar = Function(DG1)
q1_hat_bar = Function(DG1)

q2_bar = Function(DG1)
q2_hat_bar = Function(DG1)


# q+- factor in alpha


q1_plus = Function(DG1)
w1 = TestFunction(DG1)
q1_plus_num = Function(DG1)
q1_plus_form = both(Fn1 * w1) * dS + Fn1 * w1 *q * ds
assemble(q1_plus_form, tensor=q1_plus_num)
q1_plus.assign((1/c_plus) * q1_plus_num)

q1_minus = Function(DG1)
q1_minus_num = Function(DG1)
q1_minus_form = -both(Fn1 * w1)  * dS - Fn1 * w1 *q_in * ds
assemble(q1_minus_form, tensor=q1_minus_num)
q1_minus.assign((1/c_minus) * q1_minus_num)


q2_plus = Function(DG1)
w2 = TestFunction(DG1)
q2_plus_num = Function(DG1)
q2_plus_form = both(Fn2 * w2)  * dS + Fn2 * w2 * q * ds
assemble(q2_plus_form, tensor=q2_plus_num)
q2_plus.assign((1/c_plus) * q2_plus_num)

q2_minus = Function(DG1)
q2_minus_num = Function(DG1)
q2_minus_form = -both(Fn2 * w2) * dS - Fn2 * w2 * q_in * ds
assemble(q2_minus_form, tensor=q2_minus_num)
q2_minus.assign((1/c_minus) * q2_minus_num)




q_plus = Function(DG1)
w = TestFunction(DG1)
q_plus_num = Function(DG1)
q_plus_form = both(Fn * w)  * dS + Fn * w * q *ds
assemble(q_plus_form, tensor=q_plus_num)
q_plus.assign((1/c_plus) * q_plus_num)

q_minus = Function(DG1)
q_minus_num = Function(DG1)
q_minus_form =- both(Fn * w)  * dS - Fn * w * q_in  *ds
assemble(q_minus_form, tensor=q_minus_num)
q_minus.assign((1/c_minus) * q_minus_num)


#maximum bound for q
qmax = Constant(2.0)
qmin = Constant(1.0)

#set alpha
#alpha_expr = Min(1, ((1 + c_minus - c_plus)* qmax - q_hat_bar * (1 - c_plus) - c_minus * q_minus) / (c_plus * (q_hat_bar - q_plus)))

#alpha1_expr = Min(1, ((1 + c_minus - c_plus)* qmax - q1_hat_bar * (1 - c_plus) - c_minus * q1_minus) / (c_plus * (q1_hat_bar - q1_plus)))

#alpha2_expr = Min(1, ((1 + c_minus - c_plus)* qmax - q2_hat_bar * (1 - c_plus) - c_minus * q2_minus) / (c_plus * (q2_hat_bar - q2_plus)))


alpha_expr = 0
alpha1_expr = 0
alpha2_expr = 0


alpha_min_expr = Constant(1.0)
alpha1_min_expr = Constant(1.0)
alpha2_min_expr = Constant(1.0)
#alpha_min_expr = Min(1, (q_hat_bar * (1 - c_plus) + c_minus * q_minus - (1 + c_minus + c_plus)* qmin) / (c_plus * (q_plus - q_hat_bar)))
#alpha1_min_expr = Min(1, (q1_hat_bar * (1 - c_plus) + c_minus * q1_minus - (1 + c_minus + c_plus)* qmin) / (c_plus * (q1_plus - q1_hat_bar)))
#alpha2_min_expr = Min(1, (q2_hat_bar * (1 - c_plus) + c_minus * q2_minus - (1 + c_minus + c_plus)* qmin) / (c_plus * (q2_plus - q2_hat_bar)))


#variational problem for q
L1_q = dtc*(q*dot(grad(phi),Fsf)*dx
          - conditional(dot(Fsf, n) < 0, phi*dot(Fsf, n)*q_in, 0.0)*ds
          - conditional(dot(Fsf, n) > 0, phi*dot(Fsf, n)*q, 0.0)*ds
          - (phi('+') - phi('-'))*(Fn('+')*q('+') - Fn('-')*q('-'))*dS)
L1_q += phi * rho * q * dx


L2_q = replace(L1_q, {q: q1, Fsf: Fsf1, Fn: Fn1})
L3_q = replace(L1_q, {q: q2, Fsf: Fsf2, Fn: Fn2})
qnew = Function(V)




# set solvers for rho and q.
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1_rho = LinearVariationalProblem(a, L1_rho, drho)
solv1_rho = LinearVariationalSolver(prob1_rho, solver_parameters=params)
prob2_rho = LinearVariationalProblem(a, L2_rho, drho)
solv2_rho = LinearVariationalSolver(prob2_rho, solver_parameters=params)
prob3_rho = LinearVariationalProblem(a, L3_rho, drho)
solv3_rho = LinearVariationalSolver(prob3_rho, solver_parameters=params)


prob1_q = LinearVariationalProblem(b, L1_q, qnew)
solv1_q = LinearVariationalSolver(prob1_q, solver_parameters=params)
prob2_q = LinearVariationalProblem(b, L2_q, qnew)
solv2_q = LinearVariationalSolver(prob2_q, solver_parameters=params)
prob3_q = LinearVariationalProblem(b, L3_q, qnew)
solv3_q = LinearVariationalSolver(prob3_q, solver_parameters=params)

rho_data = File('BA_Euler_rho.pvd')
q_data = File('BA_Euler_q.pvd')

#Set Kuzmin limiter
limiter_rho = VertexBasedLimiter(V)
limiter_q = VertexBasedLimiter(V)

t = 0.0
step = 0
output_freq = 2


if step % output_freq == 0:
    rhos.append(rho.copy(deepcopy=True))
    qs.append(q.copy(deepcopy=True))
    print("t=", t)



#Apply the limiter to q and density first and find beta, alpha.
rho_bar.project(rho)
limiter_rho.apply(rho)
rho_hat_bar.project(rho)
#Here rho is the value after applying the Kuzmin limiter i.e. rho_hat
beta.assign(beta_expr)
#apply the limiting scheme to density
rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))
print("rho_max=", rho.dat.data.max())
print("rho_min=", rho.dat.data.min())

q_bar.project(q)
limiter_q.apply(q)
q_hat_bar.project(q)
alpha.assign(Min(alpha_expr,alpha_min_expr))
q.project(q_hat_bar + alpha * (q - q_hat_bar))
print("q_max=", q.dat.data.max())
print("q_min=", q.dat.data.min())

#Main body

while t < T - 0.5*dt:
    #solve the density and the bounded advection


    #first stage
    #For Flux_1, it should be solved before rho is solved depending on the way it's defined.
    Fssolver1.solve()

    #rho limiting scheme, beta1 found.
    rho_bar.project(rho)
    limiter_rho.apply(rho)
    rho_hat_bar.project(rho)
    beta.assign(beta_expr)
    #apply the limiting scheme
    rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))

    #For rho_1
    solv1_rho.solve()
    rho_new.assign(rho + drho)

    #For q_1
    solv1_q.solve()
    q.assign(qnew)


    #q limiting scheme
    q_bar.project(q)
    limiter_q.apply(q)
    q_hat_bar.project(q)
    alpha.assign(Min(alpha_expr,alpha_min_expr))
    q.project(q_hat_bar + alpha * (q - q_hat_bar))


    rho.assign(rho_new)



    print("rho_max=", rho.dat.data.max())
    print("rho_min=", rho.dat.data.min())
    
    print("q_max=", q.dat.data.max())
    print("q_min=", q.dat.data.min())

    rho_data.write(rho)
    q_data.write(q)

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
    animation_rho.save("BA_rho_1.mp4", writer="ffmpeg")
    animation_q.save("BA_q_1.mp4", writer="ffmpeg")
except:
    print("Failed to write movie! Try installing `ffmpeg`.")