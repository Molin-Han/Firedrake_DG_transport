from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Mesh
# For Flux problem to be solved correctly with BC


# Problem existing: on boundary condition, the flux solved is
# not correct!

#mesh = PeriodicUnitSquareMesh(40,40)
mesh = UnitSquareMesh(40, 40)
#space
#degree of space
deg = 1
V = FunctionSpace(mesh, "DG", deg)
#W = VectorFunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "DG", 1)

x, y = SpatialCoordinate(mesh)


# Velocity field
# divergence 0
#velocity = as_vector(( (-0.05*x - y + 0.475 ) , ( x - 0.05*y-0.525)))
# faster divergence 0
velocity = as_vector(( (0.5 - y ) , ( x - 0.5)))

# velocity satisfies the periodic boundary.
#velocity = as_vector((-sin(pi * x)* cos(pi * y), cos(pi*x)* sin(pi *y)))


u = Function(W).interpolate(velocity)

# velocity from stream function(periodic velocity field)

#stream = FunctionSpace(mesh,"CG", 2)
#stream_func = Function(stream).interpolate(1/ pi * sin(pi*x)*sin(pi*y))
#u = Function(W).interpolate(as_vector((-stream_func.dx(1),stream_func.dx(0))))


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
rho_data = File('BA_Euler_rho.pvd')
q_data = File('BA_Euler_q.pvd')



#Initial setting for time
#time period
T = 2*math.pi/40
dt = 2* math.pi /120000
dtc = Constant(dt)
rho_in = Constant(1.0)
q_in = Constant(1.0)



#trial functions and test functions
# rho^(n+1) for updated density
rho_new = Function(V)

drho_trial = TrialFunction(V)
dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*drho_trial*dx
b = phi*rho_new*dq_trial*dx

#elements
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))




#Set Kuzmin limiter
limiter_rho = VertexBasedLimiter(V)
limiter_q = VertexBasedLimiter(V)




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
print(norm(Courant_plus))

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




# Set for the Flux Limiter for Density Equation
beta = Function(DG1)

rho_bar = Function(DG1)
rho_hat_bar = Function(DG1)

#set for the expression for beta
c_plus = Courant_plus * rho / rho_hat_bar
c_minus = Courant_minus * rho / rho_hat_bar

beta_expr = Max(0, Min(1, (1 + Courant_minus - Courant_plus)/(c_minus - c_plus - Courant_minus + Courant_plus)))


# Density Equation Variational Problem
L_rho = dtc*(rho*dot(grad(phi),u)*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*rho_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*rho, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*rho('+') - un('-')*rho('-'))*dS)
drho = Function(V)




#Flux Problem
# Surface Flux equation - build RT2 out of BDM1 and TDG1
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
params_flux = {'ksp_type': 'preonly', 'pc_type': 'lu','mat_type': 'aij','pc_factor_mat_solver_type':'mumps'}
Fsproblem = LinearVariationalProblem(aFs, LFs, Fs)
Fssolver = LinearVariationalSolver(Fsproblem,solver_parameters=params_flux)
Fssolver.solve()
Fsf,phi_flux= split(Fs)
Fn  = 0.5*(dot((Fsf), n) + abs(dot((Fsf), n)))




# Flux limiting for q.
alpha = Function(DG1)
q_bar = Function(DG1)
q_hat_bar = Function(DG1)

# q+- factor in alpha

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
alpha_expr = 0
alpha_min_expr = Constant(1.0)
#alpha_min_expr = Min(1, (q_hat_bar * (1 - c_plus) + c_minus * q_minus - (1 + c_minus + c_plus)* qmin) / (c_plus * (q_plus - q_hat_bar)))



#variational problem for q
L_q = phi * rho * q * dx + dtc*(q*dot(grad(phi),Fsf)*dx
          - conditional(dot(Fsf, n) < 0, phi*dot(Fsf, n)*q_in, 0.0)*ds
          - conditional(dot(Fsf, n) > 0, phi*dot(Fsf, n)*q, 0.0)*ds
          - (phi('+') - phi('-'))*(Fn('+')*q('+') - Fn('-')*q('-'))*dS)
qnew = Function(V)




# set solvers for rho and q.
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob_rho = LinearVariationalProblem(a, L_rho, drho)
solv_rho = LinearVariationalSolver(prob_rho, solver_parameters=params)

prob_q = LinearVariationalProblem(b, L_q, qnew)
solv_q = LinearVariationalSolver(prob_q, solver_parameters=params)



# begin looping
t = 0.0
step = 0
output_freq = 2
# stage
i = 0

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
#rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))
print(f"stage{i},rho_max=", rho.dat.data.max())
print(f"stage{i},rho_min=", rho.dat.data.min())

q_bar.project(q)
limiter_q.apply(q)
q_hat_bar.project(q)
alpha.assign(Min(alpha_expr,alpha_min_expr))
q.project(q_hat_bar + alpha * (q - q_hat_bar))
print(f"stage{i},q_max=", q.dat.data.max())
print(f"stage{i},q_min=", q.dat.data.min())



# Main Body
# solve the density and the bounded advection
while t < T - 0.5*dt:

    #first stage
    #For Flux, it should be solved before rho is solved depending on the way it's defined.
    Fssolver.solve()
    #solv1_rho.solve()
    #rho_new.assign(rho + drho)
    #rho.assign(rho_new)
    #rho limiting scheme, beta1 found.
    rho_bar.project(rho)
    limiter_rho.apply(rho)
    rho_hat_bar.project(rho)
    beta.assign(beta_expr)
    #apply the limiting scheme
    rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))
    #For rho_1
    solv_rho.solve()
    rho_new.assign(rho + drho)

    #For q_1
    solv_q.solve()
    q.assign(qnew)


    #q limiting scheme
    q_bar.project(q)
    limiter_q.apply(q)
    q_hat_bar.project(q)
    alpha.assign(Min(alpha_expr,alpha_min_expr))
    q.project(q_hat_bar + alpha * (q - q_hat_bar))


    #rho.assign(rho_new)
    rho.assign(rho + drho)

    print(f'stage{i},rho_max=', rho.dat.data.max())
    print(f'stage{i},rho_min=', rho.dat.data.min())
    #print(f'stage{i},q_max=', q.dat.data.max())
    #print(f'stage{i},q_min=', q.dat.data.min())

    #$rho_data.write(rho)
    #q_data.write(q)

    #update the step and proceed to the next time step.
    i+=1
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