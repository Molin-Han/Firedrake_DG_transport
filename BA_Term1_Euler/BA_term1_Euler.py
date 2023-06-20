import firedrake as fd
import math
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


# Mesh
# For Flux problem to be solved correctly with BC

mesh = fd.PeriodicUnitSquareMesh(40, 40)
#mesh = fd.UnitSquareMesh(40, 40)
# space
# degree of space
deg = 1
V = fd.FunctionSpace(mesh, "DG", deg)
# W = fd.VectorFunctionSpace(mesh, "CG", 1)
W = fd.VectorFunctionSpace(mesh, "DG", 1)
Velo = fd.FunctionSpace(mesh, 'BDM', 1)

x, y = fd.SpatialCoordinate(mesh)


# Velocity field
# divergence 0
# velocity = fd.as_vector(( (-0.05*x - y + 0.475 ) , ( x - 0.05*y-0.525)))
# faster divergence 0
# velocity = fd.as_vector(((0.5 - y), (x - 0.5)))

# velocity satisfies the periodic boundary.
# velocity = fd.as_vector((- fd.sin(fd.pi * x) * fd.cos(fd.pi * y), fd.cos(fd.pi * x) * fd.sin(fd.pi * y)))


# Trial Velocity
# velocity = fd.as_vector((0.5 - x + y, x - y - 0.5))

# u = fd.Function(W).interpolate(velocity)

# velocity from stream function(periodic velocity field)

stream = fd.FunctionSpace(mesh, "CG", 2)
stream_func = fd.Function(stream).interpolate(1 / fd.pi * fd.sin(fd.pi * x) * fd.sin(fd.pi * y))
#u = fd.Function(Velo).interpolate(fd.as_vector((-stream_func.dx(1), stream_func.dx(0))))
velocity_psi = fd.as_vector((-stream_func.dx(1), stream_func.dx(0)))

velocity_phi = fd.as_vector((-0.2*fd.pi*fd.sin(2*fd.pi*x)*fd.cos(2*fd.pi*y), -0.2*fd.pi*fd.cos(2*fd.pi*x)*fd.sin(2*fd.pi*y)))
#velocity_psi = fd.as_vector((-0.2*fd.pi*fd.sin(2*fd.pi*x)*fd.cos(2*fd.pi*y), 0.2*fd.pi*fd.cos(2*fd.pi*x)*fd.sin(2*fd.pi*y)))
u = fd.Function(W).interpolate(velocity_phi+velocity_psi)

#u = fd.Function(W).interpolate(phi + psi)


#u = fd.Function(W).interpolate(fd.as_vector((fd.Constant(0), fd.Constant(1.0))) + 0.01*velocity_phi)
# initial condition for the atomsphere
bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+fd.cos(math.pi*fd.min_value(fd.sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - fd.min_value(fd.sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
slot_cyl = fd.conditional(fd.sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
            fd.conditional(fd.And(fd.And(x > slot_left, x < slot_right), y < slot_top),
            0.0, 1.0), 0.0)
# + bell + cone + slot_cyl
# initial condition for density equation
#rho = fd.Function(V).interpolate(1.0 + bell + cone + slot_cyl)
rho = fd.Function(V).interpolate(fd.Constant(1.0))
rho_init = fd.Function(V).assign(rho)
# initial condition for advection equation
#q = fd.Function(V).interpolate(bell)
q = fd.Function(V).interpolate(bell + cone + slot_cyl)
q_init = fd.Function(V).assign(q)
print("initial maxmimum for q", q.dat.data.max())


# solution list
rhos = []
qs = []
rho_data = fd.File('BA_Euler_rho.pvd')
q_data = fd.File('BA_Euler_q.pvd')

alpha_data = fd.File('alpha.pvd')


# Initial setting for time
# time period
T = 2 * math.pi / 8
dt = 2 * math.pi / 3200 # make it bigger
dtc = fd.Constant(dt)
rho_in = fd.Constant(1.0)
q_in = fd.Constant(1.0)


# trial functions and test functions
# rho^(n+1) for updated density
rho_new = fd.Function(V)

drho_trial = fd.TrialFunction(V)
dq_trial = fd.TrialFunction(V)
phi = fd.TestFunction(V)
a = phi*drho_trial*fd.dx
b = phi*rho_new*dq_trial*fd.dx

# elements
n = fd.FacetNormal(mesh)
un = 0.5*(fd.dot(u, n) + abs(fd.dot(u, n)))


# Set Kuzmin limiter
limiter_rho = fd.VertexBasedLimiter(V)
limiter_q = fd.VertexBasedLimiter(V)


# Courant number setting
def both(vec):
    return vec('+') + vec('-')


# Set for the Flux Limiter for Density Equation
DG0 = fd.FunctionSpace(mesh, "DG", 0)
beta = fd.Function(DG0)

rho_bar = fd.Function(DG0)
rho_hat_bar = fd.Function(DG0)
rho_bar.project(rho)
limiter_rho.apply(rho)
rho_hat_bar.project(rho)

# check

DG1 = fd.FunctionSpace(mesh, "DG", deg)
One = fd.Function(DG0).assign(1.0)
v = fd.TestFunction(DG0)


# ctil+
Courant_num_plus = fd.Function(DG0)
Courant_num_form_plus = dt*(
    both(un*v)*(fd.dS)
    + un*v*fd.ds
)
Courant_denom_plus = fd.Function(DG0)
Courant_plus = fd.Function(DG0)

fd.assemble(One*v*fd.dx, tensor=Courant_denom_plus)
fd.assemble(Courant_num_form_plus, tensor=Courant_num_plus)
Courant_plus.interpolate(Courant_num_plus/Courant_denom_plus)
print("Courant number for the initial problem", fd.norm(Courant_plus))

# ctil-
Courant_num_minus = fd.Function(DG0)
Courant_num_form_minus = -dt*(
    both((fd.inner(u, n)-un)*v)*(fd.dS)
    + (fd.inner(u, n)-un)*v*fd.ds
)
Courant_denom_minus = fd.Function(DG0)
Courant_minus = fd.Function(DG0)

fd.assemble(One*v*fd.dx, tensor=Courant_denom_minus)
fd.assemble(Courant_num_form_minus, tensor=Courant_num_minus)
Courant_minus.interpolate(Courant_num_minus / Courant_denom_minus)


# c+
c_num_plus = fd.Function(DG0)
c_num_form_plus = dt*(
    both(un*v*rho/rho_hat_bar)*(fd.dS)
    + un*v*rho/rho_hat_bar*fd.ds
)
c_denom_plus = fd.Function(DG0)
c_plus = fd.Function(DG0)

fd.assemble(One*v*fd.dx, tensor=c_denom_plus)
fd.assemble(c_num_form_plus, tensor=c_num_plus)
c_plus.interpolate(c_num_plus/c_denom_plus)


# c-
c_num_minus = fd.Function(DG0)
c_num_form_minus = -dt*(
    both((fd.inner(u, n)-un)*v*rho / rho_hat_bar)*(fd.dS)
    + (fd.inner(u, n)-un)*v*rho/rho_hat_bar*fd.ds
)
c_denom_minus = fd.Function(DG0)
c_minus = fd.Function(DG0)


fd.assemble(One*v*fd.dx, tensor=c_denom_minus)
fd.assemble(c_num_form_minus, tensor=c_num_minus)
c_minus.interpolate(c_num_minus/c_denom_minus)



print('#####################c-c+', c_plus.dat.data.max(), c_minus.dat.data.max())
# set for the expression for beta
# c_plus = Courant_plus * rho / rho_hat_bar
# c_minus = Courant_minus * rho / rho_hat_bar
#####------------------########
# two version of beta, one for colin's one for molin's

# colin's version is used here
beta_expr = fd.max_value(0, fd.min_value(1, (1 + c_minus - Courant_plus)/(c_plus - Courant_minus)))


# Density Equation Variational Problem
L_rho = dtc*(rho*fd.dot(fd.grad(phi), u)*fd.dx
             - fd.conditional(fd.dot(u, n) < 0, phi*fd.dot(u, n)*rho_in, 0.0)*fd.ds
             - fd.conditional(fd.dot(u, n) > 0, phi*fd.dot(u, n)*rho, 0.0)*fd.ds
             - (phi('+') - phi('-'))*(un('+')*rho('+') - un('-')*rho('-'))*fd.dS)
drho = fd.Function(V)


# Flux Problem
# Surface Flux equation - build RT2 out of BDM1 and TDG1
Fluxes = fd.FunctionSpace(mesh, "RT", 2)
Inners = fd.VectorFunctionSpace(mesh, "DG", 0)
W = fd.MixedFunctionSpace((Fluxes, Inners))

wF, wI = fd.TestFunctions(W)
uF, phi_flux = fd.TrialFunctions(W)

aFs = (
    0.5 * (fd.inner(wF('+'), n('+'))*fd.inner(uF('+'), n('+'))
           + fd.inner(wF('-'), n('-')) * fd.inner(uF('-'), n('-')))*fd.dS(metadata={'quadrature_degree': 4})
    + fd.inner(wF, n)*fd.inner(uF, n) * fd.ds
    + fd.inner(wI, uF) * fd.dx
    + fd.inner(wF, phi_flux) * fd.dx
    )
LFs = (
    (fd.inner(wF('+'), n('+'))*un('+')*rho('+')
     + fd.inner(wF('-'), n('-'))*un('-')*rho('-'))*fd.dS(metadata={'quadrature_degree': 4})
    + fd.inner(wF, n) * un * rho * fd.ds
    + fd.inner(wF, n) * (1-un) * rho_in * fd.ds
    + fd.inner(wI, u)*rho*fd.dx
    )
Fs = fd.Function(W)
params_flux = {'ksp_type': 'preonly', 'pc_type': 'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
Fsproblem = fd.LinearVariationalProblem(aFs, LFs, Fs)
Fssolver = fd.LinearVariationalSolver(Fsproblem, solver_parameters=params_flux)
Fssolver.solve()
Fsf, phi_flux = fd.split(Fs)
Fn = 0.5*(fd.dot((Fsf), n) + abs(fd.dot((Fsf), n)))


# Flux limiting for q.
alpha = fd.Function(DG0)
q_bar = fd.Function(DG0)
q_hat_bar = fd.Function(DG0)
q_bar.project(q)
limiter_q.apply(q)
print("Maximum value of q after applying VB limiter", q.dat.data.max())
q_hat_bar.project(q)
# q+- factor in alpha ################
# TODO : q+- are nan!: c is 0 somewhere change to c+q+=F+ product


F_plus = fd.Function(DG0)
w = fd.TestFunction(DG0)
F_plus_num = fd.Function(DG0)
F_plus_form = both(Fn * w * q) * fd.dS + Fn * w * q * fd.ds
fd.assemble(F_plus_form, tensor=F_plus_num)
F_plus.interpolate(F_plus_num)

F_minus = fd.Function(DG0)
F_minus_num = fd.Function(DG0)
F_minus_form = -(both((fd.inner(Fsf, n) - Fn) * w * q) * fd.dS
                   + (fd.inner(Fsf, n) - Fn) * w * q * fd.ds)
fd.assemble(F_minus_form, tensor=F_minus_num)
F_minus.interpolate(F_minus_num)

print('##################################F-F+', F_plus.dat.data.max(), F_minus.dat.data.max())



# maximum bound for q
# local max and min should be used?
qmax = fd.Constant(1.0)
qmin = fd.Constant(0.0)



# set alpha
alpha_expr = ((1 + c_minus - c_plus) * qmax - q_hat_bar * (1 - c_plus) - F_minus) / (c_plus * q_hat_bar - F_plus)
alpha_min_expr = (q_hat_bar * (1 - c_plus) + F_minus - (1 + c_minus - c_plus) * qmin) / (F_plus - c_plus * q_hat_bar)



# variational problem for q
L_q = phi * rho * q * fd.dx + dtc * (q * fd.dot(fd.grad(phi), Fsf) * fd.dx
                                     - fd.conditional(fd.dot(Fsf, n) < 0, phi*fd.dot(Fsf, n)*q_in, 0.0) * fd.ds
                                     - fd.conditional(fd.dot(Fsf, n) > 0, phi*fd.dot(Fsf, n)*q, 0.0) * fd.ds
                                     - (phi('+') - phi('-'))*(Fn('+')*q('+') - Fn('-')*q('-'))*fd.dS)
qnew = fd.Function(V)


# set solvers for rho and q.
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob_rho = fd.LinearVariationalProblem(a, L_rho, drho)
solv_rho = fd.LinearVariationalSolver(prob_rho, solver_parameters=params)

prob_q = fd.LinearVariationalProblem(b, L_q, qnew)
solv_q = fd.LinearVariationalSolver(prob_q, solver_parameters=params)

psi = fd.TestFunction(V)
q_trial = fd.TrialFunction(V)
c = psi * rho * q_trial * fd.dx
Q = fd.Function(V)
L_lim_q = psi*Q*fd.dx

prob_Q = fd.LinearVariationalProblem(c, L_lim_q, q)
solv_Q = fd.LinearVariationalSolver(prob_Q, solver_parameters=params)


# begin looping
t = 0.0
step = 0
output_freq = 1
# stage
i = 0

if step % output_freq == 0:
    rhos.append(rho.copy(deepcopy=True))
    qs.append(q.copy(deepcopy=True))
    print("t=", t)

# find beta, alpha.
beta.interpolate(beta_expr)
alpha.interpolate(fd.min_value(fd.conditional(c_plus*q_hat_bar-F_plus>0, alpha_expr, alpha_min_expr),1))
print("alpha_after_interpolate", alpha.dat.data.min())
rho_data.write(rho)
q_data.write(q)

omega = fd.Constant(30.0)


##########6.13 check for (37) condition in limiter notes
cond_file = fd.File('cond.pvd')
cond1_func = fd.Function(DG0)
cond0_func = fd.Function(DG0)
cond1 = (1+c_minus-c_plus) *qmax-q_hat_bar*(1-c_plus)-F_minus + F_plus - c_plus*q_hat_bar
cond0 = (1+c_minus-c_plus) *qmax-q_hat_bar*(1-c_plus)-F_minus
#cond = q_hat_bar*(1-c_plus)-F_minus
#cond = (1+c_minus-c_plus)

indicator = fd.Function(DG0)
#ind_file = fd.File('ind.pvd')

beta_file = fd.File('beta.pvd')
# Main Body
# solve the density and the bounded advection
while t < T - 0.5*dt:
    print(f'##########################stage{i} starts#############')
    # time dependent velocity field to make it compressible
    u.interpolate(0.2*fd.cos(omega*fd.Constant(t))*velocity_phi + velocity_psi)
    #u.interpolate(fd.cos(omega*fd.Constant(t))*velocity_phi + velocity_psi)
    #u.interpolate(fd.as_vector((fd.Constant(0), fd.Constant(1.0))) + 0.01*fd.cos(omega*fd.Constant(t))*velocity_phi)

    # convergence test
    #if t >= T/4:
        #u.interpolate(-0.1*fd.cos(omega*fd.Constant(t))*velocity_phi - velocity_psi)

    # update the courant numbers
    fd.assemble(Courant_num_form_plus, tensor=Courant_num_plus)
    Courant_plus.interpolate(Courant_num_plus/Courant_denom_plus)
    fd.assemble(Courant_num_form_minus, tensor=Courant_num_minus)
    Courant_minus.interpolate(Courant_num_minus/Courant_denom_minus)
    fd.assemble(c_num_form_plus, tensor=c_num_plus)
    c_plus.interpolate(c_num_plus/c_denom_plus)
    fd.assemble(c_num_form_minus, tensor=c_num_minus)
    c_minus.interpolate(c_num_minus/c_denom_minus)


    # For Flux, it should be solved before rho is solved depending on the way it's defined.
    Fssolver.solve()


    # update q+- value
    fd.assemble(F_plus_form, tensor=F_plus_num)
    F_plus.interpolate(F_plus_num)
    fd.assemble(F_minus_form, tensor=F_minus_num)
    F_minus.interpolate(F_minus_num)

    print('###########FFFFF', F_minus.dat.data.max(), F_plus.dat.data.max())
    # rho limiting scheme, beta found.
    rho_bar.project(rho)
    # VB limiter
    #limiter_rho.apply(rho)
    rho_hat_bar.project(rho)
    beta.interpolate(beta_expr)
    # TODO: test for beta and visualise beta
    #beta_file.write(beta)
    print('beta max and beta min', beta.dat.data.max(), beta.dat.data.min())
    # apply the flux limiting scheme
    #rho.project(rho_hat_bar + beta * (rho - rho_hat_bar)) # old rho after Kuzmin and flux limiter applied
    # solve for density
    solv_rho.solve()
    # rho n+1
    rho_new.interpolate(rho + drho)

    # q limiting scheme
    Q.project(rho*q)
    limiter_q.apply(Q)
    solv_Q.solve()
    # FIXME: changed qbar
    q_hat_bar.project(q * rho / rho_hat_bar)

    #cond_func.interpolate(cond)
    #print('!!!!condition',cond_func.dat.data.max(),cond_func.dat.data.min())
    #cond_file.write(cond_func)
    # TODO:Alpha negative
    cond0_func.interpolate(cond0)
    cond1_func.interpolate(cond1)
    print('cond0',cond0_func.dat.data.max(),cond0_func.dat.data.min())
    print('cond01',cond1_func.dat.data.max(),cond1_func.dat.data.min())

    alpha.interpolate(fd.conditional(cond0>0,fd.conditional(cond1>0,1,fd.conditional(c_plus*q_hat_bar-F_plus>0, alpha_expr, alpha_min_expr)), -99999))
    #alpha.interpolate(fd.min_value(fd.conditional(c_plus*q_hat_bar-F_plus>0, alpha_expr, alpha_min_expr),1))
    #alpha.interpolate(fd.max_value(fd.min_value(fd.conditional(c_plus*q_hat_bar-F_plus>0, alpha_expr, alpha_min_expr),1),0))

    ####### tesing for qmax>1

    #indicator.interpolate(fd.conditional(q-fd.Constant(1.0+1e-6)>0, 1, 0))
    #ind_file.write(indicator)
    #print('ind function',indicator.dat.data.max())
    #if i>=22:
        #alpha.interpolate(fd.Constant(0.0))
    #alpha.interpolate(fd.Max(0,fd.Min(alpha_expr, alpha_min_expr)))
    #alpha.interpolate(fd.Min(alpha_expr, alpha_min_expr))
    #TODO:check alpha negative
    print("Alpha_max and Alpha_min", alpha.dat.data.max(), alpha.dat.data.min())


    q.project(q_hat_bar + alpha * (q - q_hat_bar))
    #q.project(q_hat_bar+1*(q-q_hat_bar))
    solv_q.solve()
    q.interpolate(qnew)

    # update rho value
    rho.interpolate(rho + drho)
    limiter_rho.apply(rho)
    #rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))
    limiter_q.apply(q)
    #q.project(q_hat_bar + alpha * (q - q_hat_bar))

    print(f'stage{i},rho_max=', rho.dat.data.max())
    print(f'stage{i},rho_min=', rho.dat.data.min())
    print(f'stage{i},q_max=', q.dat.data.max())
    print(f'stage{i},q_min=', q.dat.data.min())
    print('courant number!!!!!!!!', Courant_plus.dat.data.max())

    # update the step and proceed to the next time step.
    i += 1
    step += 1
    t += dt

    if step % output_freq == 0:
        rhos.append(rho.copy(deepcopy=True))
        qs.append(q.copy(deepcopy=True))
        print("t=", t)
        rho_data.write(rho)
        q_data.write(q)
        alpha_data.write(alpha)


L2_err_rho = fd.sqrt(fd.assemble((rho - rho_init)*(rho - rho_init) * fd.dx))
L2_init_rho = fd.sqrt(fd.assemble(rho_init * rho_init * fd.dx))
print("error_rho =", L2_err_rho/L2_init_rho)

L2_err_q = fd.sqrt(fd.assemble((q - q_init)*(q - q_init) * fd.dx))
L2_init_q = fd.sqrt(fd.assemble(q_init * q_init * fd.dx))
print("error_q =", L2_err_q/L2_init_q)
