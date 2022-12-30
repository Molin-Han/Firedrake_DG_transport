import firedrake as fd
import math
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


# Mesh
# For Flux problem to be solved correctly with BC


# Problem existing: on boundary condition, the flux solved is
# not correct!

mesh = fd.PeriodicUnitSquareMesh(40, 40)
# mesh = fd.UnitSquareMesh(40, 40)
# space
# degree of space
deg = 1
V = fd.FunctionSpace(mesh, "DG", deg)
# W = fd.VectorFunctionSpace(mesh, "CG", 1)
W = fd.VectorFunctionSpace(mesh, "DG", 1)

x, y = fd.SpatialCoordinate(mesh)


# Velocity field
# divergence 0
# velocity = fd.as_vector(( (-0.05*x - y + 0.475 ) , ( x - 0.05*y-0.525)))
# faster divergence 0
# velocity = fd.as_vector(( (0.5 - y ) , ( x - 0.5)))

# velocity satisfies the periodic boundary.
velocity = fd.as_vector((- fd.sin(fd.pi * x) * fd.cos(fd.pi * y), fd.cos(fd.pi * x) * fd.sin(fd.pi * y)))


# u = fd.Function(W).interpolate(velocity)

# velocity from stream function(periodic velocity field)

stream = fd.FunctionSpace(mesh, "CG", 2)
stream_func = fd.Function(stream).interpolate(1 / fd.pi * fd.sin(fd.pi * x) * fd.sin(fd.pi * y))
u = fd.Function(W).interpolate(fd.as_vector((-stream_func.dx(1), stream_func.dx(0))))


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
rho = fd.Function(V).interpolate(1.0 + bell + cone + slot_cyl)
rho_init = fd.Function(V).assign(rho)
# initial condition for advection equation
q = fd.Function(V).interpolate(bell + cone + slot_cyl)
q_init = fd.Function(V).assign(q)
print("initial maxmimum for q", q.dat.data.max())


# solution list
rhos = []
qs = []
rho_data = fd.File('BA_Euler_rho.pvd')
q_data = fd.File('BA_Euler_q.pvd')


# Initial setting for time
# time period
T = 2 * math.pi / 40
dt = 2 * math.pi / 1200
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


DG1 = fd.FunctionSpace(mesh, "DG", deg)
One = fd.Function(DG1).assign(1.0)
v = fd.TestFunction(DG1)


# c+
Courant_num_plus = fd.Function(DG1)
Courant_num_form_plus = dt*(
    both(un*v)*(fd.dS)
    + un*v*fd.ds
)
Courant_denom_plus = fd.Function(DG1)
Courant_plus = fd.Function(DG1)

fd.assemble(One*v*fd.dx, tensor=Courant_denom_plus)
fd.assemble(Courant_num_form_plus, tensor=Courant_num_plus)
Courant_plus.assign(Courant_num_plus/Courant_denom_plus)
print("Courant number for the initial problem",fd.norm(Courant_plus))

# c-
Courant_num_minus = fd.Function(DG1)
Courant_num_form_minus = dt*(
    both(-un*v)*(fd.dS)
    - un*v*fd.ds
)
Courant_denom_minus = fd.Function(DG1)
Courant_minus = fd.Function(DG1)

fd.assemble(One*v*fd.dx, tensor=Courant_denom_minus)
fd.assemble(Courant_num_form_minus, tensor=Courant_num_minus)
Courant_minus.assign(Courant_num_minus / Courant_denom_minus)


# Set for the Flux Limiter for Density Equation
beta = fd.Function(DG1)

rho_bar = fd.Function(DG1)
rho_hat_bar = fd.Function(DG1)

# set for the expression for beta
c_plus = Courant_plus * rho / rho_hat_bar
c_minus = Courant_minus * rho / rho_hat_bar

beta_expr = fd.Max(0, fd.Min(1, (1 + Courant_minus - Courant_plus)/(c_minus - c_plus - Courant_minus + Courant_plus)))


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
alpha = fd.Function(DG1)
q_bar = fd.Function(DG1)
q_hat_bar = fd.Function(DG1)

# q+- factor in alpha

q_plus = fd.Function(DG1)
w = fd.TestFunction(DG1)
q_plus_num = fd.Function(DG1)
q_plus_form = both(Fn * w) * fd.dS + Fn * w * q * fd.ds
fd.assemble(q_plus_form, tensor=q_plus_num)
q_plus.assign((1/c_plus) * q_plus_num)

q_minus = fd.Function(DG1)
q_minus_num = fd.Function(DG1)
q_minus_form = -both(Fn * w) * fd.dS - Fn * w * q_in * fd.ds
fd.assemble(q_minus_form, tensor=q_minus_num)
q_minus.assign((1/c_minus) * q_minus_num)

# maximum bound for q
qmax = fd.Constant(2.0)
qmin = fd.Constant(1.0)
# set alpha
alpha_expr = fd.Min(1, ((1 + c_minus - c_plus) * qmax - q_hat_bar * (1 - c_plus) - c_minus * q_minus) / (c_plus * (q_hat_bar - q_plus)))
# alpha_expr = 0
# alpha_min_expr = fd.Constant(1.0)
alpha_min_expr = fd.Min(1, (q_hat_bar * (1 - c_plus) + c_minus * q_minus - (1 + c_minus + c_plus) * qmin) / (c_plus * (q_plus - q_hat_bar)))


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

# Apply the limiter to q and density first and find beta, alpha.
rho_bar.project(rho)
limiter_rho.apply(rho)
rho_hat_bar.project(rho)
# Here rho is the value after applying the Kuzmin limiter i.e. rho_hat
beta.assign(beta_expr)
# apply the limiting scheme to density
# rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))
print(f"stage{i},rho_max=", rho.dat.data.max())
print(f"stage{i},rho_min=", rho.dat.data.min())

q_bar.project(q)
limiter_q.apply(q)
print("Maximum value of q after applying limiter", q.dat.data.max())
q_hat_bar.project(q)


alpha_expr_max = fd.Function(DG1)
alpha_expr_min = fd.Function(DG1)
alpha_expr_max.assign(alpha_expr)
alpha_expr_min.assign(alpha_min_expr)
print("Alpha_max and Alpha_min", alpha_expr_max.dat.data.min(), alpha_expr_min.dat.data.min())
alpha.assign(0)
print("alpha_before", alpha.dat.data.min())
alpha.interpolate(fd.Min(alpha_expr, alpha_min_expr))
print("alpha_after_interpolate", alpha.dat.data.min())
print("q_limiter_off", q.dat.data.max())
#q.interpolate(q_hat_bar + alpha * (q - q_hat_bar))
q.project(q_hat_bar + alpha * (q - q_hat_bar))
print("q_hat_bar", q_hat_bar.dat.data.max())
print("q_limiter_on", q.dat.data.max())
print(f"stage{i},q_max=", q.dat.data.max())
print(f"stage{i},q_min=", q.dat.data.min())


# Main Body
# solve the density and the bounded advection
while t < T - 0.5*dt:

    # first stage
    # For Flux, it should be solved before rho is solved depending on the way it's defined.
    Fssolver.solve()
    # solv1_rho.solve()
    # rho_new.assign(rho + drho)
    # rho.assign(rho_new)
    # rho limiting scheme, beta1 found.
    rho_bar.project(rho)
    limiter_rho.apply(rho)
    rho_hat_bar.project(rho)
    beta.assign(beta_expr)
    # apply the limiting scheme
    rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))
    # For rho_1
    solv_rho.solve()
    rho_new.assign(rho + drho)

    # For q_1
    solv_q.solve()
    q.assign(qnew)

    # q limiting scheme
    q_bar.project(q)
    limiter_q.apply(q)
    q_hat_bar.project(q)
    alpha.interpolate(fd.Min(alpha_expr, alpha_min_expr))
    q.project(q_hat_bar + alpha * (q - q_hat_bar))

    # rho.assign(rho_new)
    rho.assign(rho + drho)
    # ## Testing
    # rho limiting scheme, beta1 found.
    rho_bar.project(rho)
    limiter_rho.apply(rho)
    rho_hat_bar.project(rho)
    beta.assign(beta_expr)
    # apply the limiting scheme
    rho.project(rho_hat_bar + beta * (rho - rho_hat_bar))

    print(f'stage{i},rho_max=', rho.dat.data.max())
    print(f'stage{i},rho_min=', rho.dat.data.min())
    print(f'stage{i},q_max=', q.dat.data.max())
    print(f'stage{i},q_min=', q.dat.data.min())

    rho_data.write(rho)
    q_data.write(q)

    # update the step and proceed to the next time step.
    i += 1
    step += 1
    t += dt

    if step % output_freq == 0:
        rhos.append(rho.copy(deepcopy=True))
        qs.append(q.copy(deepcopy=True))
        print("t=", t)


L2_err_rho = fd.sqrt(fd.assemble((rho - rho_init)*(rho - rho_init) * fd.dx))
L2_init_rho = fd.sqrt(fd.assemble(rho_init * rho_init * fd.dx))
print("error_rho =", L2_err_rho/L2_init_rho)

L2_err_q = fd.sqrt(fd.assemble((q - q_init)*(q - q_init) * fd.dx))
L2_init_q = fd.sqrt(fd.assemble(q_init * q_init * fd.dx))
print("error_q =", L2_err_q/L2_init_q)
