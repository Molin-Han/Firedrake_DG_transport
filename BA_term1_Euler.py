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


beta_expr = Max(0, Min(1, (1 + Courant_minus - Courant_plus)/(c_minus - c_plus - Courant_minus + Courant_plus)))


