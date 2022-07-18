from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mesh = UnitSquareMesh(40, 40, quadrilateral=True)

#make the function space
V = FunctionSpace(mesh, "DQ", 1)

W = VectorFunctionSpace(mesh, "CG", 1)

x,y = SpatialCoordinate(mesh)

velocity = as_vector((0.5-y, x-0.5))
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

q = Function(V).interpolate(1.0 + bell + cone + slot_cy1)
q_init = Function(V).assign(q)

qs = []

T = 2 * math
dt = T / 600.0
dtc = Constant(dt)
q_in = Constant(1.0)



dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi * dq_trial * dx

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))
