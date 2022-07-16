from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mesh = UnitSquareMesh(40, 40, quadrilateral=True)

#make the function space
V = FunctionSpace(mesh, "DQ", 1)

W = VectorFunctionSpace(mesh, "CG", 1)

