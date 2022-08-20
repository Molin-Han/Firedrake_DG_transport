from firedrake import *
mesh = UnitSquareMesh(40, 40)
# Define function spaces and basis functions
V_dg = FunctionSpace(mesh, "DG", 1)
M = FunctionSpace(mesh, "RT", 2)

# advecting velocity
u0 = Expression(('-x[1]','x[0]','0'))
u = Function(M).project(u0)

# Mesh-related functions
n = FacetNormal(mesh)

# ( dot(v, n) + |dot(v, n)| )/2.0
un = 0.5*(dot(u, n) + abs(dot(u, n)))

dt = 1.0

# D advection equation
phi = TestFunction(V_dg)
D = TrialFunction(V_dg)
a_mass = phi*D*dx
a_int = dot(grad(phi), -u*D)*dx
a_flux = ( dot(jump(phi), un('+')*D('+') - un('-')*D('-')) )*dS

arhs = (a_int + a_flux)

D1 = Function(V_dg)

D0 = Expression("exp(-pow(x[2],2) - pow(x[1],2))")
D = Function(V_dg).interpolate(D0)

t = 0.0
T = 2*pi
k = 0
dumpfreq = 50

D1problem = LinearVariationalProblem(a_mass, action(arhs,D), D1)
D1solver = LinearVariationalSolver(D1problem)
D1solver.solve()

# Surface Flux equation - build RT2 out of BDM1 and TDG1
Fluxes = FunctionSpace(mesh,"BDM",1)
Inners = VectorFunctionSpace(mesh,"DG",0)
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
    2.0*(inner(wF('+'),n('+'))*un('+')*D('+') 
         + inner(wF('-'),n('-'))*un('-')*D('-'))*dS
    + inner(wI,u)*D*dx
    )

Fs = Function(W)

Fsproblem = LinearVariationalProblem(aFs, LFs, Fs)
Fssolver = LinearVariationalSolver(Fsproblem)
Fssolver.solve()