from ufl import sin, SpatialCoordinate, FiniteElement, VectorElement, MixedElement, TestFunction, TrialFunction, split, Identity, Measure, dx, ds, grad, nabla_grad, div, dot, inner, tr, as_vector, FacetNormal

# Auxiliary packages
from petsc4py import PETSc # Linear algebra backend
from mpi4py import MPI     # MPI 
import numpy as np

# Import dolfinx and ufl modules
from dolfinx import geometry
from dolfinx import fem
from dolfinx import io
from dolfinx import la
from dolfinx.mesh import (CellType, create_unit_square, locate_entities, exterior_facet_indices, create_rectangle, meshtags)
from ufl import TestFunction, TrialFunction, dx, inner, grad, lhs, rhs
from petsc4py.PETSc import ScalarType
from functools import partial 
from matplotlib import pyplot as plt 

def tag_boundaries(msh, boundaries):
    # Identifies and marks boundaries accoring to locator function
    # Returns meshtag object
    facet_indices, facet_markers = [], [] 
    fdim = msh.topology.dim-1
    for (marker, locator) in boundaries:
        facets = locate_entities(msh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    return Measure("ds", domain=msh, subdomain_data=facet_tag) #facet_tag

def apply_boundary_conditions(msh, bc_list):
    fdim = msh.topology.dim - 1# facet_dimension
    bcs = []
    for (facets, g, subspace, collapsed_subspace) in bc_list:
        bc_dofs = fem.locate_dofs_topological((subspace, collapsed_subspace), fdim, facets)
        w_bc = fem.Function(collapsed_subspace)
        w_bc.interpolate(lambda x: g(x))
        bcs.append(fem.dirichletbc(w_bc, bc_dofs, subspace))
    return bcs    

def thermomechanical_solver(msh, t_step, t_max):
    #---------------- Compute connectivities between facets and cells -------------------
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    #---------------- Identify the facets -----------------------------------------------
    fdim = msh.topology.dim - 1 
    left_facets = locate_entities(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right_facets = locate_entities(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    bottom_facets = locate_entities(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = locate_entities(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    boundaries = [(1, lambda x: np.isclose(x[1], 0.0)),   # Neumann boundary condition
                    (1, lambda x: np.isclose(x[0], 1.0)),
                    (2, lambda x: np.isclose(x[1], 1.0))] # Robin boundary condition
    ds = tag_boundaries(msh, boundaries)

    #---------------- Function spaces ---------------------------------------------------
    degree_u = 1 
    el_u = VectorElement("CG", msh.ufl_cell(), degree_u, dim=2) 
    degree_T = 1
    el_T = FiniteElement("CG",  msh.ufl_cell(), degree_T)
    el_m  = MixedElement([el_u , el_T]) 
    W = fem.FunctionSpace(msh, el_m) 
    U, _ = W.sub(0).collapse() 
    V, _ = W.sub(1).collapse() 
    cdim = msh.topology.dim 
 
    #---------------- Problem parameters and expressions --------------------------------
    rho = 1.0                                   # density * specific heat
    kappa = 1.0                                 # thermal conductivity
    beta = 1.0                                  # convection coefficient
    alpha =  1e-12                               # thermal expansion     
    E, nu = 30e9, 0.25                          # Young modulus, Poisson ratio
    mu    = E/(2.0*(1.0 + nu))                  # Lamé constants
    lbd  = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
  
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def sigma(u, T):
        return 2*mu*epsilon(u) + lbd*div(u)*Identity(cdim) + alpha*T*Identity(cdim)
    
    # ---------------- Sources ------------------------------------------------------------
    def Q_(t, x):
        return (np.pi**2 - np.pi)*(np.sin(np.pi*x[0]) + np.sin(np.pi*x[1]))*np.exp(-np.pi*t)

    def f_(t, x):
        aux1 = (mu*np.pi**2*np.cos(np.pi*(x[0]-x[1])) - np.pi**2*(2*mu + lbd)*np.cos(np.pi*(x[0]+x[1])))*np.exp(-np.pi*t)
        aux2 = alpha*np.pi*(np.exp(-np.pi*t) - 1) 
        return (aux1 - aux2*np.cos(np.pi*x[0]), aux1 - aux2*np.cos(np.pi*x[1]))

    # ---------------- Robin and Neumann boundary conditions -------------------------------
    def T_inf(t, x):
        return (np.sin(np.pi*x[0]) + np.sin(np.pi*x[1]) - np.pi)*np.exp(-np.pi*t)

    def p_(t, x):
        aux1 = np.pi*np.sin(np.pi*x[0])
        aux2 = np.pi*np.sin(np.pi*x[1])
        return ((-aux1*mu - aux2*(2*mu+lbd))*np.exp(-np.pi*t), -(aux1*(2*mu+lbd) - aux2*mu)*np.exp(-np.pi*t))
    
    def g_(t, x):
        return np.full_like(x[0], -np.pi*np.exp(-np.pi*t)) 

    #---------------- Dirichlet boundary conditions --------------------------------------
    u_bc_facets = np.hstack([left_facets, right_facets, top_facets, bottom_facets])
    T_bc_facets = [left_facets] 

    def T_dir(t, x):
        return np.sin(np.pi*x[1])*np.exp(-np.pi*t)
        
    def u_dir(t, x):
        return (np.zeros_like(x[0]), np.zeros_like(x[0]))

    #---------------- Setting sources and boundary conditions at initial time step -------

    g = fem.Function(V)
    g.interpolate(partial(g_, 0))
    t_inf = fem.Function(V)
    t_inf.interpolate(partial(T_inf,0))
    f = fem.Function(U)
    f.interpolate(partial(f_,0))
    p = fem.Function(U)
    p.interpolate(partial(p_, 0))
    Q = fem.Function(V)
    Q.interpolate(partial(Q_, 0))    

    g_t = lambda x: T_dir(0, x)
    g_u = lambda x: u_dir(0, x)
    bc_list = [[u_bc_facets, g_u, W.sub(0), U],
                [T_bc_facets, g_t , W.sub(1), V]]

    bcs = apply_boundary_conditions(msh, bc_list)

    #---------------- Initial temperature -----------------------------------------------
    def init_temp(x):
        return (np.sin(np.pi*x[0]) + np.sin(np.pi*x[1]))

    T0 = fem.Function(V)
    T0.interpolate(lambda x: init_temp(x))
    Tinit = T0.copy()

    #---------------- Analytical solutions ----------------------------------------------
    def exact_sol_T(t,x):
        return np.exp(-np.pi*t)*(np.sin(np.pi*x[0])+np.sin(np.pi*x[1]))

    def exact_sol_u(t,x):
        aux = np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*np.exp(-np.pi*t)
        return (aux, aux)
  
    #---------------- Time step ---------------------------------------------------------
    dt = fem.Constant(msh, t_step)

    #---------------- Defining linear solver --------------------------------------------
    TrialF = TrialFunction(W); TestF = TestFunction(W)
    (u, T) = split(TrialF); (v, r) = split(TestF)

    # LHS: Bilinear forms
    a  = rho * inner(T, r) * dx  
    a += dt * kappa * inner(grad(T), grad(r)) * dx
    a += dt * beta * inner(T,r) * ds(2) # Robin
    a +=  inner(sigma(u,T), epsilon(v)) * dx 
    a = fem.form(a) 

    # RHS: Forcing terms
    L = inner(f, v) * dx
    L +=  inner(p,v) * (ds(1) + ds(2))
    L += dt * inner(Q,r) * dx
    L += dt * inner(g,r) * ds(1) # Neumann
    L += dt * inner(beta * (t_inf),r) * ds(2) # Robin
    L += rho * inner(T0,r) * dx
    L += alpha * inner(Tinit,div(v)) * dx
    L = fem.form(L) 

    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = fem.petsc.create_vector(L)
    wh = fem.Function(W)

    comm = MPI.COMM_WORLD
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    #---------------- Files for visualization with Paraview -----------------------------
    file = io.XDMFFile(MPI.COMM_WORLD, "temp.xdmf", "w")
    file.write_mesh(msh)

    file_u = io.XDMFFile(MPI.COMM_WORLD, "disp.xdmf", "w")
    file_u.write_mesh(msh)

    er = []; er_u =[]
    t = 0; ts = []

    #---------------- Main loop ----------------------------------------------------------  
    while t < t_max:
        # L2 error norms - temperature
        T_ex = fem.Function(V)
        T_ex.interpolate(partial(exact_sol_T, t))
        eh = T0 - T_ex  
        eL2form = fem.form( eh**2*dx )
        errorL2 = fem.assemble_scalar(eL2form)
        er.append(np.sqrt(errorL2))
        print('t = '+str(t)+',  L2error = '+ str(np.sqrt(errorL2)))

        ts.append(t)
        t += dt.value

        #### Update time dependent functions #### 
        g.interpolate(partial(g_, t))
        t_inf.interpolate(partial(T_inf,t))
        f.interpolate(partial(f_,t))
        p.interpolate(partial(p_, t))
        Q.interpolate(partial(Q_, t))

        # Update Dirichlet boundary conditions 
        g_T = lambda x: T_dir(t, x)
        bc_list[1][1] = g_T
        bcs = apply_boundary_conditions(msh, bc_list)

        # Assemble RHS 
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, L)

        # Apply boundary conditions 
        fem.petsc.apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)

        # Solve linear problem 
        solver.solve(b, wh.vector)
        (uh,Th) = wh.split()

        # Save solution at time step t 
        file.write_function(T0, t=t-dt)
        file_u.write_function(uh, t=t)

        # Save step
        T0.x.array[:] = wh.sub(1).collapse().x.array[:]

        # L2 error norms - displacement
        u_ex = fem.Function(U)
        u_ex.interpolate(partial(exact_sol_u, t))
        eh = uh - u_ex
        eL2form = fem.form( eh**2*dx )
        errorL2 = fem.assemble_scalar(eL2form)
        er_u.append(np.sqrt(errorL2))
        print('t = '+str(t)+', u L2error = '+ str(np.sqrt(errorL2)))
    
    ### Save solution at final step and close files ####
    file.write_function(T0, t=t)
    file.close()
    file_u.close()

    return er, er_u, ts