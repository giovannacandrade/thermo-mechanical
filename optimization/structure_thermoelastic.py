from dolfinx import fem, io, geometry
from dolfinx.mesh import (create_rectangle, CellType, locate_entities, meshtags)
import dolfinx.cpp as _cpp
from ufl import Measure, pi, conditional, le, ge,dx, Identity, nabla_grad, div, inner, sym, grad, TestFunction, VectorElement, MixedElement, FiniteElement
from mpi4py import MPI
from matplotlib import cm, pyplot as plt
import numpy as np, sys, os
import time

def tag_boundaries(msh, boundaries):
    # Identifies and marks boundaries accoring to locator function
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

class Structure:
    def __init__(self, lx, ly, Nx, Ny, E, nu, rho, kappa, beta, alpha, eps_er):
        self.Nx, self.Ny = Nx, Ny
        self.lx, self.ly = lx, ly
        
        # Mechanical Parameters
        self.mu, self.lmbda = E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
        self.eps_er = eps_er
    
        # Thermal Parameters
        self.rho, self.kappa, self.beta, self.alpha = rho, kappa, beta, alpha

        # Mesh
        self.msh  = create_rectangle(comm=MPI.COMM_WORLD,
            points=((0.0, 0.0), (lx,ly)), n=(Nx, Ny),
            cell_type=CellType.triangle, diagonal = _cpp.mesh.DiagonalType.crossed)
        self.dim = self.msh.topology.dim
        self.dx = Measure("dx", domain= self.msh)

        # Function Spaces
        degree_u = 1; degree_T = 1
        el_u = VectorElement("CG", self.msh.ufl_cell(), degree_u, dim=2) 
        el_T = FiniteElement("CG",  self.msh.ufl_cell(), degree_T)
        el_m  = MixedElement([el_u , el_T]) 
        W = fem.FunctionSpace(self.msh, el_m) 
        self.U, _ = W.sub(0).collapse() 
        self.V, _ = W.sub(1).collapse()  # self.V = fem.FunctionSpace(self.msh, ('CG', 1))   

        # Boundary Conditions
        self.msh.topology.create_connectivity(self.dim - 1, self.dim)
        left_facets = locate_entities(self.msh, self.dim - 1, lambda x: np.isclose(x[0], 0.0))
        right_facets = locate_entities(self.msh, self.dim - 1, lambda x: np.isclose(x[0], 1.0))
        bottom_facets = locate_entities(self.msh, self.dim - 1, lambda x: np.isclose(x[1], 0.0))
        top_facets = locate_entities(self.msh, self.dim - 1, lambda x: np.isclose(x[1], 1.0))

        boundaries = [(1, lambda x: np.isclose(x[1], 0.0)),   # Neumann boundary condition
                        (1, lambda x: np.isclose(x[0], 1.0)),
                        (2, lambda x: np.isclose(x[1], 1.0))] # Robin boundary condition
        
        self.ds = tag_boundaries(self.msh, boundaries)

        # Vertices Coordinates
        self.dofsV    = self.V.tabulate_dof_coordinates()[:,:-1]    
        self.px, self.py    = [(self.dofsV[:,0]/lx)*2*Nx, (self.dofsV[:,1]/ly)*2*Ny]
        self.dofsV_max = (Nx+1)*(Ny+1) + Nx*Ny

    def epsilon(self, u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def sigma(self, u, T):
        return 2*self.mu*self.epsilon(u) + self.lbd*div(u)*Identity(self.cdim) + self.alpha*T*Identity(self.cdim)
    
    def apply_bcs(self, bc_list):
        return apply_boundary_conditions(self.msh, bc_list)

    def update_time_dependent_par(self):
        return    

    def map_geometry(self, phi):
        tol = 1e-8
        self.int_dx = conditional(le(phi, tol), 1, 0)*self.dx
        self.ext_dx = conditional(ge(phi, tol), 1, 0)*self.dx
    
    def get_volume_fraction(self):
        return fem.assemble_scalar(fem.form(self.int_dx))/(self.lx*self.ly)

    def get_compliance(self):
        return
      
    def shape_derivative(self):
        xi = TestFunction(self.U); th = fem.Function(self.U)                   
        eu,ep,Du,Dp = [sym(grad(self.u)),grad(self.u),sym(grad(self.p)),grad(self.p)]
        #S1 = 
        # rhs = 
        # self.solverav.solve(th.vector(), rhs)
        return th
    
    def state_thermoelastic(self):
        # self.u = 
        # self.T = 
        return

    def adjoint_thermoelastic(self):
        # self.p = 
        # self.S = 
        return