import gmsh
from mpi4py import MPI

def generate_mesh(filename, radius, Lx, Ly, h_elem, order: int):
    if MPI.COMM_WORLD.rank == 0:
        gdim = 2
        gmsh.initialize()
        gmsh.model.add("circle")
        gmsh.option.setNumber("General.Verbosity", 0)
        # Set the mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.75*h_elem)

        # Add scatterers
        circle = gmsh.model.occ.addCircle(Lx/2, Ly/2, 0.0, radius)
        gmsh.model.occ.addCurveLoop([circle], tag=circle)

        gmsh.model.occ.addPlaneSurface([circle], tag=circle)

        # Add domain
        rectangle = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)
        # inclusive_rectangle, _ = gmsh.model.occ.fragment([(gdim, rectangle)], [(gdim, circle)]) # for circle as subdomain
        inclusive_rectangle, _ = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, circle)]) # for circle as hole hole 

        gmsh.model.occ.synchronize()

        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [circle], tag=0)
        gmsh.model.addPhysicalGroup(2, [rectangle], tag=1)

        # Generate mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.optimize("HighOrder")
        gmsh.write(filename)

        gmsh.finalize()