import numpy as np
from matplotlib import pyplot as plt
import shapely as shape

from mesh.MESH2D import *

def test_visualization_of_the_mesh():
    node_list = [(0,0), (10,1), (7, 8), (5, 10), (3, 4)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list)
    mesh.plotElectricFieldGrid()

def test_number_of_intersections_of_a_line():
    node_list = [(0,0), (10, 1)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list)
    conformalCells, nonConformalCells = mesh.getCellSeparationByType()
    #mesh.plotElectricFieldGrid()
    for i in conformalCells:
        assert np.isclose(mesh.getNumberOfIntersections(i), 2)

def test_number_of_cells_polygon():
    node_list = [(0,0.5), (10,0.5), (5,5)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list)
    # mesh.plotElectricFieldGrid()
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()
    assert len(conformalCells) + len(outsideNonConformalCells) + len(insideNonConformalCells) == mesh.boxSize**2/mesh.dx/mesh.dy

def test_number_of_cells_line():
    node_list = [(0,0.5), (10,0.5)]
    mesh = Mesh(box_size=10.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list)
    # mesh.plotElectricFieldGrid()
    conformalCells, nonConformalCells = mesh.getCellSeparationByType()
    assert len(conformalCells) + len(nonConformalCells)  == mesh.boxSize**2/mesh.dx/mesh.dy

def test_conformal_cells_with_a_PEC_line():
    node_list = [(0,0.5), (10,0.5)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list)
    # mesh.plotElectricFieldGrid()
    conformalCells, nonConformalCells = mesh.getCellSeparationByType()
    assert mesh.boxSize/mesh.dx == len(conformalCells)

def test_general_area_polygon():
    node_list = [(0,0), (10,1), (7, 8), (5, 10), (3, 4)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list)
    # mesh.plotElectricFieldGrid()
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()

    for i in conformalCells:
        assert 0 <= mesh.getCellArea(i) < mesh.dx * mesh.dy

    for j in outsideNonConformalCells:
        assert np.isclose(mesh.getCellArea(j), mesh.dx * mesh.dy)

    for k in insideNonConformalCells:
        assert np.isclose(mesh.getCellArea(k), 0)

def test_general_area_line():
    node_list = [(0,0), (9.5,1)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 10))
    mesh.plotElectricFieldGrid()
    conformalCells, nonConformalCells= mesh.getCellSeparationByType()

    for i in conformalCells:
        assert 0 <= mesh.getCellArea(i) < mesh.dx * mesh.dy

    for j in nonConformalCells:
        assert np.isclose(mesh.getCellArea(j), mesh.dx * mesh.dy)

