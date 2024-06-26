import numpy as np
from matplotlib import pyplot as plt
import shapely as shape

from mesh.MESH2D import *

def test_visualization_of_the_mesh():
    node_list = [(0.5,0.5), (0.5,1.5), (9.5, 1.5), (9.5, 0.5)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 9))
    mesh.plotElectricFieldGrid()

def test_number_of_intersections_of_a_line():
    node_list = [(0,0), (10, 1)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 9))
    conformalCells, nonConformalCells = mesh.getCellSeparationByType()
    #mesh.plotElectricFieldGrid()
    for i in conformalCells:
        assert mesh.getNumberOfIntersections(i) == 2

def test_number_of_cells_polygon():
    node_list = [(0.5,0.5), (9.5,0.5), (9.5, 9.5), (0.5, 9.5)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 9))
    # mesh.plotElectricFieldGrid()
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()
    assert np.isclose(len(conformalCells) + len(outsideNonConformalCells) + len(insideNonConformalCells), mesh.boxSize**2/mesh.dx/mesh.dy)

def test_number_of_cells_line():
    node_list = [(0,0.5), (10,0.5)]
    mesh = Mesh(box_size=10.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 9))
    # mesh.plotElectricFieldGrid()
    conformalCells, nonConformalCells = mesh.getCellSeparationByType()
    assert np.isclose(len(conformalCells) + len(nonConformalCells), mesh.boxSize**2/mesh.dx/mesh.dy)

def test_conformal_cells_with_a_PEC_line():
    node_list = [(0,0.5), (10,0.5)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 9))
    # mesh.plotElectricFieldGrid()
    conformalCells, nonConformalCells = mesh.getCellSeparationByType()
    assert np.isclose(mesh.boxSize/mesh.dx, len(conformalCells))

def test_general_area_polygon():
    node_list = [(0.5,0.5), (9.5,0.5), (9.5, 9.5), (0.5, 9.5)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 9))
    # mesh.plotElectricFieldGrid()
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()

    for i in conformalCells:
        assert 0 <= mesh.getCellArea(i) < mesh.dx * mesh.dy

    for j in outsideNonConformalCells:
        assert np.isclose(mesh.getCellArea(j), mesh.dx * mesh.dy)

    for k in insideNonConformalCells:
        assert np.isclose(mesh.getCellArea(k), 0)

def test_general_area_line():
    node_list = [(0,0), (10,1)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 9))
    # mesh.plotElectricFieldGrid()
    conformalCells, nonConformalCells= mesh.getCellSeparationByType()

    for i in conformalCells:
        assert 0 <= mesh.getCellArea(i) < mesh.dx * mesh.dy

    for j in nonConformalCells:
        assert np.isclose(mesh.getCellArea(j), mesh.dx * mesh.dy)

def test_cell_lengths_polygon_outsidePEC():
    node_list = [(0.5,0.5), (2.5,0.5), (2.5, 9.5), (0.5, 9.5)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (4, 9))
    # mesh.plotElectricFieldGrid()
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()

    for i in outsideNonConformalCells:
        assert mesh.getCellLengths(i)["left"] == mesh.dy
        assert mesh.getCellLengths(i)["right"] == mesh.dy

        assert mesh.getCellLengths(i)["top"] == mesh.dx
        assert mesh.getCellLengths(i)["bottom"] == mesh.dx

    for i in insideNonConformalCells:
        assert mesh.getCellLengths(i)["left"] == 0
        assert mesh.getCellLengths(i)["right"] == 0

        assert mesh.getCellLengths(i)["top"] == 0
        assert mesh.getCellLengths(i)["bottom"] == 0

    for i in conformalCells:
        assert mesh.getCellLengths(i)["left"] <= mesh.dy
        assert mesh.getCellLengths(i)["right"] <= mesh.dy

        assert mesh.getCellLengths(i)["top"] <= mesh.dx
        assert mesh.getCellLengths(i)["bottom"] <= mesh.dx

def test_cell_lengths_polygon_insidePEC():
    node_list = [(0.5,0.5), (2.5,0.5), (2.5, 9.5), (0.5, 9.5)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (1, 1))
    # mesh.plotElectricFieldGrid()
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()

    for i in outsideNonConformalCells:
        assert mesh.getCellLengths(i)["left"] == 0
        assert mesh.getCellLengths(i)["right"] == 0

        assert mesh.getCellLengths(i)["top"] == 0
        assert mesh.getCellLengths(i)["bottom"] == 0

    for i in insideNonConformalCells:
        assert mesh.getCellLengths(i)["left"] == mesh.dy
        assert mesh.getCellLengths(i)["right"] == mesh.dy

        assert mesh.getCellLengths(i)["top"] == mesh.dx
        assert mesh.getCellLengths(i)["bottom"] == mesh.dx

    for i in conformalCells:
        assert mesh.getCellLengths(i)["left"] <= mesh.dy
        assert mesh.getCellLengths(i)["right"] <= mesh.dy

        assert mesh.getCellLengths(i)["top"] <= mesh.dx
        assert mesh.getCellLengths(i)["bottom"] <= mesh.dx

def test_cell_lengths_polygon_sumOutsideAndInside():
    node_list = [(0.8,0.6), (2.5,0.6), (2.5, 9.5), (0.8, 9.5)]
    mesh1 = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (1, 1))
    # mesh1.plotElectricFieldGrid()
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh1.getCellSeparationByType()

    mesh2 = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (5, 5))
    # mesh2.plotElectricFieldGrid()
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh2.getCellSeparationByType()

    for i in conformalCells:
        assert np.isclose(mesh1.getCellLengths(i)["left"] + mesh2.getCellLengths(i)["left"], mesh1.dy) 
        assert np.isclose(mesh1.getCellLengths(i)["right"] + mesh2.getCellLengths(i)["right"], mesh1.dy) 

        assert np.isclose(mesh1.getCellLengths(i)["top"] + mesh2.getCellLengths(i)["top"], mesh1.dx) 
        assert np.isclose(mesh1.getCellLengths(i)["bottom"] + mesh2.getCellLengths(i)["bottom"], mesh1.dx)

    for i in insideNonConformalCells:
        assert np.isclose(mesh1.getCellLengths(i)["left"] + mesh2.getCellLengths(i)["left"], mesh1.dy) 
        assert np.isclose(mesh1.getCellLengths(i)["right"] + mesh2.getCellLengths(i)["right"], mesh1.dy) 

        assert np.isclose(mesh1.getCellLengths(i)["top"] + mesh2.getCellLengths(i)["top"], mesh1.dx) 
        assert np.isclose(mesh1.getCellLengths(i)["bottom"] + mesh2.getCellLengths(i)["bottom"], mesh1.dx) 

    for i in outsideNonConformalCells:
        assert np.isclose(mesh1.getCellLengths(i)["left"] + mesh2.getCellLengths(i)["left"], mesh1.dy) 
        assert np.isclose(mesh1.getCellLengths(i)["right"] + mesh2.getCellLengths(i)["right"], mesh1.dy) 

        assert np.isclose(mesh1.getCellLengths(i)["top"] + mesh2.getCellLengths(i)["top"], mesh1.dx) 
        assert np.isclose(mesh1.getCellLengths(i)["bottom"] + mesh2.getCellLengths(i)["bottom"], mesh1.dx)  

def test_cell_lengths_line():
    node_list = [(0,0), (10,1)]
    mesh = Mesh(box_size=10, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell= (0, 9))
    # mesh.plotElectricFieldGrid()
    conformalCells, nonConformalCells= mesh.getCellSeparationByType()

    for i in nonConformalCells:
        assert mesh.getCellLengths(i)["left"] == mesh.dy
        assert mesh.getCellLengths(i)["right"] == mesh.dy

        assert mesh.getCellLengths(i)["top"] == mesh.dx
        assert mesh.getCellLengths(i)["bottom"] == mesh.dx

    for i in conformalCells:
        assert mesh.getCellLengths(i)["left"] <= mesh.dy
        assert mesh.getCellLengths(i)["right"] <= mesh.dy

        assert mesh.getCellLengths(i)["top"] <= mesh.dx
        assert mesh.getCellLengths(i)["bottom"] <= mesh.dx