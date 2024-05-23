import numpy as np
from matplotlib import pyplot as plt
import shapely as shape
from shapely.ops import split

def lineEquation(x0, x1):
    if x0[0] == x1[0]:
        return lambda x: x0[1]+x-x
    else:
        m = (x0[1]-x1[1])/(x0[0]-x1[0])
        return lambda x: m*x - m*x0[0] + x0[1]
    
def inverseLineEquation(x0, x1):
    if x0[1] == x1[1]:
        return lambda x: x0[0]+x-x
    else:
        m_inv = (x0[0]-x1[0])/(x0[1]-x1[1])
        return lambda x: m_inv*(x - x0[1]) + x0[0]

class Mesh():
    def __init__(self, box_size, dx, dy, external_nodes_list_PEC=None, initial_wave_cell=None):
        self.dx = dx
        self.dy = dy
        self.boxSize = box_size

        self.initialCell = initial_wave_cell

        if self.initialCell is not None:
            initial_cell_list = list(self.initialCell)
            initial_cell_list[0] = initial_cell_list[0] + self.dx/2
            initial_cell_list[1] = initial_cell_list[1] + self.dy/2

            self.initialCell = tuple(initial_cell_list)

        self.gridEx = np.linspace(0, box_size, int(1 + box_size/dx))
        self.gridEy = np.linspace(0, box_size, int(1 + box_size/dy))

        self.gridHx = (self.gridEx[1:] + self.gridEx[:-1]) / 2.0
        self.gridHy = (self.gridEy[1:] + self.gridEy[:-1]) / 2.0

        self.nodesList = external_nodes_list_PEC

        self.simplificationsErrors()    
    
    def simplificationsErrors(self):
        if len(self.nodesList) == 2:
            if np.abs(self.nodesList[0][0] - self.nodesList[1][0]) != self.boxSize:
                if np.abs(self.nodesList[0][1] - self.nodesList[1][1]) != self.boxSize:
                    raise ValueError('Please insert a line that cuts all the box')
                
            if self.nodesList[0][0] == self.nodesList[1][0] and self.nodesList[0][0] in self.gridEx:
                raise ValueError('Please insert the vertical line in a value not included in the x-grid')
            
            if self.nodesList[0][1] == self.nodesList[1][1] and self.nodesList[1][1] in self.gridEy:
                raise ValueError('Please insert the horizontal line in a value not included in the y-grid')
            
        elif len(self.nodesList) > 2:
            columns_intersection_points, rows_intersection_points = self.getIntersectionPoints()
            columns_intersection_points.sort()
            rows_intersection_points.sort()

            for i in range(len(columns_intersection_points)):
                for j in range(i+1, len(columns_intersection_points)):
                    if columns_intersection_points[i][0] == columns_intersection_points[j][0]: 
                        if np.searchsorted(self.gridEy, columns_intersection_points[i][1]) == np.searchsorted(self.gridEy, columns_intersection_points[j][1]):
                            raise ValueError('Please insert a polygon who dont cut the same line of the subcell two times or more')

            for i in range(len(rows_intersection_points)):
                for j in range(i+1, len(rows_intersection_points)):
                    if rows_intersection_points[i][1] == rows_intersection_points[j][1]: 
                        if np.searchsorted(self.gridEx, rows_intersection_points[i][0]) == np.searchsorted(self.gridEx, rows_intersection_points[j][0]):
                            raise ValueError('Please insert a polygon who dont cut the same line of the subcell two times or more')

            for i in range(len(self.gridEx)-1):
                for j in range(len(self.gridEy)-1):
                    if self.getNumberOfIntersections((i,j)) > 2:
                        raise ValueError('Please insert a polygon with no more than two intersections per cell')

    def electricFieldGridCreation(self):
        self.columns = []
        self.rows = []

        for i in self.gridEx:
            self.columns.append(self.gridEy.tolist())

        for i in self.gridEy:
            self.rows.append(self.gridEx.tolist())
        return self.columns, self.rows

    def getIntersectionPoints(self):
        self.columns_intersection_points = []
        self.rows_intersection_points = []

        if self.nodesList is not None:

            for i in range(len(self.nodesList)):
                current_node = self.nodesList[i]
                next_node = self.nodesList[(i+1) % len(self.nodesList)]

                for j in range(len(self.gridEx)):
                    if min(current_node[0], next_node[0]) <= self.gridEx[j] <= max(current_node[0], next_node[0]):
                        self.columns_intersection_points.append((self.gridEx[j], lineEquation(current_node, next_node)(self.gridEx[j])))

                for k in range(len(self.gridEy)):
                    if min(current_node[1], next_node[1]) <= self.gridEy[k] <= max(current_node[1], next_node[1]):
                        self.rows_intersection_points.append((inverseLineEquation(current_node, next_node)(self.gridEy[k]), self.gridEy[k]))

        return self.columns_intersection_points, self.rows_intersection_points


    def getNumberOfIntersections(self, cell):
        left_column = cell[0]*self.dx
        right_column = (cell[0]+1)*self.dx

        bottom_row = cell[1]*self.dy
        top_row = (cell[1]+1)*self.dy

        cell_polygon = shape.Polygon([(left_column, bottom_row), (right_column, bottom_row), (right_column, top_row), (left_column, top_row)])

        PEC_lines = []
        non_repeated_intersections = []
        number_of_intersections = 0

        if self.nodesList is not None:
            if len(self.nodesList) == 2:
                PEC_lines.append(shape.LineString([self.nodesList[0], self.nodesList[1]]))
            else:
                for i in range(len(self.nodesList)):
                    current_node = self.nodesList[i]
                    next_node = self.nodesList[(i+1) % len(self.nodesList)]

                    PEC_lines.append(shape.LineString([current_node, next_node]))
            
            for line in PEC_lines:
                if cell_polygon.intersects(line):
                    intersection = shape.intersection(cell_polygon, line)
                    
                    for i in range(len(intersection.coords)):
                        if intersection.coords[i] not in non_repeated_intersections:
                            if (intersection.coords[i][0] % self.dx == 0) or (intersection.coords[i][1] % self.dy == 0):
                                non_repeated_intersections.append(intersection.coords[i])
                                number_of_intersections += 1

        return number_of_intersections
    
    def getCellArea(self, cell):
        left_column = cell[0]*self.dx
        right_column = (cell[0]+1)*self.dx

        bottom_row = cell[1]*self.dy
        top_row = (cell[1]+1)*self.dy

        cell_polygon = shape.Polygon([(left_column, bottom_row), (right_column, bottom_row), (right_column, top_row), (left_column, top_row)])

        if len(self.nodesList) > 2:
            PEC_polygon = shape.Polygon(self.nodesList)

            return shape.area(shape.difference(cell_polygon, PEC_polygon))
        
        elif len(self.nodesList) == 2:
            if self.initialCell is None:
                raise ValueError('To calculate area of the conformal cell is necessary the start cell of the initial wave')

            if self.getNumberOfIntersections(cell) > 1:
                split_cells = [subcell for subcell in split(cell_polygon, shape.LineString(self.nodesList)).geoms]

                auxiliar_polygon_list = []
                intersection_cells = []
                area_intersection = []

                for i in range(len(self.nodesList)):
                    auxiliar_polygon_list.append(self.nodesList[i])

                auxiliar_polygon_list.append(self.initialCell)

                auxiliar_polygon = shape.Polygon(auxiliar_polygon_list)

                intersection_cells = [shape.intersection(subcell, auxiliar_polygon) for subcell in split_cells]
                area_intersection = [shape.area(subcell) for subcell in intersection_cells]
                max_area_intersection = max(area_intersection)

                if max_area_intersection == area_intersection[0]:
                    return shape.area(split_cells[0])
                
                else:
                    return shape.area(split_cells[1])
                
            else:
                return shape.area(cell_polygon)
            
    def getCellLengths(self, cell):
        left_column = cell[0]*self.dx
        right_column = (cell[0]+1)*self.dx

        bottom_row = cell[1]*self.dy
        top_row = (cell[1]+1)*self.dy

        cell_polygon = shape.Polygon([(left_column, bottom_row), (right_column, bottom_row), (right_column, top_row), (left_column, top_row)])

        left_edge = shape.LineString([(left_column, bottom_row), (left_column, top_row)])
        right_edge = shape.LineString([(right_column, bottom_row), (right_column, top_row)])
        top_edge = shape.LineString([(right_column, top_row), (left_column, top_row)])
        bottom_edge = shape.LineString([(left_column, bottom_row), (right_column, bottom_row)])

        if len(self.nodesList) > 2:
            edge_lengths = {"left": self.dy, "right": self.dy, "top": self.dx, "bottom": self.dx}
            PEC_polygon = shape.Polygon(self.nodesList)
            difference_cell = shape.difference(cell_polygon, PEC_polygon)

            if shape.intersects(difference_cell, left_edge):
                edge_lengths["left"] = shape.intersection(difference_cell, left_edge).length
            if shape.intersects(difference_cell, right_edge):
                edge_lengths["right"] = shape.intersection(difference_cell, right_edge).length
            if shape.intersects(difference_cell, top_edge):
                edge_lengths["top"] = shape.intersection(difference_cell, top_edge).length
            if shape.intersects(difference_cell, bottom_edge):
                edge_lengths["bottom"] = shape.intersection(difference_cell, bottom_edge).length

        elif len(self.nodesList) == 2:
            edge_lengths = {"left": 0, "right": 0, "top": 0, "bottom": 0}
            PEC_line = shape.LineString(self.nodesList)
            subcells = split(cell_polygon, PEC_line)

            if len(subcells.geoms) == 1:
                split_cell = subcells

            elif len(subcells.geoms) == 2:
                if shape.area(subcells.geoms[0]) == self.getCellArea(cell):
                    split_cell = subcells.geoms[0]

                elif shape.area(subcells.geoms[1]) == self.getCellArea(cell):
                    split_cell = subcells.geoms[1]

            if shape.intersects(split_cell, left_edge):
                edge_lengths["left"] = shape.intersection(split_cell, left_edge).length
            if shape.intersects(split_cell, right_edge):
                edge_lengths["right"] = shape.intersection(split_cell, right_edge).length
            if shape.intersects(split_cell, top_edge):
                edge_lengths["top"] = shape.intersection(split_cell, top_edge).length
            if shape.intersects(split_cell, bottom_edge):
                edge_lengths["bottom"] = shape.intersection(split_cell, bottom_edge).length

        return edge_lengths
    
    def getCellSeparationByType(self):
        conformal_cells = []
        non_conformal_cells = []
        outside_non_conformal_cells = []
        inside_non_conformal_cells = []

        for i in range(len(self.gridEx)-1):
            for j in range(len(self.gridEy)-1):
                number_of_intersections = self.getNumberOfIntersections((i,j))
                if number_of_intersections > 1:
                    conformal_cells.append((i,j))
                else:
                    if len(self.nodesList) != 2:
                        if np.isclose(self.getCellArea((i,j)), self.dx * self.dy):
                            outside_non_conformal_cells.append((i,j))
                        else:
                            inside_non_conformal_cells.append((i,j))
                      
                    else:
                        non_conformal_cells.append((i,j))
                    
        if len(self.nodesList) > 2:
            return conformal_cells, outside_non_conformal_cells, inside_non_conformal_cells
        
        elif len(self.nodesList) == 2:
            return conformal_cells, non_conformal_cells

        

    def plotElectricFieldGrid(self):
        columns, rows = self.electricFieldGridCreation()
        columns_intersection, rows_intersection = self.getIntersectionPoints()

        fig, ax = plt.subplots()
        for i, vec in enumerate(columns):
            x_coords = [self.gridEx[i]] * len(vec) 
            y_coords = vec 
            ax.plot(x_coords, y_coords, marker='o', color='b')

        for j, vec in enumerate(rows):
            y_coords = [self.gridEy[j]] * len(vec)  
            x_coords = vec                      
            ax.plot(x_coords, y_coords, marker='s', color='b')

        if self.nodesList is not None:
            for k in range(len(self.nodesList)):
                current_node = self.nodesList[k]
                next_node = self.nodesList[(k+1) % len(self.nodesList)]

                x_vals = np.linspace(current_node[0], next_node[0], 100)
                y_vals = np.linspace(current_node[1], next_node[1], 100)
                ax.plot(x_vals, y_vals, color='r')

            for i, vec in enumerate(columns_intersection):
                x_coords = columns_intersection[i][0]  
                y_coords = columns_intersection[i][1]  
                ax.plot(x_coords, y_coords, marker='^', color='black')

            for j, vec in enumerate(rows_intersection):
                x_coords = rows_intersection[j][0]
                y_coords = rows_intersection[j][1]                      
                ax.plot(x_coords, y_coords, marker='^', color='black')

        if self.initialCell is not None:
            ax.plot(self.initialCell[0], self.initialCell[1], marker='x' , color='green')

            

        ax.set_xlabel('Grid Ex')
        ax.set_ylabel('Grid Ey')
        ax.set_title('Mesh Plot')
        ax.grid(True)
        plt.show()