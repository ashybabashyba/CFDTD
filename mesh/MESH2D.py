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

        self.gridEx = np.linspace(0, box_size, int(1 + box_size/dx))
        self.gridEy = np.linspace(0, box_size, int(1 + box_size/dy))

        self.gridHx = (self.gridEx[1:] + self.gridEx[:-1]) / 2.0
        self.gridHy = (self.gridEy[1:] + self.gridEy[:-1]) / 2.0

        self.nodesList = external_nodes_list_PEC
    
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

        if len(self.nodesList) != 2:
            PEC_polygon = shape.Polygon(self.nodesList)

            return shape.area(shape.difference(cell_polygon, PEC_polygon))
        
        elif len(self.nodesList) == 2:
            if self.getNumberOfIntersections(cell) > 1:
                split_cells = [subcell for subcell in split(cell_polygon, shape.LineString(self.nodesList)).geoms]

                auxiliar_polygon_list = []
                for i in range(len(self.nodesList)):
                    auxiliar_polygon_list.append(self.nodesList[i])

                auxiliar_polygon_list.append(self.initialCell)

                auxiliar_polygon = shape.Polygon(auxiliar_polygon_list)

                intersection_cells = [shape.intersection(subcell, auxiliar_polygon) for subcell in split_cells]
                area_intersection = [shape.area(subcell) for subcell in intersection_cells]
                max_area_intersection = max(area_intersection)

                if max_area_intersection == area_intersection[0]:
                    return area_intersection[0]
                
                else:
                    return area_intersection[1]
                
            else:
                return shape.area(cell_polygon)
            


    
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
                    
        if len(self.nodesList) != 2:
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

            

        ax.set_xlabel('Grid Ex')
        ax.set_ylabel('Grid Ey')
        ax.set_title('Mesh Plot')
        ax.grid(True)
        plt.show()
