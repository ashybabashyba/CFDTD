import numpy as np
from matplotlib import pyplot as plt

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
    def __init__(self, box_size, dx, dy, external_nodes_list_PEC=None):
        self.dx = dx
        self.dy = dy
        self.boxSize = box_size

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

        if self.nodesList is not None:

            for i in range(len(self.nodesList)):
                current_node = self.nodesList[i]
                next_node = self.nodesList[(i+1) % len(self.nodesList)]

                for j in range(len(self.gridEx)):
                    if min(current_node[0], next_node[0]) <= self.gridEx[j] <= max(current_node[0], next_node[0]):
                        self.columns[j].append(lineEquation(current_node, next_node)(self.gridEx[j]))
                        self.columns[j].sort()

                for k in range(len(self.gridEy)):
                    if min(current_node[1], next_node[1]) <= self.gridEy[k] <= max(current_node[1], next_node[1]):
                        self.rows[k].append(inverseLineEquation(current_node, next_node)(self.gridEy[k]))
                        self.rows[k].sort()

        return self.columns, self.rows

    def plotElectricFieldGrid(self):
        columns, rows = self.electricFieldGridCreation()
        fig, ax = plt.subplots()
        for i, vec in enumerate(columns):
            x_coords = [self.gridEx[i]] * len(vec)  # Coordenadas x iguales para cada vector
            y_coords = vec                      # Valores del vector como coordenadas y
            ax.plot(x_coords, y_coords, marker='o', color='b')

        for j, vec in enumerate(rows):
            y_coords = [self.gridEy[j]] * len(vec)  # Coordenadas y iguales para cada vector
            x_coords = vec                      # Valores del vector como coordenadas x
            ax.plot(x_coords, y_coords, marker='s', color='b')

        if self.nodesList is not None:
            for k in range(len(self.nodesList)):
                current_node = self.nodesList[k]
                next_node = self.nodesList[(k+1) % len(self.nodesList)]

                x_vals = np.linspace(current_node[0], next_node[0], 100)
                y_vals = np.linspace(current_node[1], next_node[1], 100)
                ax.plot(x_vals, y_vals, color='r')

        ax.set_xlabel('Grid Ex')
        ax.set_ylabel('Grid Ey')
        ax.set_title('Mesh Plot')
        ax.legend()
        ax.grid(True)
        plt.show()


node_list = [(0,0), (1,10)]
mesh = Mesh(box_size=10, dx=1.0, dy=0.5, external_nodes_list_PEC=node_list)
mesh.plotElectricFieldGrid()
