# CFDTD

In this repository you will find the implementation of the Conformal Finite Difference Time Domain (CFDTD) method for both one and two dimensions. In the case of one dimension, both the mesh class, the solver and the initial pulse are in the same file (cfdtd/CFDTD1D.py). For the two-dimensional algorithm, each class is separated in its respective file. To formulate the iterative equation of the 2D magnetic field, we worked in the transversal electric (TE) mode using the Dey-Mittra algorithm.

<div align="center">
  <img src="https://github.com/ashybabashyba/CFDTD/blob/main/TFM/Imagenes/RectangularResonantCavity.gif" alt="Rectangular Resonant Cavity GIF">
</div>