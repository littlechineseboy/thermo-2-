# 2D Heat Equation Solver

Numerical solution of the steady-state heat equation on a 2D grid
using finite difference discretization.

## Methods
- LU decomposition (direct solver)
- Gauss-Seidel (iterative solver)

## How to run
pip install -r requirements.txt
streamlit run app.py

## Project structure
- solver.py   → core solver functions + experiments
- app.py      → interactive Streamlit app

## Authors
- mariem khieri— matrix assembly, solvers, experiments
- mazen mathlouthi — analysis, visualization, Streamlit app
