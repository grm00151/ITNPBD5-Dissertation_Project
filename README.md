# Optimising shot strategy in golf using evolutionary algorithms

## Overview

My project implements a golf shot simulation and optimisation framework in python. It uses evolutionary algorithms to discover optimal golf strategies over multiple stokes on a heightmapped golf hole.

The two optimisation algorithms used are:

**Genetic Algorithm (GA)** - Single objective optimisation 
**Non-dominated Sorting Genetic Algorithm 2 (NSGA2)** - Multi objective optimisation

The simulations models ball flight, roll after landing, terrain elevation, surface types, player handicap, club selection, shot power and direction.

Features include physics based golf shot simulation, heightmap terrain, surface map classification, 14 different golf clubs, handicap based shot variability, club restrictions, 3D visualisation using PyVista, Strategy optimisation and automatic saving of optimisation results. 

## Requirements

Python 3.10+

Required packages:

```
numpy
pyvista
pillow
jmetalpy
tqdm
```

Install using pip:

```bash
pip install numpy pyvista pillow jmetalpy tqdm
```

## Running the optimisation 

Run the following commands from a terminal (Command Prompt, PowerShell, Terminal) in the projects root directory. 

### Genetic Algorithm 

```bash
python src/solver.py configs/GA-config-params.txt
```

### Non-dominated Sorting Genetic Algorithm 2

```bash
python src/solver.py configs/NSGA2-config-params.txt
```

## Config files 

Both optimisation methods use config files located in the 'configs' directory.

### Parameters 

```
task - GA or NSGA2
runs - Number of independent algorithm executions
lowerBounds - Minimum values for power, direction and club
lowerBounds - Maximum values for power, direction and club
popSize - Population size
generations - Number of generations
objectives - Number of objectives (1 for GA, 2+ for NSGA2)
trace - Enable console output
heightmap - Terrain height image 
surfacemap - Surface classification image 
maxStrokes - Maximum stokes per strategy
handicap - Player handicap
```

## Decision Variables

Each shot is represented by three variables:

```
power - 0-100
direction - 0-360
club index - 0-13
```

## Objectives

### Genetic Algorithm 

Minimises the distance to the hole + 200 times the number of strokes encouraging reaching the hole and using fewer strokes

### Non-dominated Sorting Genetic Algorithm 2

Optimises two objectives at the same time, distance to the hole and number of strokes. This produces a pareto front of non-dominated strategies. 

## Golf simulation 

Each shot consists of club selection, launch angle calculation, inital ball speed, ball flight, aerodynamic drag, backspin lift, shot curvature, ball landing, rolling and surface interaction. Surface types include fairway, rough, green, bunker and out of bounds.

Rules include: 
Putter is automatically selected on the green. 
Sand wedge is automatically selected in the bunkers. 
Penalty for going out of bounds. 

## Player model

Higher handicaps introduce greater randomness in shot direction and ball speed. 

## Visualisation 

PyVista is used to display the terrain, surface colours, shot trajectories, club specific shot colours and overall strategy visualisations

## Results

Each optimisation run creates a new directory:

```
results/
    GA/
        test-1/
        test-2/
        ...
    NSGA2/
        test-1/
```

Where each test folder contains:

```
best-strategy.png
(algorithm)-config-params.txt
(algorithm)-convergence.csv
(algorithm)-results.csv
```

best-strategy.png shows the best strategy found across all runs of the algorithm.

(algorithm)-config-params.txt contains the configuation parameters used for the experiment. 

(algorithm)-convergence.csv file show the convergence behaviour across all runs included in the experiment.

(algorithm)-results.csv file contains run number, distance to hole, stokes, if it reached the hole, fitness depending on algorithm and runtime.

## Future improvements

Possible extensions include:

```
Wind effects/Weather conditions
Additional optimisation algorithms
Improved modelling
Multiple hole support
```

## Author

Graeme MacDonald 

Dissertation Project - Optimising shot strategy in golf using evolutionary algorithms

Developed in Python. 