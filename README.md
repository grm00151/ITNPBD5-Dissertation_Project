# Optimising shot strategy in golf using evolutionary algorithms

## Overview

My project implements a golf shot simulation and optimisation framework in python. It uses evolutionary algorithms to discover optimal golf strategies over multiple shots on a heightmapped golf hole.

The two optimisation algorithms used are:

**Genetic Algorithm (GA)** - Single objective optimisation 
**Non-dominated Sorting Genetic Algorithm 2 (NSGA2)** - Multi objective optimisation

The simulations models ball flight, roll after landing, terrain elevation, surface types, player handicap, club selection, shot power and direction.

Features include physics based golf shot simulation, heightmap terrain, surface map classification, 14 different golf clubs, handicap based shot variability, club restrictions, 3D visualisation using PyVista, Strategy optimisation and automatic saving of optimisation results. 

## Requirements

Python 3.10+

Required packages:

numpy
pyvista
pillow
jmetalpy
tqdm

Install using pip:

pip install numpy pyvista pillow jmetalpy tqdm

## Running the optimisation 

### Genetic Algorithm 

python src/solver.py configs/GA-config-params.txt

### Non-dominated Sorting Genetic Algorithm 2

python src/solver.py configs/NSGA2-config-params.txt

## Config files 

Both optimisation methods use config files located in the 'configs' directory.

### Parameters 

task - GA or NSGA2
lowerBounds - Minimum values for power, direction and club
lowerBounds - Maximum values for power, direction and club
popSize - Population size
generations - Number of generations
objectives - Number of objectives (1 for GA, 2+ for NSGA2)
trace - Enable console output
heightmap - Terrain height image 
surfacemap - Surface classification image 
maxShots - Maximum shots per strategy
handicap - Player handicap

## Decision Variables

Each shot is represented by three variables:

power - 0-100
direction - 0-360
club index - 0-13

## Objectives

### Genetic Algorithm 

Minimises the distance to the hole + 20 times the number of strokes encouraging reaching the hole and using fewer shots

### Non-dominated Sorting Genetic Algorithm 2

Optimises two objectives at the same time, distance to the hole and number of strokes. This produces a pareto front on non-dominated strategies. 

## Golf simulation 

Each shot consists of club selection, launch angle calculation, inital ball speed, ball flight, aerodynamic drag, backspin lift, shot curvature, ball landing, rolling and surface interaction. Surface types include fairway, rough, green, bunker and out of bounds.

Rules include: 
Putter is automatically selected on the green. 
Sand wedge is automatically selected in the bunkers. 

## Player model

Higher handicaps introduce greater randomness in shot direction, launch angle and ball speed. 

## Visualisation 

PyVista is used to display the terrain, surface colours, shot trajectories, club specific shot colours and overall strategy visualisations

## Results

Each optimisation run creates a new directory:

results/
    GA/
        test-1
        test-2
        ...
    NSGA2
        test-1

Where each test folder contains:

config file used
results.txt
strategy.png

results.txt contains runtime, objective score(s) and solution variables

## Future improvements
Wind effects/Weather conditions
Additional optimisation algorithms
Multiple hole support

## Author

Graeme MacDonald 

Dissertation Project - Optimising shot strategy in golf using evolutionary algorithms