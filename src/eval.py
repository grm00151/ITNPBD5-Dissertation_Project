import numpy as np
import sys

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from hole import Hole, Player

def eprint(*args, **kwargs):
	print(*args,file=sys.stderr,**kwargs)

class Eval(FloatProblem):

	def __init__(self,obj,lowerBounds,upperBounds, hole):
		super(Eval, self).__init__()
		
		# Take a note of the upper and lower allowable values for the thresholds		
		self.lower_bound = lowerBounds
		self.upper_bound = upperBounds 
		
		# How many objectives do we have (1 by default for a GA and more for NSGA2)
		self.objectives = obj
		self.constraints = 0

		self.hole = hole
		
		# What are the names for the objectives
		self.obj_labels = ['Distance', 'Strokes']

	def name(self) -> str:
		return 'Evaluator'
		
	def describe(self):
		eprint("Allocator")
		
	def number_of_variables(self):
		return len(self.lower_bound)

	def number_of_objectives(self):
		return int(self.objectives)

	def number_of_constraints(self):
		return int(self.constraints)

	# --- evalMaxOnes ---
	# Trivially simple optimisation problem that checks the 
	# code is functioning correctly. The goal is simply to 
	# get the highest sum for the array of numbers by 
	# working out that each one should be set to the highest value
	# it is allowed to take. 
	#
	# You can make different fitness functions like this that take
	# an array of values and return a score (e.g. feed them into a neural
	# network and make a prediction, then return that prediction value).
	def evalMaxOnes(self,variables):
		score = 0
		for v in variables:
			score = score + v
		return score
	
	def evalAltVal(self,variables):
		score = 0
		high = (variables[0] > 0.5)	
		for v in range(1,len(variables)):
			if high and (variables[v] > 0.5):
				score = score + 1
			else:
				score = score - 1
			high = (variables[v] > 0.5)
		return score 
	
	def evalGolfStrategy(self, variables):
		
		position = np.copy(self.hole.tee_position)
		
		HOLE_RADIUS = 1.0
		strokes = 0

		trajectories = []
		
		for i in range(0, len(variables), 3):
			
			power = variables[i]
			direction = variables[i + 1]
			
			club = int(round(variables[i + 2]))
			club = int(np.clip(club, 0, len(self.hole.player.clubs) - 1))

			club = self.hole.get_allowed_club(position, club)
			variables[i + 2] = club
			
			position, trajectory, out_of_bounds, club = self.hole.simulate_shot(position, power, direction, club)

			trajectories.append((trajectory, club))

			if out_of_bounds:
				return (1e6, 1e6, trajectories)

			strokes += 1
			
			distance = np.linalg.norm(position - self.hole.hole_position)
			
			# Hole reached
			if distance <= HOLE_RADIUS:
				return 0.0, strokes, trajectories
			
		# Hole not reached
		final_distance = np.linalg.norm(position - self.hole.hole_position)
			
		return final_distance, strokes, trajectories

	# evaluate is called by the code in solver.py but you only need to edit
	# code in here and try out different fitness functions.
	def evaluate(self, solution: FloatSolution) -> FloatSolution:		
		# By default, the optimiser will try to minimise the fitness score
		# so we flip this to make it a maximisation problem
		# (a larger fitness score will result in smaller overall result which 
		# is what the optimiser is trying to get from calling this function with
		# different possible solutions.) For it, a score of 0 is good.

		# You can change evalMaxOnes to a different problem and keep a set of
		# these functions in this file.

		# solution.objectives[0] = 8 - self.evalMaxOnes(solution.variables)
		# solution.objectives[0] = self.evalMaxOnes(solution.variables)
		# solution.objectives[0] = self.evalGolfStrategyGA(solution.variables)
		distance, strokes, trajectories = self.evalGolfStrategy(solution.variables)

		solution.distance = distance
		solution.strokes = strokes
		solution.trajectories = trajectories
		solution.holed = (distance <= 1.0)
		
		if (len(solution.objectives) > 1):
			solution.objectives[0] = distance
			solution.objectives[1] = strokes
		else:
			solution.objectives[0] = distance + (strokes * 200)

		return solution

# Test the function
if __name__ == "__main__":
	
	player = Player(-52)
	
	hole = Hole(player, "images/heightmap.png", "images/surfacemap.png")
	
	hole.show_shot(hole.tee_position, power=100, direction=269, club_index=0)