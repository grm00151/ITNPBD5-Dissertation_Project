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

		#store the golf hole so it can be used when evaluating strategies 
		self.hole = hole
		
		# What are the names for the objectives
		# The two objectives are distance from the hole and number of strokes
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

		# make a copy of the tee position so that it can be updated 
		position = np.copy(self.hole.tee_position)

		# the radius within which the ball is considered to be in the hole
		HOLE_RADIUS = 1.0

		# start the number of strokes at zero 
		strokes = 0

		# create an empty list to store the trajectory of every shot
		trajectories = []

		# Each shot used 3 variables: 1 = power, 2 = direction and 3 = club
		for i in range(0, len(variables), 3):

			# gets the power value for the current shot
			power = variables[i]
			# gets the direction value for the current shot
			direction = variables[i + 1]

			# get the club value and round it because the optimiser produces a floating point number and is within range
			club = int(round(variables[i + 2]))
			club = int(np.clip(club, 0, len(self.hole.player.clubs) - 1))

			# check whether the selected club is allowed from the balls current position
			club = self.hole.get_allowed_club(position, club)
			# store the corrected club value back into the variable
			variables[i + 2] = club

			#Simulate the golf shit using the current position, power, direction and club of current shot
			position, trajectory, out_of_bounds, club = self.hole.simulate_shot(position, power, direction, club)

			# add current shots trajectory and club to the trajectory list
			trajectories.append((trajectory, club))

			# check if the ball is out of bounds 
			if out_of_bounds:
				# return large vales so the optimiser treats this silution as bad
				return (1e6, 1e6, trajectories)

			# the player has taken another stroke
			strokes += 1

			# calculates the distance between the current ball position and the hole
			distance = np.linalg.norm(position - self.hole.hole_position)
			
			# check if the ball is close enough to count as holed
			if distance <= HOLE_RADIUS:
				return 0.0, strokes, trajectories
			
		# calculate the final distance between the ball and the hole if not reached
		final_distance = np.linalg.norm(position - self.hole.hole_position)

		# return remaining distance, strokes and trajectories
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

		# evaluate the current solution as a golf strategy
		distance, strokes, trajectories = self.evalGolfStrategy(solution.variables)

		# store variables inside the solution object
		solution.distance = distance
		solution.strokes = strokes
		solution.trajectories = trajectories
		solution.holed = (distance <= 1.0)

		# check whether how many objective the problem has
		if (len(solution.objectives) > 1):
			# distance to the hole, smaller distance is better
			solution.objectives[0] = distance
			# strokes taken, fewer is better
			solution.objectives[1] = strokes
		else:
			# combine distance and strokes into one fitness score
			solution.objectives[0] = distance + (strokes * 200)

		return solution

# Test the function
if __name__ == "__main__":
	# tests one shot, used alot throughout development
	
	player = Player(-32)
	
	hole = Hole(player, "images/heightmap.png", "images/surfacemap.png")
	
	hole.show_shot(hole.tee_position, power=100, direction=259, club_index=0)