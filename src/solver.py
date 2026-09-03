import sys
import io
import timeit
import time
import logging
import os
import csv
import shutil

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.mutation import PolynomialMutation, BitFlipMutation
from jmetal.operator.crossover import SBXCrossover, SPXCrossover
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.problem.multiobjective.unconstrained import OneZeroMax
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.core.solution import BinarySolution, FloatSolution


from jmetal.util.termination_criterion import StoppingByEvaluations
# from examples.singleobjective import genetic_algorithm
from jmetal import algorithm
from tqdm import tqdm 


from eval import Eval
from hole import Player, Hole

def eprint(*args, **kwargs):
	print(*args,file=sys.stderr,**kwargs)

# watches the optimisation process and record distance to hole, strokes and fitness depending on algorithm. 
class Observer:

    def __init__(self, results_dir, algorithm_name, run):

		# starts number at 1 instead of 0
        self.run = run + 1
		# name of algorithm used
        self.algorithm_name = algorithm_name

		# start with large values so first improvement can be saved 
        self.best_fitness = float("inf")
        self.best_distance = float("inf")
        self.best_strokes = float("inf")

		# create path for the convergence CSV file 
        self.csv_path = os.path.join(
            results_dir,
            f"{algorithm_name}-convergence.csv"
        )

		# check whether the CSV file already exists
        file_exists = os.path.exists(self.csv_path)

		# open the convergence file in append mode
		# new results aded and not overwriting old results
        with open(self.csv_path, "a", newline="") as file:
			# create CSV writer
            writer = csv.writer(file)

			# only create the column headings if this is a new file
            if not file_exists:
                writer.writerow([
                    "Run",
                    "Evaluations",
                    "Distance",
                    "Strokes",
                    "Fitness"
                ])

	# jMetal calls update when the algorithms produces new solutions
    def update(self, *args, **kwargs):

		# get current silutions and evaluations performed so far
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

		# check if jMetal has supplied multiple solutions
        if isinstance(solutions, list):

			# if the list has no solutions there is nothing to record
            if len(solutions) == 0:
                return

			# chose the solution with the smallest distance and if same go with lowest shots 
            solution = min(
                solutions,
                key=lambda s: (s.distance, s.strokes)
            )

		# if only one solution was supplied use it directly
        else:

            solution = solutions

		# check which algorithm is running 
        if self.algorithm_name == "GA":

			# get the fitness value
            fitness = solution.objectives[0]

			# if its better record it, if not dont 
            if fitness >= self.best_fitness:
                return

            self.best_fitness = fitness

		# dealing with NSGA-II
        else:
			# same again for NSGA-II but doesnt have fitness
            if (
                solution.distance > self.best_distance or
                (
                    solution.distance == self.best_distance
                    and solution.strokes >= self.best_strokes
                )
            ):
				# no improvement dont record 
                return

			# store the new best distance and strokes 
            self.best_distance = solution.distance
            self.best_strokes = solution.strokes

            fitness = ""

		# open the convergence file and append the new best results
        with open(self.csv_path, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                self.run,
                evaluations,
                round(solution.distance, 3),
                solution.strokes,
                round(fitness, 3) if fitness != "" else ""
            ])
	
class Solver:

	# default parameters for the optimisation 
	prm = {
		'task':'test',

		# number of times the algorithm runs
		'runs':1,
		
		'variables':0,
		
		'popSize':0,	
		'generations':0,
		
		'trace':1,
		
		'lowerBounds':[0.0,0.0],
		'upperBounds':[2.0,2.0],
		
		}

	# caclulates maximum number of evaluations
	prm['maxEval'] = prm['generations']*prm['popSize']
	objectives = 0

	# convert comma seperated string of bounds into a list of numbers
	def makeBoundsArray(self,bounds):
		variables = bounds.split(',')
		vbounds = [0] * len(variables)
		for i in range(0,len(variables)):
			vbounds[i] = float(variables[i])
		return vbounds

	# assign parameters read from provided configuration file
	def assignParams(self,lines):
		for fl in lines:
			l = fl.strip()
			if len(l) == 0:	continue
			if (l.startswith("//")): continue
			toks=l.split("=")
			if (len(toks) < 2): continue
			
			# print("%s = %s" % (toks[0],toks[1]))
			var = toks[0].strip()
			if (var == 'lowerBounds' or var == 'upperBounds'):
				self.prm[var] = self.makeBoundsArray(toks[1].strip())
				continue
			
			try:
				self.prm[var] = float(toks[1].strip())
			except Exception as e:
				self.prm[var] = toks[1].strip()
					
		self.prm['maxEval'] = self.prm['generations']*self.prm['popSize']

	# return whether tracing/output has been enabled 
	def tracing(self):
		return self.prm['trace']==1

	# load the provided parameter file
	def loadFile(self,paramFile):
		f = io.open(paramFile)
		lines = f.readlines()
		f.close()
		return lines

	# print all the current parameters 
	def printParams(self):
		for p in self.prm:	
			print("{}:{}".format(p,self.prm[p]))

	# print information about the final optimisation result
	def printResults(self,problem,algorithm,result,runtime):
		eprint("\nAlgorithm")
		eprint("Population: %.0f" % (self.prm['popSize']))
		eprint('Algorithm: {}'.format(algorithm.get_name()))
		eprint('Problem: {}'.format(problem.name()))
		eprint('Fitness: {}'.format(result.objectives))
		eprint('Solution {}'.format(result.variables))
		eprint('Computing time: {}'.format(algorithm.total_computing_time))
		eprint("Mean Cost = %.3f sec/ eval" % (algorithm.total_computing_time/self.prm['maxEval']))
	
		if type(result) is not list:
			result = [result]
		
		for r in result:
			print(r)

	# print the objectives, variables and runtime
	def logResults(self,problem,objectives,result,runtime):
		
		# Scores
		first = True
		#print('[',end="", flush=True)
		for o in objectives:
			if not first: print(",",end="",flush=True)
			first = False
			print("%.4f" % (o), end="")
		print(",\t\t",end="")
		# print('{}:'.format(result.objectives),end="")
		
		# Solution
		first = True
		for r in result.variables:
			if not first: print(', ',end="", flush=True)
			print("%.3f" % (r),end="", flush=True)
			first = False
		print("\t,%.2f" % (runtime),flush=True)

	#AI used to add progress bar to solveGA and solveNSGA2
	def solveGA(self,problem,results_dir, run):
		# get max number of evaluations for stopping criteria 
		max_eval = int(self.prm['maxEval'])
		# create progress bar with tqdm
		pbar = tqdm(total=max_eval, desc="GA", unit="eval")
		# save the original evaluation method 
		original_evaluate = problem.evaluate
		# define a wrapper around the original evaluation function
		def wrapped_evaluate(solution):
			# evaluate the solution normally
			result = original_evaluate(solution)
			# increase the progress bar by one evaluation
			pbar.update(1)
			# return evaluated solution
			return result
		# replace the problem's evaluate method with wrapped version
		problem.evaluate = wrapped_evaluate
		# record start time
		tic = timeit.default_timer()
		#create the Genetic Algorithm
		algorithm = GeneticAlgorithm(
			problem=problem, # problem to solve
			population_size=int(self.prm['popSize']), # set population size
			offspring_population_size=int(self.prm['popSize']), # number of offspring produced 
			mutation=PolynomialMutation(1.0 / problem.number_of_variables(), distribution_index=20.0), # polynomial mutation
			crossover=SBXCrossover(1.0, distribution_index=20.0), # use simulated binary crossover
			selection=BinaryTournamentSelection(), # use binary tournament selection to select parents
			termination_criterion=StoppingByEvaluations(max_evaluations=int(self.prm['maxEval'])) # stop at max evals
		)
		# record improvements 
		algorithm.observable.register(Observer(results_dir, "GA", run))
		try:
			# run the GA
			algorithm.run()
		finally:
			# restore original evaluate method and close progress bar
			problem.evaluate = original_evaluate
			pbar.close()
		# record the end time
		toc = timeit.default_timer()
		# get final results from the algorithm
		results = algorithm.result()
		# save results to CSV file
		self.saveResults(results_dir, results, toc - tic, run)
		# print results
		self.logResults(problem,results.objectives,results,toc - tic)
		if self.tracing():
			self.printResults(problem,algorithm,results,toc - tic)
			eprint("",flush=True)
			time.sleep(0.5)

		return results
				
	def solveNSGA2(self,problem,results_dir, run):
		max_eval = int(self.prm['maxEval']) # max number of evals
		# create progress bar
		pbar = tqdm(total=max_eval, desc="NSGA-II", unit="eval")
		# save original evaluation method
		original_evaluate = problem.evaluate
		# wrapper that updates progress bar
		def wrapped_evaluate(solution):
			result = original_evaluate(solution)
			pbar.update(1)
			return result
		problem.evaluate = wrapped_evaluate
		# start timer
		tic = timeit.default_timer()
		# binary_string_length = 32
		#problem = OneZeroMax(binary_string_length)

		algorithm = NSGAII(
			problem=problem, # give algorithm the problem
			population_size=int(self.prm['popSize']), # set population size 
			offspring_population_size=int(self.prm['popSize']), # set offspring pop size
			mutation=PolynomialMutation(1.0 / problem.number_of_variables(), distribution_index=20.0), # use polynomial mutation
			crossover=SBXCrossover(1.0, distribution_index=20.0), # use simulated binary crossover
			# crossover=SPXCrossover(probability=1.0),
			termination_criterion=StoppingByEvaluations(max_evaluations=int(self.prm['maxEval'])) #stop at max evals
			)
		# observe improvements
		algorithm.observable.register(Observer(results_dir, "NSGA2", run))
		try:
			# run NSGA-II
			algorithm.run()
		finally:
			# restore orginal evaluation method and close progress bar
			problem.evaluate = original_evaluate
			pbar.close()
		# record the end time
		toc = timeit.default_timer()
		# get the pareto front returned 
		front = algorithm.result()
		# save the results 
		self.saveResults(results_dir, front, toc - tic, run)
		# select the solution with the smallest distnce, if same use smallest strokes
		best = min(front, key=lambda s: (s.objectives[0], s.objectives[1])) 
		# Save results to file
		#print_function_values_to_file(front, 'FUN.' + algorithm.label + ".txt")
		#print_variables_to_file(front, 'VAR.'+ algorithm.label + ".txt")
		
		# print every solution in pareto front
		if type(front) is not list:
			front = [front]
		for p in front:
			self.logResults(problem,p.objectives,p,toc - tic)
		print()

		return best

	# not used but runs a test using manual variables 
	def test(self,problem,testvalues: FloatSolution):
		problem.trace = True
		sol = FloatSolution(problem.lower_bound,problem.upper_bound,problem.number_of_objectives())
		sol.variables = testvalues
		result = problem.evaluate(sol)
		eprint("Score: %.2f" % (result.objectives[0]))

	# process command line arguments
	def checkCommands(self,args):
		if len(args) < 1: return
		self.assignParams(args)

	# create a folder to store the results of a test
	def create_results_folder(self, config):
		# get algorithm name
		algorithm = self.prm["task"]
		# create a base results directory 
		base_dir = os.path.join("results", algorithm)
		# create the directory if it doesnt exisit
		os.makedirs(base_dir, exist_ok=True)
		# start looking for test 1
		test_num = 1
		# keep increasing the number while the folder already exists
		while os.path.exists(os.path.join(base_dir, f"test-{test_num}")):
			test_num += 1 # move to next test number
		# create the path for the new test folder
		results_dir = os.path.join(base_dir, f"test-{test_num}")
		# create the folder
		os.makedirs(results_dir)
		# copy config file into results folder
		shutil.copy2(config, results_dir)

		# return the results directory 
		return results_dir

	# save optimisation results to a CSV file
	def saveResults(self, results_dir, result, runtime, run):
		# algorithm name 
		algorithm = self.prm["task"]
		# create the filename for the result CSV
		csv_path = os.path.join(results_dir, f"{algorithm}-results.csv")
		# check whther the file already exisits 
		file_exists = os.path.exists(csv_path)
		# open file in append mode
		with open(csv_path, "a", newline="") as f:
			# create CSV writer
			writer = csv.writer(f)
			# add headers if its a new file
			if not file_exists:
				# NSGA-II returns multiple solutions
				if isinstance(result, list):
					writer.writerow(["Run", "Solution", "Distance", "Strokes", "Holed", "Runtime"])
				# GA reutns one solution
				else:
					writer.writerow(["Run", "Distance", "Strokes", "Holed", "Fitness", "Runtime"])

			# if multiple solutions
			if isinstance(result, list):
				for i, solution in enumerate(result, start=1):
					writer.writerow([
						run + 1,
						i,
						solution.distance,
						solution.strokes,
						solution.holed,
						round(runtime, 1)
					])
			# GA returns a single solution
			else:
				writer.writerow([
					run + 1,
					result.distance,
					result.strokes,
					result.holed,
					result.objectives[0],
					round(runtime, 1)
				])

	# process the requested optimisation task
	def processTask(self,options):

		# start with no best solution
		best_solution = None

		# The first command line parameter must be a parameter file specifying the task
		if len(options)<=1:
			print("Parameter file required:")
			print("  solver alloc-params.txt")
			return
		
		# Load the task parameters from the supplied param file
		config = self.loadFile(options[1])
		# apply the configuration values to the solver parameters
		self.assignParams(config)
		# create a folder for current experiment
		results_dir = self.create_results_folder(options[1])
		# Now override file parameters with possible command line parameters 
		self.checkCommands(options)

		# get maximum number of allowed strokes 
		maxStrokes = int(self.prm['maxStrokes'])

		# repeat the bounds for every possible stroke 
		# this creates bound for every decision variable in every shot
		lowerBounds = self.prm['lowerBounds'] * maxStrokes
		upperBounds = self.prm['upperBounds'] * maxStrokes

		# get players handicap
		handicap = float(self.prm['handicap'])
		# create player object
		player = Player(handicap)

		# get the filename of heightmap and surface map
		heightmap = self.prm["heightmap"]
		surfacemap = self.prm["surfacemap"]

		# create the golf hole using the player and hole maps
		hole = Hole(player, heightmap, surfacemap)

		# create the optimisation problem 
		problem = Eval(self.prm['objectives'], lowerBounds, upperBounds, hole)
			
		# get the type of task GA/NSGA-II/test
		task = self.prm['task']
		# if tracing is enabled print config info 
		if (self.tracing()): # if trace is true, print out the config for this task 
			eprint("Task: %s" % (task))
			problem.describe()
			self.printParams()

		# gets the number of independant runs 
		runs = int(self.prm['runs'])

		# run the selected algorithm the requested number of times
		for run in range(runs):

			print(f"\nRun {run + 1}/{runs}")

			# Now carry out the task	
			if (task == "test"): self.test(problem,[1, 0, 0, 0])

			# run GA and store solution
			if (task == "GA"):
				solution = self.solveGA(problem, results_dir, run)
				if best_solution is None or solution.objectives[0] < best_solution.objectives[0]:
					best_solution = solution

			# run NSGA-II and store best solution
			if (task == "NSGA2"):
				solution = self.solveNSGA2(problem, results_dir, run)
				if (best_solution is None or (solution.objectives[0], solution.objectives[1]) < (best_solution.objectives[0], best_solution.objectives[1])):
					best_solution = solution

		# Once all runs have finished, check if a solution exisits
		if best_solution is not None:
			# display the best strategy and save a screenshot of it
			problem.hole.show_strategy(best_solution, screenshot=os.path.join(results_dir, "best-strategy.png"))
			# open interactive environemnt 
			problem.hole.show_strategy(best_solution)

# runs when solver.py is executed
if __name__ == "__main__":
	# disables loggin messages 
	logging.disable()
	# creates solver object
	ev = Solver()
	# pass the command line arguments to process task
	# python src/solver.py configs/GA-config-params.txt
	ev.processTask(sys.argv)