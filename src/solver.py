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
from observer import Observer

def eprint(*args, **kwargs):
	print(*args,file=sys.stderr,**kwargs)
	
class Solver:
	
	prm = {
		'task':'test',

		'runs':1,
		
		'variables':0,
		
		'popSize':0,	
		'generations':0,
		
		'trace':1,
		
		'lowerBounds':[0.0,0.0],
		'upperBounds':[2.0,2.0],
		
		}
	
	prm['maxEval'] = prm['generations']*prm['popSize']
	objectives = 0

	def makeBoundsArray(self,bounds):
		variables = bounds.split(',')
		vbounds = [0] * len(variables)
		for i in range(0,len(variables)):
			vbounds[i] = float(variables[i])
		return vbounds
	
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
				
	def tracing(self):
		return self.prm['trace']==1
	
	def loadFile(self,paramFile):
		f = io.open(paramFile)
		lines = f.readlines()
		f.close()
		return lines

	def printParams(self):
		for p in self.prm:	
			print("{}:{}".format(p,self.prm[p]))
				
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

	def solveGA(self,problem,results_dir, run):
		max_eval = int(self.prm['maxEval'])
		pbar = tqdm(total=max_eval, desc="GA", unit="eval")
		original_evaluate = problem.evaluate
		def wrapped_evaluate(solution):
			result = original_evaluate(solution)
			pbar.update(1)
			return result
		problem.evaluate = wrapped_evaluate
		tic = timeit.default_timer()
		algorithm = GeneticAlgorithm(
			problem=problem,
			population_size=int(self.prm['popSize']),
			offspring_population_size=int(self.prm['popSize']),
			mutation=PolynomialMutation(1.0 / problem.number_of_variables(), distribution_index=20.0),
			crossover=SBXCrossover(1.0, distribution_index=20.0),
			selection=BinaryTournamentSelection(),
			termination_criterion=StoppingByEvaluations(max_evaluations=int(self.prm['maxEval']))
		)
		algorithm.observable.register(Observer(results_dir, "GA", run))
		try:
			algorithm.run()
		finally:
			problem.evaluate = original_evaluate
			pbar.close()
		toc = timeit.default_timer()
		results = algorithm.result()
		self.saveResults(results_dir, results, toc - tic, run)
		self.logResults(problem,results.objectives,results,toc - tic)
		if self.tracing():
			self.printResults(problem,algorithm,results,toc - tic)
			eprint("",flush=True)
			time.sleep(0.5)

		return results
				
	def solveNSGA2(self,problem,results_dir, run):
		max_eval = int(self.prm['maxEval'])
		pbar = tqdm(total=max_eval, desc="NSGA-II", unit="eval")
		original_evaluate = problem.evaluate
		def wrapped_evaluate(solution):
			result = original_evaluate(solution)
			pbar.update(1)
			return result
		problem.evaluate = wrapped_evaluate
		tic = timeit.default_timer()
		# binary_string_length = 32
		#problem = OneZeroMax(binary_string_length)

		algorithm = NSGAII(
			problem=problem,
			population_size=int(self.prm['popSize']),
			offspring_population_size=int(self.prm['popSize']),
			mutation=PolynomialMutation(1.0 / problem.number_of_variables(), distribution_index=20.0),
			crossover=SBXCrossover(1.0, distribution_index=20.0),
			# crossover=SPXCrossover(probability=1.0),
			termination_criterion=StoppingByEvaluations(max_evaluations=int(self.prm['maxEval']))
			)
		algorithm.observable.register(Observer(results_dir, "NSGA2", run))
		try:
			algorithm.run()
		finally:
			problem.evaluate = original_evaluate
			pbar.close()
		toc = timeit.default_timer()
		front = algorithm.result()
		self.saveResults(results_dir, front, toc - tic, run)
		best = min(front, key=lambda s: (s.objectives[0], s.objectives[1])) 
		# Save results to file
		#print_function_values_to_file(front, 'FUN.' + algorithm.label + ".txt")
		#print_variables_to_file(front, 'VAR.'+ algorithm.label + ".txt")
		
		if type(front) is not list:
			front = [front]
		for p in front:
			self.logResults(problem,p.objectives,p,toc - tic)
		print()

		return best

	def test(self,problem,testvalues: FloatSolution):
		problem.trace = True
		sol = FloatSolution(problem.lower_bound,problem.upper_bound,problem.number_of_objectives())
		sol.variables = testvalues
		result = problem.evaluate(sol)
		eprint("Score: %.2f" % (result.objectives[0]))

	def checkCommands(self,args):
		if len(args) < 1: return
		self.assignParams(args)

	def create_results_folder(self, config):
		algorithm = self.prm["task"]

		base_dir = os.path.join("results", algorithm)
		os.makedirs(base_dir, exist_ok=True)

		test_num = 1
		while os.path.exists(os.path.join(base_dir, f"test-{test_num}")):
			test_num += 1

		results_dir = os.path.join(base_dir, f"test-{test_num}")
		os.makedirs(results_dir)

		shutil.copy2(config, results_dir)

		return results_dir

	def saveResults(self, results_dir, result, runtime, run):

		algorithm = self.prm["task"]

		csv_path = os.path.join(results_dir, f"{algorithm}-results.csv")

		file_exists = os.path.exists(csv_path)

		with open(csv_path, "a", newline="") as f:

			writer = csv.writer(f)

			if not file_exists:
				if isinstance(result, list):
					writer.writerow(["Run", "Solution", "Distance", "Strokes", "Holed", "Runtime"])
				else:
					writer.writerow(["Run", "Distance", "Strokes", "Holed", "Fitness", "Runtime"])

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
			else:
				writer.writerow([
					run + 1,
					result.distance,
					result.strokes,
					result.holed,
					result.objectives[0],
					round(runtime, 1)
				])

	def processTask(self,options):

		best_solution = None

		# The first command line parameter must be a parameter file specifying the task
		if len(options)<=1:
			print("Parameter file required:")
			print("  solver alloc-params.txt")
			return
		
		# Load the task parameters from the supplied param file
		config = self.loadFile(options[1])
		self.assignParams(config)

		results_dir = self.create_results_folder(options[1])
		# Now override file parameters with possible command line parameters 
		self.checkCommands(options)

		maxStrokes = int(self.prm['maxStrokes'])

		lowerBounds = self.prm['lowerBounds'] * maxStrokes
		upperBounds = self.prm['upperBounds'] * maxStrokes

		handicap = float(self.prm['handicap'])
		player = Player(handicap)
		
		heightmap = self.prm["heightmap"]
		surfacemap = self.prm["surfacemap"]
		hole = Hole(player, heightmap, surfacemap)

		problem = Eval(self.prm['objectives'], lowerBounds, upperBounds, hole)
			
		# What's our task?
		task = self.prm['task']
		if (self.tracing()): # if trace is true, print out the config for this task 
			eprint("Task: %s" % (task))
			problem.describe()
			self.printParams()

		runs = int(self.prm['runs'])

		for run in range(runs):

			print(f"\nRun {run + 1}/{runs}")

			# Now carry out the task	
			if (task == "test"): self.test(problem,[1, 0, 0, 0])

			if (task == "GA"):
				solution = self.solveGA(problem, results_dir, run)
				if best_solution is None or solution.objectives[0] < best_solution.objectives[0]:
					best_solution = solution

			if (task == "NSGA2"):
				solution = self.solveNSGA2(problem, results_dir, run)
				if (best_solution is None or (solution.objectives[0], solution.objectives[1]) < (best_solution.objectives[0], best_solution.objectives[1])):
					best_solution = solution

		if best_solution is not None:
			problem.hole.show_strategy(best_solution, screenshot=os.path.join(results_dir, "best-strategy.png"))
			problem.hole.show_strategy(best_solution)

if __name__ == "__main__":
	logging.disable()
	ev = Solver()
	ev.processTask(sys.argv)
