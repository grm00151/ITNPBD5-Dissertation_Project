import sys
import io
import timeit
import time
import logging

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
	
class Solver:
	
	prm = {
		'task':'test',
		
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

	def solveGA(self,problem):
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
		try:
			algorithm.run()
		finally:		
			pbar.close()
		toc = timeit.default_timer()
		results = algorithm.result()
		problem.hole.show_strategy(results.variables)
		self.logResults(problem,results.objectives,results,toc - tic)
		if self.tracing():
			self.printResults(problem,algorithm,results,toc - tic)
			eprint("",flush=True)
			time.sleep(0.5)
			
		# Uncomment the following line to show the best solution
		score = original_evaluate(results)
		eprint("Score: %.2f" % (results.objectives[0]))
				
	def solveNSGA2(self,problem):
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
		try:
			algorithm.run()
		finally:
			pbar.close()
		front = algorithm.result()
		best = min(front, key=lambda s: (s.objectives[0], s.objectives[1])) 
		problem.hole.show_strategy(best.variables)
		# Save results to file
		#print_function_values_to_file(front, 'FUN.' + algorithm.label + ".txt")
		#print_variables_to_file(front, 'VAR.'+ algorithm.label + ".txt")
		
		toc = timeit.default_timer()
		if type(front) is not list:
			front = [front]
		for p in front:
			self.logResults(problem,p.objectives,p,toc - tic)
		print()

	def test(self,problem,testvalues: FloatSolution):
		problem.trace = True
		sol = FloatSolution(problem.lower_bound,problem.upper_bound,problem.number_of_objectives())
		sol.variables = testvalues
		result = problem.evaluate(sol)
		eprint("Score: %.2f" % (result.objectives[0]))

	def checkCommands(self,args):
		if len(args) < 1: return
		self.assignParams(args)
			
	def processTask(self,options):
		# The first command line parameter must be a parameter file specifying the task
		if len(options)<=1:
			print("Parameter file required:")
			print("  solver alloc-params.txt")
			return
		
		# Load the task parameters from the supplied param file
		config = self.loadFile(options[1])
		self.assignParams(config)
		# Now override file parameters with possible command line parameters 
		self.checkCommands(options)

		maxShots = int(self.prm['maxShots'])

		lowerBounds = self.prm['lowerBounds'] * maxShots
		upperBounds = self.prm['upperBounds'] * maxShots

		handicap = self.prm['handicap']
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

		# Now carry out the task	
		if (task == "test"): self.test(problem,[1, 0, 0, 0])
		if (task == "GA"): self.solveGA(problem)
		if (task == "NSGA2"): self.solveNSGA2(problem)

if __name__ == "__main__":
	logging.disable()
	ev = Solver()
	ev.processTask(sys.argv)
