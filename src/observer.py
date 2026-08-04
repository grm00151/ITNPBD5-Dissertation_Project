import csv
import os


class Observer:

    def __init__(self, results_dir, algorithm_name, run):

        self.run = run + 1
        self.algorithm_name = algorithm_name

        self.best_fitness = float("inf")
        self.best_distance = float("inf")
        self.best_strokes = float("inf")

        self.csv_path = os.path.join(
            results_dir,
            f"{algorithm_name}-convergence.csv"
        )

        file_exists = os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "Run",
                    "Evaluations",
                    "Distance",
                    "Strokes",
                    "Fitness"
                ])

    def update(self, *args, **kwargs):

        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if isinstance(solutions, list):

            if len(solutions) == 0:
                return

            solution = min(
                solutions,
                key=lambda s: (s.distance, s.strokes)
            )

        else:

            solution = solutions

        if self.algorithm_name == "GA":

            fitness = solution.objectives[0]

            if fitness >= self.best_fitness:
                return

            self.best_fitness = fitness

        else:

            if (
                solution.distance > self.best_distance or
                (
                    solution.distance == self.best_distance
                    and solution.strokes >= self.best_strokes
                )
            ):
                return

            self.best_distance = solution.distance
            self.best_strokes = solution.strokes

            fitness = ""

        with open(self.csv_path, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                self.run,
                evaluations,
                round(solution.distance, 3),
                solution.strokes,
                round(fitness, 3) if fitness != "" else ""
            ])