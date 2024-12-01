import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.termination import get_termination
import os

from train import train_model, evaluate_model

"""
    Espacio de busqueda
    d = 36-64
    s = 8-20
    m = 1-5
"""
class NASProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=3,
            n_obj=2,
            n_constr=0,
            xl=np.array([36, 8, 1]),  
            xu=np.array([64, 20, 5]),
            vtype=int
        )
        self.current_gen = 1

    def _evaluate(self, X, out, *args, **kwargs):
        psnr_dataset1 = []
        psnr_dataset2 = []
        logs = []

        for solution in X:
            d, s, m = map(int, solution)
            print(d,s,m)
            model = train_model(d=int(d),s=int(s),m=int(m), num_workers=8, train_file="T91atacado.h5", eval_file="Set5_limpio.h5", outputs_dir="./models")
            
            psnr_limpio = evaluate_model(model, "Set5_limpio.h5")
            psnr_atacado = evaluate_model(model, "Set5_atacado.h5")

            psnr_dataset1.append(psnr_limpio)
            psnr_dataset2.append(psnr_atacado)
            logs.append(f"{d}-{s}-{m} {psnr_limpio}, {psnr_atacado}\n")         

        out["F"] = np.column_stack([psnr_dataset1, psnr_dataset2])
        self.visualize_gen(out["F"], logs, self.current_gen)

        self.current_gen +=1 

    def visualize_gen(self, results, logs, gen):
        plot_folder = "plots"
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        psnr_dataset1 = results[:, 0]
        psnr_dataset2 = results[:, 1]
        plt.scatter(psnr_dataset1, psnr_dataset2, alpha=0.5)
        plt.xlabel("PSNR (Limpio)")
        plt.ylabel("PSNR (Atacado)")
        plt.title(f"Generacion {gen}: PSNR")
        plt.grid(True)
        
        plot_folder = "plots"
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        
        plt.savefig(f"plots/gen_{gen}_psnr.png")
        plt.close()

        with open(f"plots/gen_{gen}_psnr.txt", "w") as f:
            f.write("d-s-m, PSNR Limpio, PSNR Atacado\n")
            f.writelines(logs)

    @staticmethod
    def evaluate_model(d, s, m, dataset):
        """
        Funcion dummy
        """
        
        base_psnr = 30 + 0.1 * d + 0.05 * s + 0.2 * m 

        noise = np.random.normal(loc=0, scale=1.0)

        if dataset == "dataset1":
            return base_psnr + noise
        else:
            return base_psnr + 2 + noise

algorithm = NSGA2(
    pop_size=5,
    sampling=LHS(),
    crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
    mutation=PolynomialMutation(eta=20, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True,
    save_history=True
)


termination = get_termination("n_gen", 20)

problem = NASProblem()

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,  
    verbose=True, 
)

for i, f in enumerate(res.F):
    print(f"Solucion {i+1}: Objetivos = {f}, Variables = {res.X[i]}")
