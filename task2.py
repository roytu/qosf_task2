
""" Solution for task 2 """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.aqua.components.optimizers import SPSA
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo

VERBOSE = False

class Task2:

    POSSIBLE_OUTCOMES = ["00", "01", "10", "11"]

    def __init__(self):
        self.py = Parameter("θ_y")
        self.px = Parameter("θ_x")
        self.ansatz = self.make_ansatz()
        self.backend = Aer.get_backend("qasm_simulator")

        # Get noise model
        vigo_backend = FakeVigo()
        self.noise_model = NoiseModel.from_backend(vigo_backend)
        self.coupling_map = vigo_backend.configuration().coupling_map
        self.basis_gates = self.noise_model.basis_gates

        # Extra arrays for debugging
        self.cost_history = []

    def cost(self, params, M):
        """ Calculate the cost associated with running the simulator
        with parameters `params` for M iterations. `params` is in the
        order [θ_y, θ_x].

        Args:
            params: list of values to assign parameters as
            M: number of measurements per iteration

        Returns:
            cost (as float)
        """

        if VERBOSE:
            print("Trying params: \t{0}\t{1}".format(params[0], params[1]))

        # Assign parameters
        qc = self.ansatz.assign_parameters({
            self.py: params[0],
            self.px: params[1]
        })

        # Run the circuit M times and construct a probability distribution
        result = execute(
                qc,
                self.backend,
                shots=M,
                noise_model=self.noise_model,
                coupling_map=self.coupling_map,
                basis_gates=self.basis_gates
        ).result()
        counts = result.get_counts(qc)
        observed_pd = {}

        for k in Task2.POSSIBLE_OUTCOMES:
            if k in counts:
                observed_pd[k] = counts[k] / M
            else:
                observed_pd[k] = 0

        if VERBOSE:
            print(observed_pd)

        # Compare observed probability distribution
        # with target and calculate cost
        target_pd = {
                "00": 0,
                "01": 0.5,
                "10": 0.5,
                "11": 0
        }

        cost = np.sum([np.abs(observed_pd[k] - target_pd[k]) for k in Task2.POSSIBLE_OUTCOMES])

        if VERBOSE:
            print(f"Cost: {cost}")

        self.cost_history.append(cost)

        return cost

    def run(self, M):
        """ Performs the optimization.

        Args:
            M: number of measurements per iteration

        Returns:
            (best_params, cost_history)
                where

            best_params: calculated values for ["θ_y", "θ_x"] (as list)
            cost_history: costs for each iteration (as list)
        """

        # Clear costs log
        self.cost_history = []

        # Use SPSA as our classical optimizer
        optimizer = SPSA()

        # Define cost function
        def cost_(params):
            return self.cost(params, M)

        # Randomize initial point
        initial_point = [np.random.rand() * np.pi for _ in range(2)]

        # Perform optimization
        best_params, _, _ = optimizer.optimize(
                num_vars=2,
                objective_function=cost_,
                variable_bounds=[(0, 2 * np.pi)] * 2,
                initial_point=initial_point)

        return best_params, self.cost_history

    def make_ansatz(self):
        """ Creates the following QC:

                ┌─────────┐      ░ ┌─┐
           |0>: ┤ RY(θ_y) ├──■───░─┤M├───
                ├─────────┤┌─┴─┐ ░ └╥┘┌─┐
           |0>: ┤ RX(θ_x) ├┤ X ├─░──╫─┤M├
                └─────────┘└───┘ ░  ║ └╥┘
        meas: 2/════════════════════╩══╩═
                            0  1

        """

        qc = QuantumCircuit(2)
        qc.ry(self.py, 0)
        qc.rx(self.px, 1)
        qc.cnot(0, 1)
        qc.measure_all()

        return qc

    def bonus_ansatz(self):
        """ Creates the following QC:
                                ┌───┐           ░ ┌─┐
           |0>: ────────────────┤ X ├───────────░─┤M├──────
                ┌─────────┐┌───┐└─┬─┘┌───┐      ░ └╥┘┌─┐
           |0>: ┤ RY(θ_y) ├┤ H ├──■──┤ H ├──■───░──╫─┤M├───
                ├─────────┤└───┘     └───┘┌─┴─┐ ░  ║ └╥┘┌─┐
           |0>: ┤ RX(θ_x) ├───────────────┤ X ├─░──╫──╫─┤M├
                └─────────┘               └───┘ ░  ║  ║ └╥┘
        meas: 3/═══════════════════════════════════╩══╩══╩═
        """

        qc = QuantumCircuit(3)
        qc.ry(self.py, 1)
        qc.h(1)
        qc.cx(1, 0)
        qc.h(1)
        qc.rx(self.px, 2)
        qc.cnot(1, 2)
        qc.measure_all()

        return qc


    def show(self):
        """ Show circuit for debugging purposes
        """
        self.ansatz.draw("mpl")
        plt.show()

if __name__ == "__main__":

    task2 = Task2()

    for M in [1, 10, 100, 1000]:
        print(f"Running test: M={M}")
        best_params, cost_history = task2.run(M)

        theta_y, theta_x = best_params
        theta_y_div_pi = theta_y / np.pi
        theta_x_div_pi = theta_x / np.pi

        print("=" * 20)
        print("Params:")
        print(f"    theta_y: {theta_y} ( {theta_y_div_pi} * pi )")
        print(f"    theta_x: {theta_x} ( {theta_x_div_pi} * pi )")
        print("")

        np.save(f"results/costs_{M}.npy", cost_history)
