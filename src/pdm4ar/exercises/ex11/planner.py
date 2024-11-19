from calendar import c
from dataclasses import dataclass, field
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises.ex11.discretization import *
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius - to be updated during the iterations
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant

    # Time limits
    max_time: float = 80.0  # max time for the algorithm
    min_time: float = 0.0  # min time for the algorithm


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()
        self.eta = self.params.tr_radius

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: DynObstacleState, my_tol: float
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state
        self.my_tol = my_tol

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # TODO: Implement SCvx algorithm or comparable
        # Iter for the max number of iterations
        for iteration in range(self.params.max_iterations):
            print(f"Iteration: {iteration}")

            # 1. Convexify the dynamic around the current trajectory and assign the values to the problem parameters
            self._convexification()

            """if self.problem.is_dcp():
                print("Il problema è DCP.")
            else:
                print("Il problema NON è DCP.")"""

            """if self.problem.is_dpp():
                print("Il problema è DPP.")
            else:
                print("Il problema NON è DPP.")"""

            # 2. Solve the problem

            check_for_nan_and_inf(self.problem_parameters["A_bar"].value, "A_bar")
            check_for_nan_and_inf(self.problem_parameters["B_plus_bar"].value, "B_plus_bar")
            check_for_nan_and_inf(self.problem_parameters["B_minus_bar"].value, "B_minus_bar")
            check_for_nan_and_inf(self.problem_parameters["F_bar"].value, "F_bar")
            check_for_nan_and_inf(self.problem_parameters["r_bar"].value, "r_bar")

            # Print A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar
            print(f"A_bar: {self.problem_parameters['A_bar'].value}")
            print(f"B_plus_bar: {self.problem_parameters['B_plus_bar'].value}")
            print(f"B_minus_bar: {self.problem_parameters['B_minus_bar'].value}")
            print(f"F_bar: {self.problem_parameters['F_bar'].value}")
            print(f"r_bar: {self.problem_parameters['r_bar'].value}")

            try:
                self.error = self.problem.solve(
                    verbose=self.params.verbose_solver, solver=self.params.solver, max_iters=1000
                )
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            # 3. Check convergence
            if self._check_convergence():
                print("Convergenza raggiunta.")
                break

            # 4. Update trust region
            self._update_trust_region()

        # self._unnormalize_variables()
        mycmds, mystates = self._sequence_from_array()

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        X = np.ones((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.zeros((self.spaceship.n_p))

        X[-1, :] = self.sp.m_v
        X[6, :] = 0.0

        # Linear interpolation between initial and goal state
        for k in range(self.params.K):
            X[:6, k] = self.init_state.as_ndarray()[:6] * (1 - k / K) + self.goal_state.as_ndarray() * (k / K)

        # Define initial guess for p
        p = np.ones((1)) * (self.params.max_time + self.params.min_time) / 2

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
            "nu": cvx.Variable((self.spaceship.n_x, self.params.K - 1)),
            # "nu_s": ..., # to define
            "nu_ic": cvx.Variable(self.spaceship.n_x),
            "nu_tc": cvx.Variable(self.spaceship.n_x - 2),
        }

        self.n_x = self.spaceship.n_x
        self.n_u = self.spaceship.n_u
        self.n_p = self.spaceship.n_p

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x),
            "init_input": cvx.Parameter(self.spaceship.n_u),
            "goal": cvx.Parameter(6),
            "tollerance": cvx.Parameter(),
            "A_bar": cvx.Parameter((self.n_x * self.n_x, self.params.K - 1)),
            "B_plus_bar": cvx.Parameter((self.n_x * self.n_u, self.params.K - 1)),
            "B_minus_bar": cvx.Parameter((self.n_x * self.n_u, self.params.K - 1)),
            "F_bar": cvx.Parameter((self.spaceship.n_x * self.n_p, self.params.K - 1)),
            "r_bar": cvx.Parameter((self.spaceship.n_x, self.params.K - 1)),
            "eta": cvx.Parameter(),
            "X_bar": cvx.Parameter((self.n_x, self.params.K)),
            "U_bar": cvx.Parameter((self.n_u, self.params.K)),
            "p_bar": cvx.Parameter(self.n_p),
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        coeff_toll_pos: float = 1.0
        coeff_toll_rot: float = 1.0
        coeff_toll_vel: float = 1.0
        coeff_toll_ang_vel: float = 1.0

        constraints = []
        constraints.append(self.variables["p"] <= self.params.max_time)  # time constraint
        constraints.append(self.variables["p"] >= self.params.min_time)  # time constraint
        constraints.append(self.variables["U"][:, 0] == np.zeros(self.n_u))  # initial control condition
        constraints.append(self.variables["U"][:, -1] == np.zeros(self.n_u))  # terminal control condition

        # BOUDARY CONDITIONS
        # Terminal condition
        for i in range(6):
            constraints.append(
                self.variables["X"][i, -1] - self.problem_parameters["goal"][i] + self.variables["nu_tc"][i]
                <= self.problem_parameters["tollerance"] * 5
            )
        # Initial condition
        for i in range(self.n_x):
            constraints.append(
                self.variables["X"][i, 0] - self.problem_parameters["init_state"][i] + self.variables["nu_ic"][i] == 0
            )

        # PROBLEM CONSTRAINTS
        for k in range(self.params.K):
            constraints.append(self.variables["X"][7, k] >= self.sg.m)  # mass condition
            constraints.append(self.variables["U"][0, k] >= self.sp.thrust_limits[0])  # thrust condition
            constraints.append(self.variables["U"][0, k] <= self.sp.thrust_limits[1])  # thrust condition
            constraints.append(self.variables["X"][6, k] >= self.sp.delta_limits[0])  # nozzle angle condition
            constraints.append(self.variables["X"][6, k] <= self.sp.delta_limits[1])  # nozzle angle condition
            constraints.append(self.variables["U"][1, k] >= self.sp.ddelta_limits[0])  # nozzle omega condition
            constraints.append(self.variables["U"][1, k] <= self.sp.ddelta_limits[1])  # nozzle omega condition

        # DYNAMICS CONSTRAINTS
        for k in range(self.params.K - 1):
            constraints.append(
                self.variables["X"][:, k + 1]
                == (
                    cvx.reshape(self.problem_parameters["A_bar"][:, k], (self.n_x, self.n_x))
                    @ self.variables["X"][:, k]
                    + cvx.reshape(self.problem_parameters["B_plus_bar"][:, k], (self.n_x, self.n_u))
                    @ self.variables["U"][:, k + 1]
                    + cvx.reshape(self.problem_parameters["B_minus_bar"][:, k], (self.n_x, self.n_u))
                    @ self.variables["U"][:, k]
                    + cvx.reshape(self.problem_parameters["F_bar"][:, k], (self.n_x, self.n_p)) @ self.variables["p"]
                    + self.problem_parameters["r_bar"][:, k]
                    + self.variables["nu"][:, k]
                )
            )

        # TRUST REGION CONSTRAINT
        constraints.append(
            cvx.sum_squares(self.variables["X"] - self.problem_parameters["X_bar"])
            + cvx.sum_squares(self.variables["U"] - self.problem_parameters["U_bar"])
            + cvx.sum_squares(self.variables["p"] - self.problem_parameters["p_bar"])
            <= self.problem_parameters["eta"]
        )  # trust region constraint

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Define terminal cost penalized
        phi_lambda = (
            self.params.weight_p @ self.variables["p"]
            + self.params.lambda_nu * cvx.norm1(self.variables["nu_ic"][:])
            + self.params.lambda_nu * cvx.norm1(self.variables["nu_tc"][:])
        )

        # Define gamma lambda
        gamma_lambda = []
        for k in range(self.params.K - 1):
            gamma_lambda.append(self.params.lambda_nu * cvx.norm1(self.variables["nu"][:, k]))
            # gamma_lambda.append(
            #    self.params.lambda_nu
            #    * cvx.norm1(self.variables["nu"][:, k] + self.params.lambda_nu * cvx.norm1(self.variables["nu_s"][:, k]))
            # )
        # gamma_lambda.append(self.params.lambda_nu * cvx.norm1(self.variables["nu_s"][:, self.params.K]))

        # Compute trapezoidal integration
        delta_t = 1.0 / self.params.K
        gamma = 0
        for k in range(self.params.K - 2):
            gamma += cvx.multiply(delta_t / 2, (gamma_lambda[k] + gamma_lambda[k + 1]))
        # Define objective
        objective = phi_lambda + gamma

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        # Populate Problem Parameters
        self.problem_parameters["init_state"].value = self.init_state.as_ndarray()
        self.problem_parameters["goal"].value = self.goal_state.as_ndarray()
        self.problem_parameters["tollerance"].value = self.my_tol
        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar
        self.problem_parameters["eta"].value = self.eta
        self.problem_parameters["X_bar"].value = self.X_bar
        self.problem_parameters["U_bar"].value = self.U_bar
        self.problem_parameters["p_bar"].value = self.p_bar

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        delta_x = [np.linalg.norm(self.variables["X"].value[:, i] - self.X_bar[:, i]) for i in range(self.params.K)]
        delta_p = np.linalg.norm(self.variables["p"].value - self.p_bar)

        return bool(delta_p + np.max(delta_x) <= self.params.stop_crit)

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        # Compute rho
        rho: float = self._compute_rho()  # to define

        # Update trust region considering the computed rho
        if rho <= self.params.rho_0:
            self.eta = max(self.params.min_tr_radius, self.eta / self.params.alpha)
        elif self.params.rho_0 <= rho <= self.params.rho_1:
            self.eta = max(self.params.min_tr_radius, self.eta / self.params.alpha)
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
        elif self.params.rho_1 <= rho <= self.params.rho_2:
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
        else:
            self.eta = min(self.params.max_tr_radius, self.params.beta * self.eta)
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

    def _compute_rho(self) -> float:
        """
        Compute rho for trust region update.
        """
        flow_map = [
            np.reshape(self.problem_parameters["A_bar"][:, k].value, (self.n_x, self.n_x))
            @ self.variables["X"][:, k].value
            + np.reshape(self.problem_parameters["B_plus_bar"][:, k].value, (self.n_x, self.n_u))
            @ self.variables["U"][:, k + 1].value
            + np.reshape(self.problem_parameters["B_minus_bar"][:, k].value, (self.n_x, self.n_u))
            @ self.variables["U"][:, k].value
            + np.reshape(self.problem_parameters["F_bar"][:, k].value, (self.n_x, self.n_p)) @ self.variables["p"].value
            + self.problem_parameters["r_bar"][:, k].value
            for k in range(self.params.K - 1)
        ]

        delta = [self.variables["X"][:, k + 1].value - flow_map[k] for k in range(self.params.K - 1)]

        # Define terminal cost penalized for p_bar
        phi_lambda_bar = (
            self.params.weight_p @ self.problem_parameters["p_bar"].value
            + self.params.lambda_nu
            * np.linalg.norm(self.variables["X"][:, 0].value - self.problem_parameters["init_state"].value, ord=1)
            + self.params.lambda_nu
            * np.linalg.norm(self.variables["X"][:6, -1].value - self.problem_parameters["goal"].value, ord=1)
        )

        # Define terminal cost penalized for p_opt
        phi_lambda_opt = (
            self.params.weight_p @ self.variables["p"].value
            + self.params.lambda_nu
            * np.linalg.norm(self.variables["X"][:, 0].value - self.problem_parameters["init_state"].value, ord=1)
            + self.params.lambda_nu
            * np.linalg.norm(self.variables["X"][:6, -1].value - self.problem_parameters["goal"].value, ord=1)
        )

        # Define gamma lambda
        gamma_lambda = []
        for k in range(self.params.K - 1):
            gamma_lambda.append(self.params.lambda_nu * np.linalg.norm(delta[k], ord=1))

        # Compute trapezoidal integration
        delta_t = self.variables["p"].value / self.params.K
        gamma = 0
        for k in range(self.params.K - 2):
            gamma += delta_t / 2 * (gamma_lambda[k] + gamma_lambda[k + 1])

        # Define cost function for bar
        cost_func_bar = phi_lambda_bar + gamma

        # Define cost function for opt
        cost_func_opt = phi_lambda_opt + gamma

        # Compute rho
        rho = (cost_func_bar[0] - cost_func_opt[0]) / (cost_func_bar[0] - self.error)

        return rho

    def _unnormalize_variables(self):
        """
        Normalize variables for SCvx.
        """
        # Define normalization parameters
        S_x = np.zeros((self.n_x, self.n_x))
        S_u = np.zeros((self.n_u, self.n_u))
        S_p = np.zeros((self.n_p, self.n_p))

        c_x = np.zeros((self.n_x, 1))
        c_u = np.zeros((self.n_u, 1))
        c_p = np.zeros((self.n_p, 1))

        # Define new variables
        self.new_X = np.zeros((self.n_x, self.params.K))
        self.new_U = np.zeros((self.n_u, self.params.K))
        self.new_p = np.zeros((self.n_p, 1))

        # Compute normalization parameters

        # Unnormalize variables
        for k in range(self.params.K):
            self.new_X[:, k] = S_x @ self.variables["X"].value[:, k] + c_x
            self.new_U[:, k] = S_u @ self.variables["U"].value[:, k] + c_u
            self.new_p = S_p @ self.variables["p"].value + c_p

    def _sequence_from_array(self) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        # Sequence from an array
        # 1. Create the timestaps
        ts = range(self.params.K)

        # 2. Create the sequences for commands
        F = self.variables["U"].value[0, :]
        ddelta = self.new_U[1, :]
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # 3. Create the sequences for states
        npstates = self.variables["X"].value.T
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        return mycmds, mystates


def check_for_nan_and_inf(parameter, name):
    if np.any(np.isnan(parameter)):
        print(f"Attenzione: {name} contiene valori NaN")
    if np.any(np.isinf(parameter)):
        print(f"Attenzione: {name} contiene valori infiniti")
    else:
        print(f"{name} è valido")
