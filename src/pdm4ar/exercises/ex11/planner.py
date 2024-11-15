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

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

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

        # TODO: Implement SCvx algorithm or comparable
        # Iter for the max number of iterations
        for iteration in range(self.params.max_iterations):
            print(f"Iteration: {iteration}")
            # 1. Convexify the dynamic around the current trajectory and assign the values to the problem parameters
            self._convexification()
            # 2. Solve the problem
            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
            # 3. Check convergence
            if self._check_convergence():
                print("Convergenza raggiunta.")
                break
            # 4. Update trust region
            self._update_trust_region()

        # Sequence from an array
        # 1. Create the timestaps
        ts = range(self.params.K)
        # 2. Create the sequences for commands
        F = self.variables["U"].value[0, :]
        ddelta = self.variables["U"].value[1, :]
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)
        # 3. Create the sequences for states
        npstates = self.variables["X"].value.T
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.zeros((self.spaceship.n_p))

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
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x),
            "init_input": cvx.Parameter(self.spaceship.n_u),
            "goal": cvx.Parameter((6, 1)),
            "tollerance": cvx.Parameter(1),
            "A_bar": cvx.Parameter((self.spaceship.n_x, self.spaceship.n_x)),
            "B_plus_bar": cvx.Parameter((self.spaceship.n_x, self.spaceship.n_u)),
            "B_minus_bar": cvx.Parameter((self.spaceship.n_x, self.spaceship.n_u)),
            "F_bar": cvx.Parameter((self.spaceship.n_x, 1)),
            "r_bar": cvx.Parameter((self.spaceship.n_x, 1)),
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        coeff_toll_pos: float = 1.0
        coeff_toll_rot: float = 1.0
        coeff_toll_vel: float = 1.0

        constraints = [
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],  # initial boundary condition
            self.variables["U"][:, 0] == self.problem_parameters["init_input"],
            cvx.norm2(self.variables["X"][:2, -1] - self.problem_parameters["goal"][:2])
            <= coeff_toll_pos * self.problem_parameters["tollerance"],  # final position boundary condition
            cvx.norm2(self.variables["X"][2, -1] - self.problem_parameters["goal"][2])
            <= coeff_toll_rot * self.problem_parameters["tollerance"],  # final orientation boundary condition
            cvx.norm2(self.variables["X"][3:5, -1] - self.problem_parameters["goal"][3:5])
            <= coeff_toll_vel * self.problem_parameters["tollerance"],  # final velocity boundary condition
            cvx.max(self.variables["X"][-1, :] - [self.sp.m_v for _ in range(self.params.K)])
            >= 0,  # mass boundary condition
            cvx.min(self.variables["U"][0, :] - [self.sp.thrust_limits[0] for _ in range(self.params.K)])
            >= 0,  # thrust boundary condition
            cvx.max(self.variables["U"][0, :] - [self.sp.thrust_limits[1] for _ in range(self.params.K)])
            <= 0,  # thrust boundary condition
            cvx.min(self.variables["X"][6, :] - [self.sp.delta_limits[0] for _ in range(self.params.K)])
            >= 0,  # nozzle angle boundary condition
            cvx.max(self.variables["X"][6, :] - [self.sp.delta_limits[1] for _ in range(self.params.K)])
            <= 0,  # nozzle angle boundary condition
            cvx.min(self.variables["U"][1, :] - [self.sp.ddelta_limits[0] for _ in range(self.params.K)])
            >= 0,  # nozzle angular velocity boundary condition
            cvx.max(self.variables["U"][1, :] - [self.sp.ddelta_limits[1] for _ in range(self.params.K)])
            <= 0,  # nozzle angular velocity boundary condition
            ] 
        
            constraints.append(self.variables["X"][:, k + 1]
                == self.problem_parameters["A_bar"] @ self.variables["X"][:, k]
                + self.problem_parameters["B_plus_bar"] @ self.variables["U"][:, k]
                + self.problem_parameters["B_minus_bar"] @ self.variables["U"][:, k]
                + self.problem_parameters["F_bar"] @ self.variables["p"]
                + self.problem_parameters["r_bar"]
                for k in range(self.params.K - 1)
              )  # dynamics constraints
        
            constraints.append(
                cvx.norm2(self.variables["X"] - self.X_bar)
                + cvx.norm2(self.variables["U"] - self.U_bar)
                + cvx.norm2(self.variables["p"] - self.p_bar)
                <= self.eta
            )  # trust region constraint
        
        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        objective = self.params.weight_p @ self.variables["p"]

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

        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        self.problem_parameters["init_input"].value = self.U_bar[:, 0]
        self.problem_parameters["goal"].value = self.goal_state
        self.problem_parameters["tollerance"].value = self.my_tol
        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        delta_x = [np.linalg.norm(self.variables["X"].value[:, i] - self.X_bar[:, i]) for i in range(self.params.K)]
        delta_p = np.linalg.norm(self.variables["P"].value - self.p_bar)

        return bool(delta_p + np.max(delta_x) <= self.params.stop_crit)

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        # Compute rho
        rho: float = self._compute_rho()  # to define

        # Update trust region considering the computed rho
        if rho < self.params.rho_0:
            self.eta = max(self.params.min_tr_radius, self.eta / self.params.alpha)
        elif self.params.rho_0 <= rho < self.params.rho_1:
            self.eta = max(self.params.min_tr_radius, self.eta / self.params.alpha)
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
        elif self.params.rho_1 <= rho < self.params.rho_2:
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
        # Compute rho
        rho: float = 0.0  # to define

        return rho
