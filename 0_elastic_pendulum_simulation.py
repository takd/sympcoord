# Implementation of the elastic pendulum simulation in various formulations and coordinate systems.

from inspect import signature
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

class ElasticPendulumSimulation(ABC):
    ## Lagrange x-y
    def _f_lagrange_xy_damped(self, x_circ_n_, x_n_, y_circ_n_, y_n_, t_):
        F_dx_n_ = - self.k_ * (x_circ_n_**2 + y_circ_n_**2)**(self.mu/2) * x_circ_n_
        F_dy_n_ = - self.k_ * (x_circ_n_**2 + y_circ_n_**2)**(self.mu/2) * y_circ_n_

        x_circdot_n_ = - self.s_/self.m_ * (1 - self.l_/(np.sqrt(x_n_**2 + y_n_**2))) * x_n_ + F_dx_n_ / self.m_
        x_dot_n_ = x_circ_n_
        y_circdot_n_ = - self.s_/self.m_ * (1 - self.l_/(np.sqrt(x_n_**2 + y_n_**2))) * y_n_ - self.g_ + F_dy_n_ / self.m_
        y_dot_n_ = y_circ_n_

        return np.array([x_circdot_n_, x_dot_n_, y_circdot_n_, y_dot_n_])

    def _f_lagrange_xy_undamped(self, x_circ_n_, x_n_, y_circ_n_, y_n_, t_):
        x_circdot_n_ = - self.s_/self.m_ * (1 - self.l_/(np.sqrt(x_n_**2 + y_n_**2))) * x_n_
        x_dot_n_ = x_circ_n_
        y_circdot_n_ = - self.s_/self.m_ * (1 - self.l_/(np.sqrt(x_n_**2 + y_n_**2))) * y_n_ - self.g_
        y_dot_n_ = y_circ_n_

        return np.array([x_circdot_n_, x_dot_n_, y_circdot_n_, y_dot_n_])

    ## Lagrange r-phi
    def _f_lagrange_rphi_damped(self, r_circ_n_, r_n_, phi_circ_n_, phi_n_, t_):
        F_dr_n_ = - self.k_ * (r_circ_n_**2 + r_n_**2 * phi_circ_n_**2)**(self.mu/2) * r_circ_n_
        F_dphi_n_ = - self.k_ * (r_circ_n_**2 + r_n_**2 * phi_circ_n_**2)**(self.mu/2) * r_n_**2 * phi_circ_n_

        r_circdot_n_ = r_n_ * phi_circ_n_**2 - self.s_ / self.m_ * (r_n_- self.l_) + self.g_ * np.cos(phi_n_) + F_dr_n_ / self.m_
        r_dot_n_ = r_circ_n_
        phi_circdot_n_ = -2*r_circ_n_ / r_n_ * phi_circ_n_ - self.g_ / r_n_ * np.sin(phi_n_) + F_dphi_n_ / (r_n_**2 * self.m_)
        phi_dot_n_ = phi_circ_n_

        return np.array([r_circdot_n_, r_dot_n_, phi_circdot_n_, phi_dot_n_])

    def _f_lagrange_rphi_undamped(self, r_circ_n_, r_n_, phi_circ_n_, phi_n_, t_):
        r_circdot_n_ = r_n_ * phi_circ_n_**2 - self.s_ / self.m_ * (r_n_- self.l_) + self.g_ * np.cos(phi_n_)
        r_dot_n_ = r_circ_n_
        phi_circdot_n_ = -2*r_circ_n_ / r_n_ * phi_circ_n_ - self.g_ / r_n_ * np.sin(phi_n_)
        phi_dot_n_ = phi_circ_n_

        return np.array([r_circdot_n_, r_dot_n_, phi_circdot_n_, phi_dot_n_])

    ## Hamilton x-y
    def _dH_dp_x_xy_(self, x_, y_, p_x_, p_y_):
        return p_x_ / self.m_

    def _dH_dp_y_xy_(self, x_, y_, p_x_, p_y_):
        return p_y_ / self.m_

    def _dH_dx_xy_(self, x_, y_, p_x_, p_y_):
        return self.s_ * ( 1 - self.l_/np.sqrt(x_**2 + y_**2)) * x_

    def _dH_dy_xy_(self, x_, y_, p_x_, p_y_):
        return self.s_ * ( 1 - self.l_/np.sqrt(x_**2 + y_**2)) * y_ + self.m_*self.g_

    def _f_hamilton_xy_damped(self, x_n_, y_n_, p_x_n_, p_y_n_, t_):
        F_dx_n_ = - self.k_ * ((p_x_n_ / self.m_)**2 + (p_y_n_ / self.m_)**2 )**(self.mu/2) * p_x_n_/self.m_
        F_dy_n_ = - self.k_ * ((p_x_n_ / self.m_)**2 + (p_y_n_ / self.m_)**2 )**(self.mu/2) * p_y_n_/self.m_

        x_dot_n_ = self._dH_dp_x_xy_(x_n_, y_n_, p_x_n_, p_y_n_)
        y_dot_n_ = self._dH_dp_y_xy_(x_n_, y_n_, p_x_n_, p_y_n_)
        p_x_dot_n_ = - self._dH_dx_xy_(x_n_, y_n_, p_x_n_, p_y_n_) + F_dx_n_
        p_y_dot_n_ = - self._dH_dy_xy_(x_n_, y_n_, p_x_n_, p_y_n_) + F_dy_n_

        return np.array([x_dot_n_, y_dot_n_, p_x_dot_n_, p_y_dot_n_])

    def _f_hamilton_xy_undamped(self, x_n_, y_n_, p_x_n_, p_y_n_, t_):
        x_dot_n_ = self._dH_dp_x_xy_(x_n_, y_n_, p_x_n_, p_y_n_)
        y_dot_n_ = self._dH_dp_y_xy_(x_n_, y_n_, p_x_n_, p_y_n_)
        p_x_dot_n_ = - self._dH_dx_xy_(x_n_, y_n_, p_x_n_, p_y_n_)
        p_y_dot_n_ = - self._dH_dy_xy_(x_n_, y_n_, p_x_n_, p_y_n_)

        return np.array([x_dot_n_, y_dot_n_, p_x_dot_n_, p_y_dot_n_])

    ## Hamilton r-phi
    def _dH_dr_rphi_(self, r_, phi_, p_r_, p_phi_):
        return -1 / self.m_ * p_phi_**2 / r_**3 + self.s_ * (r_ - self.l_) - self.m_ * self.g_ * np.cos(phi_)

    def _dH_dphi_rphi_(self, r_, phi_, p_r_, p_phi_):
        return self.m_ * self.g_ * r_ * np.sin(phi_)

    def _dH_dp_r_rphi_(self, r_, phi_, p_r_, p_phi_):
        return 1 / self.m_ * p_r_

    def _dH_dp_phi_rphi_(self, r_, phi_, p_r_, p_phi_):
        return 1 / self.m_ * p_phi_ / r_**2

    def _f_hamilton_rphi_damped(self, r_n_, phi_n_, p_r_n_, p_phi_n_, t_):
        F_dr_n_ = - self.k_ * ((p_r_n_ / self.m_)**2 + (p_phi_n_ / self.m_ / r_n_)**2)**(self.mu/2) * p_r_n_ / self.m_
        F_dphi_n_ = - self.k_ * ((p_r_n_ / self.m_)**2 + (p_phi_n_ / self.m_ / r_n_)**2)**(self.mu/2) * p_phi_n_ / self.m_

        r_dot_n_ = p_r_n_ / self.m_
        phi_dot_n_ = 1 / r_n_**2 * p_phi_n_ / self.m_
        p_r_dot_n_ = 1 / self.m_ * p_phi_n_**2 / r_n_**3 - self.s_ * (r_n_ - self.l_) + self.m_ * self.g_ * np.cos(phi_n_) + F_dr_n_
        p_phi_dot_n_ = - self.m_ * self.g_ * r_n_ * np.sin(phi_n_) + F_dphi_n_

        return np.array([r_dot_n_, phi_dot_n_, p_r_dot_n_, p_phi_dot_n_])

    def _f_hamilton_rphi_undamped(self, r_n_, phi_n_, p_r_n_, p_phi_n_, t_):
        r_dot_n_ = p_r_n_ / self.m_
        phi_dot_n_ = 1 / r_n_**2 * p_phi_n_ / self.m_
        p_r_dot_n_ = 1 / self.m_ * p_phi_n_**2 / r_n_**3 - self.s_ * (r_n_ - self.l_) + self.m_ * self.g_ * np.cos(phi_n_)
        p_phi_dot_n_ = - self.m_ * self.g_ * r_n_ * np.sin(phi_n_)

        return np.array([r_dot_n_, phi_dot_n_, p_r_dot_n_, p_phi_dot_n_])

    def __init__(self, *, model, model_variable_names, model_parameters, simulation_parameters):
        if callable(model):
            self.f = model
            self.model_name = '[callable]'
        else:
            self.model_name = model

            if model == 'lagrange_xy_damped':
                self.f = self._f_lagrange_xy_damped
            elif model == 'lagrange_xy_undamped':
                self.f = self._f_lagrange_xy_undamped
            elif model == 'lagrange_rphi_damped':
                self.f = self._f_lagrange_rphi_damped
            elif model == 'lagrange_rphi_undamped':
                self.f = self._f_lagrange_rphi_undamped
            elif model == 'hamilton_xy_damped':
                self.f = self._f_hamilton_xy_damped
            elif model == 'hamilton_xy_undamped':
                self.f = self._f_hamilton_xy_undamped
            elif model == 'hamilton_rphi_damped':
                self.f = self._f_hamilton_rphi_damped
            elif model == 'hamilton_rphi_undamped':
                self.f = self._f_hamilton_rphi_undamped
            else:
                raise ValueError('Unknown uncallable model {0}'.format(model))

        self.var_dim = len(signature(self.f).parameters) - 1  ## Number of variables is len(args(f)) - 1 (self not counted)

        self.model_variable_names = model_variable_names
        self.model_parameters = model_parameters
        self.simulation_parameters = simulation_parameters

        self.initialize_model_parameters()
        self.initialize_simulation_parameters()

    def initialize_model_parameters(self):
        self.l_ = self.model_parameters['l_']  # Rescaled length
        self.m_ = self.model_parameters['m_']  # Rescaled mass
        self.g_ = self.model_parameters['g_']  # Rescaled gravity
        self.k_ = self.model_parameters['k_']  # Rescaled damping coefficient
        self.s_ = self.model_parameters['s_']  # Rescaled spring constant

        if 'mu' in self.model_parameters:
            self.mu = self.model_parameters['mu']

    @abstractmethod
    def initialize_simulation_parameters(self):
        pass

    @abstractmethod
    def run_simulation(self, *, ic=None, show_progress=True):
        pass


class ElasticPendulumSimulationRK(ElasticPendulumSimulation):
    def initialize_simulation_parameters(self):
        self.J = self.simulation_parameters['J']  # Number of timesteps
        self.Dt_ = self.simulation_parameters['Dt_']  # Timestep (rescaled)
        self.s_rk = self.simulation_parameters['s_rk']  # Stages of Runge-Kutta method
        self.name = self.simulation_parameters['name']  # Name of simulation

    def _get_rk_params(self, *, stages):
        if stages==1:
            aij = np.array([np.nan])
            bi = np.array([1])
            ci = np.array([0])
        elif stages==4:
            aij = np.array([[np.nan, np.nan, np.nan, np.nan],
                            [0.5,    np.nan, np.nan, np.nan],
                            [0,      0.5,    np.nan, np.nan],
                            [0,      0,      1,      np.nan]])
            bi = np.array([1/6, 1/3, 1/3, 1/6])
            ci = np.array([0,   1/2, 1/2, 1  ])
        else:
            raise NotImplementedError("RK with {0} stages is not implemented.".format(stages))
        return aij, bi, ci

    def run_simulation(self, *, ic, show_progress=True):
        tn_ = np.linspace( 0, self.J*self.Dt_, self.J+1 )  ## defines the discretely many instants; 1-dimensional array
        yn_ = np.zeros([self.var_dim, self.J+1])  ## defines the four variables; 2-dimensional array

        # Initial conditions as a dict
        for i, mv in enumerate(self.model_variable_names):
            yn_[i, 0] = ic[mv]

        aij, bi, ci = self._get_rk_params(stages=self.s_rk)

        h_ = self.Dt_

        for j in tqdm(range(0, self.J), disable=(not show_progress)):  ## the main loop
            ## Runge-Kutta with s stages:
            ## y[n+1] = y[n] + h*sum_{i=1}^{s}(b_i*k_i)
            ## k_l = f(t_n + c_l*h, y_n + h*sum_{i=0}^{l-1}(a_{li}*k_i)

            tj_ = j*h_  ## Actual timestep

            k_l = np.zeros((self.s_rk, self.var_dim))  ## Runge-Kutta evaluations

            for l in range(0, self.s_rk): # 0-tól s-1-ig
                if l==0:
                    k_l[l, :] = self.f(*(yn_[:, j]), tj_)
                elif l>0:
                    k_l[l, :] = self.f(*(yn_[:, j] + h_*np.sum(aij[l, :l][:, np.newaxis]*k_l[:l, :], axis=0)), tj_ + ci[l]*h_)
                else:
                    raise ValueError("l cannot be negative")

            yn_[:, j+1] = yn_[:, j] + h_ * np.sum(bi[:, np.newaxis]*k_l, axis=0)

        return tn_, yn_


class ElasticPendulumSimulationSymplecticEuler(ElasticPendulumSimulation):
    def initialize_simulation_parameters(self):
        self.J = self.simulation_parameters['J']  # Number of timesteps
        self.Dt_ = self.simulation_parameters['Dt_']  # Timestep (rescaled)
        self.name = self.simulation_parameters['name']  # Name of simulation

        self.symplectic_euler_variant = self.simulation_parameters['symplectic_euler_variant']  # 'momentum_first' or 'position_first'

    def run_simulation(self, *, ic, show_progress=True):
        if 'hamilton' not in self.model_name:
            raise ValueError('Symplectic Euler method can only be used for Hamiltonian model'.format(self.model_name))

        if 'undamped' not in self.model_name:
            raise ValueError('Model {0} is not valid for symplectic Euler method'.format(self.model_name))

        if ('xy' in self.model_name) and ('rphi' in self.model_name):
            raise ValueError("Model cannot be in both 'xy' and 'rphi' coordinate system.")

        tn_ = np.linspace( 0, self.J*self.Dt_, self.J+1 )  ## defines the discretely many instants; 1-dimensional array
        yn_ = np.zeros([self.var_dim, self.J+1])  ## defines the four variables; 2-dimensional array

        # Initial conditions as a dict
        for i, mv in enumerate(self.model_variable_names):
            yn_[i, 0] = ic[mv]

        h_ = self.Dt_

        if 'xy' in self.model_name:
            if self.symplectic_euler_variant == 'momentum_first':
                def make_step(x_n_, y_n_, p_x_n_, p_y_n_):
                    # Momentum step
                    p_x_np1_ = p_x_n_ - h_ * self._dH_dx_xy_(x_n_, y_n_, None, None)
                    p_y_np1_ = p_y_n_ - h_ * self._dH_dy_xy_(x_n_, y_n_, None, None)

                    # Position step
                    x_np1_ = x_n_ + h_ * self._dH_dp_x_xy_(None, None, p_x_np1_, p_y_np1_)
                    y_np1_ = y_n_ + h_ * self._dH_dp_y_xy_(None, None, p_x_np1_, p_y_np1_)

                    return np.array([x_np1_, y_np1_, p_x_np1_, p_y_np1_])

            elif self.symplectic_euler_variant == 'position_first':
                def make_step(x_n_, y_n_, p_x_n_, p_y_n_):
                    # Position step
                    x_np1_ = x_n_ + h_ * self._dH_dp_x_xy_(None, None, p_x_n_, p_y_n_)
                    y_np1_ = y_n_ + h_ * self._dH_dp_y_xy_(None, None, p_x_n_, p_y_n_)

                    # Momentum step
                    p_x_np1_ = p_x_n_ - h_ * self._dH_dx_xy_(x_np1_, y_np1_, None, None)
                    p_y_np1_ = p_y_n_ - h_ * self._dH_dy_xy_(x_np1_, y_np1_, None, None)

                    return np.array([x_np1_, y_np1_, p_x_np1_, p_y_np1_])

        elif 'rphi' in self.model_name:
            if self.symplectic_euler_variant == 'momentum_first':
                def make_step(r_n_, phi_n_, p_r_n_, p_phi_n_):
                    # Momentum step
                    p_phi_np1_ = p_phi_n_ - h_ * self._dH_dphi_rphi_(r_n_, phi_n_, None, None)
                    p_r_np1_ = p_r_n_ - h_ * self._dH_dr_rphi_(r_n_, phi_n_, None, p_phi_np1_)

                    # Position step
                    r_np1_ = r_n_ + h_ * self._dH_dp_r_rphi_(None, None, p_r_np1_, None)
                    phi_np1_ = phi_n_ + h_ * self._dH_dp_phi_rphi_(r_n_, None, None, p_phi_np1_)

                    return np.array([r_np1_, phi_np1_, p_r_np1_, p_phi_np1_])

            elif self.symplectic_euler_variant == 'position_first':
                def make_step(r_n_, phi_n_, p_r_n_, p_phi_n_):
                    # Position step
                    # Ez csak ilyen sorrendben explicit.
                    r_np1_ = r_n_ + h_ * self._dH_dp_r_rphi_(None, None, p_r_n_, None)
                    phi_np1_ = phi_n_ + h_ * self._dH_dp_phi_rphi_(r_np1_, None, None, p_phi_n_)

                    # Momentum step
                    p_r_np1_ = p_r_n_ - h_ * self._dH_dr_rphi_(r_np1_, phi_np1_, None, p_phi_n_)
                    p_phi_np1_ = p_phi_n_ - h_ * self._dH_dphi_rphi_(r_np1_, phi_np1_, None, None)

                    return np.array([r_np1_, phi_np1_, p_r_np1_, p_phi_np1_])

        else:
            raise ValueError("Unknown symplectic Euler variant '{0}', valid variants are 'momentum_first' and 'position_first'".format(self.symplectic_euler_variant))

        for j in tqdm(range(0, self.J), disable=(not show_progress)):  ## the main loop
            yn_[:, j+1] = make_step(*yn_[:, j])

        return tn_, yn_


class ElasticPendulumSimulationStormerVerlet(ElasticPendulumSimulation):
    def initialize_simulation_parameters(self):
        self.J = self.simulation_parameters['J']  # Number of timesteps
        self.Dt_ = self.simulation_parameters['Dt_']  # Timestep (rescaled)
        self.name = self.simulation_parameters['name']  # Name of simulation

        self.stormer_verlet_variant = self.simulation_parameters['stormer_verlet_variant']  # 'momentum_first' or 'position_first'

    def run_simulation(self, *, ic, show_progress=True):
        if 'hamilton' not in self.model_name:
            raise ValueError('Störmer-Verlet method can only be used for Hamiltonian model'.format(self.model_name))

        if 'undamped' not in self.model_name:
            raise ValueError('Model {0} is not valid for Störmer-Verlet method'.format(self.model_name))

        if ('xy' in self.model_name) and ('rphi' in self.model_name):
            raise ValueError("Model cannot be in both 'xy' and 'rphi' coordinate system.")

        tn_ = np.linspace( 0, self.J*self.Dt_, self.J+1 )  ## defines the discretely many instants; 1-dimensional array
        yn_ = np.zeros([self.var_dim, self.J+1])  ## defines the four variables; 2-dimensional array
        # Initial conditions as a dict
        for i, mv in enumerate(self.model_variable_names):
            yn_[i, 0] = ic[mv]

        h_ = self.Dt_
        hh_ = h_ / 2  # Half step

        if 'xy' in self.model_name:
            if self.stormer_verlet_variant == 'momentum_first':
                def make_step(x_n_, y_n_, p_x_n_, p_y_n_):
                    # First momentum half-step
                    p_x_nph_ = p_x_n_ - hh_*self._dH_dx_xy_(x_n_, y_n_, None, None)
                    p_y_nph_ = p_y_n_ - hh_*self._dH_dy_xy_(x_n_, y_n_, None, None)

                    # Position step
                    x_np1_ = x_n_ + hh_*(2*self._dH_dp_x_xy_(None, None, p_x_nph_, p_y_nph_)) 
                    y_np1_ = y_n_ + hh_*(2*self._dH_dp_y_xy_(None, None, p_x_nph_, p_y_nph_))

                    # Second momentum half-step
                    p_x_np1_ = p_x_nph_ - hh_*self._dH_dx_xy_(x_np1_, y_np1_, None, None)
                    p_y_np1_ = p_y_nph_ - hh_*self._dH_dy_xy_(x_np1_, y_np1_, None, None)

                    return np.array([x_np1_, y_np1_, p_x_np1_, p_y_np1_])

            elif self.stormer_verlet_variant == 'position_first':
                def make_step(x_n_, y_n_, p_x_n_, p_y_n_):
                    # First position half-step
                    x_nph_ = x_n_ + hh_*self._dH_dp_x_xy_(None, None, p_x_n_, p_y_n_)
                    y_nph_ = y_n_ + hh_*self._dH_dp_y_xy_(None, None, p_x_n_, p_y_n_)

                    # Momentum step
                    p_x_np1_ = p_x_n_ - hh_*(2*self._dH_dx_xy_(x_nph_, y_nph_, None, None))
                    p_y_np1_ = p_y_n_ - hh_*(2*self._dH_dy_xy_(x_nph_, y_nph_, None, None))

                    # Second position half-step
                    x_np1_ = x_nph_ + hh_*self._dH_dp_x_xy_(None, None, p_x_np1_, p_y_np1_)
                    y_np1_ = y_nph_ + hh_*self._dH_dp_y_xy_(None, None, p_x_np1_, p_y_np1_)

                    return np.array([x_np1_, y_np1_, p_x_np1_, p_y_np1_])

        elif 'rphi' in self.model_name:
            if self.stormer_verlet_variant == 'momentum_first':
                def make_step(r_n_, phi_n_, p_r_n_, p_phi_n_):
                    # First momentum half-step
                    p_phi_nph_ = p_phi_n_ - hh_*self._dH_dphi_rphi_(r_n_, phi_n_, None, None)
                    p_r_nph_ = p_r_n_ - hh_*self._dH_dr_rphi_(r_n_, phi_n_, None, p_phi_nph_)

                    # Position step
                    r_np1_ = r_n_ + hh_*(2*self._dH_dp_r_rphi_(None, None, p_r_nph_, None))  # 2-es szorzó a dupla számítás helyett
                    phi_np1_ = phi_n_ + hh_*(self._dH_dp_phi_rphi_(r_n_, None, None, p_phi_nph_) +
                                             + self._dH_dp_phi_rphi_(r_np1_, None, None, p_phi_nph_))

                    # Second momentum half-step
                    p_phi_np1_ = p_phi_nph_ - hh_*self._dH_dphi_rphi_(r_np1_, phi_np1_, None, None)
                    p_r_np1_ = p_r_nph_ - hh_*self._dH_dr_rphi_(r_np1_, phi_np1_, None, p_phi_nph_)

                    return np.array([r_np1_, phi_np1_, p_r_np1_, p_phi_np1_])

            elif self.stormer_verlet_variant == 'position_first':
                def make_step(r_n_, phi_n_, p_r_n_, p_phi_n_):
                    # First position half-step
                    r_nph_ = r_n_ + hh_*self._dH_dp_r_rphi_(None, None, p_r_n_, None)
                    phi_nph_ = phi_n_ + hh_*self._dH_dp_phi_rphi_(r_nph_, None, None, p_phi_n_)

                    # Momentum step
                    p_phi_np1_ = p_phi_n_ - hh_*(2*self._dH_dphi_rphi_(r_nph_, phi_nph_, None, None))
                    p_r_np1_ = p_r_n_ - hh_*(self._dH_dr_rphi_(r_nph_, phi_nph_, None, p_phi_n_) +
                                             + self._dH_dr_rphi_(r_nph_, phi_nph_, None, p_phi_np1_))

                    # Second position half-step
                    r_np1_ = r_nph_ + hh_*self._dH_dp_r_rphi_(None, None, p_r_np1_, None)
                    phi_np1_ = phi_nph_ + hh_*self._dH_dp_phi_rphi_(r_nph_, None, None, p_phi_np1_)

                    return np.array([r_np1_, phi_np1_, p_r_np1_, p_phi_np1_])

        else:
            raise ValueError("Unknown Störmer-Verlet variant '{0}', valid variants are 'momentum_first' and 'position_first'".format(self.symplectic_euler_variant))

        for j in tqdm(range(0, self.J), disable=(not show_progress)):  ## the main loop
            yn_[:, j+1] = make_step(*yn_[:, j])

        return tn_, yn_
