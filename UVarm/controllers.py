import UVarm.geometry.plucker as plucker
from UVarm.geometry.symbols import construct_syms as sys_syms
from UVarm.urdfparser import URDFparser
import casadi as cs
import numpy as np
import copy


class RobotControllers(n_joints):
    def __init__(self):
        self.ssyms = sys_syms(n_joints)
        print(f"number of joints = {self.ssyms.n_joints}")

        self.qref = cs.SX.sym('qref', self.ssyms.n_joints)  # Desired joint positions
        self.Kp = cs.SX.sym('Kp', self.ssyms.n_joints)      # Proportional gain
        self.Kd = cs.SX.sym('Kd', self.ssyms.n_joints)      # Derivative gain
        self.u_max = cs.SX.sym('u_max', self.ssyms.n_joints)
        self.u_min = cs.SX.sym('u_min', self.ssyms.n_joints)


    def pid(self):
        # Error between reference and actual position
        pd_err = self.qref - self.ssyms.q

        prev_err = cs.SX.sym('prev_err', self.ssyms.n_joints)
        derr_dt = (prev_err - pd_err)/self.ssyms.dt

        # Compute the PD control output
        pd_controller = cs.diag(self.Kp)@pd_err + cs.diag(self.Kd)@derr_dt

        # Apply the limit to the control output
        limited_pd_controller = cs.fmin(cs.fmax(pd_controller, self.u_min), self.u_max)

        pd = cs.Function('pid', 
                           [self.ssyms.q, self.qref, self.Kp, self.Kd, prev_err, self.ssyms.dt, self.u_max, self.u_min],
                             [limited_pd_controller, pd_err])

        return pd