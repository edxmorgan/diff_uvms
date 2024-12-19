from UVarm.geometry.symbols import construct_syms as sys_syms
import casadi as cs


class RobotControllers():
    def __init__(self, n_joints):
        self.ssyms = sys_syms(n_joints)
        print(f"number of joints = {self.ssyms.n_joints}")

        self.qref = cs.SX.sym('qref', self.ssyms.n_joints)  # Desired joint positions
        self.Kp = cs.SX.sym('Kp', self.ssyms.n_joints)      # Proportional gain
        self.u_max = cs.SX.sym('u_max', self.ssyms.n_joints)
        self.u_min = cs.SX.sym('u_min', self.ssyms.n_joints)

    def pid(self):
        # Error between reference and actual position
        err = self.qref - self.ssyms.q

        # Compute the P control output
        p_controller = cs.diag(self.Kp)@err 

        # Apply the limit to the control output
        limited_pd_controller = cs.fmin(cs.fmax(p_controller, self.u_min), self.u_max)

        p = cs.Function('pid', 
                           [self.ssyms.q, self.qref, self.Kp, self.u_max, self.u_min],
                             [limited_pd_controller, err])

        return p
