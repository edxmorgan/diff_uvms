from diffUVMS.geometry.symbols import construct_manipulator_syms as manipulator_sys_syms
from diffUVMS.geometry.symbols import construct_vehicle_syms as vehicle_sys_syms
import casadi as cs


class RobotControllers():
    def __init__(self, n_joints):
        self.arm_ssyms = manipulator_sys_syms(n_joints)
        self.fb_ssyms = vehicle_sys_syms() #floating base symbols
        print(f"number of joints = {self.arm_ssyms.n_joints}")

        self.qref = cs.SX.sym('qref', self.arm_ssyms.n_joints)  # Desired joint positions
        self.Kp = cs.SX.sym('Kp', self.arm_ssyms.n_joints)      # Proportional gain
        self.u_max = cs.SX.sym('u_max', self.arm_ssyms.n_joints)
        self.u_min = cs.SX.sym('u_min', self.arm_ssyms.n_joints)

    def pid(self):
        # ne = self.fb_ssyms.n - nd

        # i_buffer = sum_e_buffer + ne*dt

        # pid = -diag(Kp)@ne - diag(Kd)@(self.J_@x_nb) - diag(Ki)@i_buffer

        # pid_controller = self.gn + self.J_.T@pid

        # return pid_controller, i_buffer
    
        # Error between reference and actual position
        err = self.qref - self.arm_ssyms.q

        # Compute the P control output
        p_controller = cs.diag(self.Kp)@err 

        # Apply the limit to the control output
        limited_pd_controller = cs.fmin(cs.fmax(p_controller, self.u_min), self.u_max)

        p = cs.Function('pid', 
                           [self.arm_ssyms.q, self.qref, self.Kp, self.u_max, self.u_min],
                             [limited_pd_controller, err])

        return p
