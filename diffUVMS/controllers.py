from diffUVMS.geometry.symbols import construct_manipulator_syms as arm_ssyms
import casadi as cs


class RobotControllers():
    def __init__(self, n_joints):
        self.arm_ssyms = arm_ssyms(n_joints)
        print(f"number of joints = {self.arm_ssyms.n_joints}")


    def arm_position_controller(self):
        # Error between reference and actual position
        err = self.arm_ssyms.q - self.arm_ssyms.qref

        # Compute the P control output
        p_controller = -cs.diag(self.arm_ssyms.Kp)@err - cs.diag(self.arm_ssyms.Kd)@self.arm_ssyms.q_dot 

        # Apply the limit to the control output
        limited_pd_controller = cs.fmin(cs.fmax(p_controller, self.arm_ssyms.u_min), self.arm_ssyms.u_max)

        p = cs.Function('pid', 
                           [self.arm_ssyms.q, self.arm_ssyms.q_dot, self.arm_ssyms.qref, self.arm_ssyms.Kp, 
                            self.arm_ssyms.Ki, self.arm_ssyms.Kd,
                             self.arm_ssyms.sum_e_buffer, self.arm_ssyms.dt,
                              self.arm_ssyms.u_max, self.arm_ssyms.u_min],
                             [limited_pd_controller, err])

        return p
    
