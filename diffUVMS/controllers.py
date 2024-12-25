from diffUVMS.geometry.symbols import construct_manipulator_syms as manipulator_sys_syms
from diffUVMS.geometry.symbols import construct_vehicle_syms as vehicle_sys_syms
from diffUVMS.geometry.symbols import construct_uvms_syms as sys_syms
import casadi as cs


class RobotControllers():
    def __init__(self, n_joints):
        self.arm_ssyms = manipulator_sys_syms(n_joints)
        self.fb_ssyms = vehicle_sys_syms() #floating base symbols
        self.uvms_ssyms = sys_syms(n_joints)
        print(f"number of joints = {self.arm_ssyms.n_joints}")


    def pid(self):
        # Error between reference and actual position
        err = self.arm_ssyms.q - self.arm_ssyms.qref

        # Compute the P control output
        p_controller = -cs.diag(self.arm_ssyms.Kp)@err - cs.diag(self.arm_ssyms.Kd)@self.arm_ssyms.q_dot

        # Apply the limit to the control output
        limited_pd_controller = cs.fmin(cs.fmax(p_controller, self.arm_ssyms.u_min), self.arm_ssyms.u_max)

        p = cs.Function('pid', 
                           [self.arm_ssyms.q, self.arm_ssyms.q_dot, self.arm_ssyms.qref, self.arm_ssyms.Kp, self.arm_ssyms.Kd, self.arm_ssyms.u_max, self.arm_ssyms.u_min],
                             [limited_pd_controller, err])

        return p
    
    def uvms_position_controller(self):
        gn = cs.SX.sym('gn', self.fb_ssyms.uv_dof)
        gn_q = cs.SX.zeros(self.arm_ssyms.n_joints)
        g = cs.vertcat(gn, gn_q)

        J_n = cs.SX.sym('J_n', self.fb_ssyms.uv_dof, self.fb_ssyms.uv_dof)
        J_n_q = cs.SX.eye(self.arm_ssyms.n_joints)
        J = cs.SX.zeros(10, 10)
        J[:6, :6] = J_n
        J[6:, 6:] = J_n_q

        err = self.uvms_ssyms.n - self.uvms_ssyms.nref

        #i_buffer = sum_e_buffer + ne*dt
        pid = -cs.diag(self.uvms_ssyms.Kp)@err #- cs.diag(self.Kd)@(self.J_n@x_nb) #- diag(Ki)@i_buffer

        pid_controller = g + J.T@pid

        # Apply the limit to the control output
        limited_pd_controller = cs.fmin(cs.fmax(pid_controller, self.uvms_ssyms.u_min), self.uvms_ssyms.u_max)

        controller = cs.Function('pid', 
                           [self.uvms_ssyms.n, self.uvms_ssyms.nref, gn, J_n, self.uvms_ssyms.Kp, self.uvms_ssyms.u_min, self.uvms_ssyms.u_max],
                             [limited_pd_controller, err])
        
        return controller #, i_buffer