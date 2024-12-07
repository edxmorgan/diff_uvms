from casadi import SX,  vertcat, DM, horzcat, diag

class construct_syms():
    def __init__(self, n_joints):
        self.n_joints = n_joints
        uv_dof = 6
        self.dt = SX.sym("dt")
        self.uv_u = SX.sym("uv_u", uv_dof)
        self.m_u = SX.sym("m_u", n_joints)
        self.uvms_tau_rhs = vertcat(self.uv_u,self.m_u)
        self.uvms_u = vertcat(self.uv_u[3:6], self.uv_u[0:3], self.m_u)

        self.q_max = SX.sym('q_max', n_joints)
        self.q_min = SX.sym('q_min', n_joints)

        self.q = SX.sym("q", n_joints)
        self.q_dot = SX.sym("q_dot", n_joints)
        self.q_ddot = SX.sym("q_ddot", n_joints)

        self.G = SX.sym('G', n_joints) # gear ratio

        self.fw_viscous = SX.sym('fw_viscous', n_joints) # forward motion viscous friction coefficient
        self.fw_static = SX.sym('fw_static', n_joints) # forward motion Coulomb friction coefficient
        self.bw_viscous = SX.sym('bw_viscous', n_joints) # backward motion viscous friction coefficient
        self.bw_static = SX.sym('bw_static', n_joints) # backward motion Coulomb friction coefficient

        self.rotor_spatial_inertia = SX.sym("Ir", 6,6, n_joints) #rotor spatial inertia

        self.v_base = SX.sym('v_base', uv_dof) # base velocity
        self.a_base = SX.sym('a_base', uv_dof) # base acceleration
        self.baseT_xyz = SX.sym('c_xyz', 3) # manipulator-vehicle mount link xyz origin 
        self.baseT_rpy = SX.sym('c_rpy', 3) # manipulator-vehicle mount link rpy origin
        self.base_T = vertcat(self.baseT_rpy, self.baseT_xyz) # transform from origin to 1st child
        self.v_c = SX.sym('v_c', 6) # flow current velocity
        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')
        self.tr_n = vertcat(x, y, z) # x, y ,z of uv wrt to ned origin
        thet = SX.sym('thet')
        phi = SX.sym('phi')
        psi = SX.sym('psi')
        self.eul = vertcat(phi, thet, psi)  # NED euler angular velocity
        self.p_n = vertcat(self.tr_n, self.eul) # ned total states
        self.v_uv = vertcat(self.v_base[3:6],self.v_base[0:3])

        # self.uv_states = vertcat(self.p_n, self.v_uv)
        self.m_states = vertcat(self.q, self.q_dot)

        self.uvms_states = vertcat(self.p_n, self.q, self.v_uv, self.q_dot)

        self.n = self.uvms_states[:10]

        dx = SX.sym('x_dot')
        dy = SX.sym('y_dot')
        dz = SX.sym('z_dot')
        dthet = SX.sym('thet_dot')
        dphi = SX.sym('phi_dot')
        dpsi = SX.sym('psi_dot')
        self.nd =  vertcat(dx, dy, dz, dphi, dthet, dpsi, self.q_dot)  # NED velocity
        
        self.uvms_states_ned = vertcat(self.n, self.nd)   

        # hydrodynamics symbols
        # xz, yz, xy planes of symmetry configuration of links
        self.M_A_coef = SX.sym("M_A_coeff", 6,1, n_joints)
        #linear damping components in body
        self.D_u = SX.sym("D_u", 6,1, n_joints)
        #nonlinear (quadratic) damping components in body
        self.D_uu = SX.sym("D_uu", 6,1, n_joints)
        self.link_Volume = SX.sym("LV", n_joints)
        self.cob = SX.sym("cob", 3,1 ,n_joints)
        self.rho = SX.sym("rho")

        self.Ir = self.rotor_spatial_inertia
        M_A_coef = self.M_A_coef
        D_u_coeff = self.D_u
        D_uu_coeff = self.D_uu

        self.rigid_body_p = vertcat(self.G, self.Ir[3][17], self.Ir[3][16], self.Ir[3][15], self.Ir[3][14], self.Ir[3][13], self.Ir[3][12],
                        self.Ir[2][11], self.Ir[2][10], self.Ir[2][9], self.Ir[2][8], self.Ir[2][7], self.Ir[2][6],
                        self.Ir[1][10], self.Ir[1][8], self.Ir[1][7], self.Ir[1][6],
                        self.Ir[0][14], self.fw_static, self.fw_viscous, self.bw_static, self.bw_viscous)
        
        self.trivial_Ir = vertcat(self.Ir[1][11], self.Ir[1][9], self.Ir[0][12], self.Ir[0][13], self.Ir[0][16], self.Ir[0][15], self.Ir[0][17])
        
        self.hydrodynamic_p = vertcat(*M_A_coef, *D_u_coeff, *D_uu_coeff, self.link_Volume, *self.cob, self.rho)

        self.forward_dynamics_parameters = vertcat(self.rigid_body_p ,self.hydrodynamic_p, self.dt, self.base_T)
        self.forward_dynamics_parameters_fb = vertcat(self.rigid_body_p, self.trivial_Ir, self.hydrodynamic_p, self.v_c, self.dt, self.base_T)



    def __repr__(self) -> str:
        return "differentiable symbols"
