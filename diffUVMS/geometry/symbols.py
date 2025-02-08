from casadi import SX,  vertcat

class construct_manipulator_syms():
    def __init__(self, n_joints):
        self.n_joints = n_joints
        self.gravity = SX.sym('gravity')
        self.base_gravity = SX.sym('base_gravity')

        self.joint_configurations = SX.sym('joint_configs', 2**n_joints, n_joints)
        self.m_u = SX.sym("m_u", n_joints)

        self.q_max = SX.sym('q_max', n_joints)
        self.q_min = SX.sym('q_min', n_joints)

        self.q = SX.sym("q", n_joints)
        self.q_dot = SX.sym("q_dot", n_joints)
        self.q_ddot = SX.sym("q_ddot", n_joints)

        self.noise = SX.sym("nosie", n_joints)

        self.G = SX.sym('G', n_joints) # gear ratio

        self.fw_viscous = SX.sym('fw_viscous', n_joints) # forward motion viscous friction coefficient
        self.fw_static = SX.sym('fw_static', n_joints) # forward motion Coulomb friction coefficient
        self.bw_viscous = SX.sym('bw_viscous', n_joints) # backward motion viscous friction coefficient
        self.bw_static = SX.sym('bw_static', n_joints) # backward motion Coulomb friction coefficient

        self.rotor_spatial_inertia = SX.sym("Ir", 6,6, n_joints) #rotor spatial inertia

        self.m_states = vertcat(self.q, self.q_dot)

        # # hydrodynamics symbols
        # # xz, yz, xy planes of symmetry configuration of links
        # self.M_A_coef = SX.sym("M_A_coeff", 6,1, n_joints)
        # #linear damping components in body
        # self.D_u = SX.sym("D_u", 6,1, n_joints)
        # #nonlinear (quadratic) damping components in body
        # self.D_uu = SX.sym("D_uu", 6,1, n_joints)
        # self.link_Volume = SX.sym("LV", n_joints)
        # self.cob = SX.sym("cob", 3,1 ,n_joints)
        # self.rho = SX.sym("rho")


        # M_A_coef = self.M_A_coef
        # D_u_coeff = self.D_u
        # D_uu_coeff = self.D_uu

        # self.hydrodynamic_p = vertcat(*M_A_coef, *D_u_coeff, *D_uu_coeff, self.link_Volume, *self.cob, self.rho)

        # self.forward_dynamics_parameters = vertcat(self.rigid_body_p ,self.hydrodynamic_p, self.dt, self.base_T)
        # self.forward_dynamics_parameters_fb = vertcat(self.rigid_body_p, self.trivial_Ir, self.hydrodynamic_p, self.v_c, self.dt, self.base_T)

        self.Ir = self.rotor_spatial_inertia

        self.m_rigid_body_p = vertcat(self.G, 
                                      self.Ir[3][14],
                                      self.Ir[2][7], 
                                      self.Ir[1][7],
                                      self.Ir[0][14], 
                                      self.fw_static, self.fw_viscous, self.bw_static, self.bw_viscous)
        
        self.trivial_sim_p = vertcat(self.Ir[1][11], self.Ir[1][9], self.Ir[0][12],
                                      self.Ir[0][13], self.Ir[0][16], self.Ir[0][15], 
                                      self.Ir[0][17],
                                     self.Ir[3][17], self.Ir[3][16], self.Ir[3][15],  
                                     self.Ir[3][13], self.Ir[3][12], self.Ir[2][11],
                                       self.Ir[2][10], self.Ir[2][9], self.Ir[2][8],
                                       self.Ir[2][6], self.Ir[1][10], self.Ir[1][8], 
                                       self.Ir[1][6])
        
        self.dt = SX.sym("dt")
        self.sim_p = vertcat(self.m_rigid_body_p)
        self.Kp = SX.sym('Kp',self.n_joints)
        self.Kd = SX.sym('Kd',self.n_joints)
        self.Ki = SX.sym('Ki',self.n_joints)
        self.qref = SX.sym('qref', self.n_joints)  # Desired joint positions
        self.u_max = SX.sym('u_max', self.n_joints)
        self.u_min = SX.sym('u_min', self.n_joints)
        self.sum_e_buffer = SX.sym("sum_e_buffer", self.n_joints,1)

        self.baseT_xyz = SX.sym('T_xyz', 3) # manipulator-vehicle mount link xyz origin 
        self.baseT_rpy = SX.sym('T_rpy', 3) # manipulator-vehicle mount link rpy origin
        self.base_T = vertcat(self.baseT_rpy, self.baseT_xyz) # transform from origin to 1st child

        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')
        self.tr_n = vertcat(x, y, z) # x, y ,z of uv wrt to ned origin
        thet = SX.sym('thet')
        phi = SX.sym('phi')
        psi = SX.sym('psi')
        self.eul = vertcat(phi, thet, psi)  # NED euler angular velocity
        self.p_n = vertcat(self.tr_n, self.eul) # ned total states

        self.n = vertcat(self.p_n, self.q) #NED position
        self.dn = SX.sym('dn', 6+self.n_joints) # NED velocity
        self.uvms_ul = SX.sym('uvms_ul', 6+self.n_joints)
        self.uvms_ll = SX.sym('uvms_ll', 6+self.n_joints)
        self.k0 = SX.sym('k0', 6+self.n_joints)
        self.des_v = SX.sym('des_v', 6) #operational space desired velocity

    def __repr__(self) -> str:
        return "differentiable manipulator symbols"
