from casadi import SX,  vertcat

class construct_manipulator_syms():
    def __init__(self, n_joints):
        self.n_joints = n_joints

        self.joint_configurations = SX.sym('joint_configs', 2**n_joints, n_joints)
        self.m_u = SX.sym("m_u", n_joints)

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

    def __repr__(self) -> str:
        return "differentiable manipulator symbols"


class construct_vehicle_syms():
    def __init__(self):
       
        self.uv_dof = 6
        self.p_max = SX.sym('p_max', self.uv_dof) # pose upper limit
        self.p_min = SX.sym('p_min', self.uv_dof) # pose lower limit

        self.uv_u = SX.sym("uv_u", self.uv_dof)
        self.v_base = SX.sym('v_base', self.uv_dof) # base velocity in featherson notation
        self.a_base = SX.sym('a_base', self.uv_dof) # base acceleration in featherson notation
        self.baseT_xyz = SX.sym('T_xyz', 3) # manipulator-vehicle mount link xyz origin 
        self.baseT_rpy = SX.sym('T_rpy', 3) # manipulator-vehicle mount link rpy origin
        self.base_T = vertcat(self.baseT_rpy, self.baseT_xyz) # transform from origin to 1st child
        self.v_c = SX.sym('v_c', self.uv_dof) # flow current velocity
        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')
        self.tr_n = vertcat(x, y, z) # x, y ,z of uv wrt to ned origin
        thet = SX.sym('thet')
        phi = SX.sym('phi')
        psi = SX.sym('psi')
        self.eul = vertcat(phi, thet, psi)  # NED euler angular velocity
        self.p_n = vertcat(self.tr_n, self.eul) # ned total states
        self.v_uv = vertcat(self.v_base[3:6],self.v_base[0:3]) #back to fossen spatial notation
        self.a_uv = vertcat(self.a_base[3:6],self.a_base[0:3]) #back to fossen spatial notation

        self.dx = SX.sym('x_dot')
        self.dy = SX.sym('y_dot')
        self.dz = SX.sym('z_dot')
        self.dthet = SX.sym('thet_dot')
        self.dphi = SX.sym('phi_dot')
        self.dpsi = SX.sym('psi_dot')

        W = SX.sym('W')  # weight
        B = SX.sym('B')  # buoyancy
        m = SX.sym('m')  # Mass

        I_x = SX.sym('I_x')  # moment of inertia x entry
        I_y = SX.sym('I_y')  # moment of inertia y entry
        I_z = SX.sym('I_z')  # moment of inertia z entry
        I_xz = SX.sym('I_xz')  # product of inertia zx entry

        # CoG.
        x_g = SX.sym('x_g')  # Center of gravity, x-axis wrt to the CO
        y_g = SX.sym('y_g')  # Center of gravity, y-axis wrt to the CO
        z_g = SX.sym('z_g')  # Center of gravity, z-axis wrt to the CO
        r_g = vertcat(x_g, y_g, z_g) # center of gravity wrt body origin

        x_b = SX.sym('x_b')  # Center of buoyancy, x-axis
        y_b = SX.sym('y_b')  # Center of buoyancy, y-axis
        z_b = SX.sym('z_b')  # Center of buoyancy, z-axis
        r_b = vertcat(x_b, y_b, z_b) # center of buoyancy wrt body origin 

        I_o = vertcat(I_x, I_y, I_z, I_xz) # EFFECTIVE rigid body inertia wrt body origin

        X_du = SX.sym('X_du') # Added mass in surge
        X_dq = SX.sym('X_dq') # coupled Added mass in surge & pitch
        Y_dv = SX.sym('Y_dv') # Added mass in sway
        Y_dp = SX.sym('Y_dp') # coupled Added mass in sway & roll
        Z_dw = SX.sym('Z_dw') # Added mass in heave
        K_dv = SX.sym('K_dv') # coupled Added mass in roll & sway
        K_dp = SX.sym('K_dp') # Added mass in roll
        M_du = SX.sym('M_du') # coupled Added mass in pitch & surge
        M_dq = SX.sym('M_dq') # Added mass in pitch
        N_dr = SX.sym('N_dr') # Added mass in yaw

        X_u = SX.sym('X_u') # linear Drag coefficient in surge
        Y_v = SX.sym('Y_v') # linear Drag coefficient in sway
        Z_w = SX.sym('Z_w') # linear Drag coefficient in heave
        K_p = SX.sym('K_p') # linear Drag coefficient in roll
        M_q = SX.sym('M_q') # linear Drag coefficient in pitch
        N_r = SX.sym('N_r') # linear Drag coefficient in yaw

        X_uu = SX.sym('X_uu') # quadratic Drag coefficient in surge
        Y_vv = SX.sym('Y_vv') # quadratic Drag coefficient in sway
        Z_ww = SX.sym('Z_ww') # quadratic Drag coefficient in heave
        K_pp = SX.sym('K_pp') # quadratic Drag coefficient in roll
        M_qq = SX.sym('M_qq') # quadratic Drag coefficient in pitch
        N_rr = SX.sym('N_rr') # quadratic Drag coefficient in yaw

        # Current/flow velocity
        v_c = SX.sym('v_c', 6, 1)
        
        # OTHER EFFECTIVE PARAMETERS
        decoupled_added_m = vertcat(X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr) # added mass in diagonals
        coupled_added_m =  vertcat(X_dq, Y_dp, M_du, K_dv) # effective added mass in non diagonals

        linear_dc = vertcat(X_u, Y_v, Z_w, K_p,  M_q, N_r) # linear damping coefficients
        quadratic_dc = vertcat(X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr) # quadratic damping coefficients

        self.sim_p = vertcat(m, W, B, r_g, r_b, I_o,
                                decoupled_added_m, coupled_added_m,
                                linear_dc, quadratic_dc, v_c)
        
        self.Kp = SX.sym('Kp', self.uv_dof)
        self.Kd = SX.sym('Kd', self.uv_dof)
        self.Ki = SX.sym('Ki', self.uv_dof)
        self.uvref = SX.sym('uv_ref', self.uv_dof)  # Desired dof positions
        self.u_max = SX.sym('u_max', self.uv_dof)
        self.u_min = SX.sym('u_min', self.uv_dof)
        self.sum_e_buffer = SX.sym("sum_e_buffer", self.uv_dof,1)


class construct_uvms_syms():
    def __init__(self, n_joints):
        self.dt = SX.sym("dt")
        self.arm_ssyms = construct_manipulator_syms(n_joints)
        self.fb_ssyms = construct_vehicle_syms() #floating base symbols
        self.total_dof = self.fb_ssyms.uv_dof+ self.arm_ssyms.n_joints

        self.ll = vertcat(self.fb_ssyms.p_min , self.arm_ssyms.q_min) # lower limits of robot generalized coordinates
        self.ul = vertcat(self.fb_ssyms.p_max , self.arm_ssyms.q_max) # upper limits of robot generalized coordinates

        self.k0 = SX.sym('k0', self.total_dof) # secondary task rate

        self.n = vertcat(self.fb_ssyms.p_n, self.arm_ssyms.q) #NED position
        self.dn = SX.sym("dn", self.total_dof) # explicit NED velocity
        self.des_v = SX.sym("op_space_vel", 6) # operational space velocity

        self.uvms_vel = vertcat(self.fb_ssyms.v_uv, self.arm_ssyms.q_dot) #body velcity for uv and joint velocity for arm
        self.uvms_acc = vertcat(self.fb_ssyms.a_uv, self.arm_ssyms.q_ddot) #body acceleration for uv and joint acceleration for arm

        self.uvms_states = vertcat(self.n, self.uvms_vel)

        self.nref = vertcat(self.fb_ssyms.uvref, self.arm_ssyms.qref)

        self.Kp =  vertcat(self.fb_ssyms.Kp, self.arm_ssyms.Kp)
        self.Kd =  vertcat(self.fb_ssyms.Kd, self.arm_ssyms.Kd)
        self.Ki =  vertcat(self.fb_ssyms.Ki, self.arm_ssyms.Ki)
        self.sum_e_buffer = vertcat(self.fb_ssyms.sum_e_buffer, self.arm_ssyms.sum_e_buffer)

        self.u_min = vertcat(self.fb_ssyms.u_min, self.arm_ssyms.u_min)
        self.u_max = vertcat(self.fb_ssyms.u_max, self.arm_ssyms.u_max)

    def __repr__(self) -> str:
        return "differentiable uvms symbols"