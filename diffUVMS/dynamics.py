import diffUVMS.geometry.plucker as plucker
from diffUVMS.geometry.symbols import construct_uvms_syms as UVMS_syms
from diffUVMS.urdfparser import URDFparser
import casadi as cs
import numpy as np
import copy

class RobotDynamics():
    def __init__(self, path_to_urdf,  root, tip):
        self.root = root
        self.tip = tip
        self.parser = URDFparser(func_opts=None, use_jit=True)
        self.parser.from_file(path_to_urdf)
        self.joint_list, self.joint_names, self.q_max, self.q_min = self.parser.get_joint_info(
            root, tip)
        self.n_joints = self.parser.get_n_joints(root, tip)
        self.limits = self.parser.get_dynamics_limits(root, tip)

        self.sys_syms = UVMS_syms( self.n_joints) #whole body symbols

        self.arm_ssyms = self.sys_syms.arm_ssyms # manipulator symbols
        self.fb_ssyms = self.sys_syms.fb_ssyms #floating base symbols

        print(f"number of joints = {self.arm_ssyms.n_joints}")
        self.robot_desc = copy.deepcopy(self.parser.robot_desc_backup)
        self.T_Base = plucker.XT(self.fb_ssyms.baseT_xyz, self.fb_ssyms.baseT_rpy)

    def forward_kinematics(self, floating_base = False):
        q = self.arm_ssyms.q
        i_X_p, tip_offset, Si, Ic, Icom, Im = self._model_calculation(q)
        T_Base = plucker.XT(self.fb_ssyms.baseT_xyz, self.fb_ssyms.baseT_rpy)

        i_X_0s = []
        for i in range(0, self.arm_ssyms.n_joints):
            if i != 0:
                i_X_0 = plucker.spatial_mtimes(i_X_p[i],i_X_0)
            else:
                if floating_base:
                    NED_0_ = plucker.XT(self.fb_ssyms.tr_n, self.fb_ssyms.eul)
                    T_Base_X_NED_0 = plucker.spatial_mtimes(T_Base, NED_0_)
                    i_X_0 = plucker.spatial_mtimes(i_X_p[i],T_Base_X_NED_0)
                else:
                    i_X_0 = i_X_p[i]
            
            i_X_0s.append(i_X_0)  # transformation of joint i wrt origin 0

        forward_kin = plucker.spatial_mtimes(tip_offset, i_X_0)
        i_X_0s.append(forward_kin)
        return i_X_p, i_X_0s

    def get_inverse_dynamics_rnea(self, gravity=9.81, floating_base_id=None, coupled=True):
        """Returns the inverse dynamics as a casadi expression."""
        q = self.arm_ssyms.q
        q_dot = self.arm_ssyms.q_dot
        q_ddot = self.arm_ssyms.q_ddot
        fw_static, fw_viscous = self.arm_ssyms.fw_static, self.arm_ssyms.fw_viscous
        bw_static, bw_viscous = self.arm_ssyms.bw_static, self.arm_ssyms.bw_viscous
        f_ext=None
        v_base = self.fb_ssyms.v_base 
        a_base = self.fb_ssyms.a_base
        [f, fR, tau_gear, ID_exp, Si, i_X_p, i_X_0s, v, a, Ic, f_base, friction] = self.solves_rnea(q,
                                                                                                       q_dot,
                                                                                                       q_ddot,
                                                                                                       fw_static,
                                                                                                       fw_viscous,
                                                                                                       bw_static, 
                                                                                                       bw_viscous,
                                                                                                       gravity,
                                                                                                       f_ext,
                                                                                                       floating_base_id,
                                                                                                       coupled,
                                                                                                       v_base,
                                                                                                       a_base)
        if floating_base_id!=None:
            ID_exp = cs.vertcat(f_base[3:6],f_base[0:3], ID_exp) # x, y ,z , r, p ,y ,q0, q1, q2, q3
        return ID_exp
    

    def get_bias_force(self, gravity=9.81, floating_base_bias_f=None, coupled=True):
        """Returns the Coriolis vector as a casadi expression."""
        q = self.arm_ssyms.q
        q_dot = self.arm_ssyms.q_dot
        q_ddot = [0]*self.n_joints
        fw_static, fw_viscous = self.arm_ssyms.fw_static, self.arm_ssyms.fw_viscous
        bw_static, bw_viscous = self.arm_ssyms.bw_static, self.arm_ssyms.bw_viscous
        gravity = gravity
        f_ext=None
        v_base = self.fb_ssyms.v_base 
        a_base = [0]*self.fb_ssyms.a_base.size1()
        [f, fR, tau_gear, C, Si, i_X_p, i_X_0s, v, a, Ic, C_f_base, friction] = self.solves_rnea(q,
                                                                                q_dot,
                                                                                q_ddot,
                                                                                fw_static,
                                                                                fw_viscous,
                                                                                bw_static, 
                                                                                bw_viscous,
                                                                                gravity,
                                                                                f_ext,
                                                                                floating_base_bias_f,
                                                                                coupled,
                                                                                v_base,
                                                                                a_base)
        if floating_base_bias_f!=None:
            C = cs.vertcat(C_f_base[3:6], C_f_base[0:3], C) # x, y ,z , r, p ,y ,q0, q1, q2, q3
        return C

    def get_inertia_matrix(self, gravity=9.81, floating_base_id=None, floating_base_bias_f=None, coupled=True):
        ID_exp = self.get_inverse_dynamics_rnea(gravity=gravity, floating_base_id=floating_base_id, coupled=coupled)
        ddX = self.arm_ssyms.q_ddot

        if floating_base_id!=None and floating_base_bias_f==None:
            raise Exception("floating_base_bias_force required for floating base operations")
        if floating_base_id==None and floating_base_bias_f!=None:
            raise Exception("floating_base_id required for floating base operations")
        
        if floating_base_id!=None and floating_base_bias_f!=None:
            ddX = cs.vertcat(self.fb_ssyms.a_base ,ddX)
            ddX = cs.vertcat(ddX[3:6],ddX[0:3],ddX[6:]) # x, y ,z , r, p ,y ,q0, q1, q2, q3

        C = self.get_bias_force(gravity, floating_base_bias_f, coupled=coupled)

        n = ID_exp.size1()
        ID_delta = ID_exp - C

        H = cs.SX(n, n)
        for i in range(0, n):
            d_a = [0]*n
            d_a[i] = 1
            H[:, i] = cs.substitute(ID_delta, ddX, d_a)
        return H

    def _model_calculation(self, q):
        """Calculates and returns model information needed in the
        dynamics algorithms caluculations, i.e transforms, joint space
        and inertia."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        chain = self.robot_desc.get_chain(self.root, self.tip)
        self.spatial_inertias = []
        self.link_coms = []
        self.link_masses = []
        self.i_X_p = []
        self.Sis = []
        self.tip_offset = cs.DM_eye(6)
        prev_joint = None
        n_actuated = 0
        i = 0

        for item in chain:
            # print("**************************************************")
            if item in self.robot_desc.joint_map:
                prev_na = n_actuated
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    ##########################################

                    ##########################################

                    if prev_joint == "fixed":
                        XT_prev = plucker.XT(
                            joint.origin.xyz, joint.origin.rpy)@XT_prev
                    else:
                        XT_prev = plucker.XT(
                            joint.origin.xyz, joint.origin.rpy)

                    inertia_transform = XT_prev
                    prev_inertia = spatial_inertia
                elif joint.type == "prismatic":
                    # print('found prismatic')
                    if n_actuated != 0:
                        # print(f"{link.name} link added")
                        self.spatial_inertias.append(spatial_inertia)
                        self.link_coms.append(link_com)
                        self.link_masses.append(link_mass)
                    n_actuated += 1

                    ##########################################
                    XJT = plucker.XJT_prismatic(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis, q[i])

                    Si = cs.SX([0, 0, 0,
                                joint.axis[0],
                                joint.axis[1],
                                joint.axis[2]])
                    ##########################################

                    if prev_joint == "fixed":
                        XJT = cs.mtimes(XJT, XT_prev)

                    self.i_X_p.append(XJT)
                    self.Sis.append(Si)
                    i += 1

                elif joint.type in ["revolute", "continuous"]:
                    if n_actuated != 0:
                        # print(f"{link.name} link added")
                        self.spatial_inertias.append(spatial_inertia)
                        self.link_coms.append(link_com)
                        self.link_masses.append(link_mass)
                    n_actuated += 1

                    ###############################
                    XJT = plucker.XJT_revolute(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q[i])
                    Si = cs.SX([
                        joint.axis[0],
                        joint.axis[1],
                        joint.axis[2],
                        0,
                        0,
                        0])
                    ################################

                    if prev_joint == "fixed":
                        XJT = cs.mtimes(XJT, XT_prev)

                    self.i_X_p.append(XJT)
                    self.Sis.append(Si)
                    i += 1

                prev_joint = joint.type
                # print(f'parent: {joint.parent} ; child ;{joint.child}')
                # if (n_actuated - prev_na) == 1:
                #     print(f'found {joint.name} {joint.type} actuated joint')
                # else:
                #     print(f'found {joint.name} {joint.type} joint')

                if joint.child == self.tip:
                    if joint.type == "fixed":
                        self.tip_offset = XT_prev

            if item in self.robot_desc.link_map:
                link = self.robot_desc.link_map[item]
                # print(f'found {link.name}')
                # print(i)
                if link.inertial is None:
                    spatial_inertia = np.zeros((6, 6))
                else:
                    I = link.inertial.inertia
                    spatial_inertia = plucker.spatial_inertia_matrix_IO_sym(
                        I.ixx,
                        I.ixy,
                        I.ixz,
                        I.iyy,
                        I.iyz,
                        I.izz,
                        link.inertial.mass,
                        link.inertial.origin.xyz)
                    link_com = link.inertial.origin.xyz
                    link_mass = link.inertial.mass

                if prev_joint == "fixed":
                    # print('found fixed')
                    spatial_inertia = prev_inertia + \
                        inertia_transform.T@spatial_inertia@plucker.inverse_spatial_transform(
                            inertia_transform)

                if link.name == self.tip:
                    # print('is tip')
                    # print(f"{link.name} link added")
                    self.spatial_inertias.append(spatial_inertia)
                    self.link_coms.append(link_com)
                    self.link_masses.append(link_mass)
        
        return self.i_X_p, self.tip_offset, self.Sis, self.spatial_inertias, self.link_coms, self.link_masses

    def solves_rnea(self, q, q_dot, q_ddot, fw_static, fw_viscous, bw_static, bw_viscous, gravity=9.81, f_ext=None, floating_base_id=None, coupled=True, vel_base=None, acc_base=None):
        """Returns recursive newton euler algorithm."""
        if self.parser.robot_desc_backup is None:
            raise ValueError('Robot description not loaded from urdf')
        self.robot_desc = copy.deepcopy(self.parser.robot_desc_backup)

        i_X_p, tip_offset, Si, Ic, Icom, Im = self._model_calculation(q)

        n_joints = self.arm_ssyms.n_joints

        G = self.arm_ssyms.G
        Ir = self.arm_ssyms.rotor_spatial_inertia
  
        i_X_0s = []
        v = []
        vR = []
        a = []
        aR = []
        f = []
        f_withoutR = []
        fR = []
        self.Ic_i = [] # composite rigid body inertias
        fRIC = cs.SX.zeros(n_joints)
        tau_gear = cs.SX.zeros(n_joints)
        tau_motor = cs.SX.zeros(n_joints)
        F_base = cs.SX.zeros(6, 1)

        v0 = cs.SX.zeros(6, 1)
        ag = cs.SX.zeros(6, 1)
        # FORWARD ITERATION
        for i in range(0, n_joints):
            self.Ic_i.append(Ic[i])
            if i != 0:
                i_X_0 = plucker.spatial_mtimes(i_X_p[i],i_X_0)
                v0 = i_X_p[i]@v[i-1]
                a0 = i_X_p[i]@a[i-1]
            else:
                i_X_0 = plucker.spatial_mtimes(i_X_p[i],self.T_Base)
                if coupled:
                    v0 = i_X_0@v0 #v0 = i_X_0@vel_base
                    a0 = i_X_0@ag #a0 = i_X_0@acc_base
                else:
                    v0 = i_X_p[i]@v0
                    a0 = i_X_p[i]@ag
                    
            i_X_0s.append(i_X_0)  # transformation of origin 0 to joint i

            vJ = Si[i]@q_dot[i]
            v.append(v0 + vJ)  # body i velocity

            aJ = Si[i]@q_ddot[i]
            vi_x = plucker.motion_cross_product(v[i])
            a.append(a0 + aJ + vi_x@vJ)  # body i acceleration

            vi_xf = plucker.force_cross_product(v[i])
            _fb = Ic[i]@a[i] + vi_xf@Ic[i]@v[i]  # body i forces

            # body force + hydrodynamics calculated
            # _fb = self._apply_body_link_hydrodynamics(i, _fb, v, a, Icom, Im, i_X_0, gravity)

            # other external forces
            # if f_ext is not None:
            #     _fb = self._apply_external_forces(f_ext, _fb, i_X_0)  # external body forces

            f.append(_fb)
            f_withoutR.append(_fb)

            # rotor dynamics
            vRJ = Si[i]@G[i]@q_dot[i]  # rotor i velocity
            aRJ = Si[i]@G[i]@q_ddot[i]
            vRi_x = plucker.motion_cross_product(vRJ)
            aR.append(aRJ + vRi_x@vRJ)  # rotor i acceleration

            vRi_xf = plucker.force_cross_product(vRJ)
            _fR = Ir[i]@aR[i] + vRi_xf@Ir[i]@vRJ  # rotor i forces

            fR.append(_fR)
            
        # BACKWARD ITERATION
        for i in range(n_joints-1, -1, -1):
            tau_gear[i] = Si[i].T@f[i]

            # Calculate joint friction
            static_friction_coeff = self.parameter_pick(
                q_dot[i], fw_static[i], bw_static[i])
            viscous_friction_coeff = self.parameter_pick(
                q_dot[i], fw_viscous[i], bw_viscous[i])
            
            fRIC[i] = static_friction_coeff * cs.sign(q_dot[i]) + viscous_friction_coeff*q_dot[i]

            # Include friction in the motor torque calculation
            tau_motor[i] = (tau_gear[i]/G[i]) + Si[i].T@fR[i] + fRIC[i]
            # print(i)
            if i != 0:
                p_X_i_f = plucker.inverse_spatial_transform(i_X_p[i]).T
                f[i-1] = f[i-1] + p_X_i_f@f[i] + p_X_i_f@fR[i]

                f_withoutR[i-1] = f_withoutR[i-1] + p_X_i_f@f_withoutR[i]

                self.Ic_i[i-1] = self.Ic_i[i-1] + p_X_i_f@self.Ic_i[i]@i_X_p[i]
            else:
                i_x_0b = plucker.spatial_mtimes(i_X_p[i],self.T_Base) # validate
                p_X_i_f = plucker.inverse_spatial_transform(i_x_0b).T
                if floating_base_id!=None:
                    print('floating_base found')
                    if coupled:
                        F_base = floating_base_id + p_X_i_f@f[i] + p_X_i_f@fR[i]
                        # F_base = floating_base_id + p_X_i_f@f_withoutR[i]
                    else:
                        F_base = floating_base_id
                else:
                    print('no floating_base found')
                    F_base = p_X_i_f@f[i] + p_X_i_f@fR[i]
        
        F_base = cs.substitute(F_base, self.arm_ssyms.trivial_sim_p, cs.DM([0]*self.arm_ssyms.trivial_sim_p.size1()))
        tau_motor = cs.substitute(tau_motor, self.arm_ssyms.trivial_sim_p, cs.DM([0]*self.arm_ssyms.trivial_sim_p.size1()))
        
        return [f, fR, tau_gear, tau_motor, Si, i_X_p, i_X_0s, v, a, Ic, F_base, fRIC]

    def parameter_pick(self, q_dot_i, forward_param_i, backward_param_i):
        return cs.if_else(cs.sign(q_dot_i) == 1, forward_param_i, backward_param_i)

    def _apply_external_forces(self, _external_f, _fb, _i_X_0):
        """Internal function for applying external forces in dynamics
        algorithms calculations. Assumes external force act at the base origin"""
        f_i_X_0 = _i_X_0.T
        _fb -= f_i_X_0@_external_f
        return _fb

    # def _apply_body_link_hydrodynamics(self, i, _fb, v, v_dot, Icom, Im, _i_X_0, gravity):
    #     # i is the link index
    #     # v is the link body velocity vector of size 6
    #     # v_dot is the link body acceleration vector of size 6

    #     # for bodies operating at water depths below the wave-affected zone, the hydrodynamic
    #     # coefficients will be independent of the wave excitation frequency.
    #     # Consequently,  only one frequency is needed to obtain an estimate of the added mass matrix.
    #     # In addition, there will be no potential damping. However,viscous damping Bv(w) will be present.

    #     # hydrodynamic equation of motion of link about joint frame using the plücker basis
    #     # HYDRODYNAMIC PARAMTERTIZATIONS
    #     M_A_diag_coeff = self.arm_ssyms.M_A_coef[i]  # 6 by 1
    #     M_A = -cs.diag(cs.vertcat(M_A_diag_coeff))
    #     C_A = plucker.hydrod_coriolis_lag_param(M_A, v[i])

    #     # Compute the total damping forces, including both linear and nonlinear components in body
    #     linear_damping = -cs.diag(cs.vertcat(self.arm_ssyms.D_u[i]))
    #     nonlinear_damping = - cs.diag(cs.vertcat(self.arm_ssyms.D_uu[i])*cs.fabs(v[i]))  # Quadratic
    #     D_v = linear_damping + nonlinear_damping

    #     # restoring forces
    #     fng = -cs.vertcat(0, 0, Im[i]*gravity)
    #     fnb = self.arm_ssyms.rho*gravity * \
    #         cs.vertcat(0, 0, self.arm_ssyms.link_Volume[i])
    #     R_O, _, rx = plucker.extractEr(_i_X_0)
    #     r_bb = cs.inv_skew(rx) + self.arm_ssyms.cob[i]
    #     r_bg = cs.inv_skew(rx) + Icom[i]
    #     r_bb_x = cs.skew(r_bb)
    #     r_bg_x = cs.skew(r_bg)

    #     g_n = cs.vertcat((r_bg_x@fng + r_bb_x@fnb), (fng + fnb))

    #     f_i_X_0 = _i_X_0.T
    #     # apply moments and force due to hydrodynamics to rigid body link
    #     spatial_restoring_force = f_i_X_0@g_n
    #     _fb -= M_A@v_dot[i] + C_A@v[i] + D_v@v[i] + spatial_restoring_force
    #     return _fb

    def forward_dynamics(self, gravity=9.81, floating_base_id=None, floating_base_bias_f=None, J_uv=None, coupled=True):
        # uvms bias force and Inertia
        C = self.get_bias_force(gravity, floating_base_bias_f, coupled)
        H = self.get_inertia_matrix(gravity, floating_base_id, floating_base_bias_f, coupled)
        
        if C.size1() != H.size1() and H.size1() != H.size2():
            raise Exception("C and H matrices shape mismatch. Possibly not passing necessary floating base arguments")

        if C.size1() > self.arm_ssyms.q_dot.size1():
            if J_uv == None:
                raise Exception("provide uv position transformation matrix (NED)")
    
            xd = cs.vertcat(J_uv@self.fb_ssyms.v_uv ,self.arm_ssyms.q_dot)
            u = cs.vertcat(self.fb_ssyms.uv_u, self.arm_ssyms.m_u)
            base_T = self.fb_ssyms.base_T
            parameters = cs.vertcat(self.arm_ssyms.sim_p,self.fb_ssyms.sim_p, base_T)
            states = self.sys_syms.uvms_states
            ode_xdd = cs.solve(H, u - C)

        elif C.size1() == self.arm_ssyms.q_dot.size1():
            xd = self.arm_ssyms.q_dot
            u = self.arm_ssyms.m_u
            base_T = None
            parameters = self.arm_ssyms.sim_p
            states = self.arm_ssyms.m_states
            ode_xdd = cs.solve(H, u - C)

        rhs = cs.vertcat(xd, ode_xdd)  # the complete ODE vector

        # integrator to discretize the system
        sys = {}
        sys['x'] = states
        sys['u'] = u
        sys['p'] = cs.vertcat(parameters, self.sys_syms.dt)
        sys['ode'] = rhs*self.sys_syms.dt  # Time scaling

        intg = cs.integrator('intg', 'rk', sys, 0, 1, {
                            'simplify': True, 'number_of_finite_elements': 30})
        
        u_checks = copy.deepcopy(u)
        states_checks = copy.deepcopy(states)

        if u.size1() == self.arm_ssyms.q_min.size1():
            for j in range(self.arm_ssyms.n_joints):
                u_checks[j] = cs.if_else(
                    cs.logic_or(
                        # If states[j] is already too low AND commanded force is negative ...
                        cs.logic_and(states[j] <= self.arm_ssyms.q_min[j], u[j]<0 ),
                        # ... OR if states[j] is already too high AND commanded force is positive
                        cs.logic_and(states[j] >= self.arm_ssyms.q_max[j], u[j]>0 )
                        ),
                        0, # clamp to zero if either of the above conditions hold
                        u[j] # otherwise leave as is
                        )

                states_checks[j+4] = cs.if_else(
                    cs.logic_or(
                        cs.logic_and(states[j] <= self.arm_ssyms.q_min[j], states[j+4]<0), 
                        cs.logic_and(states[j] >= self.arm_ssyms.q_max[j], states[j+4]>0)
                        ),
                        0,
                        states[j+4]
                        )
            # states_checks[0:4] = cs.fmin(cs.fmax(states[0:4], self.arm_ssyms.u_min), self.arm_ssyms.u_max)
            states_checks[0:4] = cs.fmin(cs.fmax(states[0:4], self.arm_ssyms.q_min), self.arm_ssyms.q_max)
        else:
            for j in range(self.arm_ssyms.n_joints):
                u_checks[6+j] = cs.if_else(
                    cs.logic_or(
                        cs.logic_and(states[6+j] <= self.arm_ssyms.q_min[j], u[6+j]<0 ),
                        cs.logic_and(states[6+j] >= self.arm_ssyms.q_max[j], u[6+j]>0 )
                        ),
                        0,
                        u[6+j]
                        )

                states_checks[j+16] = cs.if_else(
                    cs.logic_or(
                        cs.logic_and(states[6+j] <= self.arm_ssyms.q_min[j], states[j+16]<0),
                        cs.logic_and(states[6+j] >= self.arm_ssyms.q_max[j], states[j+16]>0)
                        ),
                        0, 
                        states[j+16]
                        )

            # u_checks = cs.fmin(cs.fmax(u, self.arm_ssyms.u_min), self.arm_ssyms.u_max)
            states_checks[6:10] = cs.fmin(cs.fmax(states[6:10], self.arm_ssyms.q_min), self.arm_ssyms.q_max)

        res = intg(x0=states_checks, u=u_checks, p=cs.vertcat(parameters, self.sys_syms.dt))  # evaluate with symbols
        x_next = res['xf']

        return x_next, states, u, self.sys_syms.dt, self.arm_ssyms.q_min, self.arm_ssyms.q_max, self.arm_ssyms.sim_p, self.fb_ssyms.sim_p, base_T, states_checks, u_checks
