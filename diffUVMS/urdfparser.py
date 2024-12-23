"""This module contains a class for turning a chain in a URDF to a
casadi function.
"""
import casadi as cs
import numpy as np
from platform import machine, system
from urdf_parser_py.urdf import URDF, Pose
import copy


class URDFparser():
    def __init__(self, func_opts=None, use_jit=True):
        """Class that turns a chain from URDF to casadi functions."""
        self.actuated_types = ["prismatic", "revolute", "continuous"]
        self.func_opts = {}
        jit_func_opts = {"jit": True, "jit_options": {"flags": "-Ofast"}}
        # OS/CPU dependent specification of compiler
        if system().lower() in ["darwin", "aarch64" ,"linux"]:
            jit_func_opts["compiler"] = "shell"

        self.robot_desc = None
        self.robot_desc_backup = None
        if func_opts:
            self.func_opts = func_opts
        if use_jit:
            # NOTE: use_jit=True requires that CasADi is built with Clang
            for k, v in jit_func_opts.items():
                self.func_opts[k] = v
        print(f'jit after {self.func_opts}')

    def from_file(self, filename):
        """Uses an URDF file to get robot description."""
        self.robot_desc = URDF.from_xml_file(filename)
        self.robot_desc_backup = copy.deepcopy(self.robot_desc)

    def from_server(self, key="robot_description"):
        """Uses a parameter server to get robot description."""
        self.robot_desc = URDF.from_parameter_server(key=key)
        self.robot_desc_backup = copy.deepcopy(self.robot_desc)

    def from_string(self, urdfstring):
        """Uses a URDF string to get robot description."""
        self.robot_desc = URDF.from_xml_string(urdfstring)
        self.robot_desc_backup = copy.deepcopy(self.robot_desc)

    def get_joint_info(self, root, tip):
        """Using an URDF to extract joint information, i.e list of
        joints, actuated names and upper and lower limits."""
        chain = self.robot_desc.get_chain(root, tip)
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        joint_list = []
        upper = []
        lower = []
        actuated_names = []

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                joint_list += [joint]
                if joint.type in self.actuated_types:
                    actuated_names += [joint.name]
                    if joint.type == "continuous":
                        upper += [cs.inf]
                        lower += [-cs.inf]
                    else:
                        upper += [joint.limit.upper]
                        lower += [joint.limit.lower]
                    if joint.axis is None:
                        joint.axis = [1., 0., 0.]
                    if joint.origin is None:
                        joint.origin = Pose(xyz=[0., 0., 0.],
                                            rpy=[0., 0., 0.])
                    elif joint.origin.xyz is None:
                        joint.origin.xyz = [0., 0., 0.]
                    elif joint.origin.rpy is None:
                        joint.origin.rpy = [0., 0., 0.]

        return joint_list, actuated_names, upper, lower

    def get_dynamics_limits(self, root, tip):
        """Using an URDF to extract joint max effort and velocity"""

        chain = self.robot_desc.get_chain(root, tip)
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        max_effort = []
        max_velocity = []

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type in self.actuated_types:
                    if joint.limit is None:
                        max_effort += [cs.inf]
                        max_velocity += [cs.inf]
                    else:
                        max_effort += [joint.limit.effort]
                        max_velocity += [joint.limit.velocity]
        max_effort = [cs.inf if x is None else x for x in max_effort]
        max_velocity = [cs.inf if x is None else x for x in max_velocity]

        return max_effort, max_velocity

    def get_friction_matrices(self, root, tip):
        """Using an URDF to extract joint frictions and dampings"""

        chain = self.robot_desc.get_chain(root, tip)
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        friction = []
        damping = []

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type in self.actuated_types:
                    if joint.dynamics is None:
                        friction += [0]
                        damping += [0]
                    else:
                        friction += [joint.dynamics.friction]
                        damping += [joint.dynamics.damping]
        friction = [0 if x is None else x for x in friction]
        damping = [0 if x is None else x for x in damping]
        Fv = np.diag(friction)
        Fd = np.diag(damping)
        return Fv, Fd

    def get_n_joints(self, root, tip):
        """Returns number of actuated joints."""

        chain = self.robot_desc.get_chain(root, tip)
        n_actuated = 0

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type in self.actuated_types:
                    n_actuated += 1

        return n_actuated