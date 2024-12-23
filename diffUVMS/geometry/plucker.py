# import casadi as cs
import diffUVMS.geometry.transformation_matrix as tm
import numpy as np
from casadi import cos, SX, sin, skew, tan, inv, vertcat, horzcat, diag, inv_skew

# Returns Skew symmetric matrix defined by 3 val vector v. 
def cross_pO(v):
    S = skew(v)
    return S

# hydrodynamics coriolis_lagrange_parameterization for links
def hydrod_coriolis_lag_param(M, x_nb):
    # Decompose into angular and translational vel.
    w_nb, v_nb = x_nb[:3], x_nb[3:]
    A11 = M[:3, :3]
    A12 = M[:3, 3:] 
    A21 = M[3:, :3]
    A22 = M[3:, 3:]

    # Create coriolis matrix. Based on Eq. 6.44. 
    C_A = SX.zeros(6, 6)
    C_A[:3, :3] = -cross_pO(A11@w_nb + A12@v_nb)
    C_A[3:, :3] = -cross_pO(A21@w_nb + A22@v_nb)
    C_A[3:, 3:] = -cross_pO(A11@w_nb + A12@v_nb)
    return C_A

def motion_cross_product(v):
    """Returns the motion cross product matrix of a spatial vector."""
    mcp = SX.zeros(6, 6)
    w_x = skew(v[:3])
    vo_x = skew(v[3:])

    mcp[:3,:3] = w_x
    mcp[3:,3:] = w_x
    mcp[3:,:3] = vo_x
    return mcp


def force_cross_product(v):
    """Returns the force cross product matrix of a spatial vector."""
    return -motion_cross_product(v).T


def spatial_inertia_matrix_IO_sym(ixx, ixy, ixz, iyy, iyz, izz, mass, c):
    """Returns the 6x6 spatial inertia matrix expressed at the origin in symbolic representation"""
    IO_sym = SX.zeros(6, 6)

    _Ic = SX(3,3)
    _Ic[0, :] = horzcat(ixx, ixy, ixz)
    _Ic[1, :] = horzcat(ixy, iyy, iyz)
    _Ic[2, :] = horzcat(ixz, iyz, izz)

    _m1 = diag(vertcat(mass, mass, mass))

    c_sk = skew(c)
    IO_sym[:3,:3] = _Ic + (mass*(c_sk@c_sk.T))
    IO_sym[0,1] = -IO_sym[0,1]
    IO_sym[1,0] = -IO_sym[1,0]

    IO_sym[0,2] = -IO_sym[0,2]
    IO_sym[2,0] = -IO_sym[2,0]

    IO_sym[1,2] = -IO_sym[1,2]
    IO_sym[2,1] = -IO_sym[2,1]

    IO_sym[3:,3:] = _m1
    IO_sym[:3,3:] = mass*c_sk
    IO_sym[3:,:3] = mass*c_sk.T

    return IO_sym


# def spatial_force_transform(R, r):
#     """Returns the spatial force transform from a 3x3 rotation matrix
#     and a 3x1 displacement vector."""
#     X = cs.SX.zeros(6, 6)
#     X[:3, :3] = R.T
#     X[3:, 3:] = R.T
#     X[:3, 3:] = cs.mtimes(cs.skew(r), R.T)
#     return X

# def spatial_transform_BA(R, r):
#     """Returns the inverse spatial motion transform from a 3x3 rotation
#     matrix and a 3x1 displacement vector."""
#     X = cs.SX.zeros(6, 6)
#     X[:3, :3] = R.T
#     X[3:, 3:] = R.T
#     X[3:, :3] = cs.mtimes(cs.skew(r), R.T)
#     return X


# def XJT_revolute_BA(xyz, rpy, axis, qi):
#     """Returns the spatial transform from parent link to child link with
#     a revolute connecting joint."""
#     T = tm.revolute(xyz, rpy, axis, qi)
#     rotation_matrix = T[:3, :3]
#     displacement = T[:3, 3]
#     return spatial_transform_BA(rotation_matrix, displacement)


# def XJT_prismatic_BA(xyz, rpy, axis, qi):
#     """Returns the spatial transform from parent link to child link with
#     a prismatic connecting joint."""
#     T = tm.prismatic(xyz, rpy, axis, qi)
#     rotation_matrix = T[:3, :3]
#     displacement = T[:3, 3]
#     return spatial_transform_BA(rotation_matrix, displacement)


def XJT_revolute(xyz, rpy, axis, qi):
    """Returns the spatial transform from child link to parent link with
    a revolute connecting joint."""
    T = tm.revolute(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return spatial_transform(rotation_matrix, displacement)


def XJT_prismatic(xyz, rpy, axis, qi):
    """Returns the spatial transform from child link to parent link with
    a prismatic connecting joint."""
    T = tm.prismatic(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    # print(f"{rotation_matrix} XJT CHECKER")
    # print(f"{rotation_matrix.T} XJT CHECKER")
    return spatial_transform(rotation_matrix, displacement)

def XT(xyz, rpy):
    """Returns a general spatial transformation matrix matrix"""
    rotation_matrix = rotation_rpy(rpy[0], rpy[1], rpy[2])
    # print(f"{rotation_matrix} XJT CHECKER")
    # print(f"{rotation_matrix.T} XJT CHECKER")
    return spatial_transform(rotation_matrix, xyz)

def rotation_rpy(roll, pitch, yaw):
    R = SX(3, 3)

    R[0,0] = cos(yaw)*cos(pitch)
    R[0,1] = -sin(yaw)*cos(roll) + cos(yaw)*sin(pitch)*sin(roll)
    R[0,2] = sin(yaw)*sin(roll) + cos(yaw)*cos(roll)*sin(pitch)

    R[1,0] = sin(yaw)*cos(pitch)
    R[1,1] = cos(yaw)*cos(roll) + sin(roll)*sin(pitch)*sin(yaw)
    R[1,2] = -cos(yaw)*sin(roll) + sin(pitch)*sin(yaw)*cos(roll)

    R[2,0] = -sin(pitch)
    R[2,1] = cos(pitch)*sin(roll)
    R[2,2] = cos(pitch)*cos(roll)
    return R

def spatial_transform(R, r):
    """Returns the spatial motion transform from a 3x3 rotation matrix
    and a 3x1 displacement vector."""
    X = SX.zeros(6, 6)
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = -R@skew(r)
    return X

def spatial_to_homogeneous(X):
    R, _, rx = extractEr(X)
    p = inv_skew(rx)
    T = SX.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T, R, p

def spatial_mtimes(X_i, X_O):
    T_ixp, _, _ = spatial_to_homogeneous(X_i)
    T_pxO, _, _ = spatial_to_homogeneous(X_O)
    T_ix0 = T_pxO@T_ixp
    R = T_ix0[:3, :3]
    r = T_ix0[:3, 3]
    return spatial_transform(R,r)

def extractEr(i_X_p):
    "returns E and rx"
    E0 = i_X_p[:3,:3]
    E4 = i_X_p[3:,3:] #E0 == E4
    _Er_x = i_X_p[3:,:3]
    rx = -E0.T@_Er_x
    return E0, E4, rx


def inverse_spatial_transform(i_X_p):
    "Returns p_X_i_"
    E0 = i_X_p[:3,:3]
    E4 = i_X_p[3:,3:] #E0 == E4
    Er_x = -i_X_p[3:,:3]

    p_X_i = SX.zeros(6, 6)
    E_T = E0.T
    p_X_i[:3,:3] = E_T
    p_X_i[3:,3:] = E_T

    r_x_E_T = Er_x.T
    p_X_i[3:,:3] = r_x_E_T
    return p_X_i

    



