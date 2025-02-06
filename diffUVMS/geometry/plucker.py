# import casadi as cs
import diffUVMS.geometry.transformation_matrix as tm
import numpy as np
from casadi import cos, SX, sin, skew, tan, inv, vertcat, horzcat, diag, inv_skew, if_else, fmod, pi, asin, sqrt, atan2, acos

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

# def inverse_spatial_transform_new(X):
#     """
#     Given a 6x6 rigid-body motion transform X = i_X_p,
#     return its inverse p_X_i.
#     """
#     # Rotation:
#     R = X[:3, :3]
#     R_T = R.T

#     # 'Bottom-left' 3x3 block, often called M or S:
#     M = X[3:, :3]

#     X_inv = SX.zeros(6, 6)
#     X_inv[:3, :3]   = R_T
#     X_inv[3:, 3:]   = R_T
#     X_inv[3:, :3]   = - R_T @ M

#     return X_inv


# def inverse_spatial_transform(i_X_p):
#     "Returns p_X_i_"
#     E0 = i_X_p[:3,:3]
#     E4 = i_X_p[3:,3:] #E0 == E4
#     Er_x = -i_X_p[3:,:3]

#     p_X_i = SX.zeros(6, 6)
#     E_T = E0.T
#     p_X_i[:3,:3] = E_T
#     p_X_i[3:,3:] = E_T

#     r_x_E_T = Er_x.T
#     p_X_i[3:,:3] = r_x_E_T
#     return p_X_i

    

def rotation_matrix_to_euler(R, order='zyx'):
    """
    Convert a rotation matrix to Euler angles using CasADi symbolic expressions.

    Parameters:
    R (SX or MX): 3x3 rotation matrix
    order (str): Order of Euler angles axes. 
                 Common orders include 'zyx', 'xyz', 'zyz', etc.

    Returns:
    euler (SX or MX): 3x1 vector of Euler angles in radians
    """
    # Validate the order
    # Define supported orders
    supported_orders = [
        'zyx', 'xyz', 'zyz', 'xzx', 'yxy', 'yzy',
        'zxy', 'yxz', 'yzx', 'xyx', 'xzy', 'zyx'
    ]
    if order not in supported_orders:
        raise ValueError(f"Unsupported Euler order '{order}'. Supported orders are {supported_orders}.")

    # Ensure R is a 3x3 matrix
    assert R.shape == (3, 3), "Rotation matrix must be 3x3."

    # Helper function to compute Euler angles for 'zyx' order
    if order == 'zyx':
        # yaw (psi)   : rotation about z-axis
        # pitch (theta): rotation about y-axis
        # roll (phi)  : rotation about x-axis

        # Compute pitch
        pitch = asin(-R[2, 0])

        # To handle the singularity when cos(pitch) is close to zero
        cos_pitch = sqrt(1 - R[2, 0]**2)

        # Define a small threshold to detect singularity
        epsilon = 1e-6

        # Compute yaw and roll
        yaw = if_else(cos_pitch > epsilon,
                        atan2(R[1, 0], R[0, 0]),
                        0)  # When cos(pitch) is near zero, set yaw to zero

        roll = if_else(cos_pitch > epsilon,
                         atan2(R[2, 1], R[2, 2]),
                         atan2(-R[1, 2], R[1, 1]))

        euler = vertcat(roll, pitch, yaw)

    elif order == 'xyz':
        # Compute pitch
        pitch = asin(R[0, 2])

        # Handle singularity
        cos_pitch = sqrt(1 - R[0, 2]**2)
        epsilon = 1e-6

        roll = if_else(cos_pitch > epsilon,
                         atan2(-R[1, 2], R[2, 2]),
                         0)

        yaw = if_else(cos_pitch > epsilon,
                        atan2(-R[0, 1], R[0, 0]),
                        atan2(R[1, 0], R[1, 1]))

        euler = vertcat(roll, pitch, yaw)

    elif order == 'zyz':
        # Compute theta
        theta = acos(R[2, 2])

        # Handle singularity when theta is 0 or pi
        sin_theta = sin(theta)
        epsilon = 1e-6

        phi = if_else(sin_theta > epsilon,
                        atan2(R[2, 0], R[2, 1]),
                        0)

        psi = if_else(sin_theta > epsilon,
                        atan2(R[0, 2], -R[1, 2]),
                        atan2(-R[1, 0], R[0, 0]))

        euler = vertcat(phi, theta, psi)

    else:
        raise NotImplementedError(f"Euler order '{order}' is not implemented yet.")

    # Normalize angles to be within [-pi, pi]
    euler = fmod(euler + pi, 2 * pi) - pi

    return euler
