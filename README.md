# Differentiable Dynamics for Underwater Vehicle and Manipulator System 🦾🌊
A library for generating the kinematics and dynamics of electric underwater robotic arm under a moving base.
<!-- ![alt text]() -->
<img src="./resources/uvman.jpg" width="420"/>

## Todo/Implementation status
- [x] [whole-body forward dynamics](https://github.com/edxmorgan/diff_uvms/blob/main/usage/example/uvms_forward_dynamics.ipynb)
- [x] [whole-body inverse dynamics](https://github.com/edxmorgan/diff_uvms/blob/main/usage/example/uvms_inverse_dynamics.ipynb)
- [x] [whole-body forward kinematics](https://github.com/edxmorgan/diff_uvms/blob/main/usage/example/uvms_forward_kinematics.ipynb)
- [x] [whole-body differential inverse kinematics](https://github.com/edxmorgan/diff_uvms/blob/main/usage/example/redundancy_nullspace_projection_diK.ipynb)
- [x] [model identification regressor](https://github.com/edxmorgan/diff_uvms/blob/main/usage/example/uvms_for_identification.ipynb)
- [x] [whole-body nonlinear PID controller](https://github.com/edxmorgan/diff_uvms/blob/main/usage/example/uvms_pid.ipynb)

For usage examples of Diff_UVMS, see [Jupyter notebook](https://github.com/edxmorgan/Diff_UVMS/tree/main/usage/example).

Dynamics of moving base (underwater vehicle) used in [UVMS foward dynamics example](https://github.com/edxmorgan/diff_uvms/blob/main/usage/example/uvms_forward_dynamics.ipynb) are derived from the [Diff_UV (Differentiable Underwater Vehicle System)](https://github.com/edxmorgan/Diff_UV) project.


## References
Roy Featherstone and Kluwer Academic Publishers. 1987. Robot Dynamics Algorithm. Kluwer Academic Publishers, USA. http://dx.doi.org/10.1007/978-1-4899-7560-7

Fossen, T.I. (2011) Handbook of Marine Craft Hydrodynamics and Motion Control. John Wiley & Sons, Inc., Chichester, UK. https://doi.org/10.1002/9781119994138

## Caution ⚠️  
This project is still experimental. While the core functionalities have been implemented and tested to some extent, further validation and testing are required. Use with care, especially for safety-critical applications. Contributions and feedback are welcome!  
