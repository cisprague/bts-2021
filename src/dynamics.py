import jax.numpy as np
from functools import partial

class Dynamics:

    def __init__(self, dim_state, dim_control, bounds_state, bounds_control):

        # dimensions
        self.dim_state = dim_state
        self.dim_control = dim_control

        # bounds
        self.bounds_state = bounds_state
        self.bounds_control = bounds_control

    @jit
    @staticmethod
    def skew(vec):
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])

    @jit
    @staticmethod
    def unskew(mat):
        return np.array([
            mat[2,1],
            mat[0,2],
            mat[1,0]
        ])

    @jit
    @staticmethod
    def R_b2ned(phi, theta, psi):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        R = Rz@Ry@Rx
        return R

class Fossen(Dynamics):

    def __init__(self):

        # dimensions
        dim_state = 12
        dim_control = 6

        # bounds
        bounds_state = np.array([
            20, 20, 20, 
            2*np.pi, 2*np.pi, 2*np.pi, 
            10, 10, 10, 
            10, 10, 10
        ])
        bounds_state = (
            -bounds_state, 
            bounds_state
        )
        bounds_control = (
            np.full(dim_control, -1.0),
            np.full(dim_control, 1.0)
        )

        # inherit
        Dynamics.__init__(self, dim_state, dim_control, bounds_state, bounds_control)

    @partial(jit, static_argums=(0,))
    def f(self, state, control):

        # state and control
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state
        rpm1, rpm2, de, dr, lcg, vbs = control

        # position and velocity
        eta = np.array([x, y, z, phi, theta, psi])
        nu = np.array([u, v, w, p, q, r])

        # scaled controls
        rpm1 *= 1000.0
        rpm2 *= 1000.0
        de *= 0.05
        dr *= 0.05
        # vbs *= 1.0
        # lcg *= 1.0

        # mass and inertia matrix
        m = 14.0
        I_o = np.diag(np.array([0.0294, 1.6202, 1.6202]))

        # centre of gravity, buoyancy, and pressure positions, resp.
        r_g = np.array([0.1 + lcg*0.01, 0.0, 0.0])
        r_b = np.array([0.1, 0.0, 0.0])
        r_cp = np.array([0.1, 0.0, 0.0])

        # <insert title>
        W = m*9.81
        B = W + vbs*1.5

        # hydrodynamic coefficients
        Xuu = 5. #3. #1.0
        Yvv = 20. #10. #100.0
        Zww = 50. #100.0
        Kpp = 0.1 #10.0
        Mqq = 20.#40 #100.0
        Nrr = 20. #150.0

        # control actuators
        K_T = np.array([0.0175, 0.0175])
        Q_T = np.array([0.001, -0.001])#*0.0

        # mass and inertia matrix
        M = np.block([
        [m*np.eye(3,3), -m*skew(r_g)],
        [m*skew(r_g), I_o]
        ])
        assert M.shape == (6,6), M

        # coriolis and centripetal matrix
        nu1 = np.array([u, v, w])
        nu2 = np.array([p, q, r])
        top_right = -m*skew(nu1) - m*skew(nu2)*skew(r_g)
        bottom_left = -m*skew(nu1) + m*skew(r_g)*skew(nu2)
        bottom_right = -skew(I_o.dot(nu2))
        C_RB = np.block([
        [np.zeros((3,3)), top_right],
        [bottom_left, bottom_right]
        ])
        assert C_RB.shape == (6, 6), C_RB

        # damping matrix
        forces = np.diag(np.array([Xuu*np.abs(u), Yvv*np.abs(v), Zww*np.abs(w)]))
        moments = np.diag(np.array([Kpp*np.abs(p), Mqq*np.abs(q), Nrr*np.abs(r)]))
        coupling = np.matmul(skew(r_cp), forces)
        D = np.block([[forces, np.zeros((3, 3))], [-coupling, moments]])
        assert D.shape == (6, 6), D

        # rotational transform between body and NED in Euler        
        T_euler = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)],
        ])
        R_euler = np.array([
        [
            np.cos(psi)*np.cos(theta),
            -np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(theta)*np.sin(phi),
            np.sin(psi)*np.sin(phi)+np.cos(psi)*np.cos(phi)*np.sin(theta)
        ],
        [
            np.sin(psi)*np.cos(theta),
            np.cos(psi)*np.cos(phi)+np.sin(phi)*np.sin(theta)*np.sin(psi),
            -np.cos(psi)*np.sin(phi)+np.sin(theta)*np.sin(psi)*np.cos(phi),
        ],
        [
            -np.sin(theta),
            np.cos(theta)*np.sin(phi),
            np.cos(theta)*np.cos(phi),
        ],
        ])
        assert R_euler.shape == (3,3), R_euler
        J_eta = np.block([
        [R_euler, np.zeros((3,3))],
        [np.zeros((3,3)), T_euler]
        ])
        assert J_eta.shape == (6,6), J_eta

        # buoyancy in quaternions
        f_g = np.array([0, 0, W])
        f_b = np.array([0, 0, -B])
        row1 = np.linalg.inv(R_euler).dot(f_g + f_b)
        row2 = skew(r_g).dot(np.linalg.inv(R_euler)).dot(f_g) + \
        skew(r_b).dot(np.linalg.inv(R_euler)).dot(f_b)
        geta = np.block([row1, row2])
        assert geta.shape == (6,), geta

        # <insert title>
        F_T = K_T.dot(np.array([rpm1, rpm2]))
        M_T = Q_T.dot(np.array([rpm1, rpm2]))
        tauc = np.array([
        F_T*np.cos(de)*np.cos(dr),
        -F_T*np.sin(dr),
        F_T*np.sin(de)*np.cos(dr),
        M_T*np.cos(de)*np.cos(dr),
        -M_T*np.sin(dr),
        M_T*np.sin(de)*np.cos(dr)
        ])
        assert tauc.shape == (6,), tauc

        # velocity and acceleration 
        etadot = np.block([J_eta.dot(nu)])
        assert etadot.shape == (6,)
        nudot = np.linalg.inv(M).dot(tauc - (C_RB + D).dot(nu - geta))
        assert nudot.shape == (6,)

        # state-space
        sdot = np.block([etadot, nudot])
        return sdot