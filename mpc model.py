from msilib.schema import Upgrade
from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi

# weights for control and state variables
Q_x = 9
Q_y = 7
Q_theta = 1
R1 = 0.05
R2 = 0.05
R3 = 0.05
R4 = 0.05

step_horizon = 0.2  # time between steps in seconds
N = 20              # number of look ahead steps
rob_diam = 0.3      # diameter of the robot
wheel_radius = 0.1    # wheel radius
Lx = 0.2            # L in J Matrix (half robot x-axis length)
Ly = 0.1            # l in J Matrix (half robot y-axis length)
sim_time = 200      # simulation time (will be removed in practical application)

# initial and target position
x_init = 0
y_init = 0
theta_init = 0
x_target = 15
y_target = 10
theta_target = pi/4

# omega constraints
omega_max = pi
omega_min = -pi

# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(
    x,
    y,
    theta
)
n_states = states.numel()

# control symbolic variables
omega_w1 = ca.SX.sym('omega_w1')
omega_w2 = ca.SX.sym('omega_w2')
omega_w3 = ca.SX.sym('omega_w3')
omega_w4 = ca.SX.sym('omega_w4')
controls = ca.vertcat(
    omega_w1,
    omega_w2,
    omega_w3,
    omega_w4
)
n_controls = controls.numel()

# discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
rot_3d_z = ca.vertcat(
    ca.horzcat(cos(theta), -sin(theta), 0),
    ca.horzcat(sin(theta),  cos(theta), 0),
    ca.horzcat(         0,           0, 1)
)

# j0 plus
J = (wheel_radius/4) * ca.DM([
    [         1,         1,          1,         1],
    [        -1,         1,          1,        -1],
    [-1/(Lx+Ly), 1/(Lx+Ly), -1/(Lx+Ly), 1/(Lx+Ly)]
])

# RHS = states + J @ controls * step_horizon  # Euler discretization
RHS = rot_3d_z @ J @ controls

# maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = ca.Function('f', [states, controls], [RHS])

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)

# controls weights matrix
R = ca.diagcat(R1, R2, R3, R4)

cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation


# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)

    OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# lower and upper bounds for state and control (optimization variables)
lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

# lower and upper bound for constraint
lbg = ca.DM.zeros((n_states*(N+1), 1))
ubg = ca.DM.zeros((n_states*(N+1), 1))

lbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf      # X upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

lbx[n_states*(N+1):] = omega_min
lbx[n_states*(N+1)+1:] = omega_min
lbx[n_states*(N+1)+2:] = omega_min

ubx[n_states*(N+1):] = omega_max
ubx[n_states*(N+1)+1:] = omega_max
ubx[n_states*(N+1)+2:] = omega_max

args = {
    'lbg': lbg,
    'ubg': ubg,
    'lbx': lbx,
    'ubx': ubx
}


t0 = 0 #current time based on time steps
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state


t = ca.DM(t0) # keeps track of time step horizons

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full
