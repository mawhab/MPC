from cmath import sqrt
from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
from simulation_code import simulate

# q and r weights ffor state and controls

Q_X = 500
Q_Y = 500
Q_theta = 2000

R_omega1 = 25
R_omega2 = 25
R_omega3 = 25
R_omega4 = 25

step_horizon = 0.1  # time between steps in seconds
N = 20              # number of look ahead steps
rob_diam = 0.3      # diameter of the robot
wheel_radius = 1    # wheel radius
Lx = 0.3              # L in J Matrix (half robot x-axis length)
Ly = 0.3            # l in J Matrix (half robot y-axis length)
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
omega_min = -omega_max

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
Q = ca.diagcat(Q_X, Q_Y, Q_theta)

# controls weights matrix
R = ca.diagcat(R_omega1, R_omega2, R_omega3, R_omega4)

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
# obstacle center coordinates and diameter
obs_x = 7.5
obs_y = 4.5
obs_diam = 0

# adding obstacle inequality to g
for k in range(N+1):
    inequality = ca.SX((rob_diam/2 + obs_diam/2) - ((X[0,k] - obs_x)**2 + (X[1,k] - obs_y)**2))**1/2
    g = ca.vertcat(
        g,
        inequality
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

# lower and upper bound for constraint; they are zero because we have an equality constraint of x(k+1) - x(K+1)rk = 0
lbg = ca.DM.zeros(((n_states+1)*(N+1), 1))
ubg = ca.DM.zeros(((n_states+1)*(N+1), 1))


# adding obstacle constraints
lbg[n_states*(N+1):, :] = -np.inf
ubg[n_states*(N+1):, :] = 0

# INDEXING:
# we place x, y and theta bounds once every n_states steps so that the matrix has x, y and theta repeated N+1 times (1 set for each time step)
# e.g. N = 5 
# [x,y,theta,x,y,theta,x,y,theta,x,y,theta,x,y,theta,x,y,theta]

lbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf      # X upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound



# INDEXING:
# we place omega bounds once every n_controls steps so that the matrix has omega 1-4 repeated N times (1 set for each time step)
# difference here is that controls start from the end of states; states end at n_states*(N+1), controls start from there and end at n_states*(N+1)+n_controls*N
# states are N+1 while controls are N because u can get one extra state from controls but not vice versa, so there is always 1 more set of states than controls in multiple shooting
# e.g. N = 5 
# [x,y,theta,x,y,theta,x,y,theta,x,y,theta,x,y,theta,x,y,theta,w1,w2,w3,w4,w1,w2,w3,w4,w1,w2,w3,w4,w1,w2,w3,w4,w1,w2,w3,w4]

lbx[n_states*(N+1):n_states*(N+1)+n_controls*N:n_controls] = omega_min    # Omega 1 lower bound
lbx[n_states*(N+1)+1:n_states*(N+1)+n_controls*N:n_controls] = omega_min  # Omega 2 lower bound 
lbx[n_states*(N+1)+2:n_states*(N+1)+n_controls*N:n_controls] = omega_min  # Omega 3 lower bound 
lbx[n_states*(N+1)+3:n_states*(N+1)+n_controls*N:n_controls] = omega_min  # Omega 4 lower bound 

ubx[n_states*(N+1):n_states*(N+1)+n_controls*N:n_controls] = omega_max    # Omega 1 upper bound
ubx[n_states*(N+1)+1:n_states*(N+1)+n_controls*N:n_controls] = omega_max  # Omega 2 upper bound 
ubx[n_states*(N+1)+2:n_states*(N+1)+n_controls*N:n_controls] = omega_max  # Omega 3 upper bound 
ubx[n_states*(N+1)+3:n_states*(N+1)+n_controls*N:n_controls] = omega_max  # Omega 4 upper bound 

args = {
    'lbg': lbg,
    'ubg': ubg,
    'lbx': lbx,
    'ubx': ubx
}


t0 = 0 #current time based on time steps
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state


t = ca.DM(t0) # keeps track of time for each iteration

current_control = ca.DM.zeros((n_controls, N))  # initial control
current_state = ca.repmat(state_init, 1, N+1)         # initial state full
target_state = ca.repmat(state_target, 1, N+1)         # target state full

mpc_iteration = 0 # current iteration of system

predicted_states = np.array(current_state.full()) # storing all predicted states in a 3d matrix
computed_control = np.array(current_control[:,0].full()) # storing all control signals in 2d matrix

times = np.array([[0]]) # keeping track of time


# shift function used to set nxt state to be current state and to apply control action
def shift(t0, state, control, f):
    f_val = f(state, control[:,0])
    next_state = ca.DM(state + step_horizon*f_val)

    t0 += step_horizon
    u0 = ca.horzcat( # removing first control sequence as it is already done and duplicating final control sequence as a prediction
        control[:,1:],
        control[:,-1]
    )
    return t0, next_state, u0

main_loop = time()
# simulation loop is running while accepted error has not been met and time has not reached max simulation time
# if either condition has been met the loop terminates 
while ca.norm_fro(state_init-state_target) > 10**-2 and mpc_iteration < sim_time / step_horizon :
    t1 = time() # getting time at teh start of iteration

    args['p'] = ca.vertcat(         # passing in the current state and the reference, target, state to the solver
        state_init,
        state_target
    )

    args['x0'] = ca.vertcat(        # passing in the current optimization variables, state and control, to the solver
        ca.reshape(current_state, n_states*(N+1), 1),
        ca.reshape(current_control, n_controls*N, 1)
    )

    # running the solver to get optimization variables, state and control
    sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

    control = ca.reshape(sol['x'][n_states*(N+1):], n_controls, N) # extracting control actions from solution and shaping it as a matrix instead of a vector
    current_state = ca.reshape(sol['x'][:n_states*(N+1)], n_states, N+1) # extracting predicted states from solution and shaping it as a matrix instead of a vector

    computed_control = np.vstack(( # adding first calculated control sequence (storing applied control sequence)
        computed_control,
        np.array(control[:,0].full())
    ))

    predicted_states = np.dstack(( # adding calculated state sequence, using dstack to store it depth wise (3d)
        predicted_states,
        np.array(current_state.full())
    ))

    t0, state_init, current_control = shift(t0, state_init, control, f)  # shifting to get next state and apply control action

    t = np.vstack(( # adding timestep to total
        t,
        t0
    ))
    
    current_state = ca.horzcat( # shifting current state by one and again duplicating last state
        current_state[:, 1:],
        current_state[:,-1]
    )

    t2 = time() # geting time after iteration

    times = np.vstack(( # storing time taken for current iteration
        times,
        t2-t1
    ))

    mpc_iteration += 1 # incrementeing iteration count


ss_error = ca.norm_2(state_init - state_target) # finding error from current state - target state

main_loop_time = time() # getting time after loop

# printing results
print('\n\n')
print('Total time: ', main_loop_time - main_loop)
print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
print('final error: ', ss_error)
# visualizing system
simulate(predicted_states, computed_control, times, step_horizon, N, 
            np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]))
    