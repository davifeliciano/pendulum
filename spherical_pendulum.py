from numpy import sin, cos
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from time import time
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

''' 
    Author: Davi Feliciano
    email: dfeliciano37@gmail.com
    
    The equations for the position an velocity of the mass
    of the pendulum were obtained using spherical polar
    coordinates. The 'dot' following some variable's names stands for
    total time derivatives. The differential equations resolved by this
    algorithm were derived from the Lagrange's Equations of each
    generalized coordinate. For more information, see
    https://en.wikipedia.org/wiki/Spherical_pendulum
    This algorithm was inspired by this one about the double pendulum at
    https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
'''


class SphericalPendulum(object):

    def __init__(self,
                 init_state=(45., 0., 45., 80.),            # init_state is [phi, phidot, theta, thetadot]
                 m=1.0,
                 r=1.0,
                 g=9.81,
                 origin=(0, 0, 0),
                 tracer=False,                              # If True, the path of the pendulum is marked over time
                 tracer_len=1100,                           # How many points does the tracer have?
                 dissipation=False,                         # If True, hydrodynamic drag is added. False by default
                 beta=.005,                                                 # The coefficient of linear drag
                 gamma=.005):                                               # The coefficient of quadratic drag
        self.init_state = np.asarray(init_state, dtype='float')
        self.state = self.init_state * np.pi / 180.0
        self.params = (m, r, g, beta, gamma)
        self.origin = origin
        self.tracer = tracer
        if self.tracer:
            self.tracer_len = tracer_len
        self.dissipation = dissipation
        if self.dissipation:
            self.beta = beta
            self.gamma = gamma
        self.time_elapsed = 0.

    def position(self):     # Computes the position, in cartesian coordinates, of the mass of the pendulum

        (m, r, g, beta, gamma) = self.params

        x = np.cumsum([self.origin[0], r * sin(self.state[0]) * cos(self.state[2])])
        y = np.cumsum([self.origin[1], r * sin(self.state[0]) * sin(self.state[2])])
        z = np.cumsum([self.origin[2], - r * cos(self.state[0])])
        return x, y, z

    def energy(self):       # Computes the mechanical energy of the mass of the pendulum

        (m, r, g, beta, gamma) = self.params

        z = - r * cos(self.state[0])
        xdot = r * (cos(self.state[0]) * cos(self.state[2]) * self.state[1]
                    - sin(self.state[0]) * sin(self.state[2]) * self.state[3])
        ydot = r * (cos(self.state[0]) * sin(self.state[2]) * self.state[1] +
                    sin(self.state[0]) * cos(self.state[2]) * self.state[3])
        zdot = r * sin(self.state[0]) * self.state[1]

        potential = m * g * z
        kinetic = 0.5 * m * (xdot * xdot + ydot * ydot + zdot * zdot)

        return potential + kinetic

    def angular_momenta(self):      # Computes the angular momentum relative to the z axis
        (m, r, g, beta, gamma) = self.params
        return m * r * r * sin(self.state[0]) * sin(self.state[0]) * self.state[3]

    def dstate_dt(self, state, t):      # Computes the next state of the pendulum, based on the current state

        (m, r, g, beta, gamma) = self.params
        adot = np.zeros_like(state)

        if self.dissipation:
            alpha = beta + gamma * r * np.sqrt(state[3] ** 2 + state[1] ** 2 * sin(state[2]) ** 2)

            adot[0] = state[1]
            adot[1] = sin(state[0]) * cos(state[0]) * state[3] * state[3] - g * sin(state[0]) / r - \
                sin(state[2]) ** 2 * state[1] * alpha / m
            adot[2] = state[3]
            adot[3] = - 2 * cos(state[0]) * state[1] * state[3] / sin(state[0]) - \
                state[3] * alpha / (m * sin(state[0]) * sin(state[0]))
        else:
            adot[0] = state[1]
            adot[1] = sin(state[0]) * cos(state[0]) * state[3] * state[3] - g * sin(state[0]) / r
            adot[2] = state[3]
            adot[3] = - 2 * cos(state[0]) * state[1] * state[3] / sin(state[0])

        return adot

    def step(self, dt):     # Changes the state of the pendulum, based on the previous function
        self.state = odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.time_elapsed += dt


frame_rate = 30.
frames = 1000       # The amount of frames you want the .mp4 to have

pendulum = SphericalPendulum(tracer=True, dissipation=False)    # Creates a 'SphericalPendulum' like object
dt = 1.0 / frame_rate

mpl.rcParams['text.usetex'] = True          # You need livetex or miktex installed in your machine!
mpl.rc('text.latex', unicode=True)
plt.rc('font', family='serif', size=12)

fig = plt.figure()      # Set up the figure and stuff!
ax = Axes3D(fig)
ax.set_ylim(-1., 1.)
ax.set_xlim(-1., 1.)
ax.set_zlim(-1., 0.)
#   ax.view_init(90., 0.)

tracer_X, tracer_Y, tracer_Z = [], [], []

line, = ax.plot([], [], [], 'o-', lw=2)
if pendulum.tracer:
    tracer, = ax.plot([], [], [], ':', color='r', lw=0.45)

time_indicator = ax.text(0., .15, 6.25, '', transform=ax.transAxes)
energy_indicator = ax.text(0., .15, 6.45, '', transform=ax.transAxes)
momenta_indicator = ax.text(0., .15, 6.65, '', transform=ax.transAxes)


def init():                     # Initialize animation
    line.set_data([], [])
    line.set_3d_properties([])
    tracer.set_data([], [])
    tracer.set_3d_properties([])
    time_indicator.set_text('')
    energy_indicator.set_text('')
    momenta_indicator.set_text('')
    return line, tracer, time_indicator, energy_indicator, momenta_indicator


def animate(i):                                             # Perform animation step
    global pendulum, dt, tracer_X, tracer_Y, tracer_Z
    pendulum.step(dt)

    if pendulum.tracer:
        tracer_X.append(pendulum.position()[0][1])
        if len(tracer_X) == pendulum.tracer_len:
            tracer_X.pop(0)
        tracer_Y.append(pendulum.position()[1][1])
        if len(tracer_Y) == pendulum.tracer_len:
            tracer_Y.pop(0)
        tracer_Z.append(pendulum.position()[2][1])
        if len(tracer_Z) == pendulum.tracer_len:
            tracer_Z.pop(0)
        tracer.set_data(tracer_X, tracer_Y)
        tracer.set_3d_properties(tracer_Z)

    line.set_data(pendulum.position()[0], pendulum.position()[1])
    line.set_3d_properties(pendulum.position()[2])
    time_indicator.set_text('Tempo = %.1f' % pendulum.time_elapsed)
    energy_indicator.set_text('Energia Mecânica = %.3f J' % pendulum.energy())
    momenta_indicator.set_text('Momento Ângular = %.3f Kg '
                               '$\\textrm{m}^2\\;\\textrm{s}^{-1}$' % pendulum.angular_momenta())
    return line, tracer, time_indicator, energy_indicator, momenta_indicator


t0 = time()         # choose the interval based on dt and the time to animate one step
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, init_func=init)
anim.save('spherical_pendulum.mp4', fps=frame_rate, extra_args=['-vcodec', 'libx264'])
# save the animation as an mp4.  This requires ffmpeg or mencoder
plt.show()
