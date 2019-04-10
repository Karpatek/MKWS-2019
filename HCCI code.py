# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:56:09 2019

@author: Natalka
"""
"""
Simulation of a Hydrogen fueled Homogenous C Combustion Engine.

The use of pure hydrogen as fuel requires an unrealistically high temperature resistance.

"""

import cantera as ct
import numpy as np

#######################################################################
# Input Parameters
#######################################################################

f = 1500. / 60.  # engine speed [1/s] (3000 rpm)
V_H = .5e-3  # displaced volume [m**3]
epsilon = 30.  # compression ratio [-]
d_piston = 0.083  # piston diameter [m]

# turbocharger temperature, pressure, and composition
T_inlet = 300.  # K
p_inlet = 1.3e5  # Pa
comp_inlet = 'H2:2, O2:1, N2:3.76'

# outlet pressure
p_outlet = 1.2e5  # Pa

# ambient properties
T_ambient = 300.  # K
p_ambient = 1e5  # Pa
comp_ambient = 'O2:1, N2:3.76'

# Reaction mechanism name
reaction_mechanism = 'gri30.xml'

# Inlet valve friction coefficient, open and close timings
inlet_valve_coeff = 1.e-6
inlet_open = -2. / 180. * np.pi
inlet_close = 198. / 180. * np.pi

# Outlet valve friction coefficient, open and close timings
outlet_valve_coeff = 1.e-6
outlet_open = 522. / 180 * np.pi
outlet_close = 2. / 180. * np.pi

# Simulation time and resolution
sim_n_revolutions = 10.
sim_n_timesteps = 30000.


###################################################################

# load reaction mechanism
gas = ct.Solution(reaction_mechanism)

# define initial state
gas.TPX = T_inlet, p_inlet, comp_inlet
r = ct.IdealGasReactor(gas)

# define inlet state
gas.TPX = T_inlet, p_inlet, comp_inlet
inlet = ct.Reservoir(gas)

# define outlet pressure (temperature and composition don't matter)
gas.TPX = T_ambient, p_outlet, comp_ambient
outlet = ct.Reservoir(gas)

# define ambient pressure (temperature and composition don't matter)
gas.TPX = T_ambient, p_ambient, comp_ambient
ambient_air = ct.Reservoir(gas)

# set up connecting devices
inlet_valve = ct.Valve(inlet, r)
outlet_valve = ct.Valve(r, outlet)
piston = ct.Wall(ambient_air, r)

# convert time to crank angle
def crank_angle(t):
    return np.remainder(2 * np.pi * f * t, 4 * np.pi)

# set up IC engine parameters
V_oT = V_H / (epsilon - 1.)
A_piston = .25 * np.pi * d_piston ** 2
stroke = V_H / A_piston
r.volume = V_oT
piston.area = A_piston
def piston_speed(t):
    return - stroke / 2 * 2 * np.pi * f * np.sin(crank_angle(t))
piston.set_velocity(piston_speed)

# create a reactor network containing the cylinder
sim = ct.ReactorNet([r])

# set up output data arrays
states = ct.SolutionArray(r.thermo)
t_sim = sim_n_revolutions / f
t = (np.arange(sim_n_timesteps) + 1) / sim_n_timesteps * t_sim
V = np.zeros_like(t)
m = np.zeros_like(t)
test = np.zeros_like(t)
mdot_in = np.zeros_like(t)
mdot_out = np.zeros_like(t)
d_W_v_d_t = np.zeros_like(t)

# set parameters for the automatic time step refinement
n_last_refinement = -np.inf  # for initialization only
n_wait_coarsening = 10

# do simulation
for n1, t_i in enumerate(t):
    # define opening and closing of valves and injector
    if (np.mod(crank_angle(t_i) - inlet_open, 4 * np.pi) <
            np.mod(inlet_close - inlet_open, 4 * np.pi)):
        inlet_valve.set_valve_coeff(inlet_valve_coeff)
        test[n1] = 1
    else:
        inlet_valve.set_valve_coeff(0)
        
    if (np.mod(crank_angle(t_i) - outlet_open, 4 * np.pi) <
            np.mod(outlet_close - outlet_open, 4 * np.pi)):
        outlet_valve.set_valve_coeff(outlet_valve_coeff)
    else:
        outlet_valve.set_valve_coeff(0)

    # perform time integration, refine time step if necessary
    solved = False
    for n2 in range(4):
        try:
            sim.advance(t_i)
            solved = True
            break
        except ct.CanteraError:
            sim.set_max_time_step(1e-6 * 10. ** -n2)
            n_last_refinement = n1
    if not solved:
        raise ct.CanteraError('Refinement limit reached')
    # coarsen time step if too long ago
    if n1 - n_last_refinement == n_wait_coarsening:
        sim.set_max_time_step(1e-5)

    # write output data
    states.append(r.thermo.state)
    V[n1] = r.volume
    m[n1] = r.mass
    mdot_in[n1] = inlet_valve.mdot(0)
    mdot_out[n1] = outlet_valve.mdot(0)
    d_W_v_d_t[n1] = - (r.thermo.P - ambient_air.thermo.P) * A_piston * \
        piston_speed(t_i)

   
print('P_max:\t'+format(max(states.P/1.e5),'2.1f')+'bar')
print('T_max:\t'+format(max(states.T),'2.1f')+'K')
print('W_max:\t'+format(max(d_W_v_d_t/1.e6),'2.1f')+'MJ')

#####################################################################
# Plot Results in matplotlib
#####################################################################

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy

phi=np.zeros_like(t)
phi=crank_angle(t) * 180 / np.pi

# pressure and temperature
plt.clf()
plt.subplot(211)
plt.plot(t, states.P / 1.e5)
plt.ylabel('$p$ [bar]')
plt.xlabel('$\phi%.3$ [deg]')
plt.xticks(plt.xticks()[0], numpy.round(crank_angle(plt.xticks()[0]) * 180 / np.pi), rotation=17)
plt.savefig('ic_engine_P_(phi).png')
plt.show()

plt.subplot(212)
plt.plot(t, states.T)
plt.ylabel('$T$ [K]')
plt.xlabel('$\phi%.3$ [deg]')
plt.xticks(plt.xticks()[0], numpy.round(crank_angle(plt.xticks()[0]) * 180 / np.pi), rotation=17)
plt.savefig('ic_engine_T_(phi).png')
plt.show()

# inlet and outlet mass flow
plt.clf()
plt.plot(t, mdot_in, label='m_in')
plt.plot(t, mdot_out, label='m_out')
plt.ylabel('$m$ [g/s]')
plt.xlabel('$\phi$ [deg]')
plt.legend(loc=0)
plt.xticks(plt.xticks()[0], numpy.round(crank_angle(plt.xticks()[0]) * 180 / np.pi),
           rotation=17)
plt.savefig('ic_engine_In_Out_mass.png')
plt.show()

# p-V diagram
plt.clf()
plt.plot(V[t<2*4*np.pi/f] * 1000, states.P[t<2*4*np.pi/f] / 1.e5)
plt.xlabel('$V$ [l]')
plt.ylabel('$p$ [bar]')
plt.savefig('ic_engine_P_(V).png')
plt.show()

# p-V diagram; inlet/outlet loop
plt.clf()
plt.plot(V[t<2*4*np.pi/f] * 1000, states.P[t<2*4*np.pi/f] / 1.e5)
plt.xlabel('$V$ [l]')
plt.ylim(-0.1, 3.)
plt.ylabel('$p$ [bar]')
plt.savefig('ic_engine_P_(V)_inlet_outlet_loop.png')
plt.show()

# expansion work
plt.clf()
plt.plot(t, d_W_v_d_t/1e6, label='$\dot{W}_v$')
plt.legend(loc=0)
plt.ylabel('[MW]')
plt.xlabel('$\phi$ [deg]')
plt.xticks(plt.xticks()[0], numpy.round(crank_angle(plt.xticks()[0]) * 180 / np.pi),
           rotation=17)
plt.savefig('ic_engine_Q_W.png')
plt.show()

# gas composition
plt.clf()
plt.plot(t, states('O2').X, label='O2')
plt.plot(t, states('H2O').X, label='H2O')
plt.plot(t, states('H2').X * 10, label='H2 x10')
plt.legend(loc=0)
plt.grid(True)
plt.ylabel('$X_i$ [-]')
plt.xlabel('$\phi$ [deg]')
plt.xticks(plt.xticks()[0], numpy.round(crank_angle(plt.xticks()[0]) * 180 / np.pi),
           rotation=17)
plt.savefig('ic_engine_gas_comp.png')
plt.show()



#####################################################################
# Integral Results
#####################################################################

from scipy.integrate import trapz

# Total mass flow through inlet and oulet valve
fr_m_in=trapz(mdot_in, t)
fr_m_out=trapz(mdot_out, t)

# Difference between Inlet Flow and Outlet Flow; should be ~0
fr_diff=fr_m_in-fr_m_out

#Reactants Enthalpy
cv=r.thermo.cv_mass

#Products Enthalpy


#Difference between Inlet Enthalpy and Outlet Enthalpy


#Higher heating value for H2 [J/kg]
H2_hhv=120.1*1e6
#Total Heat Release
ni_H2=2
ni_O2=32
ni_N2=28
heat_release=(fr_m_in)*((2*ni_H2)/(2*ni_H2+1*ni_O2+3.76*ni_N2))*H2_hhv

#Total Heat Release and Expantion Work
Q = heat_release
W = trapz(d_W_v_d_t, t)

# Efficency
eta = W/Q

# Results
print('Wyniki dla 1500 RPM:')
print('Heat release rate per cylinder (estimate):\t' + format(Q / t_sim / 1000., '2.1f') + ' kW')
print('Expansion power per cylinder (estimate):\t' + format(W / t_sim / 1000., ' 2.1f') + ' kW')
print('Efficiency (estimate):\t\t\t' + format(eta * 100., ' 2.1f') + ' %')
print('Inlet Mass Flow (estimate):\t\t\t' + format(fr_m_in * 1000., ' 2.3f') + 'g')
print('Outlet Mass Flow (estimate):\t\t\t' + format(fr_m_out * 1000., ' 2.3f') + 'g')
print('Outlet Mass Flow (estimate):\t\t\t' + format(fr_diff * 1000., ' 2.3f') + 'g')


