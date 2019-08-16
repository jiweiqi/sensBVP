"""
Constant-pressure, adiabatic kinetics simulation with sensitivity analysis
"""

import sys
import time
import numpy as np

start = time.time()
import cantera as ct

gas = ct.Solution('./mech-FFCM1.cti')
temp = 900.0
pres = 10.0*ct.one_atm

gas.TPX = temp, pres, 'CH4:0.0499002,O2:0.199601,N2:0.750499'
r = ct.IdealGasConstPressureReactor(gas, name='R1')
sim = ct.ReactorNet([r])

# enable sensitivity with respect to all reactions
for i in range(0,gas.n_reactions):
    r.add_sensitivity_reaction(i)

# set the tolerances for the solution and for the sensitivity coefficients
sim.rtol = 1.0e-6
sim.atol = 1.0e-12
sim.rtol_sensitivity = 1.0e-6
sim.atol_sensitivity = 1.0e-12

#states = ct.SolutionArray(gas, extra=['t','s2'])
ignitionStateFound=False
Told=temp
TIgn=1.115367e+03 #K
tEnd=1.0  #s
tauIgn=0.0
nPts=1000
dt=tEnd/(float(nPts)-1.0)
sens=np.zeros(gas.n_reactions)
out=open("ignitionSensitivities.dat","w")

for t in np.arange(0, tEnd, dt):

    TOld=r.T
    sim.advance(t)
    TCurrent=r.T
    for i in range(0,gas.n_reactions):
        sens[i] = sim.sensitivity('temperature', i)

    if(ignitionStateFound==False):
        if(r.T>=TIgn):
            print("Ignition state found!\n")
            print("T=%15.6e K\n"%(TCurrent))
            print("tau=%15.6e s\n"%(t))
            ignitionStateFound=True
            tauIgn=t
            dTdtau=(TCurrent-TOld)/dt
            for i in range(0,gas.n_reactions):
                sens[i]=sens[i]*(-1.0e0/dTdtau)*(TCurrent/tauIgn)
                out.write("%15d\t%15.6e\n"%(i,sens[i]))
            break

out.close()
end = time.time()
print("Elapsed time: %15.6e s\n"%(end-start))
