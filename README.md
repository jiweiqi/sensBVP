#sensBVP

This is the public repository for the code sensBVP as cited in the paper
[Direct Sensitivity Analysis for ignition delay
times](https://doi.org/10.1016/j.combustflame.2019.08.007). Its use is
in calculating the sensitivity of ignition delay time to rate
coefficients. In sensBVP, this is carried out by transforming the
initial value problem to a boundary value problem, reducing the time to
calculate the sensitivity coefficients by at least an order of
magnitude.

In order to install sensBVP, modify the various paths in the Makefile.
Make sure that SUNDIALS (v3.1.1 or higher), Cantera (2.3 or higher), and
GSL are installed. Run make all, followed by make install in the source
directory.

After installing, the executable sensBVP can be executed as follows: 
``` 
sensBVP -a <absolute tolerance> 
	-r <relative tolerance> 
	-f <BVP residue tolerance> 
	-T <initial temperature in K> 
	-P <initial pressure in atm> 
	-m <mechanism file (cti or xml)> 
	-c <mole fraction composition> 
	-t <integration time> 
	-s <enable sensitivity analysis>
```

Execute sensBVP -h for all available options. The time evolution of
species mass fractions, temperature, and pressure can be found in
"output.dat". The ignition delay sensitivities can be found in
"ignitionSensitivities.dat".

Try the cases in the examples directory.

If you use this code in your work, please cite: 

```
@article{GURURAJAN2019478,
title = "Direct sensitivity analysis for ignition delay times",
journal = "Combustion and Flame",
volume = "209",
pages = "478 - 480",
year = "2019",
issn = "0010-2180",
doi = "https://doi.org/10.1016/j.combustflame.2019.08.007",
url = "http://www.sciencedirect.com/science/article/pii/S0010218019303645",
author = "Vyaas Gururajan and Fokion N. Egolfopoulos",
keywords = "Ignition, Sensitivity, Jacobian, Chemical kinetics",
}


```
