#sensBVP

This is the public repository for the code sensBVP. Its use is in
calculating sensitivity coefficients of ignition delay time to rate
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
sensBVP -a <absolute tolerance> -r <relative tolerance> -f <BVP
residue tolerance> -T <initial temperature in K> -P <initial pressure in
atm> -m <mechanism file (cti or xml)> -c <mole fraction composition> -t
<integration time> -s <enable sensitivity analysis>
```

Execute sensBVP -h for all available options.

Try the cases in the examples directory.
