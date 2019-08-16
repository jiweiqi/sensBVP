#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

/*Cantera include files*/
#include <cantera/IdealGasMix.h>

/*Sundials include files*/
#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <cvode/cvode_direct.h>        /* access to CVDls interface            */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */
#include <sundials/sundials_math.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_spline.h>

static int imaxarg1,imaxarg2;
#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
        (imaxarg1) : (imaxarg2))

static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
        (iminarg1) : (iminarg2))

/*quick-sort related routines*/
static unsigned long *lvector(long nl, long nh);
static void free_lvector(unsigned long *v, long nl, long nh);
static void sort(unsigned long n, double arr[], int ind[]);

#define ZERO  RCONST(0.0)
#define HALF  RCONST(0.5)
#define ONE   RCONST(1.0)
#define TWO   RCONST(2.0)
#define THREE RCONST(3.0)
#define TEN   RCONST(10.0)
#define BUFSIZE 1000
#define EPSILON RCONST(0.1)

/* In order to keep begin the index numbers from 1 instead of 0, we define
 * macros here. Also, we define macros to ease referencing various variables in
 * the sundials nvector.
 */

#define Y0(k)  NV_Ith_S(data->Y0,k-1)
#define x(i)   NV_Ith_S(data->x,i-1)

#define t(i)   NV_Ith_S(data->tArray,i)
#define P(i)   NV_Ith_S(data->PArray,i)
#define dPdt(i)   NV_Ith_S(data->dPdtArray,i)

#define T(i)   NV_Ith_S(y,((i-1)*data->nvar)+data->nt)
#define Y(i,k) NV_Ith_S(y,((i-1)*data->nvar)+data->ny+k-1)

#define Tdot(i)   NV_Ith_S(ydot,((i-1)*data->nvar)+data->nt)
#define Ydot(i,k) NV_Ith_S(ydot,((i-1)*data->nvar)+data->ny+k-1)

#define Pp(i)      NV_Ith_S(data->Pp,i-1)
#define dPdtp(i)   NV_Ith_S(data->dPdtp,i-1)

#define Tres(i)   NV_Ith_S(res,((i-1)*(data->nvar+1))+data->nt)
#define Yres(i,k) NV_Ith_S(res,((i-1)*(data->nvar+1))+data->ny+k-1)
#define taures(i) NV_Ith_S(res,((i-1)*(data->nvar+1))+data->ntau)

#define wdot(i) wdot[i-1]
#define enthalpy(i) enthalpy[i-1]


typedef struct {
	/* This structure contains all information relevant to evaluating the
	 * residue.
	 */
	Cantera::IdealGasMix* gas;//Ideal gas object containing thermodynamic and
				  //kinetic data
				
	realtype P;		//Pressure (in Pascals)
	realtype T0;		//Initial Temperature (in K)
	N_Vector Y0;		//Initial Mass Fractions
	realtype rho0;		//initial density
	realtype TIgn;		//Ignition Temperature (in K)
	realtype tauIgn;	//Ignition delay time  (in s)
	N_Vector x;		//The grid
	N_Vector tArray;
	N_Vector PArray;
	N_Vector dPdtArray;
	N_Vector Pp;
	N_Vector dPdtp;
	realtype dPdt;
	int tArraySize;
	bool constantVolume;	//boolean to enable/disable constant volume
				//version of the ignition delay problem
	bool imposedPdt;	//boolean to enable/disable manual entry for P and dPdt
	int jguess;		//saved index to accelerate lookup via "hunt"
	bool writeSpeciesSensitivities;	//boolean to enable/disable writing
				//of species sensitivities
	bool IVPSuccess;
	int nsp,neq,npts,nvar;	//key quantities required in for loops:
				//nsp-> no: of chemical species
				//neq-> number of governing equations
				//npts-> no: of grid points
				//nvar->no: of variables
	int nt,ny,ntau;		//array indices for various variables: 
				//nt-> temperature (0)
				//ny-> species mass fractions
				//ntau-> ignition delay time
	int KSneq;		//No: of equations in the BVP formulation of
				//the ignition delay problem
	realtype rel;		//the relative perturbation factor used in
				//computing difference quotients
	realtype atol;
	realtype rtol;
	
  	realtype t1;		//Maximum time to run simulation
	int nreac;		
	int pert_index;		//index of reaction whose rate is perturbed
	bool sensitivityAnalysis;// boolean to activate perturbations for
				//sensitivity analysis
	bool sens;		//if true, perform sensitivity analysis
	bool secondOrder;	//if true, use second order finite differences
	FILE *output;		//output file for temperature and species profiles
	FILE *ignSensOutput;	//output file for ignition sensitivities
	FILE *speSensOutput;	//output file for species sensitivities
	FILE *dPdtInput;	//output file for species sensitivities

	gsl_interp_accel* acc;
	gsl_interp_accel* accdot;
	gsl_spline* spline;
	gsl_spline* splinedot;
} *UserData;


/* Set the initial conditions using this subroutine: */
static int setInitialProfile(UserData data, N_Vector y);

/* Evaluate the residue for the IVP using this subroutine: */
static int residue(realtype t, N_Vector y, N_Vector ydot, void *user_data);

static int parseInput(UserData data, int argc, char *argv[]);

static int hunt(realtype x, realtype xx[], int n, int jguess);

static void polint(realtype *xdata, realtype *f, int n, realtype x,realtype *y, realtype *dy);

static void lookupDpdt(UserData data, realtype t, realtype *P, realtype *dPdt);

/* Subroutine that prints the column titles in the output file: */
static void printInstructions();

/* Subroutine that prints the column titles in the output file: */
static void printHeader(UserData data);
static void printSensitivitiesHeader(UserData data);

/* Subroutine that prints the output of the IVP into the output file contained
 * in the UserData structure: */
static void printOutput(realtype t, N_Vector y, UserData data);

/* Subroutine that prints the sensitivities into the output file (ignSensOutput)
 * contained in the UserData structure: */
static void printIgnSensitivities(realtype sensCoeffs[], int indices[], UserData data);

static int parsedPdt(UserData data);

/* Print solver statistics for the IVP: */
static void PrintFinalStats(void *cvode_mem);

/* Subroutine that reports failure if memory hasn't been successfully allocated
 * to various objects: */
static int check_flag(void *flagvalue, const char *funcname, int opt);

int main(int argc, char *argv[])
{
	clock_t start, end;
     	double cpu_time_used;
     	start = clock();

  	int ier;			//error flag

  	UserData data;			//User defined data 
  	data = NULL;

  	/* Create and load problem data block. */
  	data = (UserData) malloc(sizeof *data);
	ier=parseInput(data,argc,argv);
	if(ier==-1)return(-1);

	{
		/* Solver Memory:*/
  		void *mem;	
		mem=NULL;

		/* Save the initial mass fractions: */
		data->Y0=N_VNew_Serial(data->nsp);	
		for(int k=1;k<=data->nsp;k++){
  			Y0(k)=data->gas->massFraction(k-1); 	
		}

		/*Assign variable indices:*/
		data->nt=0;
		data->ny=data->nt+1;
		data->ntau=data->ny+data->nsp;
		
		/*Get and save the no: of reactions:*/
		data->nreac=data->gas->nReactions();

		/* Solution vector of the IVP: */
  		N_Vector y;
  		y = NULL;
  		y = N_VNew_Serial(data->neq);
  		ier=check_flag((void *)y, "N_VNew_Serial", 0);

  		realtype t0, tret;			//times
		t0=tret=ZERO;

		/*Set the initial values:*/
		setInitialProfile(data, y);
		/*Print out the column names and the initial values:*/
		printHeader(data);
		printOutput(0.0e0,y,data);

  		/* Create a CVode solver object for solution of the IVP:  */
  		mem = CVodeCreate(CV_BDF,CV_NEWTON);
  		ier=check_flag((void *)mem, "CVodeCreate", 0);
		
		/*Associate the user data with the solver object: */
  		ier = CVodeSetUserData(mem, data);
  		ier=check_flag(&ier, "CVodeSetUserData", 1);

		/* Initialize the solver object by connecting it to the solution vector
		 * y: */
  		ier = CVodeInit(mem, residue, t0, y);
  		ier=check_flag(&ier, "CVodeInit", 1);
		
		/*Set the tolerances: */
  		ier = CVodeSStolerances(mem, data->rtol, data->atol);
  		ier=check_flag(&ier, "IDASStolerances", 1);

      		/* Create dense SUNMatrix for use in linear solves: */
  		SUNMatrix A;
      		A = SUNDenseMatrix(data->neq, data->neq);
      		ier=check_flag((void *)A, "SUNDenseMatrix", 0);

      		/* Create dense SUNLinearSolver object for use by CVode: */
  		SUNLinearSolver LS;
      		LS = SUNDenseLinearSolver(y, A);
      		ier=check_flag((void *)LS, "SUNDenseLinearSolver", 0);

		/* Call CVDlsSetLinearSolver to attach the matrix and linear solver to
		 * CVode: */
      		ier = CVDlsSetLinearSolver(mem, LS, A);
      		ier=check_flag(&ier, "CVDlsSetLinearSolver", 1);

      		/* Use a difference quotient Jacobian: */
      		ier = CVDlsSetJacFn(mem, NULL);
      		ier=check_flag(&ier, "CVDlsSetJacFn", 1);
		
		/*Begin IVP solution; Save solutions in the temporary solution vector;
		 * stop problem when the temperature reaches a certain value (like 400
		 * K) corresponding to the time for ignition :*/
		int i=1;
		bool delayFound=false;
		while (tret<=data->t1) {
			if(T(1)>=data->TIgn && !delayFound){
				printf("Ignition Delay: %15.6es\n", tret);
				data->tauIgn=tret;	//Save the ignition delay time
				delayFound=true;
			}
  			ier = CVode(mem, data->t1, y, &tret, CV_ONE_STEP);
  			if(check_flag(&ier, "CVode", 1)) {
				data->IVPSuccess=false;
				break;
			}
			else{
				data->IVPSuccess=true;
			}
			printOutput(tret,y,data);
		}

		if(data->IVPSuccess && data->sens){
			/*Enable sensitivity analysis:*/
			data->sensitivityAnalysis=true;
			data->rel=EPSILON;
			realtype oneOverRel=ONE/(data->rel);

			/*Create an array to store the logarithmic sensitivity
			 * coefficients:*/
			realtype sensCoeffs[data->nreac];

			/*Create an array to store the indices of the reactions (needed
			 * for sorting the sensitivity coefficients):*/
			int indices[data->nreac];
			double tauIgnPerturbedForward=0.0e0;
			double tauIgnPerturbedBackward=0.0e0;
			
			for(int j=0;j<data->nreac;j++){

				/*Save the index of the reaction whose collision
				 * frequency (A in the Arrhenius law k=A*exp(-Eₐ/RT))
				 * is to be perturbed:*/ 
				data->pert_index=j;	

				/*Perturb forward:*/
				data->rel=EPSILON;
				/*Set the initial values:*/
				setInitialProfile(data, y);

				/* Initialize the solver object by connecting it to the solution vector
				 * y: */
  				ier = CVodeReInit(mem, t0, y);
  				ier=check_flag(&ier, "CVodeReInit", 1);
				
				/*Rerun the ignition delay problem!*/
				printf("Solving forward perturbed problem %d:\n",j);
				tret=0.0e0;
				delayFound=false;
				while (tret<=data->t1) {
					if(T(1)>=data->TIgn && !delayFound){
						printf("\tIgnition Delay(forward): %15.6es\n", tret);
						tauIgnPerturbedForward=tret;	//Save the ignition delay time
						delayFound=true;
						/*No point continuing solution*/
						break;
					}
  					ier = CVode(mem, data->t1, y, &tret, CV_ONE_STEP);
  					if(check_flag(&ier, "CVode", 1)) {
						data->IVPSuccess=false;
						break;
					}
					else{
						data->IVPSuccess=true;
					}
					//printOutput(tret,y,data);
				}
				if(!delayFound)tauIgnPerturbedForward=1e+99;
				
				if(data->secondOrder){
					/*Perturb backward:*/
					data->rel=-EPSILON;
					/*Set the initial values:*/
					setInitialProfile(data, y);

					/* Initialize the solver object by connecting it to the solution vector
					 * y: */
  					ier = CVodeReInit(mem, t0, y);
  					ier=check_flag(&ier, "CVodeReInit", 1);
					
					/*Rerun the ignition delay problem!*/
					printf("Solving backward perturbed problem %d:\n",j);
					tret=0.0e0;
					delayFound=false;
					while (tret<=data->t1) {
						if(T(1)>=data->TIgn && !delayFound){
							printf("\tIgnition Delay(backward): %15.6es\n", tret);
							tauIgnPerturbedBackward=tret;	//Save the ignition delay time
							delayFound=true;
							/*No point continuing solution*/
							break;
						}
  						ier = CVode(mem, data->t1, y, &tret, CV_ONE_STEP);
  						if(check_flag(&ier, "CVode", 1)) {
							data->IVPSuccess=false;
							break;
						}
						else{
							data->IVPSuccess=true;
						}
						//printOutput(tret,y,data);
					}
					if(!delayFound)tauIgnPerturbedBackward=1e+99;

					/*Take the finite difference quotient as an
					 * approximation to the logarithmic sensitivity
					 * coefficient:*/
					sensCoeffs[j]=oneOverRel*(tauIgnPerturbedForward-tauIgnPerturbedBackward)/(2.0e0*data->tauIgn);
				}else{
					sensCoeffs[j]=oneOverRel*(tauIgnPerturbedForward-data->tauIgn)/(data->tauIgn);
				}

				indices[j]=j;
				printf("\n");
			}

			/*Sort the sensitivities in ascending order. Note the advancing
			 * of the beginning indices of the sensCoeffs and indices
			 * arrays. This is due to the Numerical recipes convention for
			 * array indexing used in sort subroutine:*/
			//sort(data->nreac,sensCoeffs-1,indices-1);

			/*Print out the sensitivities:*/
			printIgnSensitivities(sensCoeffs,indices,data);
		}

  		/* Print remaining counters and free memory. */
		PrintFinalStats(mem);
  		CVodeFree(&mem);
  		N_VDestroy_Serial(y);
	}

	/*Free memory and delete all the vectors and user data:*/
	if(data->Y0!=NULL){
  		N_VDestroy_Serial(data->Y0);
		printf("Initial Mass fraction Vector Deleted!\n");
	}
	if(data->x!=NULL){
  		N_VDestroy_Serial(data->x);
		printf("Grid for BVP deleted!\n");
	}
	if(data->Pp!=NULL){
  		N_VDestroy_Serial(data->Pp);
		printf("P array deleted!\n");
	}
	if(data->dPdtp!=NULL){
  		N_VDestroy_Serial(data->dPdtp);
		printf("dPdt array deleted!\n");
	}
	if(data->gas!=NULL){
		delete data->gas;
		printf("Gas Deleted!\n");

	}
	if(data->imposedPdt){
		if(data->acc!=NULL){
			gsl_interp_accel_free(data->acc);
		}
		if(data->accdot!=NULL){
			gsl_interp_accel_free(data->accdot);
		}
		if(data->spline!=NULL){
			gsl_spline_free(data->spline);
		}
		if(data->splinedot!=NULL){
			gsl_spline_free(data->splinedot);
		}
	}
	if(data!=NULL){
		/* Free the user data */
  		free(data);             
		printf("User data structure deleted!\n");
	}

 	end = clock();
     	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Elapsed cpu time: %15.6e s\n", cpu_time_used);
	return(0);
}

static int parseInput(UserData data, int argc, char *argv[]){

	/*Set defaults here:*/
	/*****************************************************/
	/*Relative tolerance for IVP:*/
  	data->rtol = RCONST(1e-14);
	/*Absolute tolerance for IVP:*/
  	data->atol = RCONST(1e-14);
	/*Final Time:*/
	data->t1   = RCONST(10.0);
	/*Solve constant pressure problem:*/
	data->constantVolume=false;	
	/*Do not impose a P vs t curve:*/
	data->imposedPdt=false;		
	/*Disable writing species sensitivities:*/
	data->writeSpeciesSensitivities=false;
	/*Disable sensitivity analysis:*/
	data->sens=false;
	/*Index to start with for pressure lookup (see hunt):*/
	data->jguess=0;			
	/*Set rate of change of pressure to zero:*/
	data->dPdt=ZERO;
	/*Disable sensitivity analysis:*/
	data->sensitivityAnalysis=false;
	/*Use first order finite differences:*/
	data->secondOrder=false;
	/*Find the relative perturbation constant:*/
	//data->rel=SUNRsqrt(UNIT_ROUNDOFF);
	data->rel=0.1;
	/*Set flags that indicate success of various stages:*/
	data->IVPSuccess=false;
	/*****************************************************/

	int ier;
	int opt;
	char mech[BUFSIZE+1];
	char comp[BUFSIZE+1];
	bool enteredT0, enteredP, enteredMech, enteredComp;
	enteredT0=enteredP=enteredMech=enteredComp=false;
	while((opt=getopt(argc,argv,"a:r:T:P:m:c:t:vsd2")) != -1){
		switch(opt){
			case 'a':
				data->atol=RCONST(atof(optarg));
				break;
			case 'r':
				data->rtol=RCONST(atof(optarg));
				break;
			case 'T':
				data->T0=RCONST(atof(optarg));
				enteredT0=true;
				break;
			case 'P':
				data->P=RCONST(atof(optarg))*Cantera::OneAtm; 
				enteredP=true;
				break;
			case 'm':
				snprintf(mech,BUFSIZE,"%s",optarg);
				enteredMech=true;
				break;
			case 'c':
				snprintf(comp,BUFSIZE,"%s",optarg);
				enteredComp=true;
				break;
			case 't':
				data->t1=RCONST(atof(optarg));
				enteredT0=true;
				break;
			case 'v':
				data->constantVolume=true;
				break;
			case 'S':
				data->writeSpeciesSensitivities=true;
				break;
			case 's':
				data->sens=true;
				break;
			case 'd':
				data->imposedPdt=true;
				break;
			case '2':
				data->secondOrder=true;
				break;
			default:
				printInstructions();
				return(-1);
		}
	}
	if(!enteredT0){
		printf("Not specified initial Temperature! Exiting...\n");
		printInstructions();
		return(-1);
	}
	else if(!enteredP){
		printf("Not specified Pressure! Exiting...\n");
		printInstructions();
		return(-1);
	}
	else if(!enteredMech){
		printf("Not specified Mechanism! Exiting...\n");
		printInstructions();
		return(-1);
	}
	else if(!enteredComp){
		printf("Not specified Composition! Exiting...\n");
		printInstructions();
		return(-1);
	}
	else{
		if(data->imposedPdt){
			ier=parsedPdt(data);
			if(ier==-1){
				return(-1);
			}
		}

		printf("Required inputs provided:\n"); 
		printf("\natol: %15.6e\n", data->atol); 
		printf("\nrtol: %15.6e\n", data->rtol); 
		printf("\nT0  : %15.6e K\n", data->T0); 
		printf("\nP   : %15.6e Pa\n", data->P); 
		printf("\nmech: %s\n", mech); 
		printf("\ncomp: %s\n", comp); 
		printf("\nt1  : %15.6e s\n", data->t1); 
		printf("\nconstantVolume  : %d\n", data->constantVolume); 
		printf("\nimposedPdt  : %d\n", data->imposedPdt); 
		printf("\nwriteSpeciesSensitivities  : %d\n", 
				data->writeSpeciesSensitivities); 
		printf("Proceeding...\n\n");
	}

	/*Define Gas here:*/
	data->gas = new Cantera::IdealGasMix(mech);

	/* Set the initial state of the gas: */
	data->gas->setState_TPX(data->T0,
				data->P,
				comp);

  	/* Create output file for the solution of the IVP: */
	data->output=fopen("output.dat","w");

	/* Create output file for the ignition sensitivities (post BVP): */
	data->ignSensOutput=fopen("ignitionSensitivities.dat","w");

	/* Create output file for the species sensitivities (post BVP): */
	data->speSensOutput=fopen("speciesSensitivities.dat","w");

	data->rho0=data->gas->density();
	if(data->constantVolume){
		data->gas->equilibrate("UV");
	}
	else{
		data->gas->equilibrate("HP");
	}
	data->TIgn=data->T0+(data->gas->temperature()
			    -data->T0)*RCONST(0.20);
	printf("Ignition Temperature: %15.6e K\n",data->TIgn);
	data->gas->setState_TPX(data->T0,
				data->P,
				comp);

	/*Get and save the no: of species:*/
	data->nsp=data->gas->nSpecies();	
	/*set the no: of variables for the IVP:*/
	data->nvar=data->nsp+1;			 
	/*set the no: of equations for the IVP:*/
	data->neq=data->nvar;			

	return(0);
}

static int setInitialProfile(UserData data, N_Vector y)
{
	/*This routine sets the initial temperature and mass fractions for the
	 * solution of the initial value problem:*/
  	N_VConst(ZERO, y);

	T(1)=data->T0;
	for (int k = 1; k <=data->nsp; k++) {
  		Y(1,k)=Y0(k);
	}

  	return(0);

}

static int residue(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{

	/*This is the sub-routine that computes the right hand side F(y) in the
	 * problem ∂y/∂t = F(y). 
	 *
	 * The energy conservation equation is 
	 * 	  ∂T/∂t = (1/ρcₚ)(dP/dt -∑ ώᵢhᵢ)
	 *
	 * T has units of K
	 * hᵢ has units J/kmol, so we must multiply the enthalpy
	 * defined above (getEnthalpy_RT) by T (K) and the gas constant
	 * (J/kmol/K) to get the right units.
	 * ρ has units of kg/m³
	 * cₚ has units J/kg/K
	 * ώᵢ has units of kmol/m³/s
	 * dP/dt has units of Pa/s
	 *
	 * In the case of a constant volume problem, the energy conservation
	 * equation can be re-written as
	 * 	  ∂T/∂t = -(1/ρcᵥ)∑ ώᵢeᵢ
	 * eᵢ has units J/kmol
	 * cᵥ has units J/kg/K
	 *
	 *
	 * The species conservation equation is
	 * 	∂Yᵢ/∂t = ώᵢWᵢ/ρ
	 * Wᵢ has units of kg/kmol
	 */

	/*Assign data structure type to user_data:*/
	UserData data;
	data=(UserData)user_data;

	/*Get no: of species:*/
	int nsp=data->nsp;

	/*Create temporary arrays to store enthalpy (non-dimensional) and ώᵢ:*/
	realtype wdot[nsp];
	realtype enthalpy[nsp];

    	try {
		/*If sensitivity analysis has been enabled, perturb the collision
		 * frequency by using cantera's setMultiplier function:*/
		if(data->sensitivityAnalysis){
			data->gas->setMultiplier(data->pert_index,ONE+data->rel);
			//printf("%d\t%15.9e\n",data->pert_index,ONE+data->rel);
		}

		/*Set the gas conditions:*/
		if(data->constantVolume){
  			data->gas->setMassFractions_NoNorm(&Y(1,1)); 	
  			data->gas->setTemperature(T(1)); 	        
  			data->gas->setDensity(data->rho0);
			data->P=data->gas->pressure();
		}
		else if(data->imposedPdt){
  			data->gas->setMassFractions_NoNorm(&Y(1,1)); 	
  			data->gas->setTemperature(T(1)); 	        
			realtype P,dPdt;
			P=dPdt=0.0e0;
			lookupDpdt(data, t, &P, &dPdt);
			data->P=P*Cantera::OneAtm;
			data->dPdt=dPdt*Cantera::OneAtm;
  			data->gas->setPressure(data->P);
		}
		else{
  			data->gas->setMassFractions_NoNorm(&Y(1,1)); 	
  			data->gas->setTemperature(T(1)); 	        
  			data->gas->setPressure(data->P);
		}

		realtype rho=data->gas->density();	//get ρ
		realtype cp=data->gas->cp_mass();	//get cₚ
		data->gas->getNetProductionRates(wdot);	//get ώᵢ
		data->gas->getEnthalpy_RT(enthalpy);	//get hᵢ/RT

		if(data->constantVolume){
			for(int k=1;k<=nsp;k++){
				enthalpy(k)=enthalpy(k)-ONE;
			}
			cp=data->gas->cv_mass();
		}

		realtype sum=ZERO;
		for(int k=1;k<=nsp;k++){
			Ydot(1,k)=wdot(k)*(data->gas->molecularWeight(k-1))/rho;
			sum=sum+wdot(k)*enthalpy(k)*T(1);
		}
		sum=sum*Cantera::GasConstant;
		Tdot(1)=(data->dPdt-sum)/(rho*cp);

		/*If sensitivity analysis has been enabled, *un*perturb the collision
		 * frequency:*/
		if(data->sensitivityAnalysis){
			data->gas->setMultiplier(data->pert_index,ONE);
		}

    	} catch (Cantera::CanteraError& err) {
	    printf("Error:\n");
	    printf("%s\n",err.what());
	    return(-1);
    	}

	return(0);
}

static void lookupDpdt(UserData data, realtype t, realtype *P, realtype *dPdt)
{

	realtype *tData, *PData, *dPdtData;
	tData = N_VGetArrayPointer_Serial(data->tArray);
	PData = N_VGetArrayPointer_Serial(data->PArray);
	dPdtData = N_VGetArrayPointer_Serial(data->dPdtArray);
	int jguess=data->jguess;
	int k=0;
	int safel=0;
	int nOrder=4;

	realtype dy;
	int npts=data->tArraySize;

	if(t<=data->t1){
		*P=gsl_spline_eval(data->spline,t,data->acc);
		*dPdt=gsl_spline_eval(data->splinedot,t,data->accdot);
	}
	else{
		*P=P(npts-1);
		*dPdt=dPdt(npts-1);
	}
}


static int hunt(realtype x, realtype xx[], int n, int jguess)
{
	int jlo=jguess;
	int jm,jhi,inc;
	int ascnd;

	ascnd=(xx[n-1] >= xx[0]);
	if (jlo <= 0 || jlo > n-1) {
		jlo=0;
		jhi=n;
	} else {
		inc=1;
		if ((x >= xx[jlo]) == ascnd) {
			if (jlo == n-1) return(jlo);
			jhi=(jlo)+1;
			while ((x >= xx[jhi]) == ascnd) {
				jlo=jhi;
				inc += inc;
				jhi=(jlo)+inc;
				if (jhi > n-1) {
					jhi=n;
					break;
				}
			}
		} else {
			if (jlo == 1) {
				jlo=0;
				return (jlo);
			}
			jhi=(jlo)--;
			while ((x < xx[jlo]) == ascnd) {
				jhi=(jlo);
				inc <<= 1;
				if (inc >= jhi) {
					jlo=0;
					break;
				}
				else jlo=jhi-inc;
			}
		}
	}
	while (jhi-(jlo) != 1) {
		jm=(jhi+(jlo)) >> 1;
		if ((x >= xx[jm]) == ascnd)
			jlo=jm;
		else
			jhi=jm;
	}
	if (x == xx[n-1]) jlo=n-2;
	if (x == xx[0]) jlo=1;

	return(jlo);
}

static void polint(realtype *xdata, realtype *f, int n, realtype x, realtype *y, realtype *dy){
	int i,m,ns=1;
	realtype den,dif,dift,ho,hp,w;
	realtype c[n+1],d[n+1];
	dif=fabs(x-xdata[1]);
	for (i=1;i<=n;i++) {
		if ( (dift=fabs(x-xdata[i])) < dif) {
			ns=i;
			dif=dift;
		}
		c[i]=f[i];
		d[i]=f[i];
	}
	*y=f[ns--];
	for (m=1;m<n;m++) {
		for (i=1;i<=n-m;i++) {
			ho=xdata[i]-x;
			hp=xdata[i+m]-x;
			w=c[i+1]-d[i];
			if ( (den=ho-hp) == 0.0) printf("Error in routine polint!\n");
			den=w/den;
			d[i]=hp*den;
			c[i]=ho*den;
		}
		*y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
	}
}

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
#define M 7
#define NSTACK 50
#define NR_END 1
#define FREE_ARG char*

static unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
	unsigned long *v;

	v=(unsigned long *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long)));
	if (!v) printf("allocation failure in lvector()");
	return v-nl+NR_END;
}

static void free_lvector(unsigned long *v, long nl, long nh)
/* free an unsigned long vector allocated with lvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

static void sort(unsigned long n, double arr[], int ind[])
{
	unsigned long i,ir=n,j,k,l=1,*istack;
	int jstack=0;
	double a,temp;
	int b;

	istack=lvector(1,NSTACK);
	for (;;) {
		if (ir-l < M) {
			for (j=l+1;j<=ir;j++) {
				a=arr[j];
				b=ind[j];
				for (i=j-1;i>=l;i--) {
					if (arr[i] <= a) break;
					arr[i+1]=arr[i];
					ind[i+1]=ind[i];
				}
				arr[i+1]=a;
				ind[i+1]=b;
			}
			if (jstack == 0) break;
			ir=istack[jstack--];
			l=istack[jstack--];
		} else {
			k=(l+ir) >> 1;
			SWAP(arr[k],arr[l+1])
			SWAP(ind[k],ind[l+1])
			if (arr[l] > arr[ir]) {
				SWAP(arr[l],arr[ir])
				SWAP(ind[l],ind[ir])
			}
			if (arr[l+1] > arr[ir]) {
				SWAP(arr[l+1],arr[ir])
				SWAP(ind[l+1],ind[ir])
			}
			if (arr[l] > arr[l+1]) {
				SWAP(arr[l],arr[l+1])
				SWAP(ind[l],ind[l+1])
			}
			i=l+1;
			j=ir;
			a=arr[l+1];
			b=ind[l+1];
			for (;;) {
				do i++; while (arr[i] < a);
				do j--; while (arr[j] > a);
				if (j < i) break;
				SWAP(arr[i],arr[j]);
				SWAP(ind[i],ind[j]);
			}
			arr[l+1]=arr[j];
			ind[l+1]=ind[j];
			arr[j]=a;
			ind[j]=b;
			jstack += 2;
			if (jstack > NSTACK) printf("NSTACK too small in sort.");
			if (ir-i+1 >= j-l) {
				istack[jstack]=ir;
				istack[jstack-1]=i;
				ir=j-1;
			} else {
				istack[jstack]=j-1;
				istack[jstack-1]=l;
				l=i;
			}
		}
	}
	free_lvector(istack,1,NSTACK);
}
#undef M
#undef NSTACK
#undef SWAP

static void printInstructions(){
	printf("\nInputs either incomplete or wrong!\n");
	printf("\n-a <absolute tolerance for IVP>\n");
	printf("\n-r <relative tolerance for IVP>\n");
	printf("\n-T <Initial Temperature in Kelvin>\n");
	printf("\n-P <Initial Pressure in atm>\n");
	printf("\n-m <mechanism file (cti or xml)>\n");
	printf("\n-c <composition in mole fractions>\n");
	printf("\n-t <Total simulation time in seconds>\n");
	printf("\n-v :Enables constant volume simulation\n");
	printf("\n-s :Enables ignition delay sensitivity output\n");
	printf("\n-S :Enables species mass fraction sensitivity output (works only with -s)\n");
	printf("\n-d :Enables manual dPdt entryspecies sensitivity output\n");
	printf("\n-2 :Enables 2nd order accurate finite differences\n");
	printf("\nexample: <executable> ");
	printf("-T 1200.0 -P 1.0 -m gri30.cti");
	printf(" -c H2:1.0,N2:3.76,O2:1.0");
	printf(" -t 1e-03\n");
}

static void printHeader(UserData data)
{
	fprintf((data->output), "%15s\t","#1");
	for (int k = 1; k <=data->nvar; k++) {
		fprintf((data->output), "%15d\t",k+1);
	}
	fprintf((data->output), "\n");

	fprintf((data->output), "%15s\t%15s\t%15s\t",
			"#time(s)","Temp(K)","Pressure(Pa)");
	for (int k = 1; k <=data->nsp; k++) {
		fprintf((data->output), "%15s\t",
				data->gas->speciesName(k-1).c_str());
	}
	fprintf((data->output), "\n");

}

static void printSensitivitiesHeader(UserData data)
{
	fprintf((data->speSensOutput), "%15s\t","#1");
	for (int k = 1; k <=data->nvar; k++) {
		fprintf((data->speSensOutput), "%15d\t",k+1);
	}
	fprintf((data->speSensOutput), "\n");

	fprintf((data->speSensOutput), "%15s\t%15s\t",
			"#time(s)","Temp(K)");
	for (int k = 1; k <=data->nsp; k++) {
		fprintf((data->speSensOutput), "%15s\t",
				data->gas->speciesName(k-1).c_str());
	}
	fprintf((data->speSensOutput), "\n");

}

static void printOutput(realtype t, N_Vector y, UserData data)
{
	fprintf((data->output), "%15.6e\t%15.6e\t%15.6e\t",t,T(1),data->P);
	for (int k = 1; k <=data->nsp; k++) {
		fprintf((data->output), "%15.6e\t",Y(1,k));
	}
	fprintf((data->output), "\n");
}

static void printIgnSensitivities(realtype sensCoeffs[], int indices[], UserData data)
{
	char buf[100];
	std::string line;
	for(int j=0;j<data->nreac;j++){
		line=data->gas->reactionString(indices[j]);
		sprintf(buf,"%s",line.c_str());
		fprintf((data->ignSensOutput), "%d\t\"%35s\"\t%15.6e\n",indices[j],buf,sensCoeffs[j]);
	}
	//for(int j=0;j<data->nreac;j++){
	//	fprintf((data->ignSensOutput), "%d\t%15.6e\n",indices[j],sensCoeffs[j]);
	//}
}

static int parsedPdt(UserData data){
  	/* Open file containing P and dPdt as functions of time: */
	data->dPdtInput=fopen("dPdt.dat","r");
	if(data->dPdtInput==NULL){
		printf("The dPdt.dat file wasn't found!\n");
		return(-1);
	}
	char buf[1000];
	char comment[1];
	char *ret;
	int i=0;
	int j=0;
	while (fgets(buf,100, data->dPdtInput)!=NULL){
		comment[0]=buf[0];
		if(strncmp(comment,"#",1)!=0 &&
		   strncmp(comment,"\n",1)!=0){
			i++;
		}
	}
	printf("No: of rows with numbers: %d\n",i);
	rewind(data->dPdtInput);
	int nPts=i;
	data->tArraySize=nPts;
	i=0;

  	data->tArray 	= N_VNew_Serial(nPts);
  	data->PArray 	= N_VNew_Serial(nPts);
  	data->dPdtArray = N_VNew_Serial(nPts);

	while (fgets(buf,100, data->dPdtInput)!=NULL){
		comment[0]=buf[0];
		if(strncmp(comment,"#",1)==0 ||
		   strncmp(comment,"\n",1)==0){
			printf("Comment! Skip Line!\n");
		}
		else{
			j=0;
			ret=strtok(buf,", \t");
			t(i)=atof(ret);
			j++;
			while(ret!=NULL){
   		  		ret=strtok(NULL,", \t");
				if(j==1){
					P(i)=atof(ret);
				}
				else if(j==2){
					dPdt(i)=atof(ret);
				}
				j++;
			}
			i++;
   		}
	}
	fclose(data->dPdtInput);
	//for(int k=0;k<nPts;k++){
	//	printf("%15.6e\t%15.6e\t%15.6e\n",t(k),P(k),dPdt(k));
	//}
	data->t1=t(nPts-1);
	data->P=P(0)*Cantera::OneAtm;

	/*check the polynomial interpolation (testing only)*/
	//realtype t,P,dPdt;
	//t=1e-03;
	//P=dPdt=0.0e0;
	//lookupDpdt(data, t, &P, &dPdt);
	//printf("Interpolated P value %15.6e. \n",P);
	//printf("Interpolated dPdt value %15.6e. \n",dPdt);
	//
	
	//GSL additions here:
	data->acc=gsl_interp_accel_alloc();
	data->spline=gsl_spline_alloc(gsl_interp_steffen,data->tArraySize);

	data->accdot=gsl_interp_accel_alloc();
	data->splinedot=gsl_spline_alloc(gsl_interp_steffen,data->tArraySize);

	double* Pdata;
	double* dPdtdata;
	double* tdata;

	tdata=N_VGetArrayPointer_Serial(data->tArray);
	Pdata=N_VGetArrayPointer_Serial(data->PArray);
	dPdtdata=N_VGetArrayPointer_Serial(data->dPdtArray);

	gsl_spline_init(data->spline,tdata,Pdata,data->tArraySize);
	gsl_spline_init(data->splinedot,tdata,dPdtdata,data->tArraySize);

	return(0);
}

static void PrintFinalStats(void *cvode_mem)
{
	long int nst, nfe, nsetups, nje, nfeLS, nni, ncfn, netf, nge;
	int flag;
	
	flag = CVodeGetNumSteps(cvode_mem, &nst);
	check_flag(&flag, "CVodeGetNumSteps", 1);
	flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
	check_flag(&flag, "CVodeGetNumRhsEvals", 1);
	flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
	check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
	flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
	check_flag(&flag, "CVodeGetNumErrTestFails", 1);
	flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
	check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
	flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
	check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);
	
	flag = CVDlsGetNumJacEvals(cvode_mem, &nje);
	check_flag(&flag, "CVDlsGetNumJacEvals", 1);
	flag = CVDlsGetNumRhsEvals(cvode_mem, &nfeLS);
	check_flag(&flag, "CVDlsGetNumRhsEvals", 1);
	
	flag = CVodeGetNumGEvals(cvode_mem, &nge);
	check_flag(&flag, "CVodeGetNumGEvals", 1);
	
	printf("\nFinal CVode Statistics:\n");
	printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n",
	       nst, nfe, nsetups, nfeLS, nje);
	printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n \n",
		 nni, ncfn, netf, nge);
}

static int check_flag(void *flagvalue, const char *funcname, int opt)
{
	int *errflag;
	
	/* Check if SUNDIALS function returned NULL pointer - no memory allocated */
	if (opt == 0 && flagvalue == NULL) {
	  fprintf(stderr, 
	          "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
	          funcname);
	  return(1);
	}
	
	/* Check if flag < 0 */
	else if (opt == 1) {
	  errflag = (int *) flagvalue;
	  if (*errflag < 0) {
	    fprintf(stderr,
	            "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
	            funcname, *errflag);
	    return(1); 
	  }
	}
	
	/* Check if function returned NULL pointer - no memory allocated */
	else if (opt == 2 && flagvalue == NULL) {
	  fprintf(stderr,
	          "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
	          funcname);
	  return(1);
	}
	
	return(0);
}
