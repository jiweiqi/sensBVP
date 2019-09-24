/*
sensBVP: A program that calculates the sensitivity of ignition delay time to
kinetic rate parameters by using a boundary value problem formulation.
Copyright (C) 2019  Vyaas Gururajan

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
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

#include <kinsol/kinsol.h>             /* access to KINSOL func., consts. */
//#include <kinsol/kinsol_impl.h>        /* access to KINSOL data structs */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix        */
#include <kinsol/kinsol_direct.h>      /* access to KINDls interface      */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver  */
//#include <sunlinsol/sunlinsol_lapackband.h>  /* access to band SUNLinearSolver  */

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

#define Ttemp(i)   NV_Ith_S(ytemp,((i-1)*(data->nvar+1))+data->nt)
#define Ytemp(i,k) NV_Ith_S(ytemp,((i-1)*(data->nvar+1))+data->ny+k-1)
#define tautemp(i) NV_Ith_S(ytemp,((i-1)*(data->nvar+1))+data->ntau)

#define Tp(i)   NV_Ith_S(yp,((i-1)*(data->nvar+1))+data->nt)
#define Yp(i,k) NV_Ith_S(yp,((i-1)*(data->nvar+1))+data->ny+k-1)
#define taup(i) NV_Ith_S(yp,((i-1)*(data->nvar+1))+data->ntau)

#define Pp(i)      NV_Ith_S(data->Pp,i-1)
#define dPdtp(i)   NV_Ith_S(data->dPdtp,i-1)

#define Tres(i)   NV_Ith_S(res,((i-1)*(data->nvar+1))+data->nt)
#define Yres(i,k) NV_Ith_S(res,((i-1)*(data->nvar+1))+data->ny+k-1)
#define taures(i) NV_Ith_S(res,((i-1)*(data->nvar+1))+data->ntau)

#define tautmp1(i) NV_Ith_S(tmp1,((i-1)*(data->nvar+1))+data->ntau)
#define tautmp2(i) NV_Ith_S(tmp2,((i-1)*(data->nvar+1))+data->ntau)

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
	bool BVPSuccess;
	bool sensSuccess;
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
	realtype ftol;
	
  	realtype t1;		//Maximum time to run simulation
	int nreac;		
	int pert_index;		//index of reaction whose rate is perturbed
	bool sensitivityAnalysis;// boolean to activate perturbations for
				//sensitivity analysis
	bool sens;		//if true, perform sensitivity analysis
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

/* Evaluate the residue for the BVP using this subroutine: */
static int residueKS(N_Vector yp, N_Vector res, void *user_data);

static int parseInput(UserData data, int argc, char *argv[]);

static int hunt(realtype x, realtype xx[], int n, int jguess);

static void polint(realtype *xdata, realtype *f, int n, realtype x,realtype *y, realtype *dy);

static void lookupDpdt(UserData data, realtype t, realtype *P, realtype *dPdt);

/* Subroutine to compute the Jacobian using numerical differencing: */
static int kinDlsBandDQJac(N_Vector u, N_Vector fu, SUNMatrix Jac, UserData data, N_Vector scale, N_Vector tmp1, N_Vector tmp2);

/* Subroutine that prints the column titles in the output file: */
static void printInstructions();

/* Subroutine that prints the column titles in the output file: */
static void printHeader(UserData data);
static void printSensitivitiesHeader(UserData data);

/* Subroutine that prints the output of the IVP into the output file contained
 * in the UserData structure: */
static void printOutput(realtype t, N_Vector y, UserData data);

/* Subroutine that prints the output of the BVP into the output file contained
 * in the UserData structure: */
//static void printOutputKS(N_Vector yp, UserData data);

/* Subroutine that prints the residue of the BVP into the output file contained
 * in the UserData structure: */
//static void printResidueKS(N_Vector res, UserData data);

/* Subroutine that prints the species sensitivities into the output file
 * (speSensOutput) contained in the UserData structure: */
static void printSpeciesSensitivities(int index, N_Vector yp, N_Vector res, UserData data);

/* Subroutine that prints the sensitivities into the output file (ignSensOutput)
 * contained in the UserData structure: */
static void printIgnSensitivities(realtype sensCoeffs[], int indices[], UserData data);

static int parsedPdt(UserData data);

/* Print solver statistics for the IVP: */
static void PrintFinalStats(void *cvode_mem);

/* Print solver statistics for the BVP: */
static void PrintFinalStatsKS(void *kmem);

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

	/* Set the maximum number of time-steps that will be used in the BVP
	 * formulation of the problem: */
	int nout=50000;

	/* Create a temporary solution array to save the results of the IVP for
	 * use in the BVP:*/
  	N_Vector ytemp;
	data->npts=nout;
	data->KSneq=(data->nvar+1)*data->npts;
	ytemp=N_VNew_Serial(data->KSneq);

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
		
		/*Save the initial values in the temporary solution vector:*/
		Ttemp(1)=T(1);
		for(int k=1;k<=data->nsp;k++){
			Ytemp(1,k)=Y(1,k);
		}
		tautemp(1)=tret;

		/*Begin IVP solution; Save solutions in the temporary solution vector;
		 * stop problem when the temperature reaches a certain value (like 400
		 * K) corresponding to the time for ignition :*/
		int i=1;
		bool BVPCutOff=false;
		int skip=10;
		int count=0;
		while (tret<=data->t1) {
			if(i<data->npts){
				if(!BVPCutOff){
					Ttemp(i+1)=T(1);
					for(int k=1;k<=data->nsp;k++){
						Ytemp(i+1,k)=Y(1,k);
					}
					tautemp(i+1)=tret;
				}
			}
			else{
				printf("Need more points!\n");
				printf("Hard-coded npts=%d not enough!\n",data->npts);
				data->IVPSuccess=false;
				break;
			}
			if(T(1)>=data->TIgn && !BVPCutOff){
				printf("Ignition Delay: %15.6es\n", tret);
				BVPCutOff=true;
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

			count++;

			if(tret>1e-06 && !BVPCutOff && count%skip==0){
				i++;
				count=0;
			}

		}

		data->npts=i;
		data->KSneq=(data->nvar+1)*data->npts;

  		/* Print remaining counters and free memory. */
		PrintFinalStats(mem);
  		CVodeFree(&mem);
  		N_VDestroy_Serial(y);
	}
	
	/*Create a solution vector yp, dimensioned such that the maximum no: of
	 * grid points associated with it is the no: of time-steps (i) used
	 * above. Fill in its values using the temporary solution vector. */
  	N_Vector yp;
	if(data->IVPSuccess){
		yp=N_VNew_Serial(data->KSneq);
		for(int j=1;j<=data->npts;j++){
			Tp(j)=Ttemp(j);
			for(int k=1;k<=data->nsp;k++){
				if(fabs(Ytemp(j,k))>1e-16){
					Yp(j,k)=Ytemp(j,k);
				}
				else{
					Yp(j,k)=ZERO;
				}
			}
			taup(j)=tautemp(j);
		}

		data->tauIgn=taup(data->npts);	//Save the ignition delay time
		data->TIgn=Tp(data->npts);	//Save the temperature corresponding to
						//the ignition delay
		/* Create a grid (in time) to be used in the solution of the BVP: */
  		data->x = N_VNew_Serial(data->npts);
  		if(check_flag((void *)data->x, "N_VNew_Serial", 0)) return(1);

		if(data->imposedPdt){
  			data->Pp    = N_VNew_Serial(data->npts);
  			if(check_flag((void *)data->Pp, "N_VNew_Serial", 0)) return(1);
  			data->dPdtp = N_VNew_Serial(data->npts);
  			if(check_flag((void *)data->dPdtp, "N_VNew_Serial", 0)) return(1);
		}

		realtype P,dPdt;
		for (int j = 1; j <=data->npts; j++) {
			if(data->imposedPdt){	
				//P=dPdt=ZERO;
				lookupDpdt(data, taup(j), &P, &dPdt);
				Pp(j)=P*Cantera::OneAtm;
				dPdtp(j)=dPdt*Cantera::OneAtm;
				//printf("%15.6e\t%15.6e\n",Pp(j),dPdtp(j));
			}
			x(j)=taup(j)/data->tauIgn;
			taup(j)=data->tauIgn;
		}
		if(data->tArray!=NULL){
  			N_VDestroy_Serial(data->tArray);
		}
		if(data->PArray!=NULL){
  			N_VDestroy_Serial(data->PArray);
		}
		if(data->dPdtArray!=NULL){
  			N_VDestroy_Serial(data->dPdtArray);
		}
		//printOutputKS(yp,data);
	}

  	N_VDestroy_Serial(ytemp);	//Destroy the temporary solution vector
	printf("No: of time-steps: %d\n",data->npts);
	
	/********************************************************/

	/*Begin solution of the BVP here:*/
	if(data->IVPSuccess && data->sens){
		/*Create a KINSOL solver object for the solution of the BVP:*/
		/* Solver Memory:*/
  		void *mem; mem = NULL;
  		mem = KINCreate();
  		//if (check_flag((void *)mem, "KINCreate", 0)) return(1);

		/*Associate the user data with the solver object: */
  		ier = KINSetUserData(mem, data);
  		ier=check_flag(&ier, "KINSetUserData", 1);

		/* Initialize the solver object by connecting it to the solution vector
		 * yp and the residue "residueKS": */
  		ier = KINInit(mem, residueKS, yp);
  		ier = check_flag(&ier, "KINInit", 1);

  		/* Specify stopping tolerance based on residual: */
  		ier = KINSetFuncNormTol(mem, data->ftol);
  		ier = check_flag(&ier, "KINSetFuncNormTol", 1);
		
		/* Create banded SUNMatrix for use in linear solves; bandwidths are
		 * dictated by the dependence of the solution at a given time on one
		 * time-step ahead and one time-step behind:*/
		SUNMatrix J;
		int mu,ml;
		mu = data->nvar+2; ml = data->nvar+2;
  		J = SUNBandMatrix(data->KSneq, mu, ml, mu+ml);
  		ier = check_flag((void *)J, "SUNBandMatrix", 0);

      		/* Create dense SUNLinearSolver object for use by KINSOL: */
  		SUNLinearSolver LSK;
  		LSK = SUNBandLinearSolver(yp, J);
  		ier = check_flag((void *)LSK, "SUNBandLinearSolver", 0);

		/* Call KINDlsSetLinearSolver to attach the matrix and linear solver to
		 * KINSOL: */
  		ier = KINDlsSetLinearSolver(mem, LSK, J);
  		ier = check_flag(&ier, "KINDlsSetLinearSolver", 1);

  		/* No scaling used: */
		N_Vector scale;
		scale=NULL;
  		scale = N_VNew_Serial(data->KSneq);
  		//ier = check_flag((void *)scale, "N_VNew_Serial", 0);
  		N_VConst_Serial(ONE,scale);

  		/* Force a Jacobian re-evaluation every mset iterations: */
  		int mset = 100;
  		ier = KINSetMaxSetupCalls(mem, mset);
  		//ier = check_flag(&ier, "KINSetMaxSetupCalls", 1);

  		/* Every msubset iterations, test if a Jacobian evaluation
  		   is necessary: */
  		int msubset = 1;
  		ier = KINSetMaxSubSetupCalls(mem, msubset);
  		//ier = check_flag(&ier, "KINSetMaxSubSetupCalls", 1);

		ier=KINSetPrintLevel(mem,2);

  		/* Solve the BVP! */
  		ier = KINSol(mem,     /* KINSol memory block */
  		             yp,      /* initial guess on input; solution vector */
  		             KIN_NONE,//KIN_LINESEARCH,/* global strategy choice */
  		             scale,   /* scaling vector, for the variable cc */
  		             scale);  /* scaling vector for function values fval */

  		if (check_flag(&ier, "KINSol", 1)){
			data->BVPSuccess=false;
		}else{
			data->BVPSuccess=true;
			/* Get scaled norm of the system function */
			ier = KINGetFuncNorm(mem, &data->ftol);
			//ier = check_flag(&ier, "KINGetfuncNorm", 1);
			printf("\nComputed solution (||F|| = %g):\n\n",data->ftol);
			printf("KinSOL Ignition Delay: %15.6es\n", taup(1));
		}
  		//ier=check_flag(&ier, "KINSol", 1);

		//printOutputKS(yp,data);
		fclose(data->output);
		PrintFinalStatsKS(mem);
  		KINFree(&mem);
  		N_VDestroy_Serial(scale);
	}

	/********************************************************/
	/*Begin sensitivity analysis here:*/
	if(data->BVPSuccess){
		/* Create banded SUNMatrix for use in linear solves; bandwidths
		 * are dictated by the dependence of the solution at a given
		 * time on one time-step ahead and one time-step behind:*/
		SUNMatrix J;
		int mu,ml;
		mu = data->nvar+1; ml = data->nvar+1;
  		J = SUNBandMatrix(data->KSneq, mu, ml, mu+ml);
  		ier = check_flag((void *)J, "SUNBandMatrix", 0);

		/*Create a residue vector and two temporary vectors:*/
  		N_Vector res,tmp1,tmp2, scale;
		res=N_VNew_Serial(data->KSneq);
		tmp1=N_VNew_Serial(data->KSneq);
		tmp2=N_VNew_Serial(data->KSneq);
  		scale = N_VNew_Serial(data->KSneq);
  		N_VConst_Serial(ONE,scale);

		/*Evaluate the residue using the solution computed by solving
		 * the BVP above:*/
		ier = residueKS(yp, res, data);
		ier = check_flag(&ier, "residueKS", 1);
		//printResidueKS(res,data);
       
       		/*Compute the Jacobian and store it in J:*/	
		kinDlsBandDQJac(yp, res, J, data, scale, tmp1, tmp2);

		/*Create a linear solver object for the banded system; in the
		 * problem Ax=b, A is J and x is tmp2:*/
  		SUNLinearSolver LS;
  		LS = SUNBandLinearSolver(res, J);

		/*Initialize the solver and perform an LU factorization by
		 * running SUNLinSolSetup:*/
		ier = SUNLinSolInitialize_Band(LS);
		ier = check_flag(&ier, "SUNLinSolInitialize", 1);
		ier = SUNLinSolSetup_Band(LS,J);
		ier = check_flag(&ier, "SUNLinSolSetup", 1);

		/*Create an array to store the logarithmic sensitivity
		 * coefficients:*/
		realtype sensCoeffs[data->nreac];

		/*Create an array to store the indices of the reactions (needed
		 * for sorting the sensitivity coefficients):*/
		int indices[data->nreac];

		/*The sensitivities are computed as follows:
		 * For a system equations
		 *    F(y;α)=0
		 *    where F is the residue of the BVP
		 *    	    y is the solution of the BVP
		 *    	    α is a parameter 
		 * => (∂F/∂y)(∂y/∂α)+(∂F/∂α) = 0
		 * => (∂F/∂y)(∂y/∂α)=-(∂F/∂α)
		 * Therefore, the sensitivity coefficient (∂y/∂α) can be
		 * computed by solving a system Ax=b, where A=(∂F/∂y),
		 * x=(∂y/∂α), and b=-(∂F/∂α)!*/

		/*Run a loop over all the reactions: */
		printf("\nRunning Sensitivity Analysis...\n");
		/*Enable sensitivity analysis:*/
		data->sensitivityAnalysis=true;
		realtype oneOverRel=ONE/(data->rel);
		for(int j=0;j<data->nreac;j++){

			/*Save the index of the reaction whose collision
			 * frequency (A in the Arrhenius law k=A*exp(-Eₐ/RT))
			 * is to be perturbed:*/ 
			data->pert_index=j;	

			/*Compute the perturbed residue and store it in tmp1:*/
			ier=residueKS(yp, tmp1, data);
			ier=check_flag(&ier, "residueKS", 1);
			//printResidueKS(tmp1,data);

			/*Compute the difference F(α+Δα)-F(α) and store it in
			 * tmp2:*/
			N_VLinearSum(ONE,tmp1,-ONE,res,tmp2);

			/*Divide the difference quotient by -Δα and store the
			 * result in tmp1. This is the numerical approximation
			 * to -(∂F/∂α):*/
			N_VScale(-oneOverRel,tmp2,tmp1);

			/*Solve the linear system (thankfully the LU
			 * factiorization has already been carried out once and
			 * for all above!) and store the solution in tmp2:*/
			ier=SUNLinSolSolve_Band(LS,J,tmp2,tmp1,ZERO);
			ier=check_flag(&ier, "SUNLinSolSolve", 1);

			/*Divide the result by the ignition delay time to get
			 * the logarithmic sensitivity coefficient and save it
			 * in sensCoeffs:*/
			sensCoeffs[j]=tautmp2(1)/taup(1);

			if(sensCoeffs[j]!=sensCoeffs[j]){
				printf("\nNaN! Quitting Program! Try different tolerances!\n");
				data->sensSuccess=false;
				break;
			}else{
				data->sensSuccess=true;
				printf("Reaction %5d: %15.6e\n",j,sensCoeffs[j]);
			}

			/*Print out the temperature and species mass-fraction
			 * sensitivities:*/
			if(data->writeSpeciesSensitivities){
				printSpeciesSensitivities(j, yp, tmp2, data);
			}

			/*Save the index of the reaction:*/
			indices[j]=j;
		}

		if(data->sensSuccess){
			/*Sort the sensitivities in ascending order. Note the advancing
			 * of the beginning indices of the sensCoeffs and indices
			 * arrays. This is due to the Numerical recipes convention for
			 * array indexing used in sort subroutine:*/
			//sort(data->nreac,sensCoeffs-1,indices-1);

			/*Print out the sensitivities:*/
			printIgnSensitivities(sensCoeffs,indices,data);
		}

  		N_VDestroy_Serial(scale);
  		N_VDestroy_Serial(res);
  		N_VDestroy_Serial(tmp1);
  		N_VDestroy_Serial(tmp2);
		fclose(data->ignSensOutput);
		fclose(data->speSensOutput);
	}
	
	/*Free memory and delete all the vectors and user data:*/
	if(yp!=NULL){
  		N_VDestroy_Serial(yp);
		printf("BVP Solution Vector Deleted!\n");
	}
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
	/*Residue tolerance for BVP:*/
	data->ftol = RCONST(1e-13);
	/*Final Time:*/
	data->t1   = RCONST(10.0);
	/*Solve constant pressure problem:*/
	data->constantVolume=false;	
	/*Do not impose a P vs t curve:*/
	data->imposedPdt=false;		
	/*Disable writing species sensitivities:*/
	data->writeSpeciesSensitivities=false;
	/*Enable sensitivity analysis:*/
	data->sens=false;
	/*Index to start with for pressure lookup (see hunt):*/
	data->jguess=0;			
	/*Set rate of change of pressure to zero:*/
	data->dPdt=ZERO;
	/*Disable sensitivity analysis:*/
	data->sensitivityAnalysis=false;
	/*Find the relative perturbation constant:*/
	data->rel=SUNRsqrt(UNIT_ROUNDOFF);
	/*Set flags that indicate success of various stages:*/
	data->IVPSuccess=false;
	data->BVPSuccess=false;
	data->sensSuccess=false;
	/*****************************************************/

	int ier;
	int opt;
	char mech[BUFSIZE+1];
	char comp[BUFSIZE+1];
	bool enteredT0, enteredP, enteredMech, enteredComp;
	enteredT0=enteredP=enteredMech=enteredComp=false;
	while((opt=getopt(argc,argv,"a:r:f:T:P:m:c:t:vsd")) != -1){
		switch(opt){
			case 'a':
				data->atol=RCONST(atof(optarg));
				break;
			case 'r':
				data->rtol=RCONST(atof(optarg));
				break;
			case 'f':
				data->ftol=RCONST(atof(optarg));
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
		printf("\nftol: %15.6e\n", data->ftol); 
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

	T(1)=data->gas->temperature();
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

    	} catch (Cantera::CanteraError& err) {
	    printf("Error:\n");
	    printf("%s\n",err.what());
	    return(-1);
    	}

	return(0);
}

static int residueKS(N_Vector yp, N_Vector res, void *user_data)
{

	/*This is the sub-routine that computes the right hand side F(y) in the
	 * problem F(y) = 0. Here the problem is specifically that of a
	 * homogeneous isobaric batch reactor formulated as a boundary value
	 * problem (BVP). The unknowns in this problem are T, Yᵢ, and τ.
	 *
	 * The energy conservation equation is 
	 * 	  ∂T/∂x = (τ/ρcₚ)(dP/dt - ∑ ώᵢhᵢ)
	 *
	 * T has units of K
	 * x is *unitless*
	 * τ has units of time
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
	 * 	  ∂T/∂x = -(τ/ρcᵥ)∑ ώᵢeᵢ
	 * eᵢ has units J/kmol
	 * cᵥ has units J/kg/K
	 *
	 * The species conservation equation is
	 * 	∂Yᵢ/∂x = τώᵢWᵢ/ρ
	 *
	 * Wᵢ has units of kg/kmol
	 *
	 * The equation (trivial) to maintain a banded structure in the linear
	 * solver is 
	 * 	∂τ/∂x = 0
	 */
	/*Assign data structure type to user_data:*/
	UserData data;
	data=(UserData)user_data;

	/*Get no: of species:*/
	int nsp=data->nsp;
	
	/*Create temporary arrays to store enthalpy (non-dimensional) and ώᵢ:*/
	realtype wdot[nsp];
	realtype enthalpy[nsp];
	
	/*Create temperorary variables to store grid spacing and summation
	 * terms:*/
	realtype dx;
	realtype sum=ZERO;
	data->dPdt=ZERO;

    	try {
		/*If sensitivity analysis has been enabled, perturb the collision
		 * frequency by using cantera's setMultiplier function:*/
		if(data->sensitivityAnalysis){
			data->gas->setMultiplier(data->pert_index,ONE+data->rel);
			//printf("%d\t%15.9e\n",data->pert_index,ONE+data->rel);
		}

		Tres(1)=Tp(1)-data->T0;
		for(int k=1;k<=nsp;k++){
			Yres(1,k)=Yp(1,k)-Y0(k);
		}
		taures(1)=taup(2)-taup(1);


		for(int i=2;i<=data->npts;i++){

			if(data->constantVolume){
  				data->gas->setMassFractions_NoNorm(&Yp(i,1)); 	
  				data->gas->setTemperature(Tp(i)); 	        
  				data->gas->setDensity(data->rho0);
			}
			else if(data->imposedPdt){
				//printf("%15.6e\t%15.6e\t%15.6e\n",x(i),Tp(i),Pp(i));
  				data->gas->setMassFractions_NoNorm(&Yp(i,1)); 	
  				data->gas->setTemperature(Tp(i)); 	        
  				data->gas->setPressure(Pp(i));

				data->dPdt=dPdtp(i);
			}
			else{
  				data->gas->setMassFractions_NoNorm(&Yp(i,1)); 	
  				data->gas->setTemperature(Tp(i)); 	        
  				data->gas->setPressure(data->P);
			}

			realtype rho=data->gas->density();
			realtype cp=data->gas->cp_mass();
			data->gas->getNetProductionRates(wdot);
			data->gas->getEnthalpy_RT(enthalpy);

			if(data->constantVolume){
				for(int k=1;k<=nsp;k++){
					enthalpy(k)=enthalpy(k)-ONE;
				}
				cp=data->gas->cv_mass();
			}

			dx=x(i)-x(i-1);
			sum=ZERO;

			for(int k=1;k<=nsp;k++){
				Yres(i,k)=(Yp(i,k)-Yp(i-1,k))/dx;
				Yres(i,k)+=-taup(i)*wdot(k)*(data->gas->molecularWeight(k-1))/rho;
				sum+=wdot(k)*enthalpy(k)*Tp(i);
			}
			sum=sum*Cantera::GasConstant;
			Tres(i)=(Tp(i)-Tp(i-1))/dx;
			Tres(i)+=taup(i)*(sum-data->dPdt)/(rho*cp);
			if(i==data->npts){
				taures(i)=Tp(i)-data->TIgn;
			}
			else{
				taures(i)=taup(i+1)-taup(i);
			}

		}

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

	realtype *dPdtData;
	dPdtData = N_VGetArrayPointer_Serial(data->dPdtArray);

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

/*------------------------------------------------------------------
  kinDlsBandDQJac

  This routine generates a banded difference quotient approximation
  to the Jacobian of F(u).  It assumes a SUNBandMatrix input stored
  column-wise, and that elements within each column are contiguous.
  This makes it possible to get the address of a column of J via the
  function SUNBandMatrix_Column() and to write a simple for loop to
  set each of the elements of a column in succession.
 
  NOTE: Any type of failure of the system function her leads to an
        unrecoverable failure of the Jacobian function and thus of
        the linear solver setup function, stopping KINSOL.
  ------------------------------------------------------------------*/
static int kinDlsBandDQJac(N_Vector u, N_Vector fu, SUNMatrix Jac,
                    UserData data, N_Vector scale, N_Vector tmp1, N_Vector tmp2)
{
	realtype inc, inc_inv;
	N_Vector futemp, utemp;
	sunindextype group, i, j, width, ngroups, i1, i2;
	sunindextype N, mupper, mlower;
	realtype *col_j, *fu_data, *futemp_data, *u_data, *utemp_data, *uscale_data;
	int retval = 0;
	//KINDlsMem kindls_mem;
	
	//KINMem kin_mem;
	//kin_mem = (KINMem) mem;
	
	///* access DlsMem interface structure */
	//kindls_mem = (KINDlsMem) kin_mem->kin_lmem;
	
	/* access matrix dimensions */
	N = SUNBandMatrix_Columns(Jac);
	mupper = SUNBandMatrix_UpperBandwidth(Jac);
	mlower = SUNBandMatrix_LowerBandwidth(Jac);
	
	/* Rename work vectors for use as temporary values of u and fu */
	futemp = tmp1;
	utemp  = tmp2;
	
	/* Obtain pointers to the data for ewt, fy, futemp, y, ytemp */
	fu_data     = N_VGetArrayPointer(fu);
	futemp_data = N_VGetArrayPointer(futemp);
	u_data      = N_VGetArrayPointer(u);
	//uscale_data = N_VGetArrayPointer(kin_mem->kin_uscale);
	uscale_data = N_VGetArrayPointer(scale);
	utemp_data  = N_VGetArrayPointer(utemp);
	
	/* Load utemp with u */
	N_VScale(ONE, u, utemp);
	
	/* Set bandwidth and number of column groups for band differencing */
	width   = mlower + mupper + 1;
	ngroups = SUNMIN(width, N);
	
	//UserData data;
	//data=(UserData) kin_mem->kin_user_data;
	
	for (group=1; group <= ngroups; group++) {
	  
	  /* Increment all utemp components in group */
	  for(j=group-1; j < N; j+=width) {
	    //inc = kin_mem->kin_sqrt_relfunc*SUNMAX(SUNRabs(u_data[j]),
	    //                                       ONE/SUNRabs(uscale_data[j]));
	    //inc = data->rel*SUNRabs(u_data[j]);
	    inc = data->rel*SUNMAX(SUNRabs(u_data[j]),
	                           ONE/SUNRabs(uscale_data[j]));
	    utemp_data[j] += inc;
	  }
	
	  /* Evaluate f with incremented u */
	  //retval = kin_mem->kin_func(utemp, futemp, kin_mem->kin_user_data);
	  retval=residueKS(utemp, futemp, data);
	  if (retval != 0) return(retval); 
	
	  /* Restore utemp components, then form and load difference quotients */
	  for (j=group-1; j < N; j+=width) {
	    utemp_data[j] = u_data[j];
	    col_j = SUNBandMatrix_Column(Jac, j);
	    //inc = kin_mem->kin_sqrt_relfunc*SUNMAX(SUNRabs(u_data[j]),
	    //                                       ONE/SUNRabs(uscale_data[j]));
	    //inc = kin_mem->kin_sqrt_relfunc*SUNRabs(u_data[j]);
	    //inc = data->rel*SUNRabs(u_data[j]);
	    inc = data->rel*SUNMAX(SUNRabs(u_data[j]),
	                           ONE/SUNRabs(uscale_data[j]));
	    inc_inv = ONE/inc;
	    i1 = SUNMAX(0, j-mupper);
	    i2 = SUNMIN(j+mlower, N-1);
	    for (i=i1; i <= i2; i++)
	      SM_COLUMN_ELEMENT_B(col_j,i,j) = inc_inv * (futemp_data[i] - fu_data[i]);
	  }
	}
	
	///* Increment counter nfeDQ */
	//kindls_mem->nfeDQ += ngroups;
	
	
	return(0);
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
	printf("\n-f <function tolerance for BVP>\n");
	printf("\n-T <Initial Temperature in Kelvin>\n");
	printf("\n-P <Initial Pressure in atm>\n");
	printf("\n-m <mechanism file (cti or xml)>\n");
	printf("\n-c <composition in mole fractions>\n");
	printf("\n-t <Total simulation time in seconds>\n");
	printf("\n-v :Enables constant volume simulation\n");
	printf("\n-s :Enables ignition delay sensitivity output\n");
	printf("\n-S :Enables species mass fraction sensitivity output (works only with -s)\n");
	printf("\n-d :Enables manual dPdt entryspecies sensitivity output\n");
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


//static void printOutputKS(N_Vector yp, UserData data)
//{
//	//printf("%d\n",data->npts);
//	fprintf(data->output, "\n\n");
//	for(int i=1;i<=data->npts;i++){
//		fprintf((data->output), "%15.6e\t%15.6e\t%15.6e\t",x(i)*data->tauIgn,Tp(i),data->P);
//		//printf("%15.6e\t%15.6e\n",x(i),Tp(i));
//		for (int k = 1; k <=data->nsp; k++) {
//			fprintf((data->output), "%15.6e\t",Yp(i,k));
//		}
//		fprintf(data->output, "\n");
//	}
//}

static void printSpeciesSensitivities(int index, N_Vector yp, N_Vector res, UserData data)
{
	char buf[50];
	std::string line;
	line=data->gas->reactionString(index);
	sprintf(buf,"%s",line.c_str());
	fprintf((data->speSensOutput), "#%d: %38s\t\n", index, buf);
	printSensitivitiesHeader(data);
	for(int i=1;i<=data->npts;i++){
		fprintf((data->speSensOutput), "%15.6e\t%15.6e\t",x(i)*taup(i),Tres(i)/Tp(i));
		for (int k = 1; k <=data->nsp; k++) {
			if(Yp(i,k)>=data->atol){
				fprintf((data->speSensOutput), "%15.6e\t",Yres(i,k)/(Yp(i,k)+1e-31));
			}
			else{
				fprintf((data->speSensOutput), "%15.6e\t",ZERO);
			}
		}
		fprintf((data->speSensOutput), "\n");
	}
	fprintf((data->speSensOutput), "\n\n");
}

//static void printResidueKS(N_Vector res, UserData data)
//{
//	fprintf((data->output), "\n\n");
//	for(int i=1;i<=data->npts;i++){
//		fprintf((data->output), "%15.6e\t%15.6e\t",x(i),Tres(i));
//		for (int k = 1; k <=data->nsp; k++) {
//			fprintf((data->output), "%15.6e\t",Yres(i,k));
//		}
//		fprintf((data->output), "\n");
//	}
//}

static void printIgnSensitivities(realtype sensCoeffs[], int indices[], UserData data)
{
	char buf[50];
	std::string line;
	for(int j=0;j<data->nreac;j++){
		line=data->gas->reactionString(indices[j]);
		sprintf(buf,"%s",line.c_str());
		fprintf((data->ignSensOutput), "%d\t\"%35s\"\t%15.6e\n",indices[j],buf,sensCoeffs[j]);
	}
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

static void PrintFinalStatsKS(void *kmem)
{
	long int nni, nfe, nje, nfeD;
	long int lenrw, leniw, lenrwB, leniwB;
	long int nbcfails, nbacktr;
	int flag;
	
	/* Main solver statistics */
	
	flag = KINGetNumNonlinSolvIters(kmem, &nni);
	check_flag(&flag, "KINGetNumNonlinSolvIters", 1);
	flag = KINGetNumFuncEvals(kmem, &nfe);
	check_flag(&flag, "KINGetNumFuncEvals", 1);
	
	/* Linesearch statistics */
	
	flag = KINGetNumBetaCondFails(kmem, &nbcfails);
	check_flag(&flag, "KINGetNumBetacondFails", 1);
	flag = KINGetNumBacktrackOps(kmem, &nbacktr);
	check_flag(&flag, "KINGetNumBacktrackOps", 1);
	
	/* Main solver workspace size */
	
	flag = KINGetWorkSpace(kmem, &lenrw, &leniw);
	check_flag(&flag, "KINGetWorkSpace", 1);
	
	/* Band linear solver statistics */
	
	flag = KINDlsGetNumJacEvals(kmem, &nje);
	check_flag(&flag, "KINDlsGetNumJacEvals", 1);
	flag = KINDlsGetNumFuncEvals(kmem, &nfeD);
	check_flag(&flag, "KINDlsGetNumFuncEvals", 1);
	
	/* Band linear solver workspace size */
	
	flag = KINDlsGetWorkSpace(kmem, &lenrwB, &leniwB);
	check_flag(&flag, "KINDlsGetWorkSpace", 1);
	
	printf("\nFinal KINSOL Statistics:\n");
	printf("nni      = %6ld    nfe     = %6ld \n", nni, nfe);
	printf("nbcfails = %6ld    nbacktr = %6ld \n", nbcfails, nbacktr);
	printf("nje      = %6ld    nfeB    = %6ld \n", nje, nfeD);
	printf("\n");
	printf("lenrw    = %6ld    leniw   = %6ld \n", lenrw, leniw);
	printf("lenrwB   = %6ld    leniwB  = %6ld \n", lenrwB, leniwB);

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
