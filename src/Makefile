compiler	=g++
CANTERA_DIR	=/opt/scientific/cantera-2.4_gnu_blas
CVODE_DIR	=/opt/scientific/sundials-3.1.1_gnu_blas
KINSOL_DIR	=/opt/scientific/sundials-3.1.1_gnu_blas
BVPEXE		=sensBVP
BRUTEEXE	=sensBrute
DESTDIR		=~/bin

CANTERA_INCLUDES=-I$(CANTERA_DIR)/include
CVODE_INCLUDES	=-I$(CVODE_DIR)/include
KINSOL_INCLUDES	=-I$(KINSOL_DIR)/include
CVODE_LIBS	=-L$(CVODE_DIR)/lib -lsundials_nvecserial -lsundials_cvode
KINSOL_LIBS	=-L$(KINSOL_DIR)/lib -lsundials_kinsol 
CANTERA_LIBS 	=-L$(CANTERA_DIR)/lib -lcantera_shared
GSL_INCLUDES	=-I/usr/include/gsl
GSL_LIBS 	=-L/usr/lib -lgsl -lgslcblas
RPATH		=-Wl,-rpath=$(CVODE_DIR)/lib,-rpath=$(KINSOL_DIR)/lib
RM=rm -f

compiler?=g++

ifeq ($(compiler),g++)
	CPPFLAGS= -Wall -O3
	CPP=g++
endif

ifeq ($(compiler),icpc)
	export GXX_INCLUDE=/usr/lib/gcc/x86_64-pc-linux-gnu/7.4.1/include/c++
	CPPFLAGS= -Wall -O3 -gxx-name=/usr/bin/g++-7 -std=c++11
	CPP=icpc
endif

all: $(BVPEXE) $(BRUTEEXE)

sensBVP.o: sensBVP.cpp
	$(CPP) $(CPPFLAGS) $(CANTERA_INCLUDES) $(CVODE_INCLUDES) \
		$(KINSOL_INCLUDES) $(GSL_INCLUDES) \
		-c sensBVP.cpp -o sensBVP.o

sensBrute.o: sensBrute.cpp
	$(CPP) $(CPPFLAGS) $(CANTERA_INCLUDES) $(CVODE_INCLUDES) \
		$(GSL_INCLUDES) \
		-c sensBrute.cpp -o sensBrute.o

$(BVPEXE): sensBVP.o
	$(CPP) $(CPPFLAGS) \
		sensBVP.o -o $(BVPEXE) $(RPATH) $(CVODE_LIBS)  \
		$(KINSOL_LIBS) $(CANTERA_LIBS) $(GSL_LIBS)

$(BRUTEEXE): sensBrute.o
	$(CPP) $(CPPFLAGS) \
		sensBrute.o -o $(BRUTEEXE) $(RPATH) $(CVODE_LIBS)  \
		$(CANTERA_LIBS) $(GSL_LIBS)

.PHONY: install
install:
	cp $(BVPEXE) $(BRUTEEXE) $(DESTDIR)

clean:
	rm -f $(BVPEXE) $(BRUTEEXE) *.o *.d

