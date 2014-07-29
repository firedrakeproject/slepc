
/*
   This is used by share/slepc/matlab/classes/SlepcInitialize() to define to
   Matlab all the functions available in the SLEPc shared library. We cannot
   simply use the regular SLEPc include files because they are too complicated
   for Matlab to parse.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
int SlepcInitializeNoPointers(int,char **,const char*,const char*);
int SlepcInitialized(int*);
typedef int MPI_Comm;
int SlepcFinalize(void);

typedef int PetscBool;
typedef long int PetscPointer;
typedef PetscPointer Vec;
typedef PetscPointer Mat;
typedef PetscPointer KSP;
typedef PetscPointer PetscViewer;

typedef PetscPointer ST;
int STCreate(MPI_Comm,ST *);
int STSetType(ST,const char*);
int STSetFromOptions(ST);
int STSetUp(ST);
int STSetOperators(ST,int,Mat*);
int STView(ST,PetscViewer);
int STGetKSP(ST,KSP*);
int STDestroy(ST*);

typedef PetscPointer EPS;
typedef int EPSProblemType;
typedef int EPSWhich;
typedef int EPSExtraction;
int EPSCreate(MPI_Comm,EPS*);
int EPSSetType(EPS,const char*);
int EPSSetFromOptions(EPS);
int EPSSetOperators(EPS,Mat,Mat);
int EPSSetProblemType(EPS,EPSProblemType);
int EPSSetWhichEigenpairs(EPS,EPSWhich);
int EPSSetTarget(EPS,double);
int EPSSetExtraction(EPS,EPSExtraction);
int EPSSetTolerances(EPS,double,int);
int EPSSetDimensions(EPS,int,int,int);
int EPSSolve(EPS);
int EPSSetUp(EPS);
int EPSGetConverged(EPS,int*);
int EPSGetEigenvalue(EPS,int,double*,double*);
int EPSGetEigenvector(EPS,int,Vec,Vec);
int EPSGetOperators(EPS,Mat*,Mat*);
int EPSComputeRelativeError(EPS,int,double*);
int EPSView(EPS,PetscViewer);
int EPSGetST(EPS,ST*);
int EPSDestroy(EPS*);

typedef PetscPointer SVD;
typedef int SVDWhich;
int SVDCreate(MPI_Comm,SVD*);
int SVDSetType(SVD,const char*);
int SVDSetFromOptions(SVD);
int SVDSetOperator(SVD,Mat);
int SVDSetWhichSingularTriplets(SVD,SVDWhich);
int SVDSetTolerances(SVD,double,int);
int SVDSetImplicitTranspose(SVD,PetscBool);
int SVDSetDimensions(SVD,int,int,int);
int SVDSolve(SVD);
int SVDSetUp(SVD);
int SVDGetConverged(SVD,int*);
int SVDGetSingularTriplet(SVD,int,double*,Vec,Vec);
int SVDGetOperator(SVD,Mat*);
int SVDComputeRelativeError(SVD,int,double*);
int SVDView(SVD,PetscViewer);
int SVDDestroy(SVD*);

typedef PetscPointer PEP;
typedef int PEPProblemType;
typedef int PEPWhich;
int PEPCreate(MPI_Comm,PEP*);
int PEPSetType(PEP,const char*);
int PEPSetFromOptions(PEP);
int PEPSetOperators(PEP,int,Mat*);
int PEPSetProblemType(PEP,PEPProblemType);
int PEPSetWhichEigenpairs(PEP,PEPWhich);
int PEPSetTolerances(PEP,double,int);
int PEPSetDimensions(PEP,int,int,int);
int PEPSolve(PEP);
int PEPSetUp(PEP);
int PEPGetConverged(PEP,int*);
int PEPGetEigenpair(PEP,int,double*,double*,Vec,Vec);
int PEPGetOperators(PEP,int,Mat*);
int PEPComputeRelativeError(PEP,int,double*);
int PEPView(PEP,PetscViewer);
int PEPDestroy(PEP*);

