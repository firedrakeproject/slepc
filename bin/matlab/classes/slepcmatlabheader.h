
/*
    This is used by bin/matlab/classes/SlepcInitialize() to define to Matlab all the functions available in the 
   SLEPc shared library. We cannot simply use the regular SLEPc include files because they are too complicated for 
   Matlab to parse.

*/
int SlepcInitializeMatlab(int,char **,const char*,const char*);
int SlepcInitializedMatlab(void);
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
int STSetOperators(ST,Mat,Mat);
int STSetUp(ST);
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
int EPSGetEigenpair(EPS,int,double*,double*,Vec,Vec);
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
int SVDSetDimensions(SVD,int,int,int);
int SVDSolve(SVD);
int SVDSetUp(SVD);
int SVDGetConverged(SVD,int*);
int SVDGetSingularTriplet(SVD,int,double*,Vec,Vec);
int SVDComputeRelativeError(SVD,int,double*);
int SVDView(SVD,PetscViewer);
int SVDDestroy(SVD*);

typedef PetscPointer QEP;
typedef int QEPProblemType;
typedef int QEPWhich;
int QEPCreate(MPI_Comm,QEP*);
int QEPSetType(QEP,const char*);
int QEPSetFromOptions(QEP);
int QEPSetOperators(QEP,Mat,Mat,Mat);
int QEPSetProblemType(QEP,QEPProblemType);
int QEPSetWhichEigenpairs(QEP,QEPWhich);
int QEPSetScaleFactor(QEP,double);
int QEPSetTolerances(QEP,double,int);
int QEPSetDimensions(QEP,int,int,int);
int QEPSolve(QEP);
int QEPSetUp(QEP);
int QEPGetConverged(QEP,int*);
int QEPGetEigenpair(QEP,int,double*,double*,Vec,Vec);
int QEPComputeRelativeError(QEP,int,double*);
int QEPView(QEP,PetscViewer);
int QEPDestroy(QEP*);

