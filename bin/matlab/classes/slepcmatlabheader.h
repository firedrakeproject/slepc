
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
int EPSCreate(MPI_Comm,EPS *);
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
int EPSComputeRelativeError(EPS,int,double*);
int EPSView(EPS,PetscViewer);
int EPSGetST(EPS,ST*);
int EPSDestroy(EPS*);

