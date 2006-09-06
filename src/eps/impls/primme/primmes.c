
/*                       
       This file implements a wrapper to the PRIMME library
*/

#include "src/eps/epsimpl.h"

EXTERN_C_BEGIN
#include "primme.h"
EXTERN_C_END

#ifndef PETSC_USE_COMPLEX
typedef double PRIMMEScalar;
#else
typedef Complex_Z PRIMMEScalar;
#endif

typedef struct {
  primme_params primme;           /* param struc */
  primme_preset_method method;    /* primme method */
  Mat A;                          /* problem matrix */
  Vec M;                          /* store reciprocal A diagonal */ 
} EPS_PRIMME;

void multMatvec_PRIMME(PRIMMEScalar *in, PRIMMEScalar *out, int *blockSize, primme_params *primme);
void applyPreconditioner_PRIMME(PRIMMEScalar *in, PRIMMEScalar *out, int *blockSize, struct primme_params *primme);
void par_GlobalSumDouble(void *sendBuf, void *recvBuf, int *count, primme_params *primme);

void par_GlobalSumDouble(void *sendBuf, void *recvBuf, int *count, primme_params *primme) {
  MPI_Allreduce((double*)sendBuf, (double*)recvBuf, *count, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_PRIMME"
PetscErrorCode EPSSetUp_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N, n, numProcs, procID;
  PetscScalar    *a; 
  Mat B;
  EPS_PRIMME     *ops = (EPS_PRIMME *)eps->data;
  primme_params  *primme = &(((EPS_PRIMME *)eps->data)->primme);

  PetscFunctionBegin;

  MPI_Comm_size(PETSC_COMM_WORLD, &numProcs);
  MPI_Comm_rank(PETSC_COMM_WORLD, &procID);
  
  /* Check some constrains and set some default values */ 
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(eps->vec_initial,&n);CHKERRQ(ierr);
  if (!eps->ncv) eps->ncv = eps->nev+primme->maxBlockSize;
  if (eps->ncv < eps->nev+primme->maxBlockSize)
    SETERRQ(PETSC_ERR_SUP,"PRIMME needs ncv >= nev+maxBlockSize");
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  ierr = STGetOperators(eps->OP, &ops->A, &B);
  if (!ops->A) SETERRQ(PETSC_ERR_SUP,"PRIMME needs a matrix A");
  if (B) SETERRQ(PETSC_ERR_SUP,"PRIMME doesn't solve generalized problem");

  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"PRIMME was only built for Hermitian problems");

  /* Transfer SLEPc options to PRIMME options */
  primme->n = N;
  primme->nLocal = n;
  primme->numEvals = eps->nev; 
  primme->matrix = &ops->A;
  primme->initSize = 1;
  /* primme->maxBasisSize = eps->ncv; */
  primme->maxMatvecs = eps->max_it; 
  primme->eps = eps->tol; 
  primme->numProcs = numProcs; 
  primme->procID = procID;
  primme->globalSumDouble = par_GlobalSumDouble;

  switch(eps->which) {
    case EPS_LARGEST_MAGNITUDE:
      primme->target = primme_largest;
      break;

    case EPS_SMALLEST_MAGNITUDE:
      primme->target = primme_smallest;
      break;
       
    default:
      SETERRQ(PETSC_ERR_SUP,"PRIMME only allows EPS_LARGEST_MAGNITUDE and EPS_SMALLEST_MAGNITUDE for 'which' value");
      break;   
  }
  primme_set_method(ops->method, primme);
  
  /* Set workspace */
  ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);
  
  /* Copy vec_initial to V[0] vector */
  ierr = VecGetArray(eps->vec_initial, &a); CHKERRQ(ierr); 
  ierr = VecCopy(eps->vec_initial, eps->V[0]); CHKERRQ(ierr);
  ierr = VecRestoreArray(eps->vec_initial, &a); CHKERRQ(ierr);
 
  if (primme->correctionParams.precondition == 1 ) {
    /* Calc reciprocal A diagonal */
    ierr = VecDuplicate(eps->vec_initial, &ops->M); CHKERRQ(ierr);
    ierr = MatGetDiagonal(ops->A, ops->M); CHKERRQ(ierr);
    ierr = VecReciprocal(ops->M); CHKERRQ(ierr);
    primme->preconditioner = &ops->M;
    primme->applyPreconditioner = applyPreconditioner_PRIMME;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_PRIMME"
PetscErrorCode EPSSolve_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME    *ops = (EPS_PRIMME *)eps->data;
  PetscScalar   *a;
#ifdef PETSC_USE_COMPLEX
  int i;
#endif

  PetscFunctionBegin;

  /* Call PRIMME solver */
  ierr = VecGetArray(eps->V[0], &a); CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
  ierr = dprimme(eps->eigr, a, eps->errest, &ops->primme);
#else
  /* In order to optimize memory, we are going to use eps->eigr and eps->eigr like a (double*) */
  ierr = zprimme((double*)eps->eigr, (Complex_Z*)a, (double*)eps->errest, &ops->primme);
#endif
  ierr = VecRestoreArray(eps->V[0], &a); CHKERRQ(ierr);
  
  switch(ierr) {
    case 0: /* Successful */
      break;

    case -1:
      SETERRQ(PETSC_ERR_SUP,"PRIMME: Failed to open output file");
      break;

    case -2:
      SETERRQ(PETSC_ERR_SUP,"PRIMME: Insufficient integer or real workspace allocated");
      break;

    case -3:
      SETERRQ(PETSC_ERR_SUP,"PRIMME: main_iter encountered a problem");
      break;

    default:
      SETERRQ(PETSC_ERR_SUP,"PRIMME: some parameters wrong configured");
      break;
  }

  eps->nconv = ops->primme.initSize>=0?ops->primme.initSize:0;
  eps->reason = EPS_CONVERGED_TOL;
  eps->its = ops->primme.stats.numMatvecs;

#ifdef PETSC_USE_COMPLEX
  /* PRIMME return real eigenvalues, but SLEPc work with complex ones, so we must cast them */
  ierr = PetscMalloc(eps->nconv*sizeof(PetscScalar), &a); CHKERRQ(ierr);
  for(i=0; i<eps->nconv; i++)
    a[i] = PetscScalar(((double*)eps->eigr)[i], 0.0);
  ierr = PetscFree(eps->eigr); CHKERRQ(ierr);
  eps->eigr = a;
#else
  ierr = PetscMemzero(eps->eigi, eps->nconv*sizeof(double)); CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

void multMatvec_PRIMME(PRIMMEScalar *in, PRIMMEScalar *out, int *blockSize, primme_params *primme)
{
  int i, ierr, N = primme->n;
  Vec x,y;
  Mat *A = (Mat*)(primme->matrix);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,primme->nLocal,N,PETSC_NULL,&x);/*CHKERRQ(ierr);*/
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,primme->nLocal,N,PETSC_NULL,&y);/*CHKERRQ(ierr);*/

  for (i=0;i<*blockSize;i++) {
    /* build vectors using 'in' an 'out' workspace */
    ierr = VecPlaceArray(x, (PetscScalar*)in+N*i );/*CHKERRQ(ierr);*/
    ierr = VecPlaceArray(y, (PetscScalar*)out+N*i );/*CHKERRQ(ierr);*/

    ierr = MatMult(*A, x, y);/*CHKERRQ(ierr);*/
    
    ierr = VecResetArray(x);/*CHKERRQ(ierr);*/
    ierr = VecResetArray(y);/*CHKERRQ(ierr);*/
  }

  ierr = VecDestroy(x);/*CHKERRQ(ierr);*/
  ierr = VecDestroy(y);/*CHKERRQ(ierr);*/
}

void applyPreconditioner_PRIMME(PRIMMEScalar *in, PRIMMEScalar *out, int *blockSize, struct primme_params *primme)
{
  int i, ierr, N = primme->n;
  Vec x,y,M = *(Vec*)(primme->preconditioner);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,primme->nLocal,N,PETSC_NULL,&x);/*CHKERRQ(ierr);*/
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,primme->nLocal,N,PETSC_NULL,&y);/*CHKERRQ(ierr);*/

  for (i=0;i<*blockSize;i++) {
    /* build vectors using 'in' an 'out' workspace */
    ierr = VecPlaceArray(x, (PetscScalar*)in+N*i );/*CHKERRQ(ierr);*/
    ierr = VecPlaceArray(y, (PetscScalar*)out+N*i );/*CHKERRQ(ierr);*/

    ierr = VecPointwiseMult(y, M, x);/*CHKERRQ(ierr);*/
    
    ierr = VecResetArray(x);/*CHKERRQ(ierr);*/
    ierr = VecResetArray(y);/*CHKERRQ(ierr);*/
  }

  ierr = VecDestroy(x);/*CHKERRQ(ierr);*/
  ierr = VecDestroy(y);/*CHKERRQ(ierr);*/
} 


#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_PRIMME"
PetscErrorCode EPSDestroy_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME    *ops = (EPS_PRIMME *)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  
  primme_Free(&ops->primme);
  ierr = PetscFree(eps->data); CHKERRQ(ierr);
  ierr = EPSFreeSolutionContiguous(eps);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_PRIMME"
PetscErrorCode EPSView_PRIMME(EPS eps,PetscViewer viewer)
{
  PetscFunctionBegin;
  SETERRQ1(1,"Viewer type %s not supported for PRIMME",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_PRIMME"
PetscErrorCode EPSSetFromOptions_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME    *ops = (EPS_PRIMME *)eps->data;
  PetscInt       op;
  PetscTruth     flg;
  const char *methodList[] = {
    "DEFAULT_MIN_TIME",
    "DEFAULT_MIN_MATVECS",
    "Arnoldi",
    "GD",
    "GD_plusK",
    "GD_Olsen_plusK",
    "JD_Olsen_plusK",
    "RQI",
    "JDQR",
    "JDQMR",
    "JDQMR_ETol",
    "SUBSPACE_ITERATION",
    "LOBPCG_OrthoBasis",
    "LOBPCG_OrthoBasis_Window"
  }, *restartList[] = {"thick", "dtr"},
  *precondList[] = {"none", "diagonal"};
  EPS_primme_preset_method methodN[] = {
    EPSPRIMME_DEFAULT_MIN_TIME,
    EPSPRIMME_DEFAULT_MIN_MATVECS,
    EPSPRIMME_Arnoldi,
    EPSPRIMME_GD,
    EPSPRIMME_GD_plusK,
    EPSPRIMME_GD_Olsen_plusK,
    EPSPRIMME_JD_Olsen_plusK,
    EPSPRIMME_RQI,
    EPSPRIMME_JDQR,
    EPSPRIMME_JDQMR,
    EPSPRIMME_JDQMR_ETol,
    EPSPRIMME_SUBSPACE_ITERATION,
    EPSPRIMME_LOBPCG_OrthoBasis,
    EPSPRIMME_LOBPCG_OrthoBasis_Window
  };
  EPS_primme_restartscheme restartN[] = {EPSPRIMME_thick, EPSPRIMME_dtr};

  PetscFunctionBegin;
  
  ierr = PetscOptionsHead("PRIMME options");CHKERRQ(ierr);

  op = ops->primme.maxBlockSize; 
  ierr = PetscOptionsInt("-eps_primme_maxBlockSize"," Max block size","EPSPRIMMESetBlockSize",op,&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetBlockSize(eps,op);CHKERRQ(ierr);}
  op = 0;
  ierr = PetscOptionsEList("-eps_primme_method","set method for solving the eigenproblem",
                           "EPSPRIMMESetMethod",methodList,14,methodList[0],&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetMethod(eps, methodN[op]);CHKERRQ(ierr);}
  ierr = PetscOptionsEList("-eps_primme_restart","set the restarting scheme",
                           "EPSPRIMMESetRestart",restartList,2,restartList[0],&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetRestart(eps, restartN[op]);CHKERRQ(ierr);}
  ierr = PetscOptionsEList("-eps_primme_precond","set precondition",
                           "EPSPRIMMESetPrecond",precondList,2,precondList[0],&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetPrecond(eps, op);CHKERRQ(ierr);}
  
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetBlockSize_PRIMME"
PetscErrorCode EPSPRIMMESetBlockSize_PRIMME(EPS eps,int bs)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (bs == PETSC_DEFAULT) ops->primme.maxBlockSize = 1;
  else if (bs <= 0) { 
    SETERRQ(1, "PRIMME: wrong block size"); 
  } else ops->primme.maxBlockSize = bs;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetBlockSize"
/*@C
    EPSPRIMMESetBlockSize - The maximum block size the code will try to use. The user should set
    this based on the architecture specifics of the target computer, 
    as well as any a priori knowledge of multiplicities. The code does 
    NOT require BlockSize > 1 to find multiple eigenvalues.  For some 
    methods, keeping BlockSize = 1 yields the best overall performance.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  bs - block size

   Options Database Key:
    .  -eps_primme_maxBlockSize - Sets the max allowed block size value

   Note:
    +   If it doesn't set it keeps the value established by primme_initialize.
    -   Inner iterations of QMR are not performed in a block fashion. Every correction equation from a block is solved independently.

   Level: advanced
.seealso: EPSPRIMMEGetBlockSize()
@*/
PetscErrorCode EPSPRIMMESetBlockSize(EPS eps,int bs)
{
  PetscErrorCode ierr, (*f)(EPS,int);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMESetBlockSize_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,bs);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetBlockSize_PRIMME"
PetscErrorCode EPSPRIMMEGetBlockSize_PRIMME(EPS eps,int *bs)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (bs) *bs = ops->primme.maxBlockSize;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetBlockSize"
/*@C
    EPSPRIMMEGetBlockSize - Get the maximum block size the code will try to use. 
    
    Collective on EPS

   Input Parameters:
    .  eps - the eigenproblem solver context
    
   Output Parameters:  
    .  bs - returned block size 

   Level: advanced
.seealso: EPSPRIMMESetBlockSize()
@*/
PetscErrorCode EPSPRIMMEGetBlockSize(EPS eps,int *bs)
{
  PetscErrorCode ierr, (*f)(EPS,int*);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMEGetBlockSize_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,bs);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetMethod_PRIMME"
PetscErrorCode EPSPRIMMESetMethod_PRIMME(EPS eps, EPS_primme_preset_method method)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (method == PETSC_DEFAULT) ops->method = DEFAULT_MIN_TIME;
  else ops->method = (primme_preset_method)method;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetMethod"
/*@C
   EPSPRIMMESetMethod - Sets the method for the PRIMME library.

   Collective on EPS

   Input Parameters:
    +  eps - the eigenproblem solver context
    -  method - method that will be used by PRIMME. It must be one of next them: EPSPRIMME_DEFAULT_MIN_TIME(EPSPRIMME_JDQMR_ETol),
    EPSPRIMME_DEFAULT_MIN_MATVECS(EPSPRIMME_GD_Olsen_plusK), EPSPRIMME_Arnoldi, EPSPRIMME_GD, EPSPRIMME_GD_plusK, EPSPRIMME_GD_Olsen_plusK, EPSPRIMME_JD_Olsen_plusK, EPSPRIMME_RQI, EPSPRIMME_JDQR, EPSPRIMME_JDQMR,
    EPSPRIMME_JDQMR_ETol, EPSPRIMME_SUBSPACE_ITERATION, EPSPRIMME_LOBPCG_OrthoBasis, EPSPRIMME_LOBPCG_OrthoBasis_Window

   Options Database Key:
    .  -eps_primme_set_method - Sets the method for the PRIMME library.

   Note: If it doesn't set it does EPSPRIMME_DEFAULT_MIN_TIME.

   Level: advanced
.seealso: EPSPRIMMEGetMethod()  
@*/
PetscErrorCode EPSPRIMMESetMethod(EPS eps, EPS_primme_preset_method method)
{
  PetscErrorCode ierr, (*f)(EPS,EPS_primme_preset_method);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMESetMethod_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,method); CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetMethod_PRIMME"
PetscErrorCode EPSPRIMMEGetMethod_PRIMME(EPS eps, EPS_primme_preset_method *method)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (method) *method = (EPS_primme_preset_method)ops->method;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetMethod"
/*@C
    EPSPRIMMEGetMethod - Gets the method for the PRIMME library.

    Mon Collective on EPS

   Input Parameters:
    .  eps - the eigenproblem solver context
    
   Output Parameters: 
    .  method - method that will be used by PRIMME. It must be one of next them: DEFAULT_MIN_TIME(JDQMR_ETol),
    DEFAULT_MIN_MATVECS(GD_Olsen_plusK), Arnoldi, GD, GD_plusK, GD_Olsen_plusK, JD_Olsen_plusK, RQI, JDQR, JDQMR,
    JDQMR_ETol, SUBSPACE_ITERATION, LOBPCG_OrthoBasis, LOBPCG_OrthoBasis_Window

    Level: advanced
.seealso: EPSPRIMMESetMethod()
@*/
PetscErrorCode EPSPRIMMEGetMethod(EPS eps, EPS_primme_preset_method *method)
{
  PetscErrorCode ierr, (*f)(EPS,EPS_primme_preset_method*);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMEGetMethod_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,method); CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetRestart_PRIMME"
PetscErrorCode EPSPRIMMESetRestart_PRIMME(EPS eps, EPS_primme_restartscheme scheme)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;
  
  if (scheme == PETSC_DEFAULT) ops->primme.restartingParams.scheme = primme_thick;
  else ops->primme.restartingParams.scheme = (primme_restartscheme)scheme;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetRestart"
/*@C
    EPSPRIMMESetRestart - Sets the restarting scheme for the PRIMME library.

    Collective on EPS

   Input Parameters:
    +  eps - the eigenproblem solver context
    -  scheme - possible values are: EPSPRIMME_thick(thick restarting) is the most efficient and robust
    in the general case, and EPSPRIMME_dtr(dynamic thick restarting) helpful without 
    preconditioning but it is expensive to implement.

   Options Database Key:
    .  -eps_primme_restart - Sets the restarting scheme for the PRIMME library

    Level: advanced

@*/
PetscErrorCode EPSPRIMMESetRestart(EPS eps, EPS_primme_restartscheme scheme)
{
  PetscErrorCode ierr, (*f)(EPS, EPS_primme_restartscheme);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMESetRestart_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps, scheme);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetRestart_PRIMME"
    PetscErrorCode EPSPRIMMEGetRestart_PRIMME(EPS eps, EPS_primme_restartscheme *scheme)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;
  
  if (scheme) *scheme = (EPS_primme_restartscheme)ops->primme.restartingParams.scheme;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetRestart"
/*@C
    EPSPRIMMEGetRestart - Gets the restarting scheme for the PRIMME library.

    Non Collective on EPS

   Input Parameters:
    .  eps - the eigenproblem solver context
    
    Output Parameters:  
    .  scheme - possible values are: EPSPRIMME_thick(thick restarting) is the most efficient and robust
    in the general case, and EPSPRIMME_dtr(dynamic thick restarting) helpful without 
    preconditioning but it is expensive to implement.

    Level: advanced
.seealso: EPSPRIMMESetRestart()
@*/
PetscErrorCode EPSPRIMMEGetRestart(EPS eps, EPS_primme_restartscheme *scheme)
{
  PetscErrorCode ierr, (*f)(EPS, EPS_primme_restartscheme*);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMEGetRestart_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps, scheme);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetPrecond_PRIMME"
PetscErrorCode EPSPRIMMESetPrecond_PRIMME(EPS eps, int pre)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (pre == PETSC_DEFAULT) ops->primme.correctionParams.precondition = 0;
  else if (pre != 0 && pre != 1) {
    SETERRQ(1, "PRIMME: wrong precondition value (allowed 0 or 1)");
  } else ops->primme.correctionParams.precondition = pre;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetPrecond"
/*@C
    EPSPRIMMESetPrecond - Sets either none or the diagonal matrix like preconditioner for the PRIMME library.

    Collective on EPS

   Input Parameters:
    +  eps - the eigenproblem solver context
    -  pre - posible values are: 0, no preconditioning and 1, diagonal matrix for preconditioning

   Options Database Key:
    .  -eps_primme_precond - Sets either none or the diagonal matrix like preconditioner for the PRIMME library

    Note:
      The default values is 0, i. e., no preconditioning.
    
    Level: advanced
@*/
PetscErrorCode EPSPRIMMESetPrecond(EPS eps, int pre)
{
  PetscErrorCode ierr, (*f)(EPS, int);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMESetPrecond_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps, pre);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetPrecond_PRIMME"
    PetscErrorCode EPSPRIMMEGetPrecond_PRIMME(EPS eps, int *pre)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (pre) *pre = ops->primme.correctionParams.precondition;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetPrecond"
/*@C
    EPSPRIMMEGetPrecond - Gets if the diagonal matrix is going to use like preconditioner for the PRIMME library.

    Collective on EPS

   Input Parameters:
    .  eps - the eigenproblem solver context
    
  Output Parameters:
    .  pre - posible values are: 0, no preconditioning and 1, diagonal matrix for preconditioning

    Level: advanced
.seealso: EPSPRIMMEGetPrecond()
@*/
PetscErrorCode EPSPRIMMEGetPrecond(EPS eps, int *pre)
{
  PetscErrorCode ierr, (*f)(EPS, int*);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMEGetPrecond_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps, pre);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_PRIMME"
PetscErrorCode EPSCreate_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME     *primme;

  PetscFunctionBegin;
  
  ierr = PetscNew(EPS_PRIMME,&primme);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_PRIMME));
  eps->data                      = (void *) primme;
  eps->ops->solve                = EPSSolve_PRIMME;
  eps->ops->setup                = EPSSetUp_PRIMME;
  eps->ops->setfromoptions       = EPSSetFromOptions_PRIMME;
  eps->ops->destroy              = EPSDestroy_PRIMME;
  eps->ops->view                 = EPSView_PRIMME;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;

  primme_initialize(&primme->primme);
  primme->primme.matrixMatvec = multMatvec_PRIMME;
  primme->method = DEFAULT_MIN_TIME;
  primme->A = 0;
  primme->M = 0;
   
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetBlockSize_C","EPSPRIMMESetBlockSize_PRIMME",EPSPRIMMESetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetMethod_C","EPSPRIMMESetMethod_PRIMME",EPSPRIMMESetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetRestart_C","EPSPRIMMESetRestart_PRIMME",EPSPRIMMESetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetPrecond_C","EPSPRIMMESetPrecond_PRIMME",EPSPRIMMESetMethod_PRIMME);CHKERRQ(ierr); 
  
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetBlockSize_C","EPSPRIMMEGetBlockSize_PRIMME",EPSPRIMMEGetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetMethod_C","EPSPRIMMEGetMethod_PRIMME",EPSPRIMMEGetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetRestart_C","EPSPRIMMEGetRestart_PRIMME",EPSPRIMMEGetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetPrecond_C","EPSPRIMMEGetPrecond_PRIMME",EPSPRIMMEGetMethod_PRIMME);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}
EXTERN_C_END


