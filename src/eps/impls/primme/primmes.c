
/*                       
       This file implements a wrapper to the PRIMME library
*/

#include "src/eps/epsimpl.h"    /*I "slepceps.h" I*/

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
  Vec x,y;                        /* auxiliar vectors */ 
} EPS_PRIMME;

static void multMatvec_PRIMME(PRIMMEScalar *in, PRIMMEScalar *out, int *blockSize, primme_params *primme);
static void applyPreconditioner_PRIMME(PRIMMEScalar *in, PRIMMEScalar *out, int *blockSize, struct primme_params *primme);

static void par_GlobalSumDouble(void *sendBuf, void *recvBuf, int *count, primme_params *primme) {
  MPI_Allreduce((double*)sendBuf, (double*)recvBuf, *count, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_PRIMME"
PetscErrorCode EPSSetUp_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N, n;
  int            numProcs, procID;
  EPS_PRIMME     *ops = (EPS_PRIMME *)eps->data;
  primme_params  *primme = &(((EPS_PRIMME *)eps->data)->primme);

  PetscFunctionBegin;

  MPI_Comm_size(PETSC_COMM_WORLD, &numProcs);
  MPI_Comm_rank(PETSC_COMM_WORLD, &procID);
  
  /* Check some constrains and set some default values */ 
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(eps->vec_initial,&n);CHKERRQ(ierr);

  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  ierr = STGetOperators(eps->OP, &ops->A, PETSC_NULL);
  if (!ops->A) SETERRQ(PETSC_ERR_SUP,"PRIMME needs a matrix A");
  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"PRIMME is only available for Hermitian problems");
  if (eps->isgeneralized)
    SETERRQ(PETSC_ERR_SUP,"PRIMME is not available for generalized problems");

  /* Transfer SLEPc options to PRIMME options */
  primme->n = N;
  primme->nLocal = n;
  primme->numEvals = eps->nev; 
  primme->matrix = ops;
  primme->initSize = 1;
  primme->maxMatvecs = eps->max_it; 
  primme->eps = eps->tol; 
  primme->numProcs = numProcs; 
  primme->procID = procID;
  primme->globalSumDouble = par_GlobalSumDouble;

  switch(eps->which) {
    case EPS_LARGEST_REAL:
      primme->target = primme_largest;
      break;

    case EPS_SMALLEST_REAL:
      primme->target = primme_smallest;
      break;
       
    default:
      SETERRQ(PETSC_ERR_SUP,"PRIMME only allows EPS_LARGEST_REAL and EPS_SMALLEST_REAL for 'which' value");
      break;   
  }
  
  primme_set_method(ops->method, primme);
  
  /* If user sets ncv, maxBasisSize is modified. If not, ncv is set as maxBasisSize */
  if (eps->ncv) primme->maxBasisSize = eps->ncv;
  else eps->ncv = primme->maxBasisSize;
  if (eps->ncv < eps->nev+primme->maxBlockSize)  
    SETERRQ(PETSC_ERR_SUP,"PRIMME needs ncv >= nev+maxBlockSize");

  /* Set workspace */
  ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);

  /* Copy vec_initial to V[0] vector */
  ierr = VecCopy(eps->vec_initial, eps->V[0]); CHKERRQ(ierr);
 
  if (primme->correctionParams.precondition) {
    /* Calc reciprocal A diagonal */
    ierr = VecDuplicate(eps->vec_initial, &ops->M); CHKERRQ(ierr);
    ierr = MatGetDiagonal(ops->A, ops->M); CHKERRQ(ierr);
    ierr = VecReciprocal(ops->M); CHKERRQ(ierr);
    primme->preconditioner = PETSC_NULL;
    primme->applyPreconditioner = applyPreconditioner_PRIMME;
  }

  /* Prepare auxiliar vectors */ 
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,n,N,PETSC_NULL,&ops->x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,n,N,PETSC_NULL,&ops->y);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_PRIMME"
PetscErrorCode EPSSolve_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME     *ops = (EPS_PRIMME *)eps->data;
  PetscScalar    *a;
#ifdef PETSC_USE_COMPLEX
  int            i;
  PetscReal      *evals;
#endif

  PetscFunctionBegin;

  /* Call PRIMME solver */
  ierr = VecGetArray(eps->V[0], &a); CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
  ierr = dprimme(eps->eigr, a, eps->errest, &ops->primme);
#else
  /* PRIMME return real eigenvalues, but SLEPc work with complex ones */
  ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&evals); CHKERRQ(ierr);
  ierr = zprimme(evals, (Complex_Z*)a, eps->errest, &ops->primme);
  for (i=0;i<eps->ncv;i++)
    eps->eigr[i] = evals[i];
  ierr = PetscFree(evals); CHKERRQ(ierr);
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

#ifndef PETSC_USE_COMPLEX
  ierr = PetscMemzero(eps->eigi, eps->nconv*sizeof(double)); CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "multMatvec_PRIMME"
static void multMatvec_PRIMME(PRIMMEScalar *in, PRIMMEScalar *out, int *blockSize, primme_params *primme)
{
  PetscErrorCode ierr; int i, N = primme->n;
  EPS_PRIMME *ops = (EPS_PRIMME *)primme->matrix; 
  Vec x = ops->x, y = ops->y;
  Mat A = ops->A;

  PetscFunctionBegin;
  for (i=0;i<*blockSize;i++) {
    /* build vectors using 'in' an 'out' workspace */
    ierr = VecPlaceArray(x, (PetscScalar*)in+N*i ); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ierr = VecPlaceArray(y, (PetscScalar*)out+N*i ); CHKERRABORT(PETSC_COMM_WORLD,ierr);

    ierr = MatMult(A, x, y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    
    ierr = VecResetArray(x); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ierr = VecResetArray(y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  }
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "applyPreconditioner_PRIMME"
static void applyPreconditioner_PRIMME(PRIMMEScalar *in, PRIMMEScalar *out, int *blockSize, struct primme_params *primme)
{
  PetscErrorCode ierr; int i, N = primme->n;
  EPS_PRIMME *ops = (EPS_PRIMME *)primme->matrix; 
  Vec x = ops->x, y = ops->y;
  Vec M = ops->M;
 
  PetscFunctionBegin;
  for (i=0;i<*blockSize;i++) {
    /* build vectors using 'in' an 'out' workspace */
    ierr = VecPlaceArray(x, (PetscScalar*)in+N*i ); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ierr = VecPlaceArray(y, (PetscScalar*)out+N*i ); CHKERRABORT(PETSC_COMM_WORLD,ierr);

    ierr = VecPointwiseMult(y, M, x); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    
    ierr = VecResetArray(x); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ierr = VecResetArray(y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  }
  PetscFunctionReturnVoid();
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
  ierr = VecDestroy(ops->x);CHKERRQ(ierr);
  ierr = VecDestroy(ops->y);CHKERRQ(ierr);
  ierr = PetscFree(eps->data); CHKERRQ(ierr);
  ierr = EPSFreeSolutionContiguous(eps);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_PRIMME"
PetscErrorCode EPSView_PRIMME(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr; PetscTruth isascii;
  primme_params *primme = &((EPS_PRIMME *)eps->data)->primme;
  const char *methodList[] = {
    "default_min_time",
    "default_min_matvecs",
    "arnoldi",
    "gd",
    "gd_plusk",
    "gd_olsen_plusk",
    "jd_olsen_plusk",
    "rqi",
    "jdqr",
    "jdqmr",
    "jdqmr_etol",
    "subspace_iteration",
    "lobpcg_orthobasis",
    "lobpcg_orthobasis_window"
  },*precondList[] = {"none", "diagonal"};
  EPSPRIMMEMethod methodN=0;
  EPSPRIMMEPrecond precondN=0;
  
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSPRIMME",((PetscObject)viewer)->type_name);
  }
  
  ierr = PetscViewerASCIIPrintf(viewer,"PRIMME solver block size: %d\n",primme->maxBlockSize);CHKERRQ(ierr);
  ierr = EPSPRIMMEGetMethod(eps, &methodN);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"PRIMME solver method: %s\n", methodList[methodN]);CHKERRQ(ierr);
  ierr = EPSPRIMMEGetPrecond(eps, &precondN);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"PRIMME solver preconditioning: %s\n", precondList[precondN]);CHKERRQ(ierr);

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
    "default_min_time",
    "default_min_matvecs",
    "arnoldi",
    "gd",
    "gd_plusk",
    "gd_olsen_plusk",
    "jd_olsen_plusk",
    "rqi",
    "jdqr",
    "jdqmr",
    "jdqmr_etol",
    "subspace_iteration",
    "lobpcg_orthobasis",
    "lobpcg_orthobasis_window"
  },*precondList[] = {"none", "diagonal"};
  EPSPRIMMEMethod methodN[] = {
    EPSPRIMME_DEFAULT_MIN_TIME,
    EPSPRIMME_DEFAULT_MIN_MATVECS,
    EPSPRIMME_ARNOLDI,
    EPSPRIMME_GD,
    EPSPRIMME_GD_PLUSK,
    EPSPRIMME_GD_OLSEN_PLUSK,
    EPSPRIMME_JD_OlSEN_PLUSK,
    EPSPRIMME_RQI,
    EPSPRIMME_JDQR,
    EPSPRIMME_JDQMR,
    EPSPRIMME_JDQMR_ETOL,
    EPSPRIMME_SUBSPACE_ITERATION,
    EPSPRIMME_LOBPCG_ORTHOBASIS,
    EPSPRIMME_LOBPCG_ORTHOBASIS_WINDOW
  };
  EPSPRIMMEPrecond precondN[] = {EPSPRIMME_NONE, EPSPRIMME_DIAGONAL};

  PetscFunctionBegin;
  
  ierr = PetscOptionsHead("PRIMME options");CHKERRQ(ierr);

  op = ops->primme.maxBlockSize; 
  ierr = PetscOptionsInt("-eps_primme_block_size"," maximum block size","EPSPRIMMESetBlockSize",op,&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetBlockSize(eps,op);CHKERRQ(ierr);}
  op = 0;
  ierr = PetscOptionsEList("-eps_primme_method","set method for solving the eigenproblem",
                           "EPSPRIMMESetMethod",methodList,14,methodList[0],&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetMethod(eps, methodN[op]);CHKERRQ(ierr);}
  ierr = PetscOptionsEList("-eps_primme_precond","set precondition",
                           "EPSPRIMMESetPrecond",precondList,2,precondList[0],&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetPrecond(eps, precondN[op]);CHKERRQ(ierr);}
  
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
/*@
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
/*@
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
PetscErrorCode EPSPRIMMESetMethod_PRIMME(EPS eps, EPSPRIMMEMethod method)
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
/*@
   EPSPRIMMESetMethod - Sets the method for the PRIMME library.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  method - method that will be used by PRIMME. It must be one of next them: EPSPRIMME_DEFAULT_MIN_TIME(EPSPRIMME_JDQMR_ETOL),
    EPSPRIMME_DEFAULT_MIN_MATVECS(EPSPRIMME_GD_OLSEN_PLUSK), EPSPRIMME_ARNOLDI,
    EPSPRIMME_GD, EPSPRIMME_GD_PLUSK, EPSPRIMME_GD_OLSEN_PLUSK, EPSPRIMME_JD_OlSEN_PLUSK,
    EPSPRIMME_RQI, EPSPRIMME_JDQR, EPSPRIMME_JDQMR, EPSPRIMME_JDQMR_ETOL, EPSPRIMME_SUBSPACE_ITERATION,
    EPSPRIMME_LOBPCG_ORTHOBASIS, EPSPRIMME_LOBPCG_ORTHOBASIS_WINDOW

   Options Database Key:
.  -eps_primme_set_method - Sets the method for the PRIMME library.

   Note: If it doesn't set it does EPSPRIMME_DEFAULT_MIN_TIME.

   Level: advanced
.seealso: EPSPRIMMEGetMethod()  
@*/
PetscErrorCode EPSPRIMMESetMethod(EPS eps, EPSPRIMMEMethod method)
{
  PetscErrorCode ierr, (*f)(EPS,EPSPRIMMEMethod);

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
PetscErrorCode EPSPRIMMEGetMethod_PRIMME(EPS eps, EPSPRIMMEMethod *method)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (method) *method = (EPSPRIMMEMethod)ops->method;
  
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
.  method - method that will be used by PRIMME. It must be one of next them: EPSPRIMME_DEFAULT_MIN_TIME(EPSPRIMME_JDQMR_ETOL),
    EPSPRIMME_DEFAULT_MIN_MATVECS(EPSPRIMME_GD_OLSEN_PLUSK), EPSPRIMME_ARNOLDI,
    EPSPRIMME_GD, EPSPRIMME_GD_PLUSK, EPSPRIMME_GD_OLSEN_PLUSK, EPSPRIMME_JD_OlSEN_PLUSK,
    EPSPRIMME_RQI, EPSPRIMME_JDQR, EPSPRIMME_JDQMR, EPSPRIMME_JDQMR_ETOL, EPSPRIMME_SUBSPACE_ITERATION,
    EPSPRIMME_LOBPCG_ORTHOBASIS, EPSPRIMME_LOBPCG_ORTHOBASIS_WINDOW

    Level: advanced
.seealso: EPSPRIMMESetMethod()
@*/
PetscErrorCode EPSPRIMMEGetMethod(EPS eps, EPSPRIMMEMethod *method)
{
  PetscErrorCode ierr, (*f)(EPS,EPSPRIMMEMethod*);

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
#define __FUNCT__ "EPSPRIMMESetPrecond_PRIMME"
    PetscErrorCode EPSPRIMMESetPrecond_PRIMME(EPS eps, EPSPRIMMEPrecond precond)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (precond == EPSPRIMME_NONE) ops->primme.correctionParams.precondition = 0;
  else ops->primme.correctionParams.precondition = 1;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetPrecond"
/*@
    EPSPRIMMESetPrecond - Sets either none or the diagonal matrix like preconditioner for the PRIMME library.

    Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  precond - posible values are: EPSPRIMME_NONE, no preconditioning and EPSPRIMME_DIAGONAL, diagonal matrix for preconditioning

   Options Database Key:
.  -eps_primme_precond - Sets either none or the diagonal matrix like preconditioner for the PRIMME library

    Note:
      The default values is 0, i. e., no preconditioning.
    
    Level: advanced
.seealso: EPSPRIMMEGetPrecond()
@*/
PetscErrorCode EPSPRIMMESetPrecond(EPS eps, EPSPRIMMEPrecond precond)
{
  PetscErrorCode ierr, (*f)(EPS, EPSPRIMMEPrecond);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMESetPrecond_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps, precond);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMEGetPrecond_PRIMME"
    PetscErrorCode EPSPRIMMEGetPrecond_PRIMME(EPS eps, EPSPRIMMEPrecond *precond)
{
  EPS_PRIMME *ops = (EPS_PRIMME *) eps->data;

  PetscFunctionBegin;

  if (precond)
    *precond = ops->primme.correctionParams.precondition ? EPSPRIMME_DIAGONAL : EPSPRIMME_NONE;
  
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
.  precond - posible values are: EPSPRIMME_NONE, no preconditioning and EPSPRIMME_DIAGONAL, diagonal matrix for preconditioning

    Level: advanced
.seealso: EPSPRIMMESetPrecond()
@*/
PetscErrorCode EPSPRIMMEGetPrecond(EPS eps, EPSPRIMMEPrecond *precond)
{
  PetscErrorCode ierr, (*f)(EPS, EPSPRIMMEPrecond*);

  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSPRIMMEGetPrecond_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps, precond);CHKERRQ(ierr);
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
  primme->x = 0;
  primme->y = 0;
   
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetBlockSize_C","EPSPRIMMESetBlockSize_PRIMME",EPSPRIMMESetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetMethod_C","EPSPRIMMESetMethod_PRIMME",EPSPRIMMESetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetPrecond_C","EPSPRIMMESetPrecond_PRIMME",EPSPRIMMESetPrecond_PRIMME);CHKERRQ(ierr); 
  
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetBlockSize_C","EPSPRIMMEGetBlockSize_PRIMME",EPSPRIMMEGetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetMethod_C","EPSPRIMMEGetMethod_PRIMME",EPSPRIMMEGetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetPrecond_C","EPSPRIMMEGetPrecond_PRIMME",EPSPRIMMEGetPrecond_PRIMME);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}
EXTERN_C_END


