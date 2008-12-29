/*
       This file implements a wrapper to the PRIMME library
       Contributed by Eloy Romero

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/epsimpl.h"    /*I "slepceps.h" I*/
#include "private/stimpl.h"

EXTERN_C_BEGIN
#include "primme.h"
EXTERN_C_END

typedef struct {
  primme_params primme;           /* param struc */
  primme_preset_method method;    /* primme method */
  Mat A;                          /* problem matrix */
  Vec w;                          /* store reciprocal A diagonal */
  Vec x,y;                        /* auxiliar vectors */ 
} EPS_PRIMME;

const char *methodList[] = {
  "dynamic",
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
};
const char *precondList[] = {"none", "diagonal"};
EPSPRIMMEMethod methodN[] = {
  EPSPRIMME_DYNAMIC,
  EPSPRIMME_DEFAULT_MIN_TIME,
  EPSPRIMME_DEFAULT_MIN_MATVECS,
  EPSPRIMME_ARNOLDI,
  EPSPRIMME_GD,
  EPSPRIMME_GD_PLUSK,
  EPSPRIMME_GD_OLSEN_PLUSK,
  EPSPRIMME_JD_OLSEN_PLUSK,
  EPSPRIMME_RQI,
  EPSPRIMME_JDQR,
  EPSPRIMME_JDQMR,
  EPSPRIMME_JDQMR_ETOL,
  EPSPRIMME_SUBSPACE_ITERATION,
  EPSPRIMME_LOBPCG_ORTHOBASIS,
  EPSPRIMME_LOBPCG_ORTHOBASISW
};
EPSPRIMMEPrecond precondN[] = {EPSPRIMME_NONE, EPSPRIMME_DIAGONAL};

static void multMatvec_PRIMME(void *in, void *out, int *blockSize, primme_params *primme);
static void applyPreconditioner_PRIMME(void *in, void *out, int *blockSize, struct primme_params *primme);

void par_GlobalSumDouble(void *sendBuf, void *recvBuf, int *count, primme_params *primme) {
  PetscErrorCode ierr;
  ierr = MPI_Allreduce((double*)sendBuf, (double*)recvBuf, *count, MPI_DOUBLE, MPI_SUM, ((PetscObject)(primme->commInfo))->comm);CHKERRABORT(((PetscObject)(primme->commInfo))->comm,ierr);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_PRIMME"
PetscErrorCode EPSSetUp_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N, n;
  PetscMPIInt    numProcs, procID;
  EPS_PRIMME     *ops = (EPS_PRIMME *)eps->data;
  primme_params  *primme = &(((EPS_PRIMME *)eps->data)->primme);

  PetscFunctionBegin;

  ierr = MPI_Comm_size(((PetscObject)eps)->comm,&numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)eps)->comm,&procID);CHKERRQ(ierr);
  
  /* Check some constraints and set some default values */ 
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(eps->vec_initial,&n);CHKERRQ(ierr);

  if (!eps->max_it) eps->max_it = PetscMax(1000,N);
  ierr = STGetOperators(eps->OP, &ops->A, PETSC_NULL);
  if (!ops->A) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"The problem matrix has to be specified first");
  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"PRIMME is only available for Hermitian problems");
  if (eps->isgeneralized)
    SETERRQ(PETSC_ERR_SUP,"PRIMME is not available for generalized problems");

  /* Transfer SLEPc options to PRIMME options */
  primme->n = N;
  primme->nLocal = n;
  primme->numEvals = eps->nev; 
  primme->matrix = ops;
  primme->commInfo = eps;
  primme->maxMatvecs = eps->max_it; 
  primme->eps = eps->tol; 
  primme->numProcs = numProcs; 
  primme->procID = procID;

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
  
  if (primme_set_method(ops->method, primme) < 0)
    SETERRQ(PETSC_ERR_SUP,"PRIMME method not valid");
  
  /* If user sets ncv, maxBasisSize is modified. If not, ncv is set as maxBasisSize */
  if (eps->ncv) primme->maxBasisSize = eps->ncv;
  else eps->ncv = primme->maxBasisSize;
  if (eps->ncv < eps->nev+primme->maxBlockSize)  
    SETERRQ(PETSC_ERR_SUP,"PRIMME needs ncv >= nev+maxBlockSize");

  if (eps->extraction) {
     ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr);
  }

  /* Set workspace */
  ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);

  if (primme->correctionParams.precondition) {
    /* Calc reciprocal A diagonal */
    ierr = VecDuplicate(eps->vec_initial, &ops->w); CHKERRQ(ierr);
    ierr = MatGetDiagonal(ops->A, ops->w); CHKERRQ(ierr);
    ierr = VecReciprocal(ops->w); CHKERRQ(ierr);
    primme->preconditioner = PETSC_NULL;
    primme->applyPreconditioner = applyPreconditioner_PRIMME;
  }

  /* Prepare auxiliary vectors */ 
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
  PetscInt       i;
  PetscReal      *evals;
#endif

  PetscFunctionBegin;

  /* Reset some parameters left from previous runs */
  ops->primme.aNorm    = 0.0;
  ops->primme.initSize = 1;
  ops->primme.iseed[0] = -1;

  /* Copy vec_initial to V[0] vector */
  ierr = VecCopy(eps->vec_initial, eps->V[0]); CHKERRQ(ierr);
 
  /* Call PRIMME solver */
  ierr = VecGetArray(eps->V[0], &a); CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
  ierr = dprimme(eps->eigr, a, eps->errest, &ops->primme);
#else
  /* PRIMME returns real eigenvalues, but SLEPc works with complex ones */
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

  eps->nconv = ops->primme.initSize >= 0 ? ops->primme.initSize : 0;
  eps->reason = eps->ncv >= eps->nev ? EPS_CONVERGED_TOL : EPS_DIVERGED_ITS;
  eps->its = ops->primme.stats.numOuterIterations;
  eps->OP->applys = ops->primme.stats.numMatvecs;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "multMatvec_PRIMME"
static void multMatvec_PRIMME(void *in, void *out, int *blockSize, primme_params *primme)
{
  PetscErrorCode ierr;
  PetscInt       i, N = primme->n;
  EPS_PRIMME     *ops = (EPS_PRIMME *)primme->matrix; 
  Vec            x = ops->x;
  Vec            y = ops->y;
  Mat            A = ops->A;

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
static void applyPreconditioner_PRIMME(void *in, void *out, int *blockSize, struct primme_params *primme)
{
  PetscErrorCode ierr;
  PetscInt       i, N = primme->n;
  EPS_PRIMME     *ops = (EPS_PRIMME *)primme->matrix; 
  Vec            x = ops->x;
  Vec            y = ops->y;
  Vec            w = ops->w;
 
  PetscFunctionBegin;
  for (i=0;i<*blockSize;i++) {
    /* build vectors using 'in' an 'out' workspace */
    ierr = VecPlaceArray(x, (PetscScalar*)in+N*i ); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ierr = VecPlaceArray(y, (PetscScalar*)out+N*i ); CHKERRABORT(PETSC_COMM_WORLD,ierr);

    ierr = VecPointwiseMult(y, w, x); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    
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
  
  if (ops->primme.correctionParams.precondition) {
    ierr = VecDestroy(ops->w);CHKERRQ(ierr);
  }
  primme_Free(&ops->primme);
  ierr = VecDestroy(ops->x);CHKERRQ(ierr);
  ierr = VecDestroy(ops->y);CHKERRQ(ierr);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSFreeSolutionContiguous(eps);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_PRIMME"
PetscErrorCode EPSView_PRIMME(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth isascii;
  primme_params *primme = &((EPS_PRIMME *)eps->data)->primme;
  EPSPRIMMEMethod methodn;
  EPSPRIMMEPrecond precondn;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSPRIMME",((PetscObject)viewer)->type_name);
  }
  
  ierr = PetscViewerASCIIPrintf(viewer,"PRIMME solver block size: %d\n",primme->maxBlockSize);CHKERRQ(ierr);
  ierr = EPSPRIMMEGetMethod(eps, &methodn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"PRIMME solver method: %s\n", methodList[methodn]);CHKERRQ(ierr);
  ierr = EPSPRIMMEGetPrecond(eps, &precondn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"PRIMME solver preconditioning: %s\n", precondList[precondn]);CHKERRQ(ierr);

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

  PetscFunctionBegin;
  
  ierr = PetscOptionsHead("PRIMME options");CHKERRQ(ierr);

  op = ops->primme.maxBlockSize; 
  ierr = PetscOptionsInt("-eps_primme_block_size"," maximum block size","EPSPRIMMESetBlockSize",op,&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetBlockSize(eps,op);CHKERRQ(ierr);}
  op = 0;
  ierr = PetscOptionsEList("-eps_primme_method","set method for solving the eigenproblem",
                           "EPSPRIMMESetMethod",methodList,15,methodList[1],&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetMethod(eps, methodN[op]);CHKERRQ(ierr);}
  ierr = PetscOptionsEList("-eps_primme_precond","set preconditioner type",
                           "EPSPRIMMESetPrecond",precondList,2,precondList[0],&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetPrecond(eps, precondN[op]);CHKERRQ(ierr);}
  
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSPRIMMESetBlockSize_PRIMME"
PetscErrorCode EPSPRIMMESetBlockSize_PRIMME(EPS eps,PetscInt bs)
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
    EPSPRIMMESetBlockSize - The maximum block size the code will try to use. 
    The user should set
    this based on the architecture specifics of the target computer, 
    as well as any a priori knowledge of multiplicities. The code does 
    NOT require BlockSize > 1 to find multiple eigenvalues.  For some 
    methods, keeping BlockSize = 1 yields the best overall performance.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  bs - block size

   Options Database Key:
.  -eps_primme_block_size - Sets the max allowed block size value

   Notes:
   If the block size is not set, the value established by primme_initialize
   is used.

   Level: advanced
.seealso: EPSPRIMMEGetBlockSize()
@*/
PetscErrorCode EPSPRIMMESetBlockSize(EPS eps,PetscInt bs)
{
  PetscErrorCode ierr, (*f)(EPS,PetscInt);

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
PetscErrorCode EPSPRIMMEGetBlockSize_PRIMME(EPS eps,PetscInt *bs)
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
PetscErrorCode EPSPRIMMEGetBlockSize(EPS eps,PetscInt *bs)
{
  PetscErrorCode ierr, (*f)(EPS,PetscInt*);

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
-  method - method that will be used by PRIMME. It must be one of:
    EPSPRIMME_DYNAMIC, EPSPRIMME_DEFAULT_MIN_TIME(EPSPRIMME_JDQMR_ETOL),
    EPSPRIMME_DEFAULT_MIN_MATVECS(EPSPRIMME_GD_OLSEN_PLUSK), EPSPRIMME_ARNOLDI,
    EPSPRIMME_GD, EPSPRIMME_GD_PLUSK, EPSPRIMME_GD_OLSEN_PLUSK, 
    EPSPRIMME_JD_OLSEN_PLUSK, EPSPRIMME_RQI, EPSPRIMME_JDQR, EPSPRIMME_JDQMR, 
    EPSPRIMME_JDQMR_ETOL, EPSPRIMME_SUBSPACE_ITERATION,
    EPSPRIMME_LOBPCG_ORTHOBASIS, EPSPRIMME_LOBPCG_ORTHOBASISW

   Options Database Key:
.  -eps_primme_set_method - Sets the method for the PRIMME library.

   Note:
   If not set, the method defaults to EPSPRIMME_DEFAULT_MIN_TIME.

   Level: advanced

.seealso: EPSPRIMMEGetMethod(), EPSPRIMMEMethod
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

  if (method)
    *method = (EPSPRIMMEMethod)ops->method;

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
.  method - method that will be used by PRIMME. It must be one of:
    EPSPRIMME_DYNAMIC, EPSPRIMME_DEFAULT_MIN_TIME(EPSPRIMME_JDQMR_ETOL),
    EPSPRIMME_DEFAULT_MIN_MATVECS(EPSPRIMME_GD_OLSEN_PLUSK), EPSPRIMME_ARNOLDI,
    EPSPRIMME_GD, EPSPRIMME_GD_PLUSK, EPSPRIMME_GD_OLSEN_PLUSK, 
    EPSPRIMME_JD_OLSEN_PLUSK, EPSPRIMME_RQI, EPSPRIMME_JDQR, EPSPRIMME_JDQMR, 
    EPSPRIMME_JDQMR_ETOL, EPSPRIMME_SUBSPACE_ITERATION,
    EPSPRIMME_LOBPCG_ORTHOBASIS, EPSPRIMME_LOBPCG_ORTHOBASISW

    Level: advanced

.seealso: EPSPRIMMESetMethod(), EPSPRIMMEMethod
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
    EPSPRIMMESetPrecond - Sets the preconditioner to be used in the PRIMME
    library. Currently, only diagonal preconditioning is supported.

    Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  precond - either EPSPRIMME_NONE (no preconditioning) or EPSPRIMME_DIAGONAL
   (diagonal preconditioning)

   Options Database Key:
.  -eps_primme_precond - Sets either 'none' or 'diagonal' preconditioner

    Note:
    The default is no preconditioning.
    
    Level: advanced

.seealso: EPSPRIMMEGetPrecond(), EPSPRIMMEPrecond
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
    EPSPRIMMEGetPrecond - Gets the preconditioner to be used in the PRIMME
    library.

    Collective on EPS

   Input Parameters:
.  eps - the eigenproblem solver context
    
  Output Parameters:
.  precond - either EPSPRIMME_NONE (no preconditioning) or EPSPRIMME_DIAGONAL
   (diagonal preconditioning)

    Level: advanced

.seealso: EPSPRIMMESetPrecond(), EPSPRIMMEPrecond
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
  primme->primme.globalSumDouble = par_GlobalSumDouble;
  primme->method = (primme_preset_method)EPSPRIMME_DEFAULT_MIN_TIME;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetBlockSize_C","EPSPRIMMESetBlockSize_PRIMME",EPSPRIMMESetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetMethod_C","EPSPRIMMESetMethod_PRIMME",EPSPRIMMESetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetPrecond_C","EPSPRIMMESetPrecond_PRIMME",EPSPRIMMESetPrecond_PRIMME);CHKERRQ(ierr); 
  
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetBlockSize_C","EPSPRIMMEGetBlockSize_PRIMME",EPSPRIMMEGetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetMethod_C","EPSPRIMMEGetMethod_PRIMME",EPSPRIMMEGetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetPrecond_C","EPSPRIMMEGetPrecond_PRIMME",EPSPRIMMEGetPrecond_PRIMME);CHKERRQ(ierr); 

  eps->which = EPS_LARGEST_REAL;

  PetscFunctionReturn(0);
}
EXTERN_C_END


