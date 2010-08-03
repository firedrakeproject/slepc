/*
       This file implements a wrapper to the PRIMME library

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "petsc.h"
#include "private/epsimpl.h"    /*I "slepceps.h" I*/
#include "private/stimpl.h"

PetscErrorCode EPSSolve_PRIMME(EPS);

EXTERN_C_BEGIN
#include "primme.h"
EXTERN_C_END

typedef struct {
  primme_params primme;           /* param struc */
  primme_preset_method method;    /* primme method */
  Mat A;                          /* problem matrix */
  EPS eps;                        /* EPS current context */
  KSP ksp;                        /* preconditioner */
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
EPSPRIMMEMethod methodN[] = {
  EPS_PRIMME_DYNAMIC,
  EPS_PRIMME_DEFAULT_MIN_TIME,
  EPS_PRIMME_DEFAULT_MIN_MATVECS,
  EPS_PRIMME_ARNOLDI,
  EPS_PRIMME_GD,
  EPS_PRIMME_GD_PLUSK,
  EPS_PRIMME_GD_OLSEN_PLUSK,
  EPS_PRIMME_JD_OLSEN_PLUSK,
  EPS_PRIMME_RQI,
  EPS_PRIMME_JDQR,
  EPS_PRIMME_JDQMR,
  EPS_PRIMME_JDQMR_ETOL,
  EPS_PRIMME_SUBSPACE_ITERATION,
  EPS_PRIMME_LOBPCG_ORTHOBASIS,
  EPS_PRIMME_LOBPCG_ORTHOBASISW
};

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
  PetscMPIInt    numProcs, procID;
  EPS_PRIMME     *ops = (EPS_PRIMME *)eps->data;
  primme_params  *primme = &(((EPS_PRIMME *)eps->data)->primme);
  PetscTruth     t;

  PetscFunctionBegin;

  ierr = MPI_Comm_size(((PetscObject)eps)->comm,&numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)eps)->comm,&procID);CHKERRQ(ierr);
  
  /* Check some constraints and set some default values */ 
  if (!eps->max_it) eps->max_it = PetscMax(1000,eps->n);
  ierr = STGetOperators(eps->OP, &ops->A, PETSC_NULL);
  if (!ops->A) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"The problem matrix has to be specified first");
  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"PRIMME is only available for Hermitian problems");
  if (eps->isgeneralized)
    SETERRQ(PETSC_ERR_SUP,"PRIMME is not available for generalized problems");
  if (!eps->which) eps->which = EPS_LARGEST_REAL;

  /* Change the default sigma to inf if necessary */
  if (eps->which == EPS_LARGEST_MAGNITUDE || eps->which == EPS_LARGEST_REAL ||
      eps->which == EPS_LARGEST_IMAGINARY) {
    ierr = STSetDefaultShift(eps->OP, 3e300); CHKERRQ(ierr);
  }

  ierr = STSetUp(eps->OP); CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)eps->OP, STPRECOND, &t); CHKERRQ(ierr);
  if (t == PETSC_FALSE) SETERRQ(PETSC_ERR_SUP, "PRIMME only works with precond spectral transformation");

  /* Transfer SLEPc options to PRIMME options */
  primme->n = eps->n;
  primme->nLocal = eps->nloc;
  primme->numEvals = eps->nev; 
  primme->matrix = ops;
  primme->commInfo = eps;
  primme->maxMatvecs = eps->max_it; 
  primme->eps = eps->tol; 
  primme->numProcs = numProcs; 
  primme->procID = procID;
  primme->printLevel = 0;
  primme->correctionParams.precondition = 1;

  if (!eps->which) eps->which = EPS_LARGEST_REAL;
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
  if (eps->mpd) PetscInfo(eps,"Warning: parameter mpd ignored\n");

  if (eps->extraction) {
     ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr);
  }

  /* Set workspace */
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);

  /* Setup the preconditioner */
  ops->eps = eps;
  if (primme->correctionParams.precondition) {
    ierr = STGetKSP(eps->OP, &ops->ksp); CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)ops->ksp, KSPPREONLY, &t); CHKERRQ(ierr);
    if (t == PETSC_FALSE) SETERRQ(PETSC_ERR_SUP, "PRIMME only works with preonly ksp of the spectral transformation");
    primme->preconditioner = PETSC_NULL;
    primme->applyPreconditioner = applyPreconditioner_PRIMME;
  }

  /* Prepare auxiliary vectors */ 
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,eps->nloc,eps->n,PETSC_NULL,&ops->x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,eps->nloc,eps->n,PETSC_NULL,&ops->y);CHKERRQ(ierr);
 
  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_PRIMME;
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
  ops->primme.initSize = eps->nini;
  ops->primme.iseed[0] = -1;

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
  PetscInt       i, N = primme->n, lits;
  EPS_PRIMME     *ops = (EPS_PRIMME *)primme->matrix; 
  Vec            x = ops->x;
  Vec            y = ops->y;
 
  PetscFunctionBegin;
  for (i=0;i<*blockSize;i++) {
    /* build vectors using 'in' an 'out' workspace */
    ierr = VecPlaceArray(x, (PetscScalar*)in+N*i ); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ierr = VecPlaceArray(y, (PetscScalar*)out+N*i ); CHKERRABORT(PETSC_COMM_WORLD,ierr);

    ierr = KSPSolve(ops->ksp, x, y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ierr = KSPGetIterationNumber(ops->ksp, &lits); CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ops->eps->OP->lineariterations+= lits;
    
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
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
 
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetBlockSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetMethod_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetBlockSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetMethod_C","",PETSC_NULL);CHKERRQ(ierr);

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

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSPRIMME",((PetscObject)viewer)->type_name);
  }
  
  ierr = PetscViewerASCIIPrintf(viewer,"PRIMME solver block size: %d\n",primme->maxBlockSize);CHKERRQ(ierr);
  ierr = EPSPRIMMEGetMethod(eps, &methodn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"PRIMME solver method: %s\n", methodList[methodn]);CHKERRQ(ierr);

  /* Display PRIMME params */
  primme_display_params(*primme);

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
  
  ierr = PetscOptionsBegin(((PetscObject)eps)->comm,((PetscObject)eps)->prefix,"PRIMME Options","EPS");CHKERRQ(ierr);

  op = ops->primme.maxBlockSize; 
  ierr = PetscOptionsInt("-eps_primme_block_size"," maximum block size","EPSPRIMMESetBlockSize",op,&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetBlockSize(eps,op);CHKERRQ(ierr);}
  op = 0;
  ierr = PetscOptionsEList("-eps_primme_method","set method for solving the eigenproblem",
                           "EPSPRIMMESetMethod",methodList,15,methodList[1],&op,&flg); CHKERRQ(ierr);
  if (flg) {ierr = EPSPRIMMESetMethod(eps, methodN[op]);CHKERRQ(ierr);}
  
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  
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
    EPS_PRIMME_DYNAMIC, EPS_PRIMME_DEFAULT_MIN_TIME(EPS_PRIMME_JDQMR_ETOL),
    EPS_PRIMME_DEFAULT_MIN_MATVECS(EPS_PRIMME_GD_OLSEN_PLUSK), EPS_PRIMME_ARNOLDI,
    EPS_PRIMME_GD, EPS_PRIMME_GD_PLUSK, EPS_PRIMME_GD_OLSEN_PLUSK, 
    EPS_PRIMME_JD_OLSEN_PLUSK, EPS_PRIMME_RQI, EPS_PRIMME_JDQR, EPS_PRIMME_JDQMR, 
    EPS_PRIMME_JDQMR_ETOL, EPS_PRIMME_SUBSPACE_ITERATION,
    EPS_PRIMME_LOBPCG_ORTHOBASIS, EPS_PRIMME_LOBPCG_ORTHOBASISW

   Options Database Key:
.  -eps_primme_set_method - Sets the method for the PRIMME library.

   Note:
   If not set, the method defaults to EPS_PRIMME_DEFAULT_MIN_TIME.

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
    EPS_PRIMME_DYNAMIC, EPS_PRIMME_DEFAULT_MIN_TIME(EPS_PRIMME_JDQMR_ETOL),
    EPS_PRIMME_DEFAULT_MIN_MATVECS(EPS_PRIMME_GD_OLSEN_PLUSK), EPS_PRIMME_ARNOLDI,
    EPS_PRIMME_GD, EPS_PRIMME_GD_PLUSK, EPS_PRIMME_GD_OLSEN_PLUSK, 
    EPS_PRIMME_JD_OLSEN_PLUSK, EPS_PRIMME_RQI, EPS_PRIMME_JDQR, EPS_PRIMME_JDQMR, 
    EPS_PRIMME_JDQMR_ETOL, EPS_PRIMME_SUBSPACE_ITERATION,
    EPS_PRIMME_LOBPCG_ORTHOBASIS, EPS_PRIMME_LOBPCG_ORTHOBASISW

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
#define __FUNCT__ "EPSCreate_PRIMME"
PetscErrorCode EPSCreate_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME     *primme;

  PetscFunctionBegin;
  
  ierr = STSetType(eps->OP, STPRECOND); CHKERRQ(ierr);
  ierr = STPrecondSetKSPHasMat(eps->OP, PETSC_TRUE); CHKERRQ(ierr);

  ierr = PetscNew(EPS_PRIMME,&primme);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_PRIMME));
  eps->data                      = (void *) primme;
  eps->ops->setup                = EPSSetUp_PRIMME;
  eps->ops->setfromoptions       = EPSSetFromOptions_PRIMME;
  eps->ops->destroy              = EPSDestroy_PRIMME;
  eps->ops->view                 = EPSView_PRIMME;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;

  primme_initialize(&primme->primme);
  primme->primme.matrixMatvec = multMatvec_PRIMME;
  primme->primme.globalSumDouble = par_GlobalSumDouble;
  primme->method = (primme_preset_method)EPS_PRIMME_DEFAULT_MIN_TIME;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetBlockSize_C","EPSPRIMMESetBlockSize_PRIMME",EPSPRIMMESetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMESetMethod_C","EPSPRIMMESetMethod_PRIMME",EPSPRIMMESetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetBlockSize_C","EPSPRIMMEGetBlockSize_PRIMME",EPSPRIMMEGetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSPRIMMEGetMethod_C","EPSPRIMMEGetMethod_PRIMME",EPSPRIMMEGetMethod_PRIMME);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
