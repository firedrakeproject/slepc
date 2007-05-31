/*
       This file implements a wrapper to the LAPACK eigenvalue subroutines.
       Generalized problems are transformed to standard ones only if necessary.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/eps/epsimpl.h"
#include "slepcblaslapack.h"

typedef struct {
  Mat OP,A,B;
} EPS_LAPACK;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LAPACK"
PetscErrorCode EPSSetUp_LAPACK(EPS eps)
{
  PetscErrorCode ierr,ierra,ierrb;
  PetscInt       N;
  EPS_LAPACK     *la = (EPS_LAPACK *)eps->data;
  PetscTruth     flg;
  Mat            A,B;
  PetscScalar    shift;
  KSP            ksp;
  PC             pc;
  
  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  eps->ncv = N;

  if (la->OP) { ierr = MatDestroy(la->OP);CHKERRQ(ierr); }
  if (la->A) { ierr = MatDestroy(la->A);CHKERRQ(ierr); }
  if (la->B) { ierr = MatDestroy(la->B);CHKERRQ(ierr); }

  ierr = PetscTypeCompare((PetscObject)eps->OP,STSHIFT,&flg);CHKERRQ(ierr);
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  
  if (flg) {
    la->OP = PETSC_NULL;
    PetscPushErrorHandler(PetscIgnoreErrorHandler,PETSC_NULL);
    ierra = SlepcMatConvertSeqDense(A,&la->A);CHKERRQ(ierr);
    if (eps->isgeneralized) {
      ierrb = SlepcMatConvertSeqDense(B,&la->B);CHKERRQ(ierr);
    } else {
      ierrb = 0;
      la->B = PETSC_NULL;
    }
    PetscPopErrorHandler();
    if (ierra == 0 && ierrb == 0) {
      ierr = STGetShift(eps->OP,&shift);CHKERRQ(ierr);
      if (shift != 0.0) {
	ierr = MatShift(la->A,shift);CHKERRQ(ierr);
      }
      /* use dummy pc and ksp to avoid problems when B is not positive definite */
      ierr = STGetKSP(eps->OP,&ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
      ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  PetscInfo(eps,"Using slow explicit operator\n");
  la->A = PETSC_NULL;
  la->B = PETSC_NULL;
  ierr = STComputeExplicitOperator(eps->OP,&la->OP);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)la->OP,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = SlepcMatConvertSeqDense(la->OP,&la->OP);CHKERRQ(ierr);
  }
  ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LAPACK"
PetscErrorCode EPSSolve_LAPACK(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       n,i,low,high;
  PetscMPIInt    size;
  PetscScalar    *array,*arrayb,*pV,*pW;
  PetscReal      *w;
  EPS_LAPACK     *la = (EPS_LAPACK *)eps->data;
  MPI_Comm       comm = eps->comm;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  
  ierr = VecGetSize(eps->vec_initial,&n);CHKERRQ(ierr);

  if (size == 1) {
    ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(sizeof(PetscScalar)*n*n,&pV);CHKERRQ(ierr);
  }
  if (eps->solverclass == EPS_TWO_SIDE && (la->OP || !eps->ishermitian)) {
    if (size == 1) {
      ierr = VecGetArray(eps->W[0],&pW);CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc(sizeof(PetscScalar)*n*n,&pW);CHKERRQ(ierr);
    }
  } else pW = PETSC_NULL;
  
  
  if (la->OP) {
    ierr = MatGetArray(la->OP,&array);CHKERRQ(ierr);
    ierr = EPSDenseNHEP(n,array,eps->eigr,eps->eigi,pV,pW);CHKERRQ(ierr);  
    ierr = MatRestoreArray(la->OP,&array);CHKERRQ(ierr);
  } else if (eps->ishermitian) {
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(n*sizeof(PetscReal),&w);CHKERRQ(ierr);
#else
    w = eps->eigr;
#endif
    ierr = MatGetArray(la->A,&array);CHKERRQ(ierr);
    if (!eps->isgeneralized) {
      ierr = EPSDenseHEP(n,array,n,w,pV);CHKERRQ(ierr);
    } else {
      ierr = MatGetArray(la->B,&arrayb);CHKERRQ(ierr);
      ierr = EPSDenseGHEP(n,array,arrayb,w,pV);CHKERRQ(ierr);  
      ierr = MatRestoreArray(la->B,&arrayb);CHKERRQ(ierr);
    } 
    ierr = MatRestoreArray(la->A,&array);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    for (i=0;i<n;i++) {
      eps->eigr[i] = w[i];
    }
    ierr = PetscFree(w);CHKERRQ(ierr);
#endif    
  } else {
    ierr = MatGetArray(la->A,&array);CHKERRQ(ierr);
    if (!eps->isgeneralized) {
      ierr = EPSDenseNHEP(n,array,eps->eigr,eps->eigi,pV,pW);CHKERRQ(ierr);
    } else {
      ierr = MatGetArray(la->B,&arrayb);CHKERRQ(ierr);
      ierr = EPSDenseGNHEP(n,array,arrayb,eps->eigr,eps->eigi,pV,pW);CHKERRQ(ierr);  
      ierr = MatRestoreArray(la->B,&arrayb);CHKERRQ(ierr);
    }
    ierr = MatRestoreArray(la->A,&array);CHKERRQ(ierr);
  }

  if (size == 1) {
    ierr = VecRestoreArray(eps->V[0],&pV);CHKERRQ(ierr);
  } else {
    for (i=0; i<eps->ncv; i++) {
      ierr = VecGetOwnershipRange(eps->V[i], &low, &high);CHKERRQ(ierr);
      ierr = VecGetArray(eps->V[i], &array);CHKERRQ(ierr);
      ierr = PetscMemcpy(array, pV+i*n+low, (high-low)*sizeof(PetscScalar));
      ierr = VecRestoreArray(eps->V[i], &array);CHKERRQ(ierr);
    }
    ierr = PetscFree(pV);CHKERRQ(ierr);
  }
  if (pW) {
    if (size == 1) {
      ierr = VecRestoreArray(eps->W[0],&pW);CHKERRQ(ierr);
    } else {
      for (i=0; i<eps->ncv; i++) {
        ierr = VecGetOwnershipRange(eps->W[i], &low, &high);CHKERRQ(ierr);
        ierr = VecGetArray(eps->W[i], &array);CHKERRQ(ierr);
        ierr = PetscMemcpy(array, pW+i*n+low, (high-low)*sizeof(PetscScalar));
        ierr = VecRestoreArray(eps->W[i], &array);CHKERRQ(ierr);
      }
      ierr = PetscFree(pW);CHKERRQ(ierr);
    }
  }

  eps->nconv = eps->ncv;
  eps->its   = 1;  
  eps->reason = EPS_CONVERGED_TOL;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_LAPACK"
PetscErrorCode EPSDestroy_LAPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LAPACK     *la = (EPS_LAPACK *)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (la->OP) { ierr = MatDestroy(la->OP);CHKERRQ(ierr); }
  if (la->A) { ierr = MatDestroy(la->A);CHKERRQ(ierr); }
  if (la->B) { ierr = MatDestroy(la->B);CHKERRQ(ierr); }
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSFreeSolutionContiguous(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LAPACK"
PetscErrorCode EPSCreate_LAPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LAPACK     *la;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LAPACK,&la);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_LAPACK));
  eps->data                      = (void *) la;
  eps->ops->solve                = EPSSolve_LAPACK;
  eps->ops->solvets              = EPSSolve_LAPACK;
  eps->ops->setup                = EPSSetUp_LAPACK;
  eps->ops->destroy              = EPSDestroy_LAPACK;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END
