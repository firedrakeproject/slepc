
/*                       
       This file implements a wrapper to the LAPACK eigenvalue subroutines.
       Currently, only LAPACK routines for standard problems are used.
       Generalized problems are transformed to standard ones.
*/
#include "src/eps/impls/lapack/lapackp.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LAPACK"
PetscErrorCode EPSSetUp_LAPACK(EPS eps)
{
  PetscErrorCode ierr;
  int            size,rank,n,N;
  EPS_LAPACK     *la = (EPS_LAPACK *)eps->data;
  MPI_Comm       comm = eps->comm;
  Mat            *T;
  IS             isrow, iscol;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->nev<1 || eps->nev>N) SETERRQ(1,"Wrong value of nev");  
  if (eps->ncv) {
    if (eps->ncv<=eps->nev) SETERRQ(1,"Wrong value of ncv");    
#ifndef PETSC_USE_COMPLEX
  } else eps->ncv = PetscMin(eps->nev + 1, N);
#else
  } else eps->ncv = eps->nev;
#endif

  if (la->BA) { ierr = MatDestroy(la->BA);CHKERRQ(ierr); }
  ierr = EPSComputeExplicitOperator(eps,&la->BA);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  if (size > 1) {
    /* assemble full matrix on every processor */
    ierr = MatGetSize(la->BA,&n,&n);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &isrow);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &iscol);CHKERRQ(ierr);
    ierr = MatGetSubMatrices(la->BA, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, &T);CHKERRQ(ierr);
    ierr = ISDestroy(isrow);CHKERRQ(ierr);
    ierr = ISDestroy(iscol);CHKERRQ(ierr);
    ierr = MatDestroy(la->BA);CHKERRQ(ierr);
    la->BA = *T;
  }
  
  ierr = PetscTypeCompare((PetscObject)la->BA, MATSEQDENSE, &flg); CHKERRQ(ierr);
  if (!flg) {
    /* convert matrix to MatSeqDense */
    ierr = MatConvert(la->BA, MATSEQDENSE, &la->BA);CHKERRQ(ierr);
  }

  ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LAPACK"
PetscErrorCode EPSSolve_LAPACK(EPS eps)
{
  PetscErrorCode ierr;
  int            n,size,rank,i,low,high;
  PetscScalar    *array,*pV;
  EPS_LAPACK     *la = (EPS_LAPACK *)eps->data;
  MPI_Comm       comm = eps->comm;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  ierr = MatGetArray(la->BA,&array);CHKERRQ(ierr);
  ierr = MatGetSize(la->BA,&n,&n);CHKERRQ(ierr);

  if (size == 1) {
    ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(sizeof(PetscScalar)*n,&pV);CHKERRQ(ierr);
  }
  
  ierr = EPSDenseNHEPSorted(n,array,eps->eigr,eps->eigi,pV,eps->ncv,eps->which);CHKERRQ(ierr);
  
  ierr = MatRestoreArray(la->BA,&array);CHKERRQ(ierr);

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

  eps->nconv = eps->ncv;
  eps->its   = 1;  
  eps->reason = EPS_CONVERGED_TOL;
  
#ifndef PETSC_USE_COMPLEX
  if (eps->eigi[eps->nconv - 1] >= 0.0 && eps->nconv != n) eps->nconv--;
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_LAPACK"
PetscErrorCode EPSDestroy_LAPACK(EPS eps)
{
  PetscErrorCode ierr;
  int            size;
  EPS_LAPACK     *la = (EPS_LAPACK *)eps->data;
  MPI_Comm       comm = eps->comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (la->BA) { ierr = MatDestroy(la->BA);CHKERRQ(ierr); }
  if (eps->data) {ierr = PetscFree(eps->data);CHKERRQ(ierr);}
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
  PetscMemzero(la,sizeof(EPS_LAPACK));
  PetscLogObjectMemory(eps,sizeof(EPS_LAPACK));
  eps->data                      = (void *) la;
  eps->ops->solve                = EPSSolve_LAPACK;
  eps->ops->setup                = EPSSetUp_LAPACK;
  eps->ops->destroy              = EPSDestroy_LAPACK;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END
