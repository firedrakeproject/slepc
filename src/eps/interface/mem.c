/*
      EPS routines related to memory management.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSAllocateSolution"
/*
  EPSAllocateSolution - Allocate memory storage for common variables such
  as eigenvalues and eigenvectors.
*/
PetscErrorCode EPSAllocateSolution(EPS eps)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->allocated_ncv != eps->ncv) {
    if (eps->allocated_ncv > 0) {
      ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
      ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
      ierr = PetscFree(eps->errest);CHKERRQ(ierr); 
      ierr = PetscFree(eps->errest_left);CHKERRQ(ierr); 
      ierr = VecDestroyVecs(eps->V,eps->allocated_ncv);CHKERRQ(ierr);
      ierr = VecDestroyVecs(eps->AV,eps->allocated_ncv);CHKERRQ(ierr);
      if (eps->solverclass == EPS_TWO_SIDE) {
        ierr = VecDestroyVecs(eps->W,eps->allocated_ncv);CHKERRQ(ierr);
        ierr = VecDestroyVecs(eps->AW,eps->allocated_ncv);CHKERRQ(ierr);
      }
    }
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigi);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest_left);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(eps->IV[0],eps->ncv,&eps->V);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(eps->IV[0],eps->ncv,&eps->AV);CHKERRQ(ierr);
    if (eps->solverclass == EPS_TWO_SIDE) {
      ierr = VecDuplicateVecs(eps->IV[0],eps->ncv,&eps->W);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(eps->IV[0],eps->ncv,&eps->AW);CHKERRQ(ierr);
    }
    eps->allocated_ncv = eps->ncv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFreeSolution"
/*
  EPSFreeSolution - Free memory storage. This routine is related to 
  EPSAllocateSolution().
*/
PetscErrorCode EPSFreeSolution(EPS eps)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->allocated_ncv > 0) {
    ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
    ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest);CHKERRQ(ierr); 
    ierr = PetscFree(eps->errest_left);CHKERRQ(ierr); 
    ierr = VecDestroyVecs(eps->V,eps->allocated_ncv);CHKERRQ(ierr);
    ierr = VecDestroyVecs(eps->AV,eps->allocated_ncv);CHKERRQ(ierr);
    if (eps->solverclass == EPS_TWO_SIDE) {
      ierr = VecDestroyVecs(eps->W,eps->allocated_ncv);CHKERRQ(ierr);
      ierr = VecDestroyVecs(eps->AW,eps->allocated_ncv);CHKERRQ(ierr);
    }
    eps->allocated_ncv = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSAllocateSolutionContiguous"
/*
  EPSAllocateSolutionContiguous - Allocate memory storage for common 
  variables such as eigenvalues and eigenvectors. In this version, all
  vectors in V (and AV) share a contiguous chunk of memory. This is 
  necessary for external packages such as Arpack.
*/
PetscErrorCode EPSAllocateSolutionContiguous(EPS eps)
{
  PetscErrorCode ierr;
  int            i;
  PetscInt       nloc;
  PetscScalar    *pV,*pW;

  PetscFunctionBegin;
  if (eps->allocated_ncv != eps->ncv) {
    if (eps->allocated_ncv > 0) {
      ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
      ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
      ierr = PetscFree(eps->errest);CHKERRQ(ierr); 
      ierr = PetscFree(eps->errest_left);CHKERRQ(ierr); 
      ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
      for (i=0;i<eps->allocated_ncv;i++) {
        ierr = VecDestroy(eps->V[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(pV);CHKERRQ(ierr);
      ierr = PetscFree(eps->V);CHKERRQ(ierr);
      ierr = VecGetArray(eps->AV[0],&pV);CHKERRQ(ierr);
      for (i=0;i<eps->allocated_ncv;i++) {
        ierr = VecDestroy(eps->AV[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(pV);CHKERRQ(ierr);
      ierr = PetscFree(eps->AV);CHKERRQ(ierr);
      if (eps->solverclass == EPS_TWO_SIDE) {
        ierr = VecGetArray(eps->W[0],&pW);CHKERRQ(ierr);
        for (i=0;i<eps->allocated_ncv;i++) {
          ierr = VecDestroy(eps->W[i]);CHKERRQ(ierr);
        }
        ierr = PetscFree(pW);CHKERRQ(ierr);
        ierr = PetscFree(eps->W);CHKERRQ(ierr);
        ierr = VecGetArray(eps->AW[0],&pW);CHKERRQ(ierr);
        for (i=0;i<eps->allocated_ncv;i++) {
          ierr = VecDestroy(eps->AW[i]);CHKERRQ(ierr);
        }
        ierr = PetscFree(pW);CHKERRQ(ierr);
        ierr = PetscFree(eps->AW);CHKERRQ(ierr);
      }
    }
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigi);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest_left);CHKERRQ(ierr);
    ierr = VecGetLocalSize(eps->IV[0],&nloc);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(Vec),&eps->V);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*nloc*sizeof(PetscScalar),&pV);CHKERRQ(ierr);
    for (i=0;i<eps->ncv;i++) {
      ierr = VecCreateMPIWithArray(eps->comm,nloc,PETSC_DECIDE,pV+i*nloc,&eps->V[i]);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(eps->ncv*sizeof(Vec),&eps->AV);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*nloc*sizeof(PetscScalar),&pV);CHKERRQ(ierr);
    for (i=0;i<eps->ncv;i++) {
      ierr = VecCreateMPIWithArray(eps->comm,nloc,PETSC_DECIDE,pV+i*nloc,&eps->AV[i]);CHKERRQ(ierr);
    }
    if (eps->solverclass == EPS_TWO_SIDE) {
      ierr = PetscMalloc(eps->ncv*sizeof(Vec),&eps->W);CHKERRQ(ierr);
      ierr = PetscMalloc(eps->ncv*nloc*sizeof(PetscScalar),&pW);CHKERRQ(ierr);
      for (i=0;i<eps->ncv;i++) {
        ierr = VecCreateMPIWithArray(eps->comm,nloc,PETSC_DECIDE,pW+i*nloc,&eps->W[i]);CHKERRQ(ierr);
      }
      ierr = PetscMalloc(eps->ncv*sizeof(Vec),&eps->AW);CHKERRQ(ierr);
      ierr = PetscMalloc(eps->ncv*nloc*sizeof(PetscScalar),&pW);CHKERRQ(ierr);
      for (i=0;i<eps->ncv;i++) {
        ierr = VecCreateMPIWithArray(eps->comm,nloc,PETSC_DECIDE,pW+i*nloc,&eps->AW[i]);CHKERRQ(ierr);
      }
    }
    eps->allocated_ncv = eps->ncv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFreeSolutionContiguous"
/*
  EPSFreeSolution - Free memory storage. This routine is related to 
  EPSAllocateSolutionContiguous().
*/
PetscErrorCode EPSFreeSolutionContiguous(EPS eps)
{
  PetscErrorCode ierr;
  int            i;
  PetscScalar    *pV,*pW;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->allocated_ncv > 0) {
    ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
    ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest_left);CHKERRQ(ierr);
    ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
    for (i=0;i<eps->allocated_ncv;i++) {
      ierr = VecDestroy(eps->V[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pV);CHKERRQ(ierr);
    ierr = PetscFree(eps->V);CHKERRQ(ierr);
    ierr = VecGetArray(eps->AV[0],&pV);CHKERRQ(ierr);
    for (i=0;i<eps->allocated_ncv;i++) {
      ierr = VecDestroy(eps->AV[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pV);CHKERRQ(ierr);
    ierr = PetscFree(eps->AV);CHKERRQ(ierr);
    if (eps->solverclass == EPS_TWO_SIDE) {
      ierr = VecGetArray(eps->W[0],&pW);CHKERRQ(ierr);
      for (i=0;i<eps->allocated_ncv;i++) {
        ierr = VecDestroy(eps->W[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(pW);CHKERRQ(ierr);
      ierr = PetscFree(eps->W);CHKERRQ(ierr);
      ierr = VecGetArray(eps->AW[0],&pW);CHKERRQ(ierr);
      for (i=0;i<eps->allocated_ncv;i++) {
        ierr = VecDestroy(eps->AW[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(pW);CHKERRQ(ierr);
      ierr = PetscFree(eps->AW);CHKERRQ(ierr);
    }
    eps->allocated_ncv = 0;
  }
  PetscFunctionReturn(0);
}
