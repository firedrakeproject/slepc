#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultGetWork"
/*
  EPSDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
+ eps  - eigensolver context
- nw   - number of work vectors to allocate

  Notes:
  Call this only if no work vectors have been allocated.

 */
PetscErrorCode EPSDefaultGetWork(EPS eps, int nw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (eps->nwork != nw) {
    if (eps->nwork > 0) {
      ierr = VecDestroyVecs(eps->work,eps->nwork); CHKERRQ(ierr);
    }
    eps->nwork = nw;
    ierr = VecDuplicateVecs(eps->vec_initial,nw,&eps->work); CHKERRQ(ierr);
    PetscLogObjectParents(eps,nw,eps->work);
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultFreeWork"
/*
  EPSDefaultFreeWork - Free work vectors.

  Input Parameters:
. eps  - eigensolver context

 */
PetscErrorCode EPSDefaultFreeWork(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->work)  {
    ierr = VecDestroyVecs(eps->work,eps->nwork); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "EPSAllocateSolution"
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
      ierr = VecDestroyVecs(eps->V,eps->allocated_ncv);CHKERRQ(ierr);
      ierr = VecDestroyVecs(eps->AV,eps->allocated_ncv);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigi);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(eps->vec_initial,eps->ncv,&eps->V);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(eps->vec_initial,eps->ncv,&eps->AV);CHKERRQ(ierr);
    eps->allocated_ncv = eps->ncv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFreeSolution"
PetscErrorCode EPSFreeSolution(EPS eps)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->allocated_ncv > 0) {
    ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
    ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest);CHKERRQ(ierr); 
    ierr = VecDestroyVecs(eps->V,eps->allocated_ncv);CHKERRQ(ierr);
    ierr = VecDestroyVecs(eps->AV,eps->allocated_ncv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSAllocateSolutionContiguous"
PetscErrorCode EPSAllocateSolutionContiguous(EPS eps)
{
  PetscErrorCode ierr;
  int            i,nloc;
  PetscScalar    *pV;

  PetscFunctionBegin;
  if (eps->allocated_ncv != eps->ncv) {
    if (eps->allocated_ncv > 0) {
      ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
      ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
      ierr = PetscFree(eps->errest);CHKERRQ(ierr); 
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
    }
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&eps->eigi);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&eps->errest);CHKERRQ(ierr);
    ierr = VecGetLocalSize(eps->vec_initial,&nloc);CHKERRQ(ierr);
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
    eps->allocated_ncv = eps->ncv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFreeSolutionContiguous"
PetscErrorCode EPSFreeSolutionContiguous(EPS eps)
{
  PetscErrorCode ierr;
  int            i;
  PetscScalar    *pV;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->allocated_ncv > 0) {
    ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
    ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
    ierr = PetscFree(eps->errest);CHKERRQ(ierr);
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
  }
  PetscFunctionReturn(0);
}
