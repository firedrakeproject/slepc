
/*
   This file contains some simple default routines.  
 */
#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultMonitor"
/*@C
   EPSDefaultEstimatesMonitor - Print the current approximate values and 
   error estimates at each iteration of the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps    - eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  dummy  - unused monitor context 

   Level: intermediate

.seealso: EPSSetMonitor()
@*/
PetscErrorCode EPSDefaultMonitor(EPS eps,int its,int nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int nest,void *dummy)
{
  PetscErrorCode ierr;
  int            i;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(eps->comm);
  ierr = PetscViewerASCIIPrintf(viewer,"%3d EPS nconv=%d Values (Errors)",its,nconv);CHKERRQ(ierr);
  for (i=0;i<nest;i++) {
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",PetscRealPart(eigr[i]),PetscImaginaryPart(eigr[i]));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer," %g",eigr[i]);CHKERRQ(ierr);
    if (eigi[i]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",eigi[i]);CHKERRQ(ierr); }
#endif
    ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)",errest[i]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_Default"
PetscErrorCode EPSDestroy_Default(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->data) {ierr = PetscFree(eps->data);CHKERRQ(ierr);}

  /* free work vectors */
  ierr = EPSDefaultFreeWork(eps);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_Default"
PetscErrorCode EPSBackTransform_Default(EPS eps)
{
  PetscErrorCode ierr;
  int            i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  for (i=0;i<eps->nconv;i++) {
    ierr = STBackTransform(eps->OP,&eps->eigr[i],&eps->eigi[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_Default"
PetscErrorCode EPSComputeVectors_Default(EPS eps)
{
  PetscErrorCode ierr;
  int            i;

  PetscFunctionBegin;
  for (i=0;i<eps->nconv;i++) {
    ierr = VecCopy(eps->V[i],eps->AV[i]);CHKERRQ(ierr);
  }
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_Schur"
PetscErrorCode EPSComputeVectors_Schur(EPS eps)
{
  PetscErrorCode ierr;
  int            i,mout,info,ncv=eps->ncv;
  PetscScalar    *Y,*work;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif
  
  PetscFunctionBegin;
  if (eps->ishermitian) {
    ierr = EPSComputeVectors_Default(eps);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#endif 

  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif

#if !defined(PETSC_USE_COMPLEX)
  LAtrevc_("R","A",PETSC_NULL,&ncv,eps->T,&ncv,PETSC_NULL,&ncv,Y,&ncv,&ncv,&mout,work,&info,1,1);
#else
  LAtrevc_("R","A",PETSC_NULL,&ncv,eps->T,&ncv,PETSC_NULL,&ncv,Y,&ncv,&ncv,&mout,work,rwork,&info,1,1);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

  for (i=0;i<eps->nconv;i++) {
    ierr = VecCopy(eps->V[i],eps->AV[i]);CHKERRQ(ierr);
  }
  ierr = EPSReverseProjection(eps,eps->AV,Y,0,ncv,eps->work);CHKERRQ(ierr);
   
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
}
