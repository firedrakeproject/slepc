
/*
   This file contains some simple default routines.  
 */
#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSGetIterationNumber"
/*@
   EPSGetIterationNumber - Gets the current iteration number. If the 
   call to EPSSolve() is complete, then it returns the number of iterations 
   carried out by the solution method.
 
   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  its - number of iterations

   Level: intermediate

   Notes:
      During the i-th iteration this call returns i-1. If EPSSolve() is 
      complete, then parameter "its" contains either the iteration number at
      which convergence was successfully reached, or failure was detected.  
      Call EPSGetConvergedReason() to determine if the solver converged or 
      failed and why.

@*/
int EPSGetIterationNumber(EPS eps,int *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidIntPointer(its,2);
  *its = eps->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetNumberLinearIterations"
/*@
   EPSGetNumberLinearIterations - Gets the total number of iterations
   required by the linear solves associated to the ST object during the 
   last EPSSolve() call.

   Not Collective

   Input Parameter:
.  eps - EPS context

   Output Parameter:
.  lits - number of linear iterations

   Notes:
   When the eigensolver algorithm invokes STApply() then a linear system 
   must be solved (except in the case of standard eigenproblems and shift
   transformation). The number of iterations required in this solve is
   accumulated into a counter whose value is returned by this function.

   The iteration counter is reset to zero at each successive call to EPSSolve().

   Level: intermediate

@*/
int EPSGetNumberLinearIterations(EPS eps,int* lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidIntPointer(lits,2);
  STGetNumberLinearIterations(eps->OP, lits);
  PetscFunctionReturn(0);
}

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
int EPSDefaultMonitor(EPS eps,int its,int nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int nest,void *dummy)
{
  int         i,ierr;
  PetscViewer viewer = (PetscViewer) dummy;

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
int  EPSDefaultGetWork(EPS eps, int nw)
{
  int         ierr;

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
int EPSDefaultFreeWork(EPS eps)
{
  int          ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->work)  {
    ierr = VecDestroyVecs(eps->work,eps->nwork); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "EPSAllocateSolution"
int EPSAllocateSolution(EPS eps)
{
  int         ierr;
  
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
int EPSFreeSolution(EPS eps)
{
  int          ierr;
  
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
int EPSAllocateSolutionContiguous(EPS eps)
{
  int         i, ierr, nloc;
  PetscScalar *pV;

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
int EPSFreeSolutionContiguous(EPS eps)
{
  int          i, ierr;
  PetscScalar* pV;
  
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
int EPSDestroy_Default(EPS eps)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->data) {ierr = PetscFree(eps->data);CHKERRQ(ierr);}

  /* free work vectors */
  ierr = EPSDefaultFreeWork(eps);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "EPSGetConvergedReason"
/*@C
   EPSGetConvergedReason - Gets the reason why the EPSSolve() iteration was 
   stopped.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged
   (see EPSConvergedReason)

   Possible values for reason:
+  EPS_CONVERGED_TOL - converged up to tolerance
.  EPS_DIVERGED_ITS - required more than its to reach convergence
.  EPS_DIVERGED_BREAKDOWN - generic breakdown in method
-  EPS_DIVERGED_NONSYMMETRIC - The operator is nonsymmetric

   Level: intermediate

   Notes: Can only be called after the call to EPSSolve() is complete.

.seealso: EPSSetTolerances(), EPSSolve(), EPSConvergedReason
@*/
int EPSGetConvergedReason(EPS eps,EPSConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *reason = eps->reason;
  PetscFunctionReturn(0);
}

