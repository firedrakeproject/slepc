
/*
   This file contains some simple default routines.  
 */
#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSGetIterationNumber"
/*@
   EPSGetIterationNumber - Gets the current iteration number. If the 
         EPSSolve() is complete, returns the number of iterations used.
 
   Not Collective

   Input Parameters:
.  eps - the eigensolver context

   Output Parameters:
.  its - number of iterations

   Level: intermediate

   Notes:
      During the i-th iteration this call returns i-1. If the 
      EPSSolve() is complete, the parameter "its" contains either the iteration number at
      which convergence was successfully reached, or failure was detected.  
      Call EPSGetConvergedReason() to determine if the solver converged or 
      failed and why.

@*/
int EPSGetIterationNumber(EPS eps,int *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *its = eps->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultEstimatesMonitor"
/*@C
   EPSDefaultEstimatesMonitor - Print the error estimates at each iteration 
   of the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps    - eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  errest - error estimates
.  nest   - number of error estimates to display
-  dummy  - unused monitor context 

   Level: intermediate

.seealso: EPSSetMonitor()
@*/
int EPSDefaultEstimatesMonitor(EPS eps,int its,int nconv,PetscReal *errest,int nest,void *dummy)
{
  int         i,ierr;
  PetscViewer viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(eps->comm);
  ierr = PetscViewerASCIIPrintf(viewer,"%3d EPS nconv=%d Error Estimates:",its,nconv);CHKERRQ(ierr);
  for (i=0;i<nest;i++) {
    ierr = PetscViewerASCIIPrintf(viewer," %10.8e",errest[i]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultValuesMonitor"
/*@C
   EPSDefaultValuesMonitor - Print the current approximate values of the
   eigenvalues.

   Collective on EPS

   Input Parameters:
+  eps    - eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues (can be PETSC_NULL)
.  neig   - number of eigenvalues to display
-  dummy  - unused monitor context 

   Level: intermediate

.seealso: EPSSetEstimatesMonitor()
@*/
int EPSDefaultValuesMonitor(EPS eps,int its,int nconv,PetscScalar *eigr,PetscScalar *eigi,int neig,void *dummy)
{
  int         i,ierr;
  PetscViewer viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(eps->comm);
  ierr = PetscViewerASCIIPrintf(viewer,"%3d EPS nconv=%d Values:",its,nconv);CHKERRQ(ierr);
  for (i=0;i<neig;i++) {
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",PetscRealPart(eigr[i]),PetscImaginaryPart(eigr[i]));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer," %g",eigr[i]);CHKERRQ(ierr);
    if (eigi && eigi[i]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",eigi[i]);CHKERRQ(ierr); }
#endif
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultGetWork"
/*
  EPSDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
. eps  - eigensolver context
. nw   - number of work vectors to allocate

  Notes:
  Call this only if no work vectors have been allocated 

 */
int  EPSDefaultGetWork(EPS eps, int nw)
{
  int ierr;

  PetscFunctionBegin;
  if (eps->work) {ierr = EPSDefaultFreeWork( eps );CHKERRQ(ierr);}
  eps->nwork = nw;
  ierr = VecDuplicateVecs(eps->vec_initial,nw,&eps->work); CHKERRQ(ierr);
  PetscLogObjectParents(eps,nw,eps->work);
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
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->work)  {
    ierr = VecDestroyVecs(eps->work,eps->nwork); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultDestroy"
/*
  EPSDefaultDestroy - Destroys an eigensolver context variable for methods 
  with no separate context. Preferred calling sequence EPSDestroy().

  Input Parameter: 
. eps - the eigensolver context

*/
int EPSDefaultDestroy(EPS eps)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->data) {ierr = PetscFree(eps->data);CHKERRQ(ierr);}

  /* free work vectors */
  EPSDefaultFreeWork( eps );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetConvergedReason"
/*@C
   EPSGetConvergedReason - Gets the reason the EPS iteration was stopped.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see EPSConvergedReason

   Possible values for reason:
+  EPS_CONVERGED_TOL - converged up to tolerance
.  EPS_DIVERGED_ITS - required more than its to reach convergence
.  EPS_DIVERGED_BREAKDOWN - generic breakdown in method
-  EPS_DIVERGED_NONSYMMETRIC - The operator is nonsymmetric

   Level: intermediate

   Notes: Can only be called after the call the EPSSolve() is complete.

.seealso: EPSSetTolerances(), EPSConvergedReason
@*/
int EPSGetConvergedReason(EPS eps,EPSConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *reason = eps->reason;
  PetscFunctionReturn(0);
}

