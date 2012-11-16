/*
      MFN routines related to the solution process.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/mfnimpl.h>   /*I "slepcmfn.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "MFNSolve"
/*@
   MFNSolve - Solves the matrix function problem. Given a vector b, the
   vector x = f(A)*b is returned.

   Collective on MFN

   Input Parameters:
+  mfn - matrix function context obtained from MFNCreate()
-  b   - the right hand side vector

   Output Parameter:
.  x   - the solution

   Options Database:
+   -mfn_view - print information about the solver used
.   -mfn_view_before - print info at the beginning of the solve
-   -mfn_view_binary - save the matrix to the default binary file

   Notes:
   The matrix A is specified with MFMSetOperator().
   The function f is specified with MFMSetFunction().

   Level: beginner

.seealso: MFNCreate(), MFNSetUp(), MFNDestroy(), MFNSetTolerances(),
          MFMSetOperator(), MFMSetFunction()
@*/
PetscErrorCode MFNSolve(MFN mfn,Vec b,Vec x) 
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)mfn)->prefix,"-mfn_view_binary",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) {
    ierr = MatView(mfn->A,PETSC_VIEWER_BINARY_(((PetscObject)mfn)->comm));CHKERRQ(ierr);
    ierr = VecView(b,PETSC_VIEWER_BINARY_(((PetscObject)mfn)->comm));CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetBool(((PetscObject)mfn)->prefix,"-mfn_view_before",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)mfn)->comm,&viewer);CHKERRQ(ierr);
    ierr = MFNView(mfn,viewer);CHKERRQ(ierr); 
  }

  /* reset the convergence flag from the previous solves */
  mfn->reason = MFN_CONVERGED_ITERATING;

  /* call setup */
  if (!mfn->setupcalled) { ierr = MFNSetUp(mfn);CHKERRQ(ierr); }
  mfn->its = 0;

  switch (mfn->function) {
    case MFN_EXP:
      //ierr = DSSetType(mfn->ds,DSEXP);CHKERRQ(ierr);
      break;
    default: SETERRQ(((PetscObject)mfn)->comm,1,"Selected function not implemented");
  }

  /* call solver */
  ierr = PetscLogEventBegin(MFN_Solve,mfn,b,x,0);CHKERRQ(ierr);
  ierr = (*mfn->ops->solve)(mfn);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MFN_Solve,mfn,b,x,0);CHKERRQ(ierr);

  if (!mfn->reason) SETERRQ(((PetscObject)mfn)->comm,PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

  ierr = PetscOptionsGetString(((PetscObject)mfn)->prefix,"-mfn_view",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(((PetscObject)mfn)->comm,filename,&viewer);CHKERRQ(ierr);
    ierr = MFNView(mfn,viewer);CHKERRQ(ierr); 
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNGetIterationNumber"
/*@
   MFNGetIterationNumber - Gets the current iteration number. If the 
   call to MFNSolve() is complete, then it returns the number of iterations 
   carried out by the solution method.
 
   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameter:
.  its - number of iterations

   Level: intermediate

   Note:
   During the i-th iteration this call returns i-1. If MFNSolve() is 
   complete, then parameter "its" contains either the iteration number at
   which convergence was successfully reached, or failure was detected.  
   Call MFNGetConvergedReason() to determine if the solver converged or 
   failed and why.

.seealso: MFNGetConvergedReason(), MFNSetTolerances()
@*/
PetscErrorCode MFNGetIterationNumber(MFN mfn,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = mfn->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNGetConvergedReason"
/*@C
   MFNGetConvergedReason - Gets the reason why the MFNSolve() iteration was 
   stopped.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged

   Possible values for reason:
+  MFN_CONVERGED_TOL - converged up to tolerance
.  MFN_DIVERGED_ITS - required more than its to reach convergence
-  MFN_DIVERGED_BREAKDOWN - generic breakdown in method

   Note:
   Can only be called after the call to MFNSolve() is complete.

   Level: intermediate

.seealso: MFNSetTolerances(), MFNSolve(), MFNConvergedReason
@*/
PetscErrorCode MFNGetConvergedReason(MFN mfn,MFNConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidIntPointer(reason,2);
  *reason = mfn->reason;
  PetscFunctionReturn(0);
}

