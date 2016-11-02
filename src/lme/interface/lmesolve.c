/*
      LME routines related to the solution process.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/lmeimpl.h>   /*I "slepclme.h" I*/

#undef __FUNCT__
#define __FUNCT__ "LMESolve"
/*@
   LMESolve - Solves the linear matrix equation.

   Collective on LME

   Input Parameters:
.  lme - linear matrix equation solver context obtained from LMECreate()

   Options Database Keys:
+  -lme_view - print information about the solver used
.  -lme_view_mat binary - save the matrix to the default binary viewer
.  -lme_view_rhs binary - save right hand side to the default binary viewer
.  -lme_view_solution binary - save computed solution to the default binary viewer
-  -lme_converged_reason - print reason for convergence, and number of iterations

   Notes:
   The matrix coefficients are specified with LMESetCoefficients().
   The right-hand side is specified with LMESetRHS().
   The placeholder for the solution is specified with LMESetSolution().

   Level: beginner

.seealso: LMECreate(), LMESetUp(), LMEDestroy(), LMESetTolerances(), LMESetCoefficients(), LMESetRHS(), LMESetSolution()
@*/
PetscErrorCode LMESolve(LME lme)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);

  /* call setup */
  ierr = LMESetUp(lme);CHKERRQ(ierr);
  lme->its = 0;

  ierr = LMEViewFromOptions(lme,NULL,"-lme_view_pre");CHKERRQ(ierr);

  /* call solver */
  if (!lme->ops->solve[lme->problem_type]) SETERRQ1(PetscObjectComm((PetscObject)lme),PETSC_ERR_SUP,"The specified solver does not support equation type %s",LMEProblemTypes[lme->problem_type]);
  ierr = PetscLogEventBegin(LME_Solve,lme,0,0,0);CHKERRQ(ierr);
  ierr = (*lme->ops->solve[lme->problem_type])(lme);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(LME_Solve,lme,0,0,0);CHKERRQ(ierr);

  if (!lme->reason) SETERRQ(PetscObjectComm((PetscObject)lme),PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

  if (lme->errorifnotconverged && lme->reason < 0) SETERRQ(PetscObjectComm((PetscObject)lme),PETSC_ERR_NOT_CONVERGED,"LMESolve has not converged");

  /* various viewers */
  ierr = LMEViewFromOptions(lme,NULL,"-lme_view");CHKERRQ(ierr);
  ierr = LMEReasonViewFromOptions(lme);CHKERRQ(ierr);
  ierr = MatViewFromOptions(lme->A,(PetscObject)lme,"-lme_view_mat");CHKERRQ(ierr);
  /*ierr = BVViewFromOptions(lme->C,(PetscObject)lme,"-lme_view_rhs");CHKERRQ(ierr);
  ierr = BVViewFromOptions(lme->X,(PetscObject)lme,"-lme_view_solution");CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMEGetIterationNumber"
/*@
   LMEGetIterationNumber - Gets the current iteration number. If the
   call to LMESolve() is complete, then it returns the number of iterations
   carried out by the solution method.

   Not Collective

   Input Parameter:
.  lme - the linear matrix equation solver context

   Output Parameter:
.  its - number of iterations

   Note:
   During the i-th iteration this call returns i-1. If LMESolve() is
   complete, then parameter "its" contains either the iteration number at
   which convergence was successfully reached, or failure was detected.
   Call LMEGetConvergedReason() to determine if the solver converged or
   failed and why.

   Level: intermediate

.seealso: LMEGetConvergedReason(), LMESetTolerances()
@*/
PetscErrorCode LMEGetIterationNumber(LME lme,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = lme->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMEGetConvergedReason"
/*@
   LMEGetConvergedReason - Gets the reason why the LMESolve() iteration was
   stopped.

   Not Collective

   Input Parameter:
.  lme - the linear matrix equation solver context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged

   Notes:

   Possible values for reason are
+  LME_CONVERGED_TOL - converged up to tolerance
.  LME_DIVERGED_ITS - required more than max_it iterations to reach convergence
-  LME_DIVERGED_BREAKDOWN - generic breakdown in method

   Can only be called after the call to LMESolve() is complete.

   Level: intermediate

.seealso: LMESetTolerances(), LMESolve(), LMEConvergedReason, LMESetErrorIfNotConverged()
@*/
PetscErrorCode LMEGetConvergedReason(LME lme,LMEConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidIntPointer(reason,2);
  *reason = lme->reason;
  PetscFunctionReturn(0);
}

