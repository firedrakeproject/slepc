/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   MFN routines related to the solution process
*/

#include <slepc/private/mfnimpl.h>   /*I "slepcmfn.h" I*/

static PetscErrorCode MFNSolve_Private(MFN mfn,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscCall(VecSetErrorIfLocked(x,3));

  /* call setup */
  PetscCall(MFNSetUp(mfn));
  mfn->its = 0;

  PetscCall(MFNViewFromOptions(mfn,NULL,"-mfn_view_pre"));

  /* check nonzero right-hand side */
  PetscCall(VecNorm(b,NORM_2,&mfn->bnorm));
  PetscCheck(mfn->bnorm,PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_WRONG,"Cannot pass a zero b vector to MFNSolve()");

  /* call solver */
  PetscCall(PetscLogEventBegin(MFN_Solve,mfn,b,x,0));
  if (b!=x) PetscCall(VecLockReadPush(b));
  PetscUseTypeMethod(mfn,solve,b,x);
  if (b!=x) PetscCall(VecLockReadPop(b));
  PetscCall(PetscLogEventEnd(MFN_Solve,mfn,b,x,0));

  PetscCheck(mfn->reason,PetscObjectComm((PetscObject)mfn),PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

  PetscCheck(!mfn->errorifnotconverged || mfn->reason>=0,PetscObjectComm((PetscObject)mfn),PETSC_ERR_NOT_CONVERGED,"MFNSolve has not converged");

  /* various viewers */
  PetscCall(MFNViewFromOptions(mfn,NULL,"-mfn_view"));
  PetscCall(MFNConvergedReasonViewFromOptions(mfn));
  PetscCall(MatViewFromOptions(mfn->A,(PetscObject)mfn,"-mfn_view_mat"));
  PetscCall(VecViewFromOptions(b,(PetscObject)mfn,"-mfn_view_rhs"));
  PetscCall(VecViewFromOptions(x,(PetscObject)mfn,"-mfn_view_solution"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MFNSolve - Solves the matrix function problem. Given a vector b, the
   vector x = f(A)*b is returned.

   Collective

   Input Parameters:
+  mfn - matrix function context obtained from MFNCreate()
-  b   - the right hand side vector

   Output Parameter:
.  x   - the solution (this may be the same vector as b, then b will be
         overwritten with the answer)

   Options Database Keys:
+  -mfn_view - print information about the solver used
.  -mfn_view_mat binary - save the matrix to the default binary viewer
.  -mfn_view_rhs binary - save right hand side vector to the default binary viewer
.  -mfn_view_solution binary - save computed solution vector to the default binary viewer
-  -mfn_converged_reason - print reason for convergence, and number of iterations

   Notes:
   The matrix A is specified with MFNSetOperator().
   The function f is specified with MFNSetFN().

   Level: beginner

.seealso: MFNCreate(), MFNSetUp(), MFNDestroy(), MFNSetTolerances(),
          MFNSetOperator(), MFNSetFN()
@*/
PetscErrorCode MFNSolve(MFN mfn,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscCheckSameComm(mfn,1,b,2);
  if (b!=x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (b!=x) PetscCheckSameComm(mfn,1,x,3);
  mfn->transpose_solve = PETSC_FALSE;
  PetscCall(MFNSolve_Private(mfn,b,x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MFNSolveTranspose - Solves the transpose matrix function problem. Given a vector b,
   the vector x = f(A^T)*b is returned.

   Collective

   Input Parameters:
+  mfn - matrix function context obtained from MFNCreate()
-  b   - the right hand side vector

   Output Parameter:
.  x   - the solution (this may be the same vector as b, then b will be
         overwritten with the answer)

   Note:
   See available options at MFNSolve().

   Level: beginner

.seealso: MFNSolve()
@*/
PetscErrorCode MFNSolveTranspose(MFN mfn,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscCheckSameComm(mfn,1,b,2);
  if (b!=x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (b!=x) PetscCheckSameComm(mfn,1,x,3);
  mfn->transpose_solve = PETSC_TRUE;
  if (!mfn->AT) PetscCall(MatCreateTranspose(mfn->A,&mfn->AT));
  PetscCall(MFNSolve_Private(mfn,b,x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MFNGetIterationNumber - Gets the current iteration number. If the
   call to MFNSolve() is complete, then it returns the number of iterations
   carried out by the solution method.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameter:
.  its - number of iterations

   Note:
   During the i-th iteration this call returns i-1. If MFNSolve() is
   complete, then parameter "its" contains either the iteration number at
   which convergence was successfully reached, or failure was detected.
   Call MFNGetConvergedReason() to determine if the solver converged or
   failed and why.

   Level: intermediate

.seealso: MFNGetConvergedReason(), MFNSetTolerances()
@*/
PetscErrorCode MFNGetIterationNumber(MFN mfn,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscAssertPointer(its,2);
  *its = mfn->its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MFNGetConvergedReason - Gets the reason why the MFNSolve() iteration was
   stopped.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged

   Notes:

   Possible values for reason are
+  MFN_CONVERGED_TOL - converged up to tolerance
.  MFN_CONVERGED_ITS - solver completed the requested number of steps
.  MFN_DIVERGED_ITS - required more than max_it iterations to reach convergence
-  MFN_DIVERGED_BREAKDOWN - generic breakdown in method

   Can only be called after the call to MFNSolve() is complete.

   Basic solvers (e.g. unrestarted Krylov iterations) cannot determine if the
   computation is accurate up to the requested tolerance. In that case, the
   converged reason is set to MFN_CONVERGED_ITS if the requested number of steps
   (for instance, the ncv value in unrestarted Krylov methods) have been
   completed successfully.

   Level: intermediate

.seealso: MFNSetTolerances(), MFNSolve(), MFNConvergedReason, MFNSetErrorIfNotConverged()
@*/
PetscErrorCode MFNGetConvergedReason(MFN mfn,MFNConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscAssertPointer(reason,2);
  *reason = mfn->reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}
