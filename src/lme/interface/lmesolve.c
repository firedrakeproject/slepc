/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   LME routines related to the solution process
*/

#include <slepc/private/lmeimpl.h>   /*I "slepclme.h" I*/
#include <slepcblaslapack.h>

/*@
   LMESolve - Solves the linear matrix equation.

   Collective on lme

   Input Parameter:
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);

  /* call setup */
  CHKERRQ(LMESetUp(lme));
  lme->its    = 0;
  lme->errest = 0.0;

  CHKERRQ(LMEViewFromOptions(lme,NULL,"-lme_view_pre"));

  /* call solver */
  PetscCheck(lme->ops->solve[lme->problem_type],PetscObjectComm((PetscObject)lme),PETSC_ERR_SUP,"The specified solver does not support equation type %s",LMEProblemTypes[lme->problem_type]);
  CHKERRQ(PetscLogEventBegin(LME_Solve,lme,0,0,0));
  CHKERRQ((*lme->ops->solve[lme->problem_type])(lme));
  CHKERRQ(PetscLogEventEnd(LME_Solve,lme,0,0,0));

  PetscCheck(lme->reason,PetscObjectComm((PetscObject)lme),PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");

  PetscCheck(!lme->errorifnotconverged || lme->reason>=0,PetscObjectComm((PetscObject)lme),PETSC_ERR_NOT_CONVERGED,"LMESolve has not converged");

  /* various viewers */
  CHKERRQ(LMEViewFromOptions(lme,NULL,"-lme_view"));
  CHKERRQ(LMEConvergedReasonViewFromOptions(lme));
  CHKERRQ(MatViewFromOptions(lme->A,(PetscObject)lme,"-lme_view_mat"));
  CHKERRQ(MatViewFromOptions(lme->C,(PetscObject)lme,"-lme_view_rhs"));
  CHKERRQ(MatViewFromOptions(lme->X,(PetscObject)lme,"-lme_view_solution"));
  PetscFunctionReturn(0);
}

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

/*@
   LMEGetErrorEstimate - Returns the error estimate obtained during solve.

   Not Collective

   Input Parameter:
.  lme - linear matrix equation solver context

   Output Parameter:
.  errest - the error estimate

   Notes:
   This is the error estimated internally by the solver. The actual
   error bound can be computed with LMEComputeError(). Note that some
   solvers may not be able to provide an error estimate.

   Level: advanced

.seealso: LMEComputeError()
@*/
PetscErrorCode LMEGetErrorEstimate(LME lme,PetscReal *errest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidRealPointer(errest,2);
  *errest = lme->errest;
  PetscFunctionReturn(0);
}

/*
   LMEComputeResidualNorm_Lyapunov - Computes the Frobenius norm of the residual matrix
   associated with the Lyapunov equation.
*/
PetscErrorCode LMEComputeResidualNorm_Lyapunov(LME lme,PetscReal *norm)
{
  PetscInt          j,n,N,k,l;
  PetscBLASInt      n_,N_,k_,l_;
  PetscScalar       *Rarray,alpha=1.0,beta=0.0;
  const PetscScalar *A,*B;
  BV                W,AX,X1,C1;
  Mat               R,X1m,C1m;
  Vec               v,w;
  VecScatter        vscat;

  PetscFunctionBegin;
  CHKERRQ(MatLRCGetMats(lme->C,NULL,&C1m,NULL,NULL));
  CHKERRQ(MatLRCGetMats(lme->X,NULL,&X1m,NULL,NULL));
  CHKERRQ(BVCreateFromMat(C1m,&C1));
  CHKERRQ(BVSetFromOptions(C1));
  CHKERRQ(BVCreateFromMat(X1m,&X1));
  CHKERRQ(BVSetFromOptions(X1));
  CHKERRQ(BVGetSizes(X1,&n,&N,&k));
  CHKERRQ(BVGetSizes(C1,NULL,NULL,&l));
  CHKERRQ(PetscBLASIntCast(n,&n_));
  CHKERRQ(PetscBLASIntCast(N,&N_));
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(l,&l_));

  /* create W to store a redundant copy of a BV in each process */
  CHKERRQ(BVCreate(PETSC_COMM_SELF,&W));
  CHKERRQ(BVSetSizes(W,N,N,k));
  CHKERRQ(BVSetFromOptions(W));
  CHKERRQ(BVGetColumn(X1,0,&v));
  CHKERRQ(VecScatterCreateToAll(v,&vscat,NULL));
  CHKERRQ(BVRestoreColumn(X1,0,&v));

  /* create AX to hold the product A*X1 */
  CHKERRQ(BVDuplicate(X1,&AX));
  CHKERRQ(BVMatMult(X1,lme->A,AX));

  /* create dense matrix to hold the residual R=C1*C1'+AX*X1'+X1*AX' */
  CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)lme),n,n,N,N,NULL,&R));

  /* R=C1*C1' */
  CHKERRQ(MatDenseGetArrayWrite(R,&Rarray));
  for (j=0;j<l;j++) {
    CHKERRQ(BVGetColumn(C1,j,&v));
    CHKERRQ(BVGetColumn(W,j,&w));
    CHKERRQ(VecScatterBegin(vscat,v,w,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(vscat,v,w,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(BVRestoreColumn(C1,j,&v));
    CHKERRQ(BVRestoreColumn(W,j,&w));
  }
  if (n) {
    CHKERRQ(BVGetArrayRead(C1,&A));
    CHKERRQ(BVGetArrayRead(W,&B));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n_,&N_,&l_,&alpha,(PetscScalar*)A,&n_,(PetscScalar*)B,&N_,&beta,Rarray,&n_));
    CHKERRQ(BVRestoreArrayRead(C1,&A));
    CHKERRQ(BVRestoreArrayRead(W,&B));
  }
  beta = 1.0;

  /* R+=AX*X1' */
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(X1,j,&v));
    CHKERRQ(BVGetColumn(W,j,&w));
    CHKERRQ(VecScatterBegin(vscat,v,w,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(vscat,v,w,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(BVRestoreColumn(X1,j,&v));
    CHKERRQ(BVRestoreColumn(W,j,&w));
  }
  if (n) {
    CHKERRQ(BVGetArrayRead(AX,&A));
    CHKERRQ(BVGetArrayRead(W,&B));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n_,&N_,&k_,&alpha,(PetscScalar*)A,&n_,(PetscScalar*)B,&N_,&beta,Rarray,&n_));
    CHKERRQ(BVRestoreArrayRead(AX,&A));
    CHKERRQ(BVRestoreArrayRead(W,&B));
  }

  /* R+=X1*AX' */
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(AX,j,&v));
    CHKERRQ(BVGetColumn(W,j,&w));
    CHKERRQ(VecScatterBegin(vscat,v,w,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(vscat,v,w,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(BVRestoreColumn(AX,j,&v));
    CHKERRQ(BVRestoreColumn(W,j,&w));
  }
  if (n) {
    CHKERRQ(BVGetArrayRead(X1,&A));
    CHKERRQ(BVGetArrayRead(W,&B));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n_,&N_,&k_,&alpha,(PetscScalar*)A,&n_,(PetscScalar*)B,&N_,&beta,Rarray,&n_));
    CHKERRQ(BVRestoreArrayRead(X1,&A));
    CHKERRQ(BVRestoreArrayRead(W,&B));
  }
  CHKERRQ(MatDenseRestoreArrayWrite(R,&Rarray));

  /* compute ||R||_F */
  CHKERRQ(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatNorm(R,NORM_FROBENIUS,norm));

  CHKERRQ(BVDestroy(&W));
  CHKERRQ(VecScatterDestroy(&vscat));
  CHKERRQ(BVDestroy(&AX));
  CHKERRQ(MatDestroy(&R));
  CHKERRQ(BVDestroy(&C1));
  CHKERRQ(BVDestroy(&X1));
  PetscFunctionReturn(0);
}

/*@
   LMEComputeError - Computes the error (based on the residual norm) associated
   with the last equation solved.

   Collective on lme

   Input Parameter:
.  lme  - the linear matrix equation solver context

   Output Parameter:
.  error - the error

   Notes:
   This function is not scalable (in terms of memory or parallel communication),
   so it should not be called except in the case of small problem size. For
   large equations, use LMEGetErrorEstimate().

   Level: advanced

.seealso: LMESolve(), LMEGetErrorEstimate()
@*/
PetscErrorCode LMEComputeError(LME lme,PetscReal *error)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidRealPointer(error,2);

  CHKERRQ(PetscLogEventBegin(LME_ComputeError,lme,0,0,0));
  /* compute residual norm */
  switch (lme->problem_type) {
    case LME_LYAPUNOV:
      CHKERRQ(LMEComputeResidualNorm_Lyapunov(lme,error));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)lme),PETSC_ERR_SUP,"Not implemented for equation type %s",LMEProblemTypes[lme->problem_type]);
  }

  /* compute error */
  /* currently we only support absolute error, so just return the norm */
  CHKERRQ(PetscLogEventEnd(LME_ComputeError,lme,0,0,0));
  PetscFunctionReturn(0);
}
