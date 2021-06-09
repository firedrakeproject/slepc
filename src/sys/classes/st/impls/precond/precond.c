/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Implements the ST class for preconditioned eigenvalue methods
*/

#include <slepc/private/stimpl.h>          /*I "slepcst.h" I*/

typedef struct {
  PetscBool ksphasmat;  /* the KSP must have the same matrix as PC */
} ST_PRECOND;

static PetscErrorCode STSetDefaultKSP_Precond(ST st)
{
  PetscErrorCode ierr;
  PC             pc;
  PCType         pctype;
  PetscBool      t0,t1;

  PetscFunctionBegin;
  ierr = KSPGetPC(st->ksp,&pc);CHKERRQ(ierr);
  ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
  if (!pctype && st->A && st->A[0]) {
    if (st->matmode == ST_MATMODE_SHELL) {
      ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    } else {
      ierr = MatHasOperation(st->A[0],MATOP_DUPLICATE,&t0);CHKERRQ(ierr);
      if (st->nmat>1) {
        ierr = MatHasOperation(st->A[0],MATOP_AXPY,&t1);CHKERRQ(ierr);
      } else t1 = PETSC_TRUE;
      ierr = PCSetType(pc,(t0 && t1)?PCBJACOBI:PCNONE);CHKERRQ(ierr);
    }
  }
  ierr = KSPSetErrorIfNotConverged(st->ksp,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode STPostSolve_Precond(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->matmode == ST_MATMODE_INPLACE && !(st->Pmat || (PetscAbsScalar(st->sigma)>=PETSC_MAX_REAL && st->nmat>1))) {
    if (st->nmat>1) {
      ierr = MatAXPY(st->A[0],st->sigma,st->A[1],st->str);CHKERRQ(ierr);
    } else {
      ierr = MatShift(st->A[0],st->sigma);CHKERRQ(ierr);
    }
    st->Astate[0] = ((PetscObject)st->A[0])->state;
    st->state   = ST_STATE_INITIAL;
    st->opready = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*
   Operator (precond):
               Op        P         M
   if nmat=1:  ---       A-sI      NULL
   if nmat=2:  ---       A-sB      NULL
*/
PetscErrorCode STComputeOperator_Precond(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;
  st->M = NULL;

  /* P = A-sigma*B */
  if (st->Pmat) {
    ierr = PetscObjectReference((PetscObject)st->Pmat);CHKERRQ(ierr);
    ierr = MatDestroy(&st->P);CHKERRQ(ierr);
    st->P = st->Pmat;
  } else {
    ierr = PetscObjectReference((PetscObject)st->A[1]);CHKERRQ(ierr);
    ierr = MatDestroy(&st->T[0]);CHKERRQ(ierr);
    st->T[0] = st->A[1];
    if (!(PetscAbsScalar(st->sigma) < PETSC_MAX_REAL) && st->nmat>1) {
      ierr = PetscObjectReference((PetscObject)st->T[0]);CHKERRQ(ierr);
      ierr = MatDestroy(&st->P);CHKERRQ(ierr);
      st->P = st->T[0];
    } else if (PetscAbsScalar(st->sigma)<PETSC_MAX_REAL) {
      ierr = STMatMAXPY_Private(st,-st->sigma,0.0,0,NULL,PetscNot(st->state==ST_STATE_UPDATED),&st->T[1]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)st->T[1]);CHKERRQ(ierr);
      ierr = MatDestroy(&st->P);CHKERRQ(ierr);
      st->P = st->T[1];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STSetUp_Precond(ST st)
{
  PetscErrorCode ierr;
  ST_PRECOND     *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  if (st->P) {
    ierr = STKSPSetOperators(st,ctx->ksphasmat?st->P:NULL,st->P);CHKERRQ(ierr);
    /* NOTE: we do not call KSPSetUp() here because some eigensolvers such as JD require a lazy setup */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STSetShift_Precond(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;
  ST_PRECOND     *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  if (st->transform && !st->Pmat) {
    ierr = STMatMAXPY_Private(st,-newshift,-st->sigma,0,NULL,PETSC_FALSE,&st->T[1]);CHKERRQ(ierr);
    if (st->P!=st->T[1]) {
      ierr = PetscObjectReference((PetscObject)st->T[1]);CHKERRQ(ierr);
      ierr = MatDestroy(&st->P);CHKERRQ(ierr);
      st->P = st->T[1];
    }
  }
  if (st->P) {
    ierr = STKSPSetOperators(st,ctx->ksphasmat?st->P:NULL,st->P);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode STPrecondSetKSPHasMat_Precond(ST st,PetscBool ksphasmat)
{
  ST_PRECOND *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  if (ctx->ksphasmat != ksphasmat) {
    ctx->ksphasmat = ksphasmat;
    st->state      = ST_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   STPrecondSetKSPHasMat - Sets a flag indicating that during STSetUp the coefficient
   matrix of the KSP linear solver (A) must be set to be the same matrix as the
   preconditioner (P).

   Collective on st

   Input Parameters:
+  st - the spectral transformation context
-  ksphasmat - the flag

   Notes:
   Often, the preconditioner matrix is used only in the PC object, but
   in some solvers this matrix must be provided also as the A-matrix in
   the KSP object.

   Level: developer

.seealso: STPrecondGetKSPHasMat(), STSetShift()
@*/
PetscErrorCode STPrecondSetKSPHasMat(ST st,PetscBool ksphasmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveBool(st,ksphasmat,2);
  ierr = PetscTryMethod(st,"STPrecondSetKSPHasMat_C",(ST,PetscBool),(st,ksphasmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode STPrecondGetKSPHasMat_Precond(ST st,PetscBool *ksphasmat)
{
  ST_PRECOND *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  *ksphasmat = ctx->ksphasmat;
  PetscFunctionReturn(0);
}

/*@
   STPrecondGetKSPHasMat - Returns the flag indicating if the coefficient
   matrix of the KSP linear system (A) is set to be the same matrix as the
   preconditioner (P).

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  ksphasmat - the flag

   Level: developer

.seealso: STPrecondSetKSPHasMat(), STSetShift()
@*/
PetscErrorCode STPrecondGetKSPHasMat(ST st,PetscBool *ksphasmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidBoolPointer(ksphasmat,2);
  ierr = PetscUseMethod(st,"STPrecondGetKSPHasMat_C",(ST,PetscBool*),(st,ksphasmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode STDestroy_Precond(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(st->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondGetKSPHasMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondSetKSPHasMat_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode STCreate_Precond(ST st)
{
  PetscErrorCode ierr;
  ST_PRECOND     *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(st,&ctx);CHKERRQ(ierr);
  st->data = (void*)ctx;

  st->usesksp = PETSC_TRUE;

  st->ops->apply           = STApply_Generic;
  st->ops->applymat        = STApplyMat_Generic;
  st->ops->applytrans      = STApplyTranspose_Generic;
  st->ops->setshift        = STSetShift_Precond;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->setup           = STSetUp_Precond;
  st->ops->computeoperator = STComputeOperator_Precond;
  st->ops->postsolve       = STPostSolve_Precond;
  st->ops->destroy         = STDestroy_Precond;
  st->ops->setdefaultksp   = STSetDefaultKSP_Precond;

  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondGetKSPHasMat_C",STPrecondGetKSPHasMat_Precond);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondSetKSPHasMat_C",STPrecondSetKSPHasMat_Precond);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

