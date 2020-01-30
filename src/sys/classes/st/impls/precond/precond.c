/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Implements the ST class for preconditioned eigenvalue methods
*/

#include <slepc/private/stimpl.h>          /*I "slepcst.h" I*/

typedef struct {
  PetscBool setmat;
  Mat       mat;
} ST_PRECOND;

static PetscErrorCode STSetDefaultKSP_Precond(ST st)
{
  PetscErrorCode ierr;
  ST_PRECOND     *ctx = (ST_PRECOND*)st->data;
  PC             pc;
  PCType         pctype;
  PetscBool      t0,t1;

  PetscFunctionBegin;
  ierr = KSPGetPC(st->ksp,&pc);CHKERRQ(ierr);
  ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
  if (!pctype && st->A && st->A[0]) {
    if (ctx->setmat || st->matmode == ST_MATMODE_SHELL) {
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
  ST_PRECOND     *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  if (st->matmode == ST_MATMODE_INPLACE && !(ctx->mat || (PetscAbsScalar(st->sigma)>=PETSC_MAX_REAL && st->nmat>1))) {
    if (st->nmat>1) {
      ierr = MatAXPY(st->A[0],st->sigma,st->A[1],st->str);CHKERRQ(ierr);
    } else {
      ierr = MatShift(st->A[0],st->sigma);CHKERRQ(ierr);
    }
    st->Astate[0] = ((PetscObject)st->A[0])->state;
    st->state = ST_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STComputeOperator_Precond(ST st)
{
  PetscErrorCode ierr;
  ST_PRECOND     *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;

  /* P = A-sigma*B */
  if (ctx->mat) {
    ierr = PetscObjectReference((PetscObject)ctx->mat);CHKERRQ(ierr);
    ierr = MatDestroy(&st->P);CHKERRQ(ierr);
    st->P = ctx->mat;
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
    ierr = KSPSetOperators(st->ksp,ctx->setmat?st->P:NULL,st->P);CHKERRQ(ierr);
    /* NOTE: we do not call KSPSetUp() here because some eigensolvers such as JD require a lazy setup */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STSetShift_Precond(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;
  ST_PRECOND     *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  if (st->transform && !ctx->mat) {
    ierr = STMatMAXPY_Private(st,-newshift,-st->sigma,0,NULL,PETSC_FALSE,&st->T[1]);CHKERRQ(ierr);
    if (st->P!=st->T[1]) {
      ierr = PetscObjectReference((PetscObject)st->T[1]);CHKERRQ(ierr);
      ierr = MatDestroy(&st->P);CHKERRQ(ierr);
      st->P = st->T[1];
    }
  }
  if (st->P) {
    if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
    ierr = KSPSetOperators(st->ksp,ctx->setmat?st->P:NULL,st->P);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode STPrecondGetMatForPC_Precond(ST st,Mat *mat)
{
  ST_PRECOND *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  *mat = ctx->mat;
  PetscFunctionReturn(0);
}

/*@
   STPrecondGetMatForPC - Returns the matrix previously set by STPrecondSetMatForPC().

   Not Collective, but the Mat is shared by all processors that share the ST

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  mat - the matrix that will be used in constructing the preconditioner or
   NULL if no matrix was set by STPrecondSetMatForPC().

   Level: advanced

.seealso: STPrecondSetMatForPC()
@*/
PetscErrorCode STPrecondGetMatForPC(ST st,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(mat,2);
  ierr = PetscUseMethod(st,"STPrecondGetMatForPC_C",(ST,Mat*),(st,mat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode STPrecondSetMatForPC_Precond(ST st,Mat mat)
{
  ST_PRECOND     *ctx = (ST_PRECOND*)st->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  STCheckNotSeized(st,1);
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->mat);CHKERRQ(ierr);
  ctx->mat    = mat;
  ctx->setmat = mat? PETSC_TRUE: PETSC_FALSE;
  st->state   = ST_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   STPrecondSetMatForPC - Sets the matrix that must be used to build the preconditioner.

   Logically Collective on st

   Input Parameter:
+  st - the spectral transformation context
-  mat - the matrix that will be used in constructing the preconditioner

   Level: advanced

   Notes:
   This matrix will be passed to the KSP object (via KSPSetOperators) as
   the matrix to be used when constructing the preconditioner.
   If no matrix is set or mat is set to NULL, A - sigma*B will
   be used to build the preconditioner, being sigma the value set by STSetShift().

.seealso: STPrecondSetMatForPC(), STSetShift()
@*/
PetscErrorCode STPrecondSetMatForPC(ST st,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscCheckSameComm(st,1,mat,2);
  ierr = PetscTryMethod(st,"STPrecondSetMatForPC_C",(ST,Mat),(st,mat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode STPrecondSetKSPHasMat_Precond(ST st,PetscBool setmat)
{
  ST_PRECOND *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  if (ctx->setmat != setmat) {
    ctx->setmat = setmat;
    st->state   = ST_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   STPrecondSetKSPHasMat - Sets a flag indicating that during STSetUp the coefficient
   matrix of the KSP linear system (A) must be set to be the same matrix as the
   preconditioner (P).

   Collective on st

   Input Parameter:
+  st - the spectral transformation context
-  setmat - the flag

   Notes:
   In most cases, the preconditioner matrix is used only in the PC object, but
   in external solvers this matrix must be provided also as the A-matrix in
   the KSP object.

   Level: developer

.seealso: STPrecondGetKSPHasMat(), STSetShift()
@*/
PetscErrorCode STPrecondSetKSPHasMat(ST st,PetscBool setmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveBool(st,setmat,2);
  ierr = PetscTryMethod(st,"STPrecondSetKSPHasMat_C",(ST,PetscBool),(st,setmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode STPrecondGetKSPHasMat_Precond(ST st,PetscBool *setmat)
{
  ST_PRECOND *ctx = (ST_PRECOND*)st->data;

  PetscFunctionBegin;
  *setmat = ctx->setmat;
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
.  setmat - the flag

   Level: developer

.seealso: STPrecondSetKSPHasMat(), STSetShift()
@*/
PetscErrorCode STPrecondGetKSPHasMat(ST st,PetscBool *setmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidBoolPointer(setmat,2);
  ierr = PetscUseMethod(st,"STPrecondGetKSPHasMat_C",(ST,PetscBool*),(st,setmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode STDestroy_Precond(ST st)
{
  ST_PRECOND     *ctx = (ST_PRECOND*)st->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&ctx->mat);CHKERRQ(ierr);
  ierr = PetscFree(st->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondGetMatForPC_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondSetMatForPC_C",NULL);CHKERRQ(ierr);
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
  ctx->setmat = PETSC_TRUE;

  st->usesksp = PETSC_TRUE;

  st->ops->setshift        = STSetShift_Precond;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->setup           = STSetUp_Precond;
  st->ops->computeoperator = STComputeOperator_Precond;
  st->ops->postsolve       = STPostSolve_Precond;
  st->ops->destroy         = STDestroy_Precond;
  st->ops->setdefaultksp   = STSetDefaultKSP_Precond;

  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondGetMatForPC_C",STPrecondGetMatForPC_Precond);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondSetMatForPC_C",STPrecondSetMatForPC_Precond);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondGetKSPHasMat_C",STPrecondGetKSPHasMat_Precond);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STPrecondSetKSPHasMat_C",STPrecondSetKSPHasMat_Precond);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

