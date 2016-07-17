/*
      Implements the Cayley spectral transform.

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

#include <slepc/private/stimpl.h>          /*I "slepcst.h" I*/

typedef struct {
  PetscScalar nu;
  PetscBool   nu_set;
  Vec         w2;
} ST_CAYLEY;

#undef __FUNCT__
#define __FUNCT__ "STApply_Cayley"
PetscErrorCode STApply_Cayley(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* standard eigenproblem: y = (A - sI)^-1 (A + tI)x */
  /* generalized eigenproblem: y = (A - sB)^-1 (A + tB)x */
  ierr = MatMult(st->T[0],x,st->w);CHKERRQ(ierr);
  ierr = STMatSolve(st,st->w,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STApplyTranspose_Cayley"
PetscErrorCode STApplyTranspose_Cayley(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* standard eigenproblem: y =  (A + tI)^T (A - sI)^-T x */
  /* generalized eigenproblem: y = (A + tB)^T (A - sB)^-T x */
  ierr = STMatSolveTranspose(st,x,st->w);CHKERRQ(ierr);
  ierr = MatMultTranspose(st->T[0],st->w,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Cayley"
static PetscErrorCode MatMult_Cayley(Mat B,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ST             st;
  ST_CAYLEY      *ctx;
  PetscScalar    nu;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&st);CHKERRQ(ierr);
  ctx = (ST_CAYLEY*)st->data;
  nu = ctx->nu;

  if (st->shift_matrix == ST_MATMODE_INPLACE) { nu = nu + st->sigma; };

  if (st->nmat>1) {
    /* generalized eigenproblem: y = (A + tB)x */
    ierr = MatMult(st->A[0],x,y);CHKERRQ(ierr);
    ierr = MatMult(st->A[1],x,ctx->w2);CHKERRQ(ierr);
    ierr = VecAXPY(y,nu,ctx->w2);CHKERRQ(ierr);
  } else {
    /* standard eigenproblem: y = (A + tI)x */
    ierr = MatMult(st->A[0],x,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,nu,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetBilinearForm_Cayley"
PetscErrorCode STGetBilinearForm_Cayley(ST st,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = STSetUp(st);CHKERRQ(ierr);
  *B = st->T[0];
  ierr = PetscObjectReference((PetscObject)*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STBackTransform_Cayley"
PetscErrorCode STBackTransform_Cayley(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  ST_CAYLEY   *ctx = (ST_CAYLEY*)st->data;
  PetscInt    j;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar t,i,r;
#endif

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  for (j=0;j<n;j++) {
    if (eigi[j] == 0.0) eigr[j] = (ctx->nu + eigr[j] * st->sigma) / (eigr[j] - 1.0);
    else {
      r = eigr[j];
      i = eigi[j];
      r = st->sigma * (r * r + i * i - r) + ctx->nu * (r - 1);
      i = - st->sigma * i - ctx->nu * i;
      t = i * i + r * (r - 2.0) + 1.0;
      eigr[j] = r / t;
      eigi[j] = i / t;
    }
  }
#else
  for (j=0;j<n;j++) {
    eigr[j] = (ctx->nu + eigr[j] * st->sigma) / (eigr[j] - 1.0);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STPostSolve_Cayley"
PetscErrorCode STPostSolve_Cayley(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->shift_matrix == ST_MATMODE_INPLACE) {
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

#undef __FUNCT__
#define __FUNCT__ "STSetUp_Cayley"
PetscErrorCode STSetUp_Cayley(ST st)
{
  PetscErrorCode ierr;
  PetscInt       n,m;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
  ierr = ST_AllocateWorkVec(st);CHKERRQ(ierr);

  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;

  if (!ctx->nu_set) ctx->nu = st->sigma;
  if (ctx->nu == 0.0 && st->sigma == 0.0) SETERRQ(PetscObjectComm((PetscObject)st),1,"Values of shift and antishift cannot be zero simultaneously");

  /* T[0] = A+nu*B */
  if (st->shift_matrix==ST_MATMODE_INPLACE) {
    ierr = MatGetLocalSize(st->A[0],&n,&m);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)st),n,m,PETSC_DETERMINE,PETSC_DETERMINE,st,&st->T[0]);CHKERRQ(ierr);
    ierr = MatShellSetOperation(st->T[0],MATOP_MULT,(void(*)(void))MatMult_Cayley);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)st,(PetscObject)st->T[0]);CHKERRQ(ierr);
  } else {
    ierr = STMatMAXPY_Private(st,ctx->nu,0.0,0,NULL,PetscNot(st->state==ST_STATE_UPDATED),&st->T[0]);CHKERRQ(ierr);
  }

  /* T[1] = A-sigma*B */
  ierr = STMatMAXPY_Private(st,-st->sigma,0.0,0,NULL,PetscNot(st->state==ST_STATE_UPDATED),&st->T[1]);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)st->T[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&st->P);CHKERRQ(ierr);
  st->P = st->T[1];
  if (st->nmat>1) {
    ierr = VecDestroy(&ctx->w2);CHKERRQ(ierr);
    ierr = MatCreateVecs(st->A[1],&ctx->w2,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)st,(PetscObject)ctx->w2);CHKERRQ(ierr);
  }
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = STCheckFactorPackage(st);CHKERRQ(ierr);
  ierr = KSPSetOperators(st->ksp,st->P,st->P);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(st->ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetShift_Cayley"
PetscErrorCode STSetShift_Cayley(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
  if (newshift==0.0 && (!ctx->nu_set || (ctx->nu_set && ctx->nu==0.0))) SETERRQ(PetscObjectComm((PetscObject)st),1,"Values of shift and antishift cannot be zero simultaneously");

  if (!ctx->nu_set) {
    if (st->shift_matrix!=ST_MATMODE_INPLACE) {
      ierr = STMatMAXPY_Private(st,newshift,ctx->nu,0,NULL,PETSC_FALSE,&st->T[0]);CHKERRQ(ierr);
    }
    ctx->nu = newshift;
  }
  ierr = STMatMAXPY_Private(st,-newshift,-st->sigma,0,NULL,PETSC_FALSE,&st->T[1]);CHKERRQ(ierr);
  if (st->P!=st->T[1]) {
    ierr = MatDestroy(&st->P);CHKERRQ(ierr);
    st->P = st->T[1];
    ierr = PetscObjectReference((PetscObject)st->P);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(st->ksp,st->P,st->P);CHKERRQ(ierr);
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetFromOptions_Cayley"
PetscErrorCode STSetFromOptions_Cayley(PetscOptionItems *PetscOptionsObject,ST st)
{
  PetscErrorCode ierr;
  PetscScalar    nu;
  PetscBool      flg;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;
  PC             pc;
  PCType         pctype;
  KSPType        ksptype;

  PetscFunctionBegin;
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = KSPGetPC(st->ksp,&pc);CHKERRQ(ierr);
  ierr = KSPGetType(st->ksp,&ksptype);CHKERRQ(ierr);
  ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
  if (!pctype && !ksptype) {
    if (st->shift_matrix == ST_MATMODE_SHELL) {
      /* in shell mode use GMRES with Jacobi as the default */
      ierr = KSPSetType(st->ksp,KSPGMRES);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    } else {
      /* use direct solver as default */
      ierr = KSPSetType(st->ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsHead(PetscOptionsObject,"ST Cayley Options");CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-st_cayley_antishift","Value of the antishift","STCayleySetAntishift",ctx->nu,&nu,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = STCayleySetAntishift(st,nu);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCayleySetAntishift_Cayley"
static PetscErrorCode STCayleySetAntishift_Cayley(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;
  ST_CAYLEY *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
  if (st->state && st->shift_matrix!=ST_MATMODE_INPLACE) {
    ierr = STMatMAXPY_Private(st,newshift,ctx->nu,0,NULL,PETSC_FALSE,&st->T[0]);CHKERRQ(ierr);
  }
  ctx->nu     = newshift;
  ctx->nu_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCayleySetAntishift"
/*@
   STCayleySetAntishift - Sets the value of the anti-shift for the Cayley
   spectral transformation.

   Logically Collective on ST

   Input Parameters:
+  st  - the spectral transformation context
-  nu  - the anti-shift

   Options Database Key:
.  -st_cayley_antishift - Sets the value of the anti-shift

   Level: intermediate

   Note:
   In the generalized Cayley transform, the operator can be expressed as
   OP = inv(A - sigma B)*(A + nu B). This function sets the value of nu.
   Use STSetShift() for setting sigma.

.seealso: STSetShift(), STCayleyGetAntishift()
@*/
PetscErrorCode STCayleySetAntishift(ST st,PetscScalar nu)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveScalar(st,nu,2);
  ierr = PetscTryMethod(st,"STCayleySetAntishift_C",(ST,PetscScalar),(st,nu));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "STCayleyGetAntishift_Cayley"
static PetscErrorCode STCayleyGetAntishift_Cayley(ST st,PetscScalar *nu)
{
  ST_CAYLEY *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
  *nu = ctx->nu;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCayleyGetAntishift"
/*@
   STCayleyGetAntishift - Gets the value of the anti-shift used in the Cayley
   spectral transformation.

   Not Collective

   Input Parameter:
.  st  - the spectral transformation context

   Output Parameter:
.  nu  - the anti-shift

   Level: intermediate

.seealso: STGetShift(), STCayleySetAntishift()
@*/
PetscErrorCode STCayleyGetAntishift(ST st,PetscScalar *nu)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidScalarPointer(nu,2);
  ierr = PetscUseMethod(st,"STCayleyGetAntishift_C",(ST,PetscScalar*),(st,nu));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STView_Cayley"
PetscErrorCode STView_Cayley(ST st,PetscViewer viewer)
{
  PetscErrorCode ierr;
  char           str[50];
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = SlepcSNPrintfScalar(str,50,ctx->nu,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Cayley: antishift: %s\n",str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STReset_Cayley"
PetscErrorCode STReset_Cayley(ST st)
{
  PetscErrorCode ierr;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->w2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STDestroy_Cayley"
PetscErrorCode STDestroy_Cayley(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(st->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STCayleySetAntishift_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STCayleyGetAntishift_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCreate_Cayley"
PETSC_EXTERN PetscErrorCode STCreate_Cayley(ST st)
{
  PetscErrorCode ierr;
  ST_CAYLEY      *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(st,&ctx);CHKERRQ(ierr);
  st->data = (void*)ctx;

  st->ops->apply           = STApply_Cayley;
  st->ops->getbilinearform = STGetBilinearForm_Cayley;
  st->ops->applytrans      = STApplyTranspose_Cayley;
  st->ops->postsolve       = STPostSolve_Cayley;
  st->ops->backtransform   = STBackTransform_Cayley;
  st->ops->setfromoptions  = STSetFromOptions_Cayley;
  st->ops->setup           = STSetUp_Cayley;
  st->ops->setshift        = STSetShift_Cayley;
  st->ops->destroy         = STDestroy_Cayley;
  st->ops->reset           = STReset_Cayley;
  st->ops->view            = STView_Cayley;
  st->ops->checknullspace  = STCheckNullSpace_Default;
  ierr = PetscObjectComposeFunction((PetscObject)st,"STCayleySetAntishift_C",STCayleySetAntishift_Cayley);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STCayleyGetAntishift_C",STCayleyGetAntishift_Cayley);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

