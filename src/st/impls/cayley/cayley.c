/*
      Implements the Cayley spectral transform.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/stimpl.h>          /*I "slepcst.h" I*/

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
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;
  PetscScalar    nu = ctx->nu;
  
  PetscFunctionBegin;
  if (st->shift_matrix == ST_MATMODE_INPLACE) { nu = nu + st->sigma; };

  if (st->B) {
    /* generalized eigenproblem: y = (A - sB)^-1 (A + tB)x */
    ierr = MatMult(st->A,x,st->w);CHKERRQ(ierr);
    ierr = MatMult(st->B,x,ctx->w2);CHKERRQ(ierr);
    ierr = VecAXPY(st->w,nu,ctx->w2);CHKERRQ(ierr);    
    ierr = STAssociatedKSPSolve(st,st->w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A - sI)^-1 (A + tI)x */
    ierr = MatMult(st->A,x,st->w);CHKERRQ(ierr);
    ierr = VecAXPY(st->w,nu,x);CHKERRQ(ierr);
    ierr = STAssociatedKSPSolve(st,st->w,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyTranspose_Cayley"
PetscErrorCode STApplyTranspose_Cayley(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;
  PetscScalar    nu = ctx->nu;
  
  PetscFunctionBegin;
  if (st->shift_matrix == ST_MATMODE_INPLACE) { nu = nu + st->sigma; };

  if (st->B) {
    /* generalized eigenproblem: y = (A + tB)^T (A - sB)^-T x */
    ierr = STAssociatedKSPSolveTranspose(st,x,st->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(st->A,st->w,y);CHKERRQ(ierr);
    ierr = MatMultTranspose(st->B,st->w,ctx->w2);CHKERRQ(ierr);
    ierr = VecAXPY(y,nu,ctx->w2);CHKERRQ(ierr);    
  }
  else {
    /* standard eigenproblem: y =  (A + tI)^T (A - sI)^-T x */
    ierr = STAssociatedKSPSolveTranspose(st,x,st->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(st->A,st->w,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,nu,st->w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBilinearMatMult_Cayley"
PetscErrorCode STBilinearMatMult_Cayley(Mat B,Vec x,Vec y)
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

  if (st->B) {
    /* generalized eigenproblem: y = (A + tB)x */
    ierr = MatMult(st->A,x,y);CHKERRQ(ierr);
    ierr = MatMult(st->B,x,ctx->w2);CHKERRQ(ierr);
    ierr = VecAXPY(y,nu,ctx->w2);CHKERRQ(ierr);    
  }
  else {
    /* standard eigenproblem: y = (A + tI)x */
    ierr = MatMult(st->A,x,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,nu,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetBilinearForm_Cayley"
PetscErrorCode STGetBilinearForm_Cayley(ST st,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n,m;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(st->B,&n,&m);CHKERRQ(ierr);
  ierr = MatCreateShell(((PetscObject)st)->comm,n,m,PETSC_DETERMINE,PETSC_DETERMINE,st,B);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_MULT,(void(*)(void))STBilinearMatMult_Cayley);CHKERRQ(ierr);
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
    if (st->B) {
      ierr = MatAXPY(st->A,st->sigma,st->B,st->str);CHKERRQ(ierr);
    } else { 
      ierr = MatShift(st->A,st->sigma);CHKERRQ(ierr); 
    }
    st->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp_Cayley"
PetscErrorCode STSetUp_Cayley(ST st)
{
  PetscErrorCode ierr;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&st->mat);CHKERRQ(ierr);

  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;

  if (!ctx->nu_set) { ctx->nu = st->sigma; }
  if (ctx->nu == 0.0 &&  st->sigma == 0.0) {
    SETERRQ(((PetscObject)st)->comm,1,"Values of shift and antishift cannot be zero simultaneously");
  }

  switch (st->shift_matrix) {
  case ST_MATMODE_INPLACE:
    st->mat = PETSC_NULL;
    if (st->sigma != 0.0) {
      if (st->B) { 
        ierr = MatAXPY(st->A,-st->sigma,st->B,st->str);CHKERRQ(ierr); 
      } else { 
        ierr = MatShift(st->A,-st->sigma);CHKERRQ(ierr); 
      }
    }
    ierr = KSPSetOperators(st->ksp,st->A,st->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  case ST_MATMODE_SHELL:
    ierr = STMatShellCreate(st,&st->mat);CHKERRQ(ierr);
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  default:
    ierr = MatDuplicate(st->A,MAT_COPY_VALUES,&st->mat);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
      if (st->B) { 
        ierr = MatAXPY(st->mat,-st->sigma,st->B,st->str);CHKERRQ(ierr); 
      } else { 
        ierr = MatShift(st->mat,-st->sigma);CHKERRQ(ierr); 
      }
    }
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  if (st->B) { 
    ierr = VecDestroy(&ctx->w2);CHKERRQ(ierr);
    ierr = MatGetVecs(st->B,&ctx->w2,PETSC_NULL);CHKERRQ(ierr); 
  }
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetShift_Cayley"
PetscErrorCode STSetShift_Cayley(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;
  MatStructure   flg;

  PetscFunctionBegin;
  if (!ctx->nu_set) { ctx->nu = newshift; }
  if (ctx->nu == 0.0 &&  newshift == 0.0) {
    SETERRQ(((PetscObject)st)->comm,1,"Values of shift and antishift cannot be zero simultaneously");
  }

  /* Nothing to be done if STSetUp has not been called yet */
  if (!st->setupcalled) PetscFunctionReturn(0);

  /* Check if the new KSP matrix has the same zero structure */
  if (st->B && st->str == DIFFERENT_NONZERO_PATTERN && (st->sigma == 0.0 || newshift == 0.0)) {
    flg = DIFFERENT_NONZERO_PATTERN;
  } else {
    flg = SAME_NONZERO_PATTERN;
  }

  switch (st->shift_matrix) {
  case ST_MATMODE_INPLACE:
    /* Undo previous operations */
    if (st->sigma != 0.0) {
      if (st->B) { 
        ierr = MatAXPY(st->A,st->sigma,st->B,st->str);CHKERRQ(ierr);
      } else {
        ierr = MatShift(st->A,st->sigma);CHKERRQ(ierr);
      }
    }
    /* Apply new shift */
    if (newshift != 0.0) {
      if (st->B) { 
        ierr = MatAXPY(st->A,-newshift,st->B,st->str);CHKERRQ(ierr);
      } else {
        ierr = MatShift(st->A,-newshift);CHKERRQ(ierr);
      }
    }
    ierr = KSPSetOperators(st->ksp,st->A,st->A,flg);CHKERRQ(ierr);
    break;
  case ST_MATMODE_SHELL:
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);    
    break;
  default:
    ierr = MatCopy(st->A,st->mat,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    if (newshift != 0.0) {   
      if (st->B) { ierr = MatAXPY(st->mat,-newshift,st->B,st->str);CHKERRQ(ierr); }
      else { ierr = MatShift(st->mat,-newshift);CHKERRQ(ierr); }
    }
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,flg);CHKERRQ(ierr);    
  }
  st->sigma = newshift;
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetFromOptions_Cayley"
PetscErrorCode STSetFromOptions_Cayley(ST st) 
{
  PetscErrorCode ierr;
  PetscScalar    nu;
  PetscBool      flg;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;
  PC             pc;
  const PCType   pctype;
  const KSPType  ksptype;

  PetscFunctionBegin;
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
      ierr = PCSetType(pc,PCREDUNDANT);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsBegin(((PetscObject)st)->comm,((PetscObject)st)->prefix,"ST Cayley Options","ST");CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-st_cayley_antishift","Value of the antishift","STCayleySetAntishift",ctx->nu,&nu,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = STCayleySetAntishift(st,nu);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCayleySetAntishift_Cayley"
PetscErrorCode STCayleySetAntishift_Cayley(ST st,PetscScalar newshift)
{
  ST_CAYLEY *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
  ctx->nu     = newshift;
  ctx->nu_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCayleyGetAntishift_Cayley"
PetscErrorCode STCayleyGetAntishift_Cayley(ST st,PetscScalar *nu)
{
  ST_CAYLEY *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
  *nu = ctx->nu;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
  ierr = PetscTryMethod(st,"STCayleyGetAntishift_C",(ST,PetscScalar*),(st,nu));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STView_Cayley"
PetscErrorCode STView_Cayley(ST st,PetscViewer viewer)
{
  PetscErrorCode ierr;
  ST_CAYLEY      *ctx = (ST_CAYLEY*)st->data;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscViewerASCIIPrintf(viewer,"  antishift: %g\n",ctx->nu);CHKERRQ(ierr);
#else
  ierr = PetscViewerASCIIPrintf(viewer,"  antishift: %g+%g i\n",PetscRealPart(ctx->nu),PetscImaginaryPart(ctx->nu));CHKERRQ(ierr);
#endif
  ierr = STView_Default(st,viewer);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STCayleySetAntishift_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Cayley"
PetscErrorCode STCreate_Cayley(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(st,ST_CAYLEY,&st->data);CHKERRQ(ierr);
  st->ops->apply           = STApply_Cayley;
  st->ops->getbilinearform = STGetBilinearForm_Cayley;
  st->ops->applytrans      = STApplyTranspose_Cayley;
  st->ops->postsolve       = STPostSolve_Cayley;
  st->ops->backtr          = STBackTransform_Cayley;
  st->ops->setfromoptions  = STSetFromOptions_Cayley;
  st->ops->setup           = STSetUp_Cayley;
  st->ops->setshift        = STSetShift_Cayley;
  st->ops->destroy         = STDestroy_Cayley;
  st->ops->reset           = STReset_Cayley;
  st->ops->view            = STView_Cayley;
  st->ops->checknullspace  = STCheckNullSpace_Default;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STCayleySetAntishift_C","STCayleySetAntishift_Cayley",STCayleySetAntishift_Cayley);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

