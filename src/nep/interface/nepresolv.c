/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   NEP routines related to resolvent T^{-1}(z) = sum_i (z-lambda_i)^{-1} x_i y_i'
*/

#include <slepc/private/nepimpl.h>       /*I "slepcnep.h" I*/

typedef struct {
  NEP         nep;
  //PetscInt    nmat,maxnmat;
  //PetscScalar *coeff;
} ResolventCtx;

static PetscErrorCode MatMult_Resolvent(Mat M,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ResolventCtx   *ctx;
  NEP            nep;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  nep = ctx->nep;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Resolvent(Mat M)
{
  PetscErrorCode ierr;
  ResolventCtx   *ctx;

  PetscFunctionBegin;
  if (M) {
    ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
    ierr = PetscFree(ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   NEPApplyResolvent - Applies the resolvent T^{-1}(z) to a given vector.

   Collective on NEP

   Input Parameters:
+  nep   - eigensolver context obtained from NEPCreate()
.  rg    - optional region
.  omega - value where the resolvent must be evaluated
-  v     - input vector

   Output Parameter:
.  r     - result vector

   Notes:
   The resolvent T^{-1}(z) = sum_i (z-lambda_i)^{-1}*x_i*y_i' is evaluated at
   z=omega and the matrix-vector multiplication r = T^{-1}(omega)*v is computed.
   Vectors x_i and y_i are right and left eigenvectors, respectively, normalized
   so that y_i'*T'(lambda_i)*x_i=1. The sum contains only eigenvectors that have
   been previously computed with NEPSolve(), and if a region rg is given then only
   those corresponding to eigenvalues inside the region are considered.

   Level: intermediate

.seealso: NEPGetLeftEigenvector(), NEPSolve()
@*/
PetscErrorCode NEPApplyResolvent(NEP nep,RG rg,PetscScalar omega,Vec v,Vec r)
{
  PetscErrorCode ierr;
  ResolventCtx   *ctx;
  PetscInt       i,inside=1;
  PetscScalar    alpha,dot;
  Vec            x,y,z,w;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(nep,omega,3);
  PetscValidHeaderSpecific(v,VEC_CLASSID,4);
  PetscValidHeaderSpecific(r,VEC_CLASSID,5);
  NEPCheckSolved(nep,1);

  ierr = PetscLogEventBegin(NEP_Resolvent,nep,0,0,0);CHKERRQ(ierr);
  if (!nep->resolvent) {
    ierr = PetscNew(&ctx);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)nep),nep->nloc,nep->nloc,nep->n,nep->n,ctx,&nep->resolvent);CHKERRQ(ierr);
    ierr = MatShellSetOperation(nep->resolvent,MATOP_MULT,(void(*)(void))MatMult_Resolvent);CHKERRQ(ierr);
    ierr = MatShellSetOperation(nep->resolvent,MATOP_DESTROY,(void(*)(void))MatDestroy_Resolvent);CHKERRQ(ierr);
  }
  ierr = NEPComputeVectors(nep);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,2);CHKERRQ(ierr);
  w = nep->work[0];
  z = nep->work[1];
  ierr = VecSet(r,0.0);CHKERRQ(ierr);
  for (i=0;i<nep->nconv;i++) {
    if (rg) {
      ierr = RGCheckInside(rg,1,&nep->eigr[i],&nep->eigi[i],&inside);CHKERRQ(ierr);
    }
    if (inside>=0) {
      ierr = BVGetColumn(nep->V,i,&x);CHKERRQ(ierr);
      ierr = BVGetColumn(nep->W,i,&y);CHKERRQ(ierr);
      ierr = NEPApplyJacobian(nep,nep->eigr[i],x,z,w,NULL);CHKERRQ(ierr);
      ierr = VecDot(v,y,&alpha);CHKERRQ(ierr);
      ierr = VecDot(w,y,&dot);CHKERRQ(ierr);
      alpha /= dot*(omega-nep->eigr[i]);
      ierr = VecAXPY(r,alpha,x);CHKERRQ(ierr);
      ierr = BVRestoreColumn(nep->V,i,&x);CHKERRQ(ierr);
      ierr = BVRestoreColumn(nep->W,i,&y);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(NEP_Resolvent,nep,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

