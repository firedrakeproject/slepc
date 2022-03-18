/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV bi-orthogonalization routines
*/

#include <slepc/private/bvimpl.h>          /*I   "slepcbv.h"   I*/

/*
   BVBiorthogonalizeMGS1 - Compute one step of Modified Gram-Schmidt bi-orthogonalization
*/
static PetscErrorCode BVBiorthogonalizeMGS1(BV V,BV W,Vec v,PetscScalar *h,PetscScalar *c)
{
  PetscInt       i;
  PetscScalar    dot;
  Vec            vi,wi;

  PetscFunctionBegin;
  for (i=-V->nc;i<V->k;i++) {
    CHKERRQ(BVGetColumn(W,i,&wi));
    /* h_i = (v, w_i) */
    CHKERRQ(VecDot(v,wi,&dot));
    CHKERRQ(BVRestoreColumn(W,i,&wi));
    /* v <- v - h_i v_i */
    CHKERRQ(BV_SetValue(V,i,0,c,dot));
    CHKERRQ(BVGetColumn(V,i,&vi));
    CHKERRQ(VecAXPY(v,-dot,vi));
    CHKERRQ(BVRestoreColumn(V,i,&vi));
  }
  CHKERRQ(BV_AddCoefficients(V,V->k,h,c));
  PetscFunctionReturn(0);
}

/*
   BVBiorthogonalizeCGS1 - Compute one step of CGS bi-orthogonalization: v = (I-V*W')*v
*/
static PetscErrorCode BVBiorthogonalizeCGS1(BV V,BV W,Vec v,PetscScalar *h,PetscScalar *c)
{
  PetscFunctionBegin;
  /* h = W'*v */
  CHKERRQ(BVDotVec(W,v,c));

  /* v = v - V h */
  CHKERRQ(BVMultVec(V,-1.0,1.0,v,c));

  CHKERRQ(BV_AddCoefficients(V,V->k,h,c));
  PetscFunctionReturn(0);
}

#define BVBiorthogonalizeGS1(a,b,c,d,e) ((V->orthog_type==BV_ORTHOG_MGS)?BVBiorthogonalizeMGS1:BVBiorthogonalizeCGS1)(a,b,c,d,e)

/*
   BVBiorthogonalizeGS - Orthogonalize with (classical or modified) Gram-Schmidt

   V, W - the two basis vectors objects
   v    - the vector to bi-orthogonalize
*/
static PetscErrorCode BVBiorthogonalizeGS(BV V,BV W,Vec v)
{
  PetscScalar    *h,*c;

  PetscFunctionBegin;
  h = V->h;
  c = V->c;
  CHKERRQ(BV_CleanCoefficients(V,V->k,h));
  CHKERRQ(BVBiorthogonalizeGS1(V,W,v,h,c));
  if (V->orthog_ref!=BV_ORTHOG_REFINE_NEVER) {
    CHKERRQ(BVBiorthogonalizeGS1(V,W,v,h,c));
  }
  PetscFunctionReturn(0);
}

/*@
   BVBiorthogonalizeColumn - Bi-orthogonalize a column of two BV objects.

   Collective on V

   Input Parameters:
+  V - first basis vectors context
.  W - second basis vectors context
-  j - index of column to be bi-orthonormalized

   Notes:
   This function bi-orthogonalizes vectors V[j],W[j] against W[0..j-1],
   and V[0..j-1], respectively, so that W[0..j]'*V[0..j] = diagonal.

   Level: advanced

.seealso: BVOrthogonalizeColumn(), BVBiorthonormalizeColumn()
@*/
PetscErrorCode BVBiorthogonalizeColumn(BV V,BV W,PetscInt j)
{
  PetscInt       ksavev,lsavev,ksavew,lsavew;
  Vec            y,z;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidHeaderSpecific(W,BV_CLASSID,2);
  PetscValidLogicalCollectiveInt(V,j,3);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidType(W,2);
  BVCheckSizes(W,2);
  PetscCheckSameTypeAndComm(V,1,W,2);
  PetscCheck(j>=0,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  PetscCheck(j<V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%" PetscInt_FMT " but V only has %" PetscInt_FMT " columns",j,V->m);
  PetscCheck(j<W->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%" PetscInt_FMT " but W only has %" PetscInt_FMT " columns",j,W->m);
  PetscCheck(V->n==W->n,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Mismatching local dimension V %" PetscInt_FMT ", W %" PetscInt_FMT,V->n,W->n);
  PetscCheck(!V->matrix && !W->matrix,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_WRONGSTATE,"V,W must not have an inner product matrix");
  PetscCheck(!V->nc && !W->nc,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_WRONGSTATE,"V,W cannot have different number of constraints");
  PetscCheck(!V->ops->gramschmidt && !W->ops->gramschmidt,PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Object has a special GS function");

  /* bi-orthogonalize */
  CHKERRQ(PetscLogEventBegin(BV_OrthogonalizeVec,V,0,0,0));
  ksavev = V->k;
  lsavev = V->l;
  ksavew = W->k;
  lsavew = W->l;
  V->k = j;
  V->l = -V->nc;  /* must also bi-orthogonalize against constraints and leading columns */
  W->k = j;
  W->l = -W->nc;
  CHKERRQ(BV_AllocateCoeffs(V));
  CHKERRQ(BV_AllocateCoeffs(W));
  CHKERRQ(BVGetColumn(V,j,&y));
  CHKERRQ(BVBiorthogonalizeGS(V,W,y));
  CHKERRQ(BVRestoreColumn(V,j,&y));
  CHKERRQ(BVGetColumn(W,j,&z));
  CHKERRQ(BVBiorthogonalizeGS(W,V,z));
  CHKERRQ(BVRestoreColumn(W,j,&z));
  V->k = ksavev;
  V->l = lsavev;
  W->k = ksavew;
  W->l = lsavew;
  CHKERRQ(PetscLogEventEnd(BV_OrthogonalizeVec,V,0,0,0));
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  CHKERRQ(PetscObjectStateIncrease((PetscObject)W));
  PetscFunctionReturn(0);
}

/*@
   BVBiorthonormalizeColumn - Bi-orthonormalize a column of two BV objects.

   Collective on V

   Input Parameters:
+  V - first basis vectors context
.  W - second basis vectors context
-  j - index of column to be bi-orthonormalized

   Output Parameters:
.  delta - (optional) value used for normalization

   Notes:
   This function first bi-orthogonalizes vectors V[j],W[j] against W[0..j-1],
   and V[0..j-1], respectively. Then, it scales the vectors with 1/delta, so
   that the resulting vectors satisfy W[j]'*V[j] = 1.

   Level: advanced

.seealso: BVOrthonormalizeColumn(), BVBiorthogonalizeColumn()
@*/
PetscErrorCode BVBiorthonormalizeColumn(BV V,BV W,PetscInt j,PetscReal *delta)
{
  PetscScalar    alpha;
  PetscReal      deltat;
  PetscInt       ksavev,lsavev,ksavew,lsavew;
  Vec            y,z;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidHeaderSpecific(W,BV_CLASSID,2);
  PetscValidLogicalCollectiveInt(V,j,3);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidType(W,2);
  BVCheckSizes(W,2);
  PetscCheckSameTypeAndComm(V,1,W,2);
  PetscCheck(j>=0,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  PetscCheck(j<V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%" PetscInt_FMT " but V only has %" PetscInt_FMT " columns",j,V->m);
  PetscCheck(j<W->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%" PetscInt_FMT " but W only has %" PetscInt_FMT " columns",j,W->m);
  PetscCheck(V->n==W->n,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Mismatching local dimension V %" PetscInt_FMT ", W %" PetscInt_FMT,V->n,W->n);
  PetscCheck(!V->matrix && !W->matrix,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_WRONGSTATE,"V,W must not have an inner product matrix");
  PetscCheck(!V->nc && !W->nc,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_WRONGSTATE,"V,W cannot have different number of constraints");
  PetscCheck(!V->ops->gramschmidt && !W->ops->gramschmidt,PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Object has a special GS function");

  /* bi-orthogonalize */
  CHKERRQ(PetscLogEventBegin(BV_OrthogonalizeVec,V,0,0,0));
  ksavev = V->k;
  lsavev = V->l;
  ksavew = W->k;
  lsavew = W->l;
  V->k = j;
  V->l = -V->nc;  /* must also bi-orthogonalize against constraints and leading columns */
  W->k = j;
  W->l = -W->nc;
  CHKERRQ(BV_AllocateCoeffs(V));
  CHKERRQ(BV_AllocateCoeffs(W));
  CHKERRQ(BVGetColumn(V,j,&y));
  CHKERRQ(BVBiorthogonalizeGS(V,W,y));
  CHKERRQ(BVRestoreColumn(V,j,&y));
  CHKERRQ(BVGetColumn(W,j,&z));
  CHKERRQ(BVBiorthogonalizeGS(W,V,z));
  CHKERRQ(BVRestoreColumn(W,j,&z));
  V->k = ksavev;
  V->l = lsavev;
  W->k = ksavew;
  W->l = lsavew;
  CHKERRQ(PetscLogEventEnd(BV_OrthogonalizeVec,V,0,0,0));

  /* scale */
  CHKERRQ(PetscLogEventBegin(BV_Scale,V,0,0,0));
  CHKERRQ(BVGetColumn(V,j,&y));
  CHKERRQ(BVGetColumn(W,j,&z));
  CHKERRQ(VecDot(z,y,&alpha));
  CHKERRQ(BVRestoreColumn(V,j,&y));
  CHKERRQ(BVRestoreColumn(W,j,&z));
  deltat = PetscSqrtReal(PetscAbsScalar(alpha));
  if (V->n) CHKERRQ((*V->ops->scale)(V,j,1.0/PetscConj(alpha/deltat)));
  if (W->n) CHKERRQ((*W->ops->scale)(W,j,1.0/deltat));
  CHKERRQ(PetscLogEventEnd(BV_Scale,V,0,0,0));
  if (delta) *delta = deltat;
  CHKERRQ(PetscObjectStateIncrease((PetscObject)V));
  CHKERRQ(PetscObjectStateIncrease((PetscObject)W));
  PetscFunctionReturn(0);
}
