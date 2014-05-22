/*
   BV orthogonalization routines.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/bvimpl.h>          /*I   "slepcbv.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeMGS1"
/*
   BVOrthogonalizeMGS1 - Compute one step of Modified Gram-Schmidt
*/
static PetscErrorCode BVOrthogonalizeMGS1(BV bv,PetscInt j,PetscScalar *H)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    dot;
  Vec            v,vi,z;

  PetscFunctionBegin;
  ierr = BVGetColumn(bv,j,&v);CHKERRQ(ierr);
  z = v;
  for (i=-bv->nc;i<j;i++) {
    ierr = BVGetColumn(bv,i,&vi);CHKERRQ(ierr);
    /* h_i = ( v, v_i ) */
    if (bv->matrix) {
      ierr = BV_IPMatMult(bv,v);CHKERRQ(ierr);
      z = bv->Bx;
    }
    ierr = VecDot(z,vi,&dot);CHKERRQ(ierr);
    /* v <- v - h_i v_i */
    if (bv->indef) dot /= bv->omega[bv->nc+i];
    ierr = VecAXPY(v,-dot,vi);CHKERRQ(ierr);
    if (bv->indef) dot *= bv->omega[bv->nc+i];
    if (H) H[bv->nc+i] += dot;
    ierr = BVRestoreColumn(bv,i,&vi);CHKERRQ(ierr);
  }
  ierr = BVRestoreColumn(bv,j,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeCGS1"
/*
   BVOrthogonalizeCGS1 - Compute |v'| (estimated), |v| and one step of CGS with
   only one global synchronization
*/
PetscErrorCode BVOrthogonalizeCGS1(BV bv,PetscInt j,PetscScalar *H,PetscReal *onorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      sum,nrm;
  Vec            v;

  PetscFunctionBegin;
  /* h = W^* v ; alpha = (v, v) */
  bv->k = j;
  if (onorm || norm) {
    bv->k++;
    ierr = BVGetColumn(bv,j,&v);CHKERRQ(ierr);
    ierr = BVDotVec(bv,v,H);CHKERRQ(ierr);
    ierr = BVRestoreColumn(bv,j,&v);CHKERRQ(ierr);
    bv->k--;
  } else {
    ierr = BVDotColumn(bv,j,H);CHKERRQ(ierr);
  }

  /* q = v - V h */
  if (bv->indef) {
    for (i=0;i<bv->nc+j;i++) H[i] /= bv->omega[i];  /* apply inverse of signature */
  }
  ierr = BVMultColumn(bv,-1.0,1.0,j,H);CHKERRQ(ierr);
  if (bv->indef) {
    for (i=0;i<bv->nc+j;i++) H[i] *= bv->omega[i];  /* revert signature */
  }

  /* compute |v| */
  if (onorm) *onorm = PetscSqrtReal(PetscRealPart(H[bv->nc+j]));

  if (bv->indef) {
    ierr = BVNormColumn(bv,j,NORM_2,&nrm);CHKERRQ(ierr);
    if (norm) *norm = nrm;
    bv->omega[bv->nc+j] = PetscSign(nrm);
  } else if (norm) {
    /* estimate |v'| from |v| */
    sum = 0.0;
    for (i=0;i<bv->nc+j;i++) sum += PetscRealPart(H[i]*PetscConj(H[i]));
    *norm = PetscRealPart(H[bv->nc+j])-sum;
    if (*norm <= 0.0) {
      ierr = BVNormColumn(bv,j,NORM_2,norm);CHKERRQ(ierr);
    } else *norm = PetscSqrtReal(*norm);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeMGS"
/*
  BVOrthogonalizeMGS - Orthogonalize with modified Gram-Schmidt
*/
static PetscErrorCode BVOrthogonalizeMGS(BV bv,PetscInt j,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscReal      onrm,nrm;
  PetscInt       k;

  PetscFunctionBegin;
  ierr = PetscMemzero(bv->h,(bv->nc+j)*sizeof(PetscScalar));CHKERRQ(ierr);
  switch (bv->orthog_ref) {

  case BV_ORTHOG_REFINE_IFNEEDED:
    /* first step */
    ierr = BVNormColumn(bv,j,NORM_2,&onrm);CHKERRQ(ierr);
    ierr = BVOrthogonalizeMGS1(bv,j,bv->h);CHKERRQ(ierr);
    ierr = BVNormColumn(bv,j,NORM_2,&nrm);CHKERRQ(ierr);
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && nrm && nrm < bv->orthog_eta*onrm) {
      k++;
      onrm = nrm;
      ierr = BVOrthogonalizeMGS1(bv,j,bv->c);CHKERRQ(ierr);
      ierr = BVNormColumn(bv,j,NORM_2,&nrm);CHKERRQ(ierr);
    }
    if (lindep) {
      if (nrm < bv->orthog_eta*onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    }
    break;

  case BV_ORTHOG_REFINE_NEVER:
    ierr = BVOrthogonalizeMGS1(bv,j,bv->h);CHKERRQ(ierr);
    /* compute |v| */
    if (norm || lindep) {
      ierr = BVNormColumn(bv,j,NORM_2,&nrm);CHKERRQ(ierr);
    }
    /* linear dependence check: just test for exactly zero norm */
    if (lindep) *lindep = nrm? PETSC_FALSE: PETSC_TRUE;
    break;

  case BV_ORTHOG_REFINE_ALWAYS:
    /* first step */
    ierr = BVOrthogonalizeMGS1(bv,j,bv->h);CHKERRQ(ierr);
    if (lindep) {
      ierr = BVNormColumn(bv,j,NORM_2,&onrm);CHKERRQ(ierr);
    }
    /* second step */
    ierr = BVOrthogonalizeMGS1(bv,j,bv->h);CHKERRQ(ierr);
    if (norm || lindep) {
      ierr = BVNormColumn(bv,j,NORM_2,&nrm);CHKERRQ(ierr);
    }
    if (lindep) {
      if (nrm==0.0 || nrm < bv->orthog_eta*onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    }
    break;
  }
  if (bv->indef) {
    ierr = BVNormColumn(bv,j,NORM_2,&nrm);CHKERRQ(ierr);
    bv->omega[bv->nc+j] = PetscSign(nrm);
  }
  if (norm) *norm = nrm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeCGS"
/*
  BVOrthogonalizeCGS - Orthogonalize with classical Gram-Schmidt
*/
static PetscErrorCode BVOrthogonalizeCGS(BV bv,PetscInt j,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscReal      onrm,nrm;
  PetscInt       i,k;

  PetscFunctionBegin;
  switch (bv->orthog_ref) {

  case BV_ORTHOG_REFINE_IFNEEDED:
    ierr = BVOrthogonalizeCGS1(bv,j,bv->h,&onrm,&nrm);CHKERRQ(ierr);
    /* ||q|| < eta ||h|| */
    k = 1;
    while (k<3 && nrm && nrm < bv->orthog_eta*onrm) {
      k++;
      ierr = BVOrthogonalizeCGS1(bv,j,bv->c,&onrm,&nrm);CHKERRQ(ierr);
      for (i=0;i<bv->nc+j;i++) bv->h[i] += bv->c[i];
    }
    if (norm) *norm = nrm;
    if (lindep) {
      if (nrm < bv->orthog_eta*onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    }
    break;

  case BV_ORTHOG_REFINE_NEVER:
    ierr = BVOrthogonalizeCGS1(bv,j,bv->h,NULL,NULL);CHKERRQ(ierr);
    /* compute |v| */
    if (norm || lindep) {
      ierr = BVNormColumn(bv,j,NORM_2,&nrm);CHKERRQ(ierr);
    }
    if (norm) *norm = nrm;
    /* linear dependence check: just test for exactly zero norm */
    if (lindep) *lindep = nrm? PETSC_FALSE: PETSC_TRUE;
    break;

  case BV_ORTHOG_REFINE_ALWAYS:
    ierr = BVOrthogonalizeCGS1(bv,j,bv->h,NULL,NULL);CHKERRQ(ierr);
    if (lindep) {
      ierr = BVOrthogonalizeCGS1(bv,j,bv->c,&onrm,&nrm);CHKERRQ(ierr);
      if (norm) *norm = nrm;
      if (nrm==0.0 || nrm < bv->orthog_eta*onrm) *lindep = PETSC_TRUE;
      else *lindep = PETSC_FALSE;
    } else {
      ierr = BVOrthogonalizeCGS1(bv,j,bv->c,NULL,norm);CHKERRQ(ierr);
    }
    for (i=0;i<bv->nc+j;i++) bv->h[i] += bv->c[i];
    break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalize"
/*@
   BVOrthogonalize - Orthogonalize one of the column vectors with respect to
   the previous ones.

   Collective on BV

   Input Parameters:
+  bv     - the basis vectors context
-  j      - index of column to be orthogonalized

   Output Parameters:
+  H      - (optional) coefficients computed during orthogonalization
.  norm   - (optional) norm of the vector after being orthogonalized
-  lindep - (optional) flag indicating that refinement did not improve the quality
            of orthogonalization

   Notes:
   This function applies an orthogonal projector to project vector V[j] onto
   the orthogonal complement of the span of the columns of V[0..j-1],
   where V[.] are the vectors of BV. The columns V[0..j-1] are assumed to be
   mutually orthonormal.

   Leading columns V[0..l-1] also participate in the orthogonalization.

   If a non-standard inner product has been specified with BVSetMatrix(),
   then the vector is B-orthogonalized, using the non-standard inner product
   defined by matrix B. The output vector satisfies V[j]'*B*V[0..j-1] = 0.

   This routine does not normalize the resulting vector.

   Level: advanced

.seealso: BVSetOrthogonalization(), BVSetMatrix(), BVSetActiveColumns()
@*/
PetscErrorCode BVOrthogonalize(BV bv,PetscInt j,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscInt       i,ksave,lsave;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  if (j<0) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  if (j>=bv->m) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%D but BV only has %D columns",j,bv->m);

  ierr = PetscLogEventBegin(BV_Orthogonalize,bv,0,0,0);CHKERRQ(ierr);
  ksave = bv->k;
  lsave = bv->l;
  bv->l = -bv->nc;  /* must also orthogonalize against constraints and leading columns */
  if (!bv->h) {
    ierr = PetscMalloc2(bv->nc+bv->m,&bv->h,bv->nc+bv->m,&bv->c);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)bv,2*bv->m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (bv->indef && !bv->omega) {
    ierr = PetscMalloc1(bv->nc+bv->m,&bv->omega);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)bv,bv->m*sizeof(PetscReal));CHKERRQ(ierr);
  }
  switch (bv->orthog_type) {
  case BV_ORTHOG_CGS:
    ierr = BVOrthogonalizeCGS(bv,j,H,norm,lindep);CHKERRQ(ierr);
    break;
  case BV_ORTHOG_MGS:
    ierr = BVOrthogonalizeMGS(bv,j,H,norm,lindep);CHKERRQ(ierr);
    break;
  }
  bv->k = ksave;
  bv->l = lsave;
  if (H) for (i=bv->l;i<j;i++) H[i-bv->l] = bv->h[bv->nc+i];
  ierr = PetscLogEventEnd(BV_Orthogonalize,bv,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeAll"
/*@
   BVOrthogonalizeAll - Orthogonalize all columns, that is, compute the
   QR decomposition.

   Collective on BV

   Input Parameter:
.  V - basis vectors

   Output Parameters:
+  V - the modified basis vectors
-  R - a sequential dense matrix (or NULL)

   Notes:
   On input, matrix R must be a sequential dense Mat, with number of rows and
   columns equal to the number of active columns of V. The output satisfies
   V0 = V*R (where V0 represent the input V) and V'*V = I.

   Can pass NULL if R is not required.

   Level: intermediate

.seealso: BVOrthogonalize(), BVSetActiveColumns()
@*/
PetscErrorCode BVOrthogonalizeAll(BV V,Mat R)
{
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  if (R) {
    PetscValidHeaderSpecific(R,MAT_CLASSID,2);
    PetscValidType(R,2);
    ierr = PetscObjectTypeCompare((PetscObject)R,MATSEQDENSE,&match);CHKERRQ(ierr);
    if (!match) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Mat argument must be of type seqdense");
    ierr = MatGetSize(R,&m,&n);CHKERRQ(ierr);
    if (m!=n) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument is not square, it has %D rows and %D columns",m,n);
    if (n!=V->k) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat size %D does not match the number of BV active columns %D",n,V->k);
  }
  if (V->matrix) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Not implemented for non-standard inner product, use BVOrthogonalize() instead");
  if (V->nc) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Not implemented for BV with constraints, use BVOrthogonalize() instead");

  ierr = PetscLogEventBegin(BV_Orthogonalize,V,R,0,0);CHKERRQ(ierr);
  ierr = (*V->ops->orthogonalize)(V,R);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Orthogonalize,V,R,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

