/*
   BV orthogonalization routines.

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

#include <slepc/private/bvimpl.h>          /*I   "slepcbv.h"   I*/
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "BV_CleanCoefficients"
/*
   BV_CleanCoefficients - Sets to zero all entries of column j of the bv buffer
*/
PETSC_STATIC_INLINE PetscErrorCode BV_CleanCoefficients(BV bv,PetscInt j,PetscScalar *h)
{
  PetscErrorCode ierr;
  PetscScalar    *hh=h,*a;
  PetscInt       i;

  PetscFunctionBegin;
  if (!h) {
    if (!bv->buffer) { ierr = BVGetBufferVec(bv,&bv->buffer);CHKERRQ(ierr); }
    ierr = VecGetArray(bv->buffer,&a);CHKERRQ(ierr);
    hh = a + j*(bv->nc+bv->m);
  }
  for (i=0;i<bv->nc+j;i++) hh[i] = 0.0;
  if (!h) {
    ierr = VecRestoreArray(bv->buffer,&a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BV_AddCoefficients"
/*
   BV_AddCoefficients - Add the contents of the scratch (0-th column) of the bv buffer
   into column j of the bv buffer
*/
PETSC_STATIC_INLINE PetscErrorCode BV_AddCoefficients(BV bv,PetscInt j,PetscScalar *h,PetscScalar *c)
{
  PetscErrorCode ierr;
  PetscScalar    *hh=h,*cc=c;
  PetscInt       i;

  PetscFunctionBegin;
  if (!h) {
    if (!bv->buffer) { ierr = BVGetBufferVec(bv,&bv->buffer);CHKERRQ(ierr); }
    ierr = VecGetArray(bv->buffer,&cc);CHKERRQ(ierr);
    hh = cc + j*(bv->nc+bv->m);
  }
  for (i=0;i<bv->nc+j;i++) hh[i] += cc[i];
  if (!h) {
    ierr = VecRestoreArray(bv->buffer,&cc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BV_SetValue"
/*
   BV_SetValue - Sets value in position j (counted after the constraints)
   of the coefficients array
*/
PETSC_STATIC_INLINE PetscErrorCode BV_SetValue(BV bv,PetscInt j,PetscScalar *h,PetscScalar value)
{
  PetscErrorCode ierr;
  PetscScalar    *hh=h;

  PetscFunctionBegin;
  if (!h) {
    if (!bv->buffer) { ierr = BVGetBufferVec(bv,&bv->buffer);CHKERRQ(ierr); }
    ierr = VecGetArray(bv->buffer,&hh);CHKERRQ(ierr);
  }
  hh[bv->nc+j] = value;
  if (!h) {
    ierr = VecRestoreArray(bv->buffer,&hh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BV_SquareSum"
/*
   BV_SquareSum - Returns the value h'*h, where h represents the contents of the
   coefficients array (up to position j)
*/
PETSC_STATIC_INLINE PetscErrorCode BV_SquareSum(BV bv,PetscInt j,PetscScalar *h,PetscReal *sum)
{
  PetscErrorCode ierr;
  PetscScalar    *hh=h;
  PetscInt       i;

  PetscFunctionBegin;
  if (!h) {
    if (!bv->buffer) { ierr = BVGetBufferVec(bv,&bv->buffer);CHKERRQ(ierr); }
    ierr = VecGetArray(bv->buffer,&hh);CHKERRQ(ierr);
  }
  *sum = 0.0;
  for (i=0;i<bv->nc+j;i++) *sum += PetscRealPart(hh[i]*PetscConj(hh[i]));
  if (!h) {
    ierr = VecRestoreArray(bv->buffer,&hh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BV_ApplySignature"
/*
   BV_ApplySignature - Computes the pointwise product h*omega, where h represents
   the contents of the coefficients array (up to position j) and omega is the signature;
   if inverse=TRUE then the operation is h/omega
*/
PETSC_STATIC_INLINE PetscErrorCode BV_ApplySignature(BV bv,PetscInt j,PetscScalar *h,PetscBool inverse)
{
  PetscErrorCode ierr;
  PetscScalar    *hh=h;
  PetscInt       i;

  PetscFunctionBegin;
  if (!h) {
    if (!bv->buffer) { ierr = BVGetBufferVec(bv,&bv->buffer);CHKERRQ(ierr); }
    ierr = VecGetArray(bv->buffer,&hh);CHKERRQ(ierr);
  }
  if (inverse) for (i=0;i<bv->nc+j;i++) hh[i] /= bv->omega[i];
  else for (i=0;i<bv->nc+j;i++) hh[i] *= bv->omega[i];
  if (!h) {
    ierr = VecRestoreArray(bv->buffer,&hh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BV_SquareRoot"
/*
   BV_SquareRoot - Returns the square root of position j (counted after the constraints)
   of the coefficients array
*/
PETSC_STATIC_INLINE PetscErrorCode BV_SquareRoot(BV bv,PetscInt j,PetscScalar *h,PetscReal *beta)
{
  PetscErrorCode ierr;
  PetscScalar    *hh=h;

  PetscFunctionBegin;
  if (!h) {
    if (!bv->buffer) { ierr = BVGetBufferVec(bv,&bv->buffer);CHKERRQ(ierr); }
    ierr = VecGetArray(bv->buffer,&hh);CHKERRQ(ierr);
  }
  ierr = BV_SafeSqrt(bv,hh[bv->nc+j],beta);CHKERRQ(ierr);
  if (!h) {
    ierr = VecRestoreArray(bv->buffer,&hh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BV_NormVecOrColumn"
/*
   BV_NormVecOrColumn - Compute the 2-norm of the working vector, irrespective of
   whether it is in a column or not
*/
PETSC_STATIC_INLINE PetscErrorCode BV_NormVecOrColumn(BV bv,PetscInt j,Vec v,PetscReal *nrm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (v) {
    ierr = BVNormVec(bv,v,NORM_2,nrm);CHKERRQ(ierr);
  } else {
    ierr = BVNormColumn(bv,j,NORM_2,nrm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotColumnInc"
/*
   BVDotColumnInc - Same as BVDotColumn() but also including column j, which
   is multiplied by itself
*/
PETSC_STATIC_INLINE PetscErrorCode BVDotColumnInc(BV X,PetscInt j,PetscScalar *q)
{
  PetscErrorCode ierr;
  PetscInt       ksave;
  Vec            y;
  PetscScalar    *a=q;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(BV_DotVec,X,0,0,0);CHKERRQ(ierr);
  ksave = X->k;
  X->k = j+1;
  if (!q) {
    if (!X->buffer) { ierr = BVGetBufferVec(X,&X->buffer);CHKERRQ(ierr); }
    ierr = VecGetArray(X->buffer,&a);CHKERRQ(ierr);
  }
  ierr = BVGetColumn(X,j,&y);CHKERRQ(ierr);
  ierr = (*X->ops->dotvec)(X,y,a);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,j,&y);CHKERRQ(ierr);
  if (!q) { ierr = VecRestoreArray(X->buffer,&a);CHKERRQ(ierr); }
  X->k = ksave;
  ierr = PetscLogEventEnd(BV_DotVec,X,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeMGS1"
/*
   BVOrthogonalizeMGS1 - Compute one step of Modified Gram-Schmidt
*/
static PetscErrorCode BVOrthogonalizeMGS1(BV bv,PetscInt j,Vec v,PetscBool *which,PetscScalar *h,PetscScalar *c,PetscReal *onrm,PetscReal *nrm)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    dot;
  Vec            vi,z,w=v;

  PetscFunctionBegin;
  if (!v) { ierr = BVGetColumn(bv,j,&w);CHKERRQ(ierr); }
  if (onrm) { ierr = BVNormVec(bv,w,NORM_2,onrm);CHKERRQ(ierr); }
  z = w;
  for (i=-bv->nc;i<j;i++) {
    if (which && i>=0 && !which[i]) continue;
    ierr = BVGetColumn(bv,i,&vi);CHKERRQ(ierr);
    /* h_i = ( v, v_i ) */
    if (bv->matrix) {
      ierr = BV_IPMatMult(bv,w);CHKERRQ(ierr);
      z = bv->Bx;
    }
    ierr = VecDot(z,vi,&dot);CHKERRQ(ierr);
    /* v <- v - h_i v_i */
    ierr = BV_SetValue(bv,i,c,dot);CHKERRQ(ierr);
    if (bv->indef) dot /= bv->omega[bv->nc+i];
    ierr = VecAXPY(w,-dot,vi);CHKERRQ(ierr);
    ierr = BVRestoreColumn(bv,i,&vi);CHKERRQ(ierr);
  }
  if (nrm) { ierr = BVNormVec(bv,w,NORM_2,nrm);CHKERRQ(ierr); }
  if (!v) { ierr = BVRestoreColumn(bv,j,&w);CHKERRQ(ierr); }
  ierr = BV_AddCoefficients(bv,j,h,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeCGS1"
/*
   BVOrthogonalizeCGS1 - Compute |v'| (estimated), |v| and one step of CGS with
   only one global synchronization
*/
static PetscErrorCode BVOrthogonalizeCGS1(BV bv,PetscInt j,Vec v,PetscBool *which,PetscScalar *h,PetscScalar *c,PetscReal *onorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscReal      sum,beta;
  Vec            w=v;

  PetscFunctionBegin;
  /* h = W^* v ; alpha = (v, v) */
  bv->k = j;
  if (onorm || norm) {
    if (!v) {
      ierr = BVDotColumnInc(bv,j,c);CHKERRQ(ierr);
      ierr = BV_SquareRoot(bv,j,c,&beta);CHKERRQ(ierr);
    } else {
      ierr = BVDotVec(bv,w,c);CHKERRQ(ierr);
      ierr = BVNormVec(bv,w,NORM_2,&beta);CHKERRQ(ierr);
    }
  } else {
    if (!v) { ierr = BVDotColumn(bv,j,c);CHKERRQ(ierr); }
    else { ierr = BVDotVec(bv,w,c);CHKERRQ(ierr); }
  }

  /* q = v - V h */
  if (bv->indef) { ierr = BV_ApplySignature(bv,j,c,PETSC_TRUE);CHKERRQ(ierr); }
  if (!v) { ierr = BVMultColumn(bv,-1.0,1.0,j,c);CHKERRQ(ierr); }
  else { ierr = BVMultVec(bv,-1.0,1.0,w,c);CHKERRQ(ierr); }
  if (bv->indef) { ierr = BV_ApplySignature(bv,j,c,PETSC_FALSE);CHKERRQ(ierr); }

  /* compute |v| */
  if (onorm) *onorm = beta;

  if (norm) {
    if (bv->indef) {
      ierr = BV_NormVecOrColumn(bv,j,w,norm);CHKERRQ(ierr);
    } else {
      /* estimate |v'| from |v| */
      ierr = BV_SquareSum(bv,j,c,&sum);CHKERRQ(ierr);
      *norm = beta*beta-sum;
      if (*norm <= 0.0) {
        ierr = BV_NormVecOrColumn(bv,j,w,norm);CHKERRQ(ierr);
      } else *norm = PetscSqrtReal(*norm);
    }
  }
  ierr = BV_AddCoefficients(bv,j,h,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BV_StoreCoefficients"
/*
   BV_StoreCoefficients - copy the contents of the coefficients array to an array
   provided by the caller (only values from l to j are copied)
*/
PETSC_STATIC_INLINE PetscErrorCode BV_StoreCoefficients(BV bv,PetscInt j,PetscScalar *h)
{
  PetscErrorCode ierr;
  PetscScalar    *hh,*cc;
  PetscInt       i;

  PetscFunctionBegin;
  if (!bv->buffer) { ierr = BVGetBufferVec(bv,&bv->buffer);CHKERRQ(ierr); }
  ierr = VecGetArray(bv->buffer,&cc);CHKERRQ(ierr);
  hh = cc + j*(bv->nc+bv->m);
  for (i=bv->l;i<j;i++) h[i-bv->l] = hh[bv->nc+i];
  ierr = VecRestoreArray(bv->buffer,&cc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define BVOrthogonalizeGS1(a,b,c,d,e,f,g,h) (mgs?BVOrthogonalizeMGS1:BVOrthogonalizeCGS1)(a,b,c,d,e,f,g,h)

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeGS"
/*
   BVOrthogonalizeGS - Orthogonalize with (classical or modified) Gram-Schmidt

   j      - the index of the column to orthogonalize (cannot use both j and v)
   v      - the vector to orthogonalize (cannot use both j and v)
   which  - logical array indicating selected columns (only used in MGS)
   norm   - (optional) norm of the vector after being orthogonalized
   lindep - (optional) flag indicating possible linear dependence
*/
static PetscErrorCode BVOrthogonalizeGS(BV bv,PetscInt j,Vec v,PetscBool *which,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscScalar    *h,*c;
  PetscReal      onrm,nrm;
  PetscInt       k,l;
  PetscBool      mgs,dolindep,signature;

  PetscFunctionBegin;
  if (v) {
    k = bv->k;
    h = bv->h;
    c = bv->c;
  } else {
    k = j;
    h = NULL;
    c = NULL;
  }

  mgs = (bv->orthog_type==BV_ORTHOG_MGS)? PETSC_TRUE: PETSC_FALSE;

  /* if indefinite inner product, skip the computation of lindep */
  if (bv->indef && lindep) *lindep = PETSC_FALSE;
  dolindep = (!bv->indef && lindep)? PETSC_TRUE: PETSC_FALSE;

  /* if indefinite and we are orthogonalizing a column, the norm must always be computed */
  signature = (bv->indef && !v)? PETSC_TRUE: PETSC_FALSE;

  ierr = BV_CleanCoefficients(bv,k,h);CHKERRQ(ierr);

  switch (bv->orthog_ref) {

  case BV_ORTHOG_REFINE_IFNEEDED:
    ierr = BVOrthogonalizeGS1(bv,k,v,which,h,c,&onrm,&nrm);CHKERRQ(ierr);
    /* repeat if ||q|| < eta ||h|| */
    l = 1;
    while (l<3 && nrm && nrm < bv->orthog_eta*onrm) {
      l++;
      if (mgs) onrm = nrm;
      ierr = BVOrthogonalizeGS1(bv,k,v,which,h,c,mgs?NULL:&onrm,&nrm);CHKERRQ(ierr);
    }
    /* linear dependence check: criterion not satisfied in the last iteration */
    if (dolindep) *lindep = PetscNot(nrm && nrm >= bv->orthog_eta*onrm);
    break;

  case BV_ORTHOG_REFINE_NEVER:
    ierr = BVOrthogonalizeGS1(bv,k,v,which,h,c,NULL,NULL);CHKERRQ(ierr);
    /* compute ||v|| */
    if (norm || dolindep || signature) {
      ierr = BV_NormVecOrColumn(bv,k,v,&nrm);CHKERRQ(ierr);
    }
    /* linear dependence check: just test for exactly zero norm */
    if (dolindep) *lindep = PetscNot(nrm);
    break;

  case BV_ORTHOG_REFINE_ALWAYS:
    ierr = BVOrthogonalizeGS1(bv,k,v,which,h,c,NULL,NULL);CHKERRQ(ierr);
    ierr = BVOrthogonalizeGS1(bv,k,v,which,h,c,dolindep?&onrm:NULL,(norm||dolindep||signature)?&nrm:NULL);CHKERRQ(ierr);
    /* linear dependence check: criterion not satisfied in the second iteration */
    if (dolindep) *lindep = PetscNot(nrm && nrm >= bv->orthog_eta*onrm);
    break;
  }
  if (signature) bv->omega[bv->nc+k] = (nrm<0.0)? -1.0: 1.0;
  if (norm) *norm = nrm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeVec"
/*@
   BVOrthogonalizeVec - Orthogonalize a given vector with respect to all
   active columns.

   Collective on BV

   Input Parameters:
+  bv     - the basis vectors context
-  v      - the vector

   Output Parameters:
+  H      - (optional) coefficients computed during orthogonalization
.  norm   - (optional) norm of the vector after being orthogonalized
-  lindep - (optional) flag indicating that refinement did not improve the quality
            of orthogonalization

   Notes:
   This function is equivalent to BVOrthogonalizeColumn() but orthogonalizes
   a vector as an argument rather than taking one of the BV columns. The
   vector is orthogonalized against all active columns (k) and the constraints.
   If H is given, it must have enough space to store k-l coefficients, where l
   is the number of leading columns.

   In the case of an indefinite inner product, the lindep parameter is not
   computed (set to false).

   Level: advanced

.seealso: BVOrthogonalizeColumn(), BVSetOrthogonalization(), BVSetActiveColumns(), BVGetNumConstraints()
@*/
PetscErrorCode BVOrthogonalizeVec(BV bv,Vec v,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscInt       i,ksave,lsave;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  PetscValidType(v,2);
  PetscCheckSameComm(bv,1,v,2);

  ierr = PetscLogEventBegin(BV_OrthogonalizeVec,bv,0,0,0);CHKERRQ(ierr);
  ksave = bv->k;
  lsave = bv->l;
  bv->l = -bv->nc;  /* must also orthogonalize against constraints and leading columns */
  ierr = BV_AllocateCoeffs(bv);CHKERRQ(ierr);
  ierr = BV_AllocateSignature(bv);CHKERRQ(ierr);
  ierr = BVOrthogonalizeGS(bv,0,v,NULL,norm,lindep);CHKERRQ(ierr);
  bv->k = ksave;
  bv->l = lsave;
  if (H) for (i=bv->l;i<bv->k;i++) H[i-bv->l] = bv->h[bv->nc+i];
  ierr = PetscLogEventEnd(BV_OrthogonalizeVec,bv,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeColumn"
/*@
   BVOrthogonalizeColumn - Orthogonalize one of the column vectors with respect to
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

   Leading columns V[0..l-1] also participate in the orthogonalization, as well
   as the constraints. If H is given, it must have enough space to
   store j-l coefficients.

   If a non-standard inner product has been specified with BVSetMatrix(),
   then the vector is B-orthogonalized, using the non-standard inner product
   defined by matrix B. The output vector satisfies V[j]'*B*V[0..j-1] = 0.

   This routine does not normalize the resulting vector.

   In the case of an indefinite inner product, the lindep parameter is not
   computed (set to false).

   Level: advanced

.seealso: BVSetOrthogonalization(), BVSetMatrix(), BVSetActiveColumns(), BVOrthogonalize(), BVOrthogonalizeVec(), BVGetNumConstraints()
@*/
PetscErrorCode BVOrthogonalizeColumn(BV bv,PetscInt j,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscInt       ksave,lsave;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  if (j<0) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  if (j>=bv->m) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%D but BV only has %D columns",j,bv->m);

  ierr = PetscLogEventBegin(BV_OrthogonalizeVec,bv,0,0,0);CHKERRQ(ierr);
  ksave = bv->k;
  lsave = bv->l;
  bv->l = -bv->nc;  /* must also orthogonalize against constraints and leading columns */
  ierr = BV_AllocateSignature(bv);CHKERRQ(ierr);
  ierr = BVOrthogonalizeGS(bv,j,NULL,NULL,norm,lindep);CHKERRQ(ierr);
  bv->k = ksave;
  bv->l = lsave;
  if (H) { ierr = BV_StoreCoefficients(bv,j,H);CHKERRQ(ierr); }
  ierr = PetscLogEventEnd(BV_OrthogonalizeVec,bv,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeSomeColumn"
/*@
   BVOrthogonalizeSomeColumn - Orthogonalize one of the column vectors with
   respect to some of the previous ones.

   Collective on BV

   Input Parameters:
+  bv     - the basis vectors context
.  j      - index of column to be orthogonalized
-  which  - logical array indicating selected columns

   Output Parameters:
+  H      - (optional) coefficients computed during orthogonalization
.  norm   - (optional) norm of the vector after being orthogonalized
-  lindep - (optional) flag indicating that refinement did not improve the quality
            of orthogonalization

   Notes:
   This function is similar to BVOrthogonalizeColumn(), but V[j] is
   orthogonalized only against columns V[i] having which[i]=PETSC_TRUE.
   The length of array which must be j at least.

   The use of this operation is restricted to MGS orthogonalization type.

   In the case of an indefinite inner product, the lindep parameter is not
   computed (set to false).

   Level: advanced

.seealso: BVOrthogonalizeColumn(), BVSetOrthogonalization()
@*/
PetscErrorCode BVOrthogonalizeSomeColumn(BV bv,PetscInt j,PetscBool *which,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscInt       ksave,lsave;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  PetscValidPointer(which,3);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  if (j<0) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  if (j>=bv->m) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%D but BV only has %D columns",j,bv->m);
  if (bv->orthog_type!=BV_ORTHOG_MGS) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Operation only available for MGS orthogonalization");

  ierr = PetscLogEventBegin(BV_OrthogonalizeVec,bv,0,0,0);CHKERRQ(ierr);
  ksave = bv->k;
  lsave = bv->l;
  bv->l = -bv->nc;  /* must also orthogonalize against constraints and leading columns */
  ierr = BV_AllocateSignature(bv);CHKERRQ(ierr);
  ierr = BVOrthogonalizeGS(bv,j,NULL,which,norm,lindep);CHKERRQ(ierr);
  bv->k = ksave;
  bv->l = lsave;
  if (H) { ierr = BV_StoreCoefficients(bv,j,H);CHKERRQ(ierr); }
  ierr = PetscLogEventEnd(BV_OrthogonalizeVec,bv,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalize_GS"
/*
   Orthogonalize a set of vectors with Gram-Schmidt, column by column.
 */
static PetscErrorCode BVOrthogonalize_GS(BV V,Mat R)
{
  PetscErrorCode ierr;
  PetscScalar    *r=NULL;
  PetscReal      norm;
  PetscInt       j,ldr;
  Vec            v,w;

  PetscFunctionBegin;
  if (R) {
    ierr = MatGetSize(R,&ldr,NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArray(R,&r);CHKERRQ(ierr);
  }
  if (V->matrix) {
    ierr = BVGetCachedBV(V,&V->cached);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(V->cached,V->l,V->k);CHKERRQ(ierr);
  }
  for (j=V->l;j<V->k;j++) {
    if (V->matrix && V->orthog_type==BV_ORTHOG_MGS) {  /* fill cached BV */
      ierr = BVGetColumn(V->cached,j,&v);CHKERRQ(ierr);
      ierr = BVGetColumn(V,j,&w);CHKERRQ(ierr);
      ierr = MatMult(V->matrix,w,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(V,j,&w);CHKERRQ(ierr);
      ierr = BVRestoreColumn(V->cached,j,&v);CHKERRQ(ierr);
    }
    if (R) {
      ierr = BVOrthogonalizeColumn(V,j,r+j*ldr+V->l,&norm,NULL);CHKERRQ(ierr);
      r[j+j*ldr] = norm;
    } else {
      ierr = BVOrthogonalizeColumn(V,j,NULL,&norm,NULL);CHKERRQ(ierr);
    }
    if (!norm) SETERRQ(PETSC_COMM_SELF,1,"Breakdown in BVOrthogonalize due to a linearly dependent column");
    if (V->matrix && V->orthog_type==BV_ORTHOG_CGS) {  /* fill cached BV */
      ierr = BVGetColumn(V->cached,j,&v);CHKERRQ(ierr);
      ierr = VecCopy(V->Bx,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(V->cached,j,&v);CHKERRQ(ierr);
    }
    ierr = BVScaleColumn(V,j,1.0/norm);CHKERRQ(ierr);
  }
  if (R) { ierr = MatDenseRestoreArray(R,&r);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorInvert"
/*
   Compute the upper Cholesky factor in R and its inverse in S.
 */
static PetscErrorCode MatCholeskyFactorInvert(Mat R,PetscInt l,Mat *S)
{
#if defined(PETSC_MISSING_LAPACK_POTRF) || defined(SLEPC_MISSING_LAPACK_TRTRI)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF/TRTRI - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,n,m,ld;
  PetscScalar    *pR,*pS;
  PetscBLASInt   info,n_,l_,m_,ld_;

  PetscFunctionBegin;
  ierr = MatGetSize(R,&m,NULL);CHKERRQ(ierr);
  n = m-l;
  ierr = PetscBLASIntCast(m,&m_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(l,&l_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ld  = m;
  ld_ = m_;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,ld,ld,NULL,S);CHKERRQ(ierr);
  ierr = MatDenseGetArray(R,&pR);CHKERRQ(ierr);
  ierr = MatDenseGetArray(*S,&pS);CHKERRQ(ierr);

  /* save a copy of matrix in S */
  for (i=l;i<m;i++) {
    ierr = PetscMemcpy(pS+i*ld+l,pR+i*ld+l,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* compute upper Cholesky factor in R */
  PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("U",&n_,pR+l*ld+l,&ld_,&info));
  ierr = PetscLogFlops((1.0*n*n*n)/3.0);CHKERRQ(ierr);

  if (info) {  /* LAPACKpotrf failed, retry on diagonally perturbed matrix */
    for (i=l;i<m;i++) {
      ierr = PetscMemcpy(pR+i*ld+l,pS+i*ld+l,n*sizeof(PetscScalar));CHKERRQ(ierr);
      pR[i+i*ld] += 50.0*PETSC_MACHINE_EPSILON;
    }
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("U",&n_,pR+l*ld+l,&ld_,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,1,"Error in Cholesky factorization, info=%D",(PetscInt)info);
    ierr = PetscLogFlops((1.0*n*n*n)/3.0);CHKERRQ(ierr);
  }

  /* compute S = inv(R) */
  ierr = PetscMemzero(pS,m*m*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=l;i<m;i++) {
    ierr = PetscMemcpy(pS+i*ld+l,pR+i*ld+l,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscStackCallBLAS("LAPACKtrtri",LAPACKtrtri_("U","N",&n_,pS+l*ld+l,&ld_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,1,"Error in xTRTRI, info=%D",(PetscInt)info);
  ierr = PetscLogFlops(1.0*n*n*n);CHKERRQ(ierr);

  /* Zero out entries below the diagonal */
  for (i=l;i<m-1;i++) {
    ierr = PetscMemzero(pR+i*ld+i+1,(m-i-1)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(pS+i*ld+i+1,(m-i-1)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(R,&pR);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(*S,&pS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalize_Chol"
/*
   Orthogonalize a set of vectors with Cholesky: R=chol(V'*V), Q=V*inv(R)
 */
static PetscErrorCode BVOrthogonalize_Chol(BV V,Mat Rin)
{
  PetscErrorCode ierr;
  Mat            S,R=Rin;

  PetscFunctionBegin;
  if (!Rin) {
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,V->k,V->k,NULL,&R);CHKERRQ(ierr);
  }
  ierr = BVDot(V,V,R);CHKERRQ(ierr);
  ierr = MatCholeskyFactorInvert(R,V->l,&S);CHKERRQ(ierr);
  ierr = BVMultInPlace(V,S,V->l,V->k);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  if (!Rin) {
    ierr = MatDestroy(&R);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalize_TSQR"
/*
   Orthogonalize a set of vectors with the Tall-Skinny QR method
 */
static PetscErrorCode BVOrthogonalize_TSQR(BV V,Mat R)
{
  PetscErrorCode ierr;
  PetscScalar    *pv,*r=NULL;

  PetscFunctionBegin;
  if (R) { ierr = MatDenseGetArray(R,&r);CHKERRQ(ierr); }
  ierr = BVGetArray(V,&pv);CHKERRQ(ierr);
  ierr = BVOrthogonalize_LAPACK_Private(V,V->n,V->k,pv+V->nc*V->n,r);CHKERRQ(ierr);
  ierr = BVRestoreArray(V,&pv);CHKERRQ(ierr);
  if (R) { ierr = MatDenseRestoreArray(R,&r);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalize"
/*@
   BVOrthogonalize - Orthogonalize all columns (except leading ones), that is,
   compute the QR decomposition.

   Collective on BV

   Input Parameter:
.  V - basis vectors

   Output Parameters:
+  V - the modified basis vectors
-  R - a sequential dense matrix (or NULL)

   Notes:
   On input, matrix R must be a sequential dense Mat, with at least as many rows
   and columns as the number of active columns of V. The output satisfies
   V0 = V*R (where V0 represent the input V) and V'*V = I.

   If V has leading columns, then they are not modified (are assumed to be already
   orthonormal) and the corresponding part of R is not referenced.

   Can pass NULL if R is not required.

   The method to be used for block orthogonalization can be set with
   BVSetOrthogonalization(). If set to GS, the computation is done column by
   column with successive calls to BVOrthogonalizeColumn().

   If V is rank-deficient or very ill-conditioned, that is, one or more columns are
   (almost) linearly dependent with respect to the rest, then the algorithm may
   break down or result in larger numerical error. Linearly dependent columns are
   essentially replaced by random directions, and the corresponding diagonal entry
   in R is set to (nearly) zero.

   Level: intermediate

.seealso: BVOrthogonalizeColumn(), BVOrthogonalizeVec(), BVSetActiveColumns(), BVSetOrthogonalization(), BVOrthogBlockType
@*/
PetscErrorCode BVOrthogonalize(BV V,Mat R)
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
    if (V->l>0 && V->orthog_block==BV_ORTHOG_BLOCK_GS) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Cannot request matrix R in Gram-Schmidt orthogonalization if l>0");
    ierr = PetscObjectTypeCompare((PetscObject)R,MATSEQDENSE,&match);CHKERRQ(ierr);
    if (!match) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Mat argument must be of type seqdense");
    ierr = MatGetSize(R,&m,&n);CHKERRQ(ierr);
    if (m!=n) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument is not square, it has %D rows and %D columns",m,n);
    if (n<V->k) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat size %D is smaller than the number of BV active columns %D",n,V->k);
  }
  if (V->nc) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Not implemented for BV with constraints, use BVOrthogonalizeColumn() instead");

  ierr = PetscLogEventBegin(BV_Orthogonalize,V,R,0,0);CHKERRQ(ierr);
  switch (V->orthog_block) {
  case BV_ORTHOG_BLOCK_GS: /* proceed column by column with Gram-Schmidt */
    ierr = BVOrthogonalize_GS(V,R);CHKERRQ(ierr);
    break;
  case BV_ORTHOG_BLOCK_CHOL:
    ierr = BVOrthogonalize_Chol(V,R);CHKERRQ(ierr);
    break;
  case BV_ORTHOG_BLOCK_TSQR:
    if (V->matrix) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Orthogonalization method not available for non-standard inner product");
    ierr = BVOrthogonalize_TSQR(V,R);CHKERRQ(ierr);
    break;
  }
  ierr = PetscLogEventEnd(BV_Orthogonalize,V,R,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

