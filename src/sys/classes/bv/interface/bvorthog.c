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
#define __FUNCT__ "BVOrthogonalizeMGS1"
/*
   BVOrthogonalizeMGS1 - Compute one step of Modified Gram-Schmidt
*/
static PetscErrorCode BVOrthogonalizeMGS1(BV bv,PetscInt k,Vec v,PetscBool *which,PetscScalar *H)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    dot;
  Vec            vi,z;

  PetscFunctionBegin;
  z = v;
  for (i=-bv->nc;i<k;i++) {
    if (which && i>=0 && !which[i]) continue;
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeCGS1"
/*
   BVOrthogonalizeCGS1 - Compute |v'| (estimated), |v| and one step of CGS with
   only one global synchronization
*/
PetscErrorCode BVOrthogonalizeCGS1(BV bv,PetscInt j,Vec v,PetscScalar *H,PetscReal *onorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      sum,nrm,beta;
  Vec            w=v;

  PetscFunctionBegin;
  /* h = W^* v ; alpha = (v, v) */
  bv->k = j;
  if (onorm || norm) {
    if (!v) {
      bv->k++;
      ierr = BVGetColumn(bv,j,&w);CHKERRQ(ierr);
    }
    ierr = BVDotVec(bv,w,H);CHKERRQ(ierr);
    if (!v) {
      ierr = BVRestoreColumn(bv,j,&w);CHKERRQ(ierr);
      bv->k--;
      ierr = BV_SafeSqrt(bv,H[bv->nc+j],&beta);CHKERRQ(ierr);
    } else {
      ierr = BVNormVec(bv,w,NORM_2,&beta);CHKERRQ(ierr);
    }
  } else {
    if (!v) { ierr = BVDotColumn(bv,j,H);CHKERRQ(ierr); }
    else { ierr = BVDotVec(bv,w,H);CHKERRQ(ierr); }
  }

  /* q = v - V h */
  if (bv->indef) {
    for (i=0;i<bv->nc+j;i++) H[i] /= bv->omega[i];  /* apply inverse of signature */
  }
  if (!v) { ierr = BVMultColumn(bv,-1.0,1.0,j,H);CHKERRQ(ierr); }
  else { ierr = BVMultVec(bv,-1.0,1.0,w,H);CHKERRQ(ierr); }
  if (bv->indef) {
    for (i=0;i<bv->nc+j;i++) H[i] *= bv->omega[i];  /* revert signature */
  }

  /* compute |v| */
  if (onorm) *onorm = beta;

  if (bv->indef) {
    if (!v) { ierr = BVNormColumn(bv,j,NORM_2,&nrm);CHKERRQ(ierr); }
    else { ierr = BVNormVec(bv,w,NORM_2,&nrm);CHKERRQ(ierr); }
    if (norm) *norm = nrm;
    bv->omega[bv->nc+j] = (nrm<0.0)? -1.0: 1.0;
  } else if (norm) {
    /* estimate |v'| from |v| */
    sum = 0.0;
    for (i=0;i<bv->nc+j;i++) sum += PetscRealPart(H[i]*PetscConj(H[i]));
    *norm = beta*beta-sum;
    if (*norm <= 0.0) {
      if (!v) { ierr = BVNormColumn(bv,j,NORM_2,norm);CHKERRQ(ierr); }
      else { ierr = BVNormVec(bv,w,NORM_2,norm);CHKERRQ(ierr); }
    } else *norm = PetscSqrtReal(*norm);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeMGS"
/*
  BVOrthogonalizeMGS - Orthogonalize with modified Gram-Schmidt
*/
static PetscErrorCode BVOrthogonalizeMGS(BV bv,PetscInt j,Vec v,PetscBool *which,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscReal      onrm,nrm;
  PetscInt       k,l;
  Vec            w;

  PetscFunctionBegin;
  if (v) {
    w = v;
    k = bv->k;
  } else {
    ierr = BVGetColumn(bv,j,&w);CHKERRQ(ierr);
    k = j;
  }
  ierr = PetscMemzero(bv->h,(bv->nc+k)*sizeof(PetscScalar));CHKERRQ(ierr);
  switch (bv->orthog_ref) {

  case BV_ORTHOG_REFINE_IFNEEDED:
    /* first step */
    ierr = BVNormVec(bv,w,NORM_2,&onrm);CHKERRQ(ierr);
    ierr = BVOrthogonalizeMGS1(bv,k,w,which,bv->h);CHKERRQ(ierr);
    ierr = BVNormVec(bv,w,NORM_2,&nrm);CHKERRQ(ierr);
    /* ||q|| < eta ||h|| */
    l = 1;
    while (l<3 && nrm && nrm < bv->orthog_eta*onrm) {
      l++;
      onrm = nrm;
      ierr = BVOrthogonalizeMGS1(bv,k,w,which,bv->c);CHKERRQ(ierr);
      ierr = BVNormVec(bv,w,NORM_2,&nrm);CHKERRQ(ierr);
    }
    if (lindep) *lindep = PetscNot(nrm >= bv->orthog_eta*onrm);
    break;

  case BV_ORTHOG_REFINE_NEVER:
    ierr = BVOrthogonalizeMGS1(bv,k,w,which,bv->h);CHKERRQ(ierr);
    /* compute |v| */
    if (norm || lindep) {
      ierr = BVNormVec(bv,w,NORM_2,&nrm);CHKERRQ(ierr);
    }
    /* linear dependence check: just test for exactly zero norm */
    if (lindep) *lindep = PetscNot(nrm);
    break;

  case BV_ORTHOG_REFINE_ALWAYS:
    /* first step */
    ierr = BVOrthogonalizeMGS1(bv,k,w,which,bv->h);CHKERRQ(ierr);
    if (lindep) {
      ierr = BVNormVec(bv,w,NORM_2,&onrm);CHKERRQ(ierr);
    }
    /* second step */
    ierr = BVOrthogonalizeMGS1(bv,k,w,which,bv->h);CHKERRQ(ierr);
    if (norm || lindep) {
      ierr = BVNormVec(bv,w,NORM_2,&nrm);CHKERRQ(ierr);
    }
    if (lindep) *lindep = PetscNot(nrm && nrm >= bv->orthog_eta*onrm);
    break;
  }
  if (bv->indef) {
    ierr = BVNormVec(bv,w,NORM_2,&nrm);CHKERRQ(ierr);
    bv->omega[bv->nc+j] = (nrm<0.0)? -1.0: 1.0;
  }
  if (!v) { ierr = BVRestoreColumn(bv,j,&w);CHKERRQ(ierr); }
  if (norm) *norm = nrm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeCGS"
/*
  BVOrthogonalizeCGS - Orthogonalize with classical Gram-Schmidt
*/
static PetscErrorCode BVOrthogonalizeCGS(BV bv,PetscInt j,Vec v,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscReal      onrm,nrm;
  PetscInt       i,k,l;

  PetscFunctionBegin;
  if (v) k = bv->k;
  else k = j;
  switch (bv->orthog_ref) {

  case BV_ORTHOG_REFINE_IFNEEDED:
    ierr = BVOrthogonalizeCGS1(bv,k,v,bv->h,&onrm,&nrm);CHKERRQ(ierr);
    /* ||q|| < eta ||h|| */
    l = 1;
    while (l<3 && nrm && nrm < bv->orthog_eta*onrm) {
      l++;
      ierr = BVOrthogonalizeCGS1(bv,k,v,bv->c,&onrm,&nrm);CHKERRQ(ierr);
      for (i=0;i<bv->nc+k;i++) bv->h[i] += bv->c[i];
    }
    if (norm) *norm = nrm;
    if (lindep) *lindep = PetscNot(nrm >= bv->orthog_eta*onrm);
    break;

  case BV_ORTHOG_REFINE_NEVER:
    ierr = BVOrthogonalizeCGS1(bv,k,v,bv->h,NULL,NULL);CHKERRQ(ierr);
    /* compute |v| */
    if (norm || lindep) {
      if (v) { ierr = BVNormVec(bv,v,NORM_2,&nrm);CHKERRQ(ierr); }
      else { ierr = BVNormColumn(bv,k,NORM_2,&nrm);CHKERRQ(ierr); }
    }
    if (norm) *norm = nrm;
    /* linear dependence check: just test for exactly zero norm */
    if (lindep) *lindep = PetscNot(nrm);
    break;

  case BV_ORTHOG_REFINE_ALWAYS:
    ierr = BVOrthogonalizeCGS1(bv,k,v,bv->h,NULL,NULL);CHKERRQ(ierr);
    if (lindep) {
      ierr = BVOrthogonalizeCGS1(bv,k,v,bv->c,&onrm,&nrm);CHKERRQ(ierr);
      if (norm) *norm = nrm;
      *lindep = PetscNot(nrm && nrm >= bv->orthog_eta*onrm);
    } else {
      ierr = BVOrthogonalizeCGS1(bv,k,v,bv->c,NULL,norm);CHKERRQ(ierr);
    }
    for (i=0;i<bv->nc+k;i++) bv->h[i] += bv->c[i];
    break;
  }
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
   vector is orthogonalized against all active columns.

   Level: advanced

.seealso: BVOrthogonalizeColumn(), BVSetOrthogonalization(), BVSetActiveColumns()
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
  switch (bv->orthog_type) {
  case BV_ORTHOG_CGS:
    ierr = BVOrthogonalizeCGS(bv,0,v,H,norm,lindep);CHKERRQ(ierr);
    break;
  case BV_ORTHOG_MGS:
    ierr = BVOrthogonalizeMGS(bv,0,v,NULL,H,norm,lindep);CHKERRQ(ierr);
    break;
  }
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

   Leading columns V[0..l-1] also participate in the orthogonalization.

   If a non-standard inner product has been specified with BVSetMatrix(),
   then the vector is B-orthogonalized, using the non-standard inner product
   defined by matrix B. The output vector satisfies V[j]'*B*V[0..j-1] = 0.

   This routine does not normalize the resulting vector.

   Level: advanced

.seealso: BVSetOrthogonalization(), BVSetMatrix(), BVSetActiveColumns(), BVOrthogonalize(), BVOrthogonalizeVec()
@*/
PetscErrorCode BVOrthogonalizeColumn(BV bv,PetscInt j,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
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

  ierr = PetscLogEventBegin(BV_OrthogonalizeVec,bv,0,0,0);CHKERRQ(ierr);
  ksave = bv->k;
  lsave = bv->l;
  bv->l = -bv->nc;  /* must also orthogonalize against constraints and leading columns */
  ierr = BV_AllocateCoeffs(bv);CHKERRQ(ierr);
  ierr = BV_AllocateSignature(bv);CHKERRQ(ierr);
  switch (bv->orthog_type) {
  case BV_ORTHOG_CGS:
    ierr = BVOrthogonalizeCGS(bv,j,NULL,H,norm,lindep);CHKERRQ(ierr);
    break;
  case BV_ORTHOG_MGS:
    ierr = BVOrthogonalizeMGS(bv,j,NULL,NULL,H,norm,lindep);CHKERRQ(ierr);
    break;
  }
  bv->k = ksave;
  bv->l = lsave;
  if (H) for (i=bv->l;i<j;i++) H[i-bv->l] = bv->h[bv->nc+i];
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

   Level: advanced

.seealso: BVOrthogonalizeColumn(), BVSetOrthogonalization()
@*/
PetscErrorCode BVOrthogonalizeSomeColumn(BV bv,PetscInt j,PetscBool *which,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscInt       i,ksave,lsave;

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
  ierr = BV_AllocateCoeffs(bv);CHKERRQ(ierr);
  ierr = BV_AllocateSignature(bv);CHKERRQ(ierr);
  ierr = BVOrthogonalizeMGS(bv,j,NULL,which,H,norm,lindep);CHKERRQ(ierr);
  bv->k = ksave;
  bv->l = lsave;
  if (H) for (i=bv->l;i<j;i++) H[i-bv->l] = bv->h[bv->nc+i];
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
  Vec            v;

  PetscFunctionBegin;
  if (R) {
    ierr = MatGetSize(R,&ldr,NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArray(R,&r);CHKERRQ(ierr);
  }
  if (V->matrix) {
    ierr = BV_AllocateCachedBV(V);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(V->cached,V->l,V->k);CHKERRQ(ierr);
  }
  for (j=V->l;j<V->k;j++) {
    if (R) {
      ierr = BVOrthogonalizeColumn(V,j,r+j*ldr+V->l,&norm,NULL);CHKERRQ(ierr);
      r[j+j*ldr] = norm;
    } else {
      ierr = BVOrthogonalizeColumn(V,j,NULL,&norm,NULL);CHKERRQ(ierr);
    }
    if (!norm) SETERRQ(PETSC_COMM_SELF,1,"Breakdown in BVOrthogonalize due to a linearly dependent column");
    if (V->matrix) { /* fill cached BV */
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
    /*if (V->ops->orthogonalize) {
      ierr = (*V->ops->orthogonalize)(V,R);CHKERRQ(ierr);
    }*/
    break;
  }
  ierr = PetscLogEventEnd(BV_Orthogonalize,V,R,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

