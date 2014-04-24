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

/*
   BVOrthogonalizeCGS1 - Compute |v'| (estimated), |v| and one step of CGS with
   only one global synchronization
*/
#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeCGS1"
PetscErrorCode BVOrthogonalizeCGS1(BV bv,PetscInt j,PetscScalar *H,PetscReal *onorm,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      sum;
  Vec            v;

  PetscFunctionBegin;
  /* h = W^* v ; alpha = (v, v) */
  bv->k = j;
  if (onorm || norm) bv->k++;
  ierr = BVGetColumn(bv,j,&v);CHKERRQ(ierr);
  ierr = BVDotVec(bv,v,H);CHKERRQ(ierr);

  /* q = v - V h */
  if (onorm || norm) bv->k--;
  ierr = BVMultVec(bv,-1.0,1.0,v,H);CHKERRQ(ierr);
  ierr = BVRestoreColumn(bv,j,&v);CHKERRQ(ierr);

  /* compute |v| */
  if (onorm) *onorm = PetscSqrtReal(PetscRealPart(H[j]));

  if (norm) {
    /* estimate |v'| from |v| */
    sum = 0.0;
    for (i=0;i<j;i++)
      sum += PetscRealPart(H[i]*PetscConj(H[i]));
    *norm = PetscRealPart(H[j])-sum;
    if (*norm <= 0.0) {
    } else *norm = PetscSqrtReal(*norm);
  }
  PetscFunctionReturn(0);
}

/*
  BVOrthogonalizeCGS - Orthogonalize with classical Gram-Schmidt
*/
#undef __FUNCT__
#define __FUNCT__ "BVOrthogonalizeCGS"
static PetscErrorCode BVOrthogonalizeCGS(BV bv,PetscInt j,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscReal      onrm,nrm;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = BVOrthogonalizeCGS1(bv,j,bv->h,NULL,NULL);CHKERRQ(ierr);
  if (lindep) {
    ierr = BVOrthogonalizeCGS1(bv,j,bv->c,&onrm,&nrm);CHKERRQ(ierr);
    if (norm) *norm = nrm;
  } else {
    ierr = BVOrthogonalizeCGS1(bv,j,bv->c,NULL,norm);CHKERRQ(ierr);
  }
  for (i=0;i<j;i++)
    bv->h[i] += bv->c[i];
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
   the orthogonal complement of the span of the columns of defl and V[0..j-1],
   where V[.] are the vectors of BV. The columns V[0..j-1] are assumed to be
   mutually orthonormal.

   This routine does not normalize the resulting vector.

   Level: advanced

.seealso: BVSetOrthogonalization()
@*/
PetscErrorCode BVOrthogonalize(BV bv,PetscInt j,PetscScalar *H,PetscReal *norm,PetscBool *lindep)
{
  PetscErrorCode ierr;
  PetscInt       ksave;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  if (j<0) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  if (j>=bv->m) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%D but BV only has %D columns",j,bv->m);

  ierr = PetscLogEventBegin(BV_Orthogonalize,bv,0,0,0);CHKERRQ(ierr);
  ksave = bv->k;
  if (!bv->h) {
    ierr = PetscMalloc2(bv->m,&bv->h,bv->m,&bv->c);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)bv,2*bv->m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = BVOrthogonalizeCGS(bv,j,H,norm,lindep);CHKERRQ(ierr);
  bv->k = ksave;
  ierr = PetscLogEventEnd(BV_Orthogonalize,bv,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

