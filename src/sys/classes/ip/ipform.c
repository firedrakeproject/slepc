/*
     Routines for setting the matrix representation of the inner product.

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

#include <slepc-private/ipimpl.h>      /*I "slepcip.h" I*/

#undef __FUNCT__
#define __FUNCT__ "IPSetMatrix"
/*@
   IPSetMatrix - Specifies the matrix representation of the inner product.

   Collective on IP

   Input Parameters:
+  ip    - the inner product context
.  mat   - the matrix (may be NULL)
-  sfact - the scale factor

   Notes:
   A NULL has the same effect as if the identity matrix was passed.

   This function is called by EPSSetProblemType() and usually need not be
   called by the user.

   Level: developer

.seealso: IPGetMatrix(), IPInnerProduct(), IPNorm(), EPSSetProblemType()
@*/
PetscErrorCode IPSetMatrix(IP ip,Mat mat,PetscScalar sfact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  if (mat) {
    PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
    PetscObjectReference((PetscObject)mat);
  }
  PetscValidLogicalCollectiveScalar(ip,sfact,3);
  ierr = IPReset(ip);CHKERRQ(ierr);
  ip->matrix  = mat;
  ip->sfactor = sfact;
  if (mat) {
    ierr = MatGetVecs(mat,&ip->Bx,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ip,(PetscObject)ip->Bx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPGetMatrix"
/*@C
   IPGetMatrix - Retrieves the matrix representation of the inner product.

   Not collective, though a parallel Mat may be returned

   Input Parameter:
.  ip    - the inner product context

   Output Parameter:
+  mat   - the matrix of the inner product (may be NULL)
-  sfact - the scale factor

   Level: developer

.seealso: IPSetMatrix(), IPInnerProduct(), IPNorm(), EPSSetProblemType()
@*/
PetscErrorCode IPGetMatrix(IP ip,Mat* mat,PetscScalar* sfact)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  if (mat)   *mat   = ip->matrix;
  if (sfact) *sfact = ip->sfactor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPApplyMatrix_Private"
PetscErrorCode IPApplyMatrix_Private(IP ip,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)x)->id != ip->xid || ((PetscObject)x)->state != ip->xstate) {
    ierr = PetscLogEventBegin(IP_ApplyMatrix,ip,0,0,0);CHKERRQ(ierr);
    ierr = MatMult(ip->matrix,x,ip->Bx);CHKERRQ(ierr);
    ierr = VecScale(ip->Bx,ip->sfactor);CHKERRQ(ierr);
    ip->xid = ((PetscObject)x)->id;
    ip->xstate = ((PetscObject)x)->state;
    ierr = PetscLogEventEnd(IP_ApplyMatrix,ip,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPApplyMatrix"
/*@
   IPApplyMatrix - Multiplies a vector by the matrix representing the IP.

   Neighbor-wise Collective on IP and Vec

   Input Parameters:
+  ip    - the inner product context
-  x     - the vector

   Output Parameter:
.  y     - the result

   Note:
   If no matrix was specified this function copies the vector.

   Level: developer

.seealso: IPSetMatrix(), IPInnerProduct(), IPNorm(), EPSSetProblemType()
@*/
PetscErrorCode IPApplyMatrix(IP ip,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    ierr = VecCopy(ip->Bx,y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
