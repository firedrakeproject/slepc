/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV routines related to Krylov decompositions
*/

#include <slepc/private/bvimpl.h>          /*I   "slepcbv.h"   I*/

/*@
   BVMatArnoldi - Computes an Arnoldi factorization associated with a matrix.

   Collective on V

   Input Parameters:
+  V - basis vectors context
.  A - the matrix
.  H - the upper Hessenberg matrix
.  ldh - leading dimension of H
.  k - number of locked columns
-  m - dimension of the Arnoldi basis

   Output Parameters:
+  m - the modified dimension
.  beta - (optional) norm of last vector before normalization
-  breakdown - (optional) flag indicating that breakdown occurred

   Notes:
   Computes an m-step Arnoldi factorization for matrix A. The first k columns
   are assumed to be locked and therefore they are not modified. On exit, the
   following relation is satisfied:

                    A * V - V * H = beta*v_m * e_m^T

   where the columns of V are the Arnoldi vectors (which are orthonormal), H is
   an upper Hessenberg matrix, e_m is the m-th vector of the canonical basis.
   On exit, beta contains the norm of V[m] before normalization.

   The breakdown flag indicates that orthogonalization failed, see
   BVOrthonormalizeColumn(). In that case, on exit m contains the index of
   the column that failed.

   The values of k and m are not restricted to the active columns of V.

   To create an Arnoldi factorization from scratch, set k=0 and make sure the
   first column contains the normalized initial vector.

   Level: advanced

.seealso: BVSetActiveColumns(), BVOrthonormalizeColumn()
@*/
PetscErrorCode BVMatArnoldi(BV V,Mat A,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *m,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscScalar    *a;
  PetscInt       j;
  PetscBool      lindep=PETSC_FALSE;
  Vec            buf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidLogicalCollectiveInt(V,k,5);
  PetscValidIntPointer(m,6);
  PetscValidLogicalCollectiveInt(V,*m,6);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidType(A,2);
  PetscCheckSameComm(V,1,A,2);

  if (k<0 || k>V->m) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument k has wrong value %D, should be between 0 and %D",k,V->m);
  if (*m<1 || *m>V->m) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument m has wrong value %D, should be between 1 and %D",*m,V->m);
  if (*m<=k) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument m should be at least equal to k+1");

  ierr = BVSetActiveColumns(V,0,*m);CHKERRQ(ierr);
  for (j=k;j<*m;j++) {
    ierr = BVMatMultColumn(V,A,j);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(V,j+1,PETSC_FALSE,beta,&lindep);CHKERRQ(ierr);
    if (lindep) {
      *m = j+1;
      break;
    }
  }
  if (breakdown) *breakdown = lindep;
  /* extract Hessenberg matrix from the BV object */
  ierr = BVGetBufferVec(V,&buf);CHKERRQ(ierr);
  ierr = VecGetArray(buf,&a);CHKERRQ(ierr);
  for (j=k;j<*m;j++) {
    ierr = PetscArraycpy(H+j*ldh,a+V->nc+(j+1)*(V->nc+V->m),j+2);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(buf,&a);CHKERRQ(ierr);

  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

