/*

   SLEPc matrix function solver: "krylov"

   Method: Arnoldi

   Algorithm:

       Single-vector Arnoldi method to build a Krylov subspace, then
       compute f(B) on the projected matrix B.

   References:

       [1] R. Sidje, "Expokit: a software package for computing matrix
           exponentials", ACM Trans. Math. Softw. 24(1):130-156, 1998.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/mfnimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MFNSetUp_Krylov"
PetscErrorCode MFNSetUp_Krylov(MFN mfn)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = MatGetSize(mfn->A,&N,NULL);CHKERRQ(ierr);
  if (!mfn->ncv) mfn->ncv = PetscMin(30,N);
  if (!mfn->max_it) mfn->max_it = PetscMax(100,2*N/mfn->ncv);
  ierr = MFNAllocateSolution(mfn,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNBasicArnoldi"
PetscErrorCode MFNBasicArnoldi(MFN mfn,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  Vec            vj,vj1;

  PetscFunctionBegin;
  ierr = BVSetActiveColumns(mfn->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(mfn->V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(mfn->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = MatMult(mfn->A,vj,vj1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(mfn->V,j,&vj);CHKERRQ(ierr);
    ierr = BVRestoreColumn(mfn->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(mfn->V,j+1,H+ldh*j,beta,breakdown);CHKERRQ(ierr);
    H[j+1+ldh*j] = *beta;
    if (*breakdown) {
      *M = j+1;
      break;
    } else {
      ierr = BVScaleColumn(mfn->V,j+1,1.0/(*beta));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSolve_Krylov"
PetscErrorCode MFNSolve_Krylov(MFN mfn,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       m,ld,j;
  Mat            H=NULL,F=NULL;
  PetscScalar    *array,*harray,*farray;
  PetscReal      beta;
  PetscBool      breakdown,set,flg,symm=PETSC_FALSE;

  PetscFunctionBegin;
  m  = mfn->ncv;
  ld = m+1;
  ierr = PetscCalloc1(ld*ld,&array);CHKERRQ(ierr);
  ierr = MFN_CreateDenseMat(m,&F);CHKERRQ(ierr);
  ierr = MFN_CreateDenseMat(m,&H);CHKERRQ(ierr);

  /* build Arnoldi decomposition with start vector b/||b|| */
  ierr = BVInsertVec(mfn->V,0,b);CHKERRQ(ierr);
  ierr = BVScaleColumn(mfn->V,0,1.0/mfn->bnorm);CHKERRQ(ierr);
  ierr = MFNBasicArnoldi(mfn,array,ld,0,&m,&beta,&breakdown);CHKERRQ(ierr);

  ierr = MatDenseGetArray(H,&harray);CHKERRQ(ierr);
  for (j=0;j<m;j++) {
    ierr = PetscMemcpy(harray+j*m,array+j*ld,m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(H,&harray);CHKERRQ(ierr);

  /* set symmetry flag of H from A */
  ierr = MatIsHermitianKnown(mfn->A,&set,&flg);CHKERRQ(ierr);
  symm = set? flg: PETSC_FALSE;
  if (symm) {
    ierr = MatSetOption(H,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* evaluate f(H) */
  ierr = FNEvaluateFunctionMat(mfn->fn,H,F);CHKERRQ(ierr);

  /* x = ||b||*V*f(H)*e_1 */
  ierr = MatDenseGetArray(F,&farray);CHKERRQ(ierr);
  for (j=0;j<m;j++) farray[j] *= mfn->bnorm;
  ierr = BVSetActiveColumns(mfn->V,0,m);CHKERRQ(ierr);
  ierr = BVMultVec(mfn->V,1.0,0.0,x,farray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(F,&farray);CHKERRQ(ierr);

  /* there is no known way to assess accuracy for general f, so just pretend it works */
  mfn->reason = MFN_CONVERGED_ITS;
  mfn->its = 1;

  ierr = MatDestroy(&H);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = PetscFree(array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNCreate_Krylov"
PETSC_EXTERN PetscErrorCode MFNCreate_Krylov(MFN mfn)
{
  PetscFunctionBegin;
  mfn->ops->solve          = MFNSolve_Krylov;
  mfn->ops->setup          = MFNSetUp_Krylov;
  PetscFunctionReturn(0);
}
