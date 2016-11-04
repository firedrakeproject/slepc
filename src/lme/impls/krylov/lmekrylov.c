/*

   SLEPc matrix equation solver: "krylov"

   Method: ....

   Algorithm:

       ....

   References:

       [1] ...

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

#include <slepc/private/lmeimpl.h>

#undef __FUNCT__
#define __FUNCT__ "LMESetUp_Krylov"
PetscErrorCode LMESetUp_Krylov(LME lme)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = MatGetSize(lme->A,&N,NULL);CHKERRQ(ierr);
  if (!lme->ncv) lme->ncv = PetscMin(30,N);
  if (!lme->max_it) lme->max_it = 100;
  ierr = LMEAllocateSolution(lme,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMEBasicArnoldi"
PetscErrorCode LMEBasicArnoldi(LME lme,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscScalar    *a;
  PetscInt       j,nc,n,m = *M;
  Vec            vj,vj1,buf;

  PetscFunctionBegin;
  ierr = BVSetActiveColumns(lme->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(lme->V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(lme->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = MatMult(lme->A,vj,vj1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(lme->V,j,&vj);CHKERRQ(ierr);
    ierr = BVRestoreColumn(lme->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(lme->V,j+1,PETSC_FALSE,beta,breakdown);CHKERRQ(ierr);
    if (*breakdown) {
      *M = j+1;
      break;
    }
  }
  /* extract Hessenberg matrix from the BV object */
  ierr = BVGetNumConstraints(lme->V,&nc);CHKERRQ(ierr);
  ierr = BVGetSizes(lme->V,NULL,NULL,&n);CHKERRQ(ierr);
  ierr = BVGetBufferVec(lme->V,&buf);CHKERRQ(ierr);
  ierr = VecGetArray(buf,&a);CHKERRQ(ierr);
  for (j=k;j<*M;j++) {
    ierr = PetscMemcpy(H+j*ldh,a+nc+(j+1)*(nc+n),(j+2)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(buf,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMESolve_Krylov_Lyapunov_Vec"
PetscErrorCode LMESolve_Krylov_Lyapunov_Vec(LME lme,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       m,ldh,ldl;
  PetscReal      bnorm,beta;
  PetscBool      breakdown;
  PetscScalar    *H,*L,*r;
  Mat            Q;

  PetscFunctionBegin;
  m  = lme->ncv;
  ldh = m+1;
  ldl = m;
  ierr = PetscCalloc3(ldh*m,&H,ldl*m,&L,m,&r);CHKERRQ(ierr);

  /* set initial vector to b/||b|| */
  ierr = VecNorm(b,NORM_2,&bnorm);CHKERRQ(ierr);
  if (!bnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot process a zero vector in the right-hand side");
  ierr = BVInsertVec(lme->V,0,b);CHKERRQ(ierr);
  ierr = BVScaleColumn(lme->V,0,1.0/bnorm);CHKERRQ(ierr);

  /* compute Arnoldi factorization */
  ierr = LMEBasicArnoldi(lme,H,ldh,0,&m,&beta,&breakdown);CHKERRQ(ierr);
  H[m+1+m*ldh] = beta;

  /* solve compressed Lyapunov equation */
  r[0] = bnorm;
  ierr = LMEDenseLyapunovChol(lme,H,m,ldh,r,L,ldl,&lme->errest);CHKERRQ(ierr);

  if (lme->errest<lme->tol) lme->reason = LME_CONVERGED_TOL;
  else lme->reason = LME_DIVERGED_ITS;

  /* Z = V(:,1:m)*L */
  ierr = MatCreateDense(PETSC_COMM_SELF,m,m,m,m,L,&Q);CHKERRQ(ierr);
  ierr = BVMult(lme->X1,1.0,0.0,lme->V,Q);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);

  ierr = PetscFree3(H,L,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMESolve_Krylov_Lyapunov"
PetscErrorCode LMESolve_Krylov_Lyapunov(LME lme)
{
  PetscErrorCode ierr;
  PetscInt       k;
  Vec            b;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(lme->C1,NULL,&k);CHKERRQ(ierr);
  if (k>1) SETERRQ(PETSC_COMM_SELF,1,"Only implemented for rank-1 right-hand side");
  ierr = BVGetColumn(lme->C1,0,&b);CHKERRQ(ierr);
  ierr = LMESolve_Krylov_Lyapunov_Vec(lme,b);CHKERRQ(ierr);
  ierr = BVRestoreColumn(lme->C1,0,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMECreate_Krylov"
PETSC_EXTERN PetscErrorCode LMECreate_Krylov(LME lme)
{
  PetscFunctionBegin;
  lme->ops->solve[LME_LYAPUNOV]      = LMESolve_Krylov_Lyapunov;
  lme->ops->setup                    = LMESetUp_Krylov;
  PetscFunctionReturn(0);
}
