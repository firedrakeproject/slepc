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
PetscErrorCode LMESolve_Krylov_Lyapunov_Vec(LME lme,Vec b,PetscBool fixed,PetscInt rrank,BV C1,BV *X1,PetscInt *col,PetscBool *fail)
{
  PetscErrorCode ierr;
  PetscInt       m,ldh,ldl,i,rank,lrank;
  PetscReal      bnorm,beta,errest;
  PetscBool      breakdown;
  PetscScalar    *H,*L,*r,*Qarray;
  Mat            Q;

  PetscFunctionBegin;
  *fail = PETSC_FALSE;
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
  ierr = LMEDenseLyapunovChol(lme,H,m,ldh,r,L,ldl,&errest);CHKERRQ(ierr);
  if (errest>lme->tol) *fail = PETSC_TRUE;
  lme->errest += errest;

  /* determine numerical rank of L */
  for (i=1;i<m && PetscAbsScalar(L[i+i*m])>PetscAbsScalar(L[0])*PETSC_MACHINE_EPSILON;i++);
  lrank = i;
  if (!fixed) {  /* X1 was not set by user, allocate it with rank columns */
    rank = lrank;
    if (*col) {
      ierr = BVResize(*X1,*col+rank,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = BVDuplicateResize(C1,rank,X1);CHKERRQ(ierr);
    }
  } else rank = PetscMin(lrank,rrank);

  /* Z = V(:,1:m)*L */
  ierr = MatCreateDense(PETSC_COMM_SELF,m,*col+m,m,*col+m,NULL,&Q);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Q,&Qarray);CHKERRQ(ierr);
  ierr = PetscMemcpy(Qarray+(*col)*m,L,m*m*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Q,&Qarray);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(*X1,*col,*col+rank);CHKERRQ(ierr);
  ierr = BVMult(*X1,1.0,0.0,lme->V,Q);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  *col += rank;

  ierr = PetscFree3(H,L,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LMESolve_Krylov_Lyapunov"
PetscErrorCode LMESolve_Krylov_Lyapunov(LME lme)
{
  PetscErrorCode ierr;
  PetscBool      fail,fixed = lme->X? PETSC_TRUE: PETSC_FALSE;
  PetscInt       i,k,rank,col=0;
  Vec            b;
  BV             X1=NULL,C1;
  Mat            X1m,X1t,C1m;

  PetscFunctionBegin;
  ierr = MatLRCGetMats(lme->C,NULL,&C1m,NULL,NULL);CHKERRQ(ierr);
  ierr = BVCreateFromMat(C1m,&C1);CHKERRQ(ierr);
  ierr = BVSetFromOptions(C1);CHKERRQ(ierr);
  ierr = BVGetActiveColumns(C1,NULL,&k);CHKERRQ(ierr);
  if (fixed) {
    ierr = MatLRCGetMats(lme->X,NULL,&X1m,NULL,NULL);CHKERRQ(ierr);
    ierr = BVCreateFromMat(X1m,&X1);CHKERRQ(ierr);
    ierr = BVSetFromOptions(X1);CHKERRQ(ierr);
    ierr = BVGetActiveColumns(X1,NULL,&rank);CHKERRQ(ierr);
    rank = rank/k;
  }
  lme->reason = LME_CONVERGED_TOL;
  for (i=0;i<k;i++) {
    ierr = BVGetColumn(C1,i,&b);CHKERRQ(ierr);
    ierr = LMESolve_Krylov_Lyapunov_Vec(lme,b,fixed,rank,C1,&X1,&col,&fail);CHKERRQ(ierr);
    ierr = BVRestoreColumn(C1,i,&b);CHKERRQ(ierr);
    lme->its++;
    if (fail) {
      lme->reason = LME_DIVERGED_ITS;
      break;
    }
  }
  ierr = BVCreateMat(X1,&X1t);CHKERRQ(ierr);
  if (fixed) {
    ierr = MatCopy(X1t,X1m,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    ierr = MatCreateLRC(NULL,X1t,NULL,NULL,&lme->X);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&X1t);CHKERRQ(ierr);
  ierr = BVDestroy(&C1);CHKERRQ(ierr);
  ierr = BVDestroy(&X1);CHKERRQ(ierr);
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
