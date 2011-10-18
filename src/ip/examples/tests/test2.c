/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Test SlepcUpdateVectors.\n\n";

#include "slepcsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  Vec            *V,t;
  Mat            A,B,C,M;
  PetscInt       i,j,n=15,k=6,s=3,e=5;
  PetscRandom    rctx;
  PetscBool      cont,qtrans=PETSC_FALSE;
  PetscScalar    *Q,*pa,*pv;
  PetscReal      nrm;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-k",&k,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-e",&e,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-qtrans",&qtrans);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"V(:,%D:%D) = V*",s,e-1);CHKERRQ(ierr); 
  if (qtrans) { ierr = PetscPrintf(PETSC_COMM_WORLD,"Q(%D:%D,:)'",s,e-1);CHKERRQ(ierr); }
  else { ierr = PetscPrintf(PETSC_COMM_WORLD,"Q(:,%D:%D)",s,e-1);CHKERRQ(ierr); }
  ierr = PetscPrintf(PETSC_COMM_WORLD," for random vectors of length %D (V has %D columns).\n",n,k);CHKERRQ(ierr); 
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*k*k,&Q);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-contiguous",&cont);CHKERRQ(ierr);

  /* with/without contiguous storage */
  if (cont) {
    ierr = SlepcVecSetTemplate(t);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"With contiguous storage.\n");CHKERRQ(ierr); 
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"With regular storage.\n");CHKERRQ(ierr); 
  }
  ierr = VecDuplicateVecs(t,k,&V);CHKERRQ(ierr);

  /* fill with random values */
  for (i=0;i<k;i++) { ierr = VecSetRandom(V[i],rctx);CHKERRQ(ierr); }
  for (i=0;i<k*k;i++) { ierr = PetscRandomGetValue(rctx,&Q[i]);CHKERRQ(ierr); }

  /* save a copy into Mat objects */
  ierr = MatCreateSeqDense(PETSC_COMM_WORLD,n,k,PETSC_NULL,&A);CHKERRQ(ierr);
  ierr = MatGetArray(A,&pa);CHKERRQ(ierr);
  for (i=0;i<k;i++) { 
    ierr = VecGetArray(V[i],&pv);CHKERRQ(ierr);
    for (j=0;j<n;j++) { 
      pa[i*n+j] = pv[j];
    }
    ierr = VecRestoreArray(V[i],&pv);CHKERRQ(ierr);
  }
  ierr = MatRestoreArray(A,&pa);CHKERRQ(ierr);
  if (qtrans) {
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD,k,e-s,PETSC_NULL,&B);CHKERRQ(ierr);
    ierr = MatGetArray(B,&pa);CHKERRQ(ierr);
    for (i=s;i<e;i++) { 
      for (j=0;j<k;j++) { 
        pa[(i-s)*k+j] = Q[i+j*k];
      }
    }
    ierr = MatRestoreArray(B,&pa);CHKERRQ(ierr);
  } else {
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD,k,e-s,Q+s*k,&B);CHKERRQ(ierr);
  }

  /* call SlepcUpdateVectors */
  ierr = SlepcUpdateVectors(k,V,s,e,Q,k,qtrans);CHKERRQ(ierr);

  /* check result */
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,1.0,&C);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_WORLD,n,e-s,PETSC_NULL,&M);CHKERRQ(ierr);
  ierr = MatGetArray(M,&pa);CHKERRQ(ierr);
  for (i=0;i<e-s;i++) { 
    ierr = VecGetArray(V[i+s],&pv);CHKERRQ(ierr);
    for (j=0;j<n;j++) { 
      pa[i*n+j] = pv[j];
    }
    ierr = VecRestoreArray(V[i+s],&pv);CHKERRQ(ierr);
  }
  ierr = MatRestoreArray(M,&pa);CHKERRQ(ierr);
  ierr = MatAXPY(M,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(M,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
  if (nrm<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test gave correct result.\n");CHKERRQ(ierr); 
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error against MatMatMult = %G.\n",nrm);CHKERRQ(ierr); 
  }

  ierr = VecDestroyVecs(k,&V);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}
