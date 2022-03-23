/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test STPRECOND operations.\n\n";

#include <slepcst.h>

int main(int argc,char **argv)
{
  Mat            A,P,mat[1];
  ST             st;
  KSP            ksp;
  Vec            v,w;
  PetscScalar    sigma;
  PetscInt       n=10,i,Istart,Iend;
  STMatMode      matmode;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nPreconditioner for 1-D Laplacian, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Compute the operator matrix for the 1-D Laplacian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&v,&w));
  CHKERRQ(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  CHKERRQ(STSetMatrices(st,1,mat));
  CHKERRQ(STSetType(st,STPRECOND));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(STSetFromOptions(st));

  /* set up */
  /* - the transform flag is necessary so that A-sigma*I is built explicitly */
  CHKERRQ(STSetTransform(st,PETSC_TRUE));
  /* - the ksphasmat flag is necessary when using STApply(), otherwise can only use PCApply() */
  CHKERRQ(STPrecondSetKSPHasMat(st,PETSC_TRUE));
  /* no need to call STSetUp() explicitly */
  CHKERRQ(STSetUp(st));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Apply the preconditioner
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* default shift */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With default shift\n"));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* change shift */
  sigma = 0.1;
  CHKERRQ(STSetShift(st,sigma));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));
  CHKERRQ(STPostSolve(st));   /* undo changes if inplace */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Test a user-provided preconditioner matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&P));
  CHKERRQ(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(P));
  CHKERRQ(MatSetUp(P));

  CHKERRQ(MatGetOwnershipRange(P,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(P,i,i,2.0,INSERT_VALUES));
  if (Istart==0) {
    CHKERRQ(MatSetValue(P,1,0,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(P,0,1,-1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

  /* apply new preconditioner */
  CHKERRQ(STSetPreconditionerMat(st,P));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With user-provided matrix\n"));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            Test a user-provided preconditioner in split form
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STGetMatMode(st,&matmode));
  if (matmode==ST_MATMODE_COPY) {
    CHKERRQ(STSetPreconditionerMat(st,NULL));
    mat[0] = P;
    CHKERRQ(STSetSplitPreconditioner(st,1,mat,SAME_NONZERO_PATTERN));

    /* apply new preconditioner */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With split preconditioner\n"));
    CHKERRQ(STApply(st,v,w));
    CHKERRQ(VecView(w,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                             Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(STDestroy(&st));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&P));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -st_matmode {{copy inplace shell}separate output}
      requires: !single

TEST*/
