/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nPreconditioner for 1-D Laplacian, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Compute the operator matrix for the 1-D Laplacian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,i,i,2.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A,&v,&w));
  PetscCall(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  PetscCall(STSetMatrices(st,1,mat));
  PetscCall(STSetType(st,STPRECOND));
  PetscCall(STGetKSP(st,&ksp));
  PetscCall(KSPSetType(ksp,KSPPREONLY));
  PetscCall(STSetFromOptions(st));

  /* set up */
  /* - the transform flag is necessary so that A-sigma*I is built explicitly */
  PetscCall(STSetTransform(st,PETSC_TRUE));
  /* - the ksphasmat flag is necessary when using STApply(), otherwise can only use PCApply() */
  PetscCall(STPrecondSetKSPHasMat(st,PETSC_TRUE));
  /* no need to call STSetUp() explicitly */
  PetscCall(STSetUp(st));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Apply the preconditioner
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* default shift */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With default shift\n"));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* change shift */
  sigma = 0.1;
  PetscCall(STSetShift(st,sigma));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));
  PetscCall(STPostSolve(st));   /* undo changes if inplace */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Test a user-provided preconditioner matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&P));
  PetscCall(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(P));
  PetscCall(MatSetUp(P));

  PetscCall(MatGetOwnershipRange(P,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(P,i,i,2.0,INSERT_VALUES));
  if (Istart==0) {
    PetscCall(MatSetValue(P,1,0,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(P,0,1,-1.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));

  /* apply new preconditioner */
  PetscCall(STSetPreconditionerMat(st,P));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With user-provided matrix\n"));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            Test a user-provided preconditioner in split form
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(STGetMatMode(st,&matmode));
  if (matmode==ST_MATMODE_COPY) {
    PetscCall(STSetPreconditionerMat(st,NULL));
    mat[0] = P;
    PetscCall(STSetSplitPreconditioner(st,1,mat,SAME_NONZERO_PATTERN));

    /* apply new preconditioner */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With split preconditioner\n"));
    PetscCall(STApply(st,v,w));
    PetscCall(VecView(w,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                             Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(STDestroy(&st));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&P));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -st_matmode {{copy inplace shell}separate output}
      requires: !single

TEST*/
