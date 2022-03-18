/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test ST with four matrices.\n\n";

#include <slepcst.h>

int main(int argc,char **argv)
{
  Mat            A,B,C,D,mat[4];
  ST             st;
  KSP            ksp;
  Vec            v,w;
  STType         type;
  PetscScalar    sigma;
  PetscInt       n=10,i,Istart,Iend;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nTest ST with four matrices, n=%" PetscInt_FMT "\n\n",n));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&D));
  CHKERRQ(MatSetSizes(D,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(D));
  CHKERRQ(MatSetUp(D));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (i>0) {
      CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
      CHKERRQ(MatSetValue(B,i,i,(PetscScalar)i,INSERT_VALUES));
    } else {
      CHKERRQ(MatSetValue(B,i,i,-1.0,INSERT_VALUES));
    }
    if (i<n-1) {
      CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    }
    CHKERRQ(MatSetValue(C,i,n-i-1,1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(D,i,i,i*.1,INSERT_VALUES));
    if (i==0) {
      CHKERRQ(MatSetValue(D,0,n-1,1.0,INSERT_VALUES));
    }
    if (i==n-1) {
      CHKERRQ(MatSetValue(D,n-1,0,1.0,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&v,&w));
  CHKERRQ(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  mat[1] = B;
  mat[2] = C;
  mat[3] = D;
  CHKERRQ(STSetMatrices(st,4,mat));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetTolerances(ksp,100*PETSC_MACHINE_EPSILON,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(STSetFromOptions(st));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Apply the transformed operator for several ST's
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* shift, sigma=0.0 */
  CHKERRQ(STSetUp(st));
  CHKERRQ(STGetType(st,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  for (i=0;i<4;i++) {
    CHKERRQ(STMatMult(st,i,v,w));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"k= %" PetscInt_FMT "\n",i));
    CHKERRQ(VecView(w,NULL));
  }
  CHKERRQ(STMatSolve(st,v,w));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"solve\n"));
  CHKERRQ(VecView(w,NULL));

  /* shift, sigma=0.1 */
  sigma = 0.1;
  CHKERRQ(STSetShift(st,sigma));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  for (i=0;i<4;i++) {
    CHKERRQ(STMatMult(st,i,v,w));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"k= %" PetscInt_FMT "\n",i));
    CHKERRQ(VecView(w,NULL));
  }
  CHKERRQ(STMatSolve(st,v,w));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"solve\n"));
  CHKERRQ(VecView(w,NULL));

  /* sinvert, sigma=0.1 */
  CHKERRQ(STPostSolve(st));
  CHKERRQ(STSetType(st,STSINVERT));
  CHKERRQ(STGetType(st,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  for (i=0;i<4;i++) {
    CHKERRQ(STMatMult(st,i,v,w));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"k= %" PetscInt_FMT "\n",i));
    CHKERRQ(VecView(w,NULL));
  }
  CHKERRQ(STMatSolve(st,v,w));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"solve\n"));
  CHKERRQ(VecView(w,NULL));

  /* sinvert, sigma=-0.5 */
  sigma = -0.5;
  CHKERRQ(STSetShift(st,sigma));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  for (i=0;i<4;i++) {
    CHKERRQ(STMatMult(st,i,v,w));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"k= %" PetscInt_FMT "\n",i));
    CHKERRQ(VecView(w,NULL));
  }
  CHKERRQ(STMatSolve(st,v,w));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"solve\n"));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STDestroy(&st));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -st_transform -st_matmode {{copy shell}}
      output_file: output/test4_1.out
      requires: !single

   test:
      suffix: 2
      args: -st_matmode {{copy shell}}
      output_file: output/test4_2.out

TEST*/
