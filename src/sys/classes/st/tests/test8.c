/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test ST with two matrices and split preconditioner.\n\n";

#include <slepcst.h>

int main(int argc,char **argv)
{
  Mat            A,B,Pa,Pb,Pmat,mat[2];
  ST             st;
  KSP            ksp;
  PC             pc;
  Vec            v,w;
  STType         type;
  PetscScalar    sigma;
  PetscInt       n=10,i,Istart,Iend;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian plus diagonal, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrices (1-D Laplacian and diagonal)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (i>0) {
      CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
      CHKERRQ(MatSetValue(B,i,i,(PetscScalar)i,INSERT_VALUES));
    } else {
      CHKERRQ(MatSetValue(B,i,i,-1.0,INSERT_VALUES));
    }
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&v,&w));
  CHKERRQ(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the split preconditioner matrices (two diagonals)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Pa));
  CHKERRQ(MatSetSizes(Pa,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(Pa));
  CHKERRQ(MatSetUp(Pa));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Pb));
  CHKERRQ(MatSetSizes(Pb,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(Pb));
  CHKERRQ(MatSetUp(Pb));

  CHKERRQ(MatGetOwnershipRange(Pa,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(Pa,i,i,2.0,INSERT_VALUES));
    if (i>0) {
      CHKERRQ(MatSetValue(Pb,i,i,(PetscScalar)i,INSERT_VALUES));
    } else {
      CHKERRQ(MatSetValue(Pb,i,i,-1.0,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(Pa,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pa,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(Pb,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pb,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  mat[1] = B;
  CHKERRQ(STSetMatrices(st,2,mat));
  mat[0] = Pa;
  mat[1] = Pb;
  CHKERRQ(STSetSplitPreconditioner(st,2,mat,SAME_NONZERO_PATTERN));
  CHKERRQ(STSetTransform(st,PETSC_TRUE));
  CHKERRQ(STSetFromOptions(st));
  CHKERRQ(STCayleySetAntishift(st,-0.2));   /* only relevant for cayley */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Form the preconditioner matrix and print it
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(STGetOperator(st,NULL));
  CHKERRQ(PCGetOperators(pc,NULL,&Pmat));
  CHKERRQ(MatView(Pmat,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Apply the operator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* sigma=0.0 */
  CHKERRQ(STSetUp(st));
  CHKERRQ(STGetType(st,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* sigma=0.1 */
  sigma = 0.1;
  CHKERRQ(STSetShift(st,sigma));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  CHKERRQ(STGetOperator(st,NULL));
  CHKERRQ(PCGetOperators(pc,NULL,&Pmat));
  CHKERRQ(MatView(Pmat,NULL));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STDestroy(&st));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&Pa));
  CHKERRQ(MatDestroy(&Pb));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -st_type {{cayley shift sinvert}separate output}
      requires: !single

TEST*/
