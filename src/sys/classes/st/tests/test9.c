/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test ST with four matrices and split preconditioner.\n\n";

#include <slepcst.h>

int main(int argc,char **argv)
{
  Mat            A,B,C,D,Pa,Pb,Pc,Pd,Pmat,mat[4];
  ST             st;
  KSP            ksp;
  PC             pc;
  Vec            v,w;
  STType         type;
  PetscScalar    sigma;
  PetscInt       n=10,i,Istart,Iend;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
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
    } else CHKERRQ(MatSetValue(B,i,i,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(C,i,n-i-1,1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(D,i,i,i*.1,INSERT_VALUES));
    if (i==0) CHKERRQ(MatSetValue(D,0,n-1,1.0,INSERT_VALUES));
    if (i==n-1) CHKERRQ(MatSetValue(D,n-1,0,1.0,INSERT_VALUES));
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
     Compute the split preconditioner matrices (four diagonals)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Pa));
  CHKERRQ(MatSetSizes(Pa,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(Pa));
  CHKERRQ(MatSetUp(Pa));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Pb));
  CHKERRQ(MatSetSizes(Pb,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(Pb));
  CHKERRQ(MatSetUp(Pb));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Pc));
  CHKERRQ(MatSetSizes(Pc,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(Pc));
  CHKERRQ(MatSetUp(Pc));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Pd));
  CHKERRQ(MatSetSizes(Pd,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(Pd));
  CHKERRQ(MatSetUp(Pd));

  CHKERRQ(MatGetOwnershipRange(Pa,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(Pa,i,i,2.0,INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(Pb,i,i,(PetscScalar)i,INSERT_VALUES));
    else CHKERRQ(MatSetValue(Pb,i,i,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(Pd,i,i,i*.1,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(Pa,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pa,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(Pb,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pb,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(Pc,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pc,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(Pd,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pd,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  mat[1] = B;
  mat[2] = C;
  mat[3] = D;
  CHKERRQ(STSetMatrices(st,4,mat));
  mat[0] = Pa;
  mat[1] = Pb;
  mat[2] = Pc;
  mat[3] = Pd;
  CHKERRQ(STSetSplitPreconditioner(st,4,mat,SUBSET_NONZERO_PATTERN));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetTolerances(ksp,100*PETSC_MACHINE_EPSILON,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(STSetTransform(st,PETSC_TRUE));
  CHKERRQ(STSetFromOptions(st));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Apply the operator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* sigma=0.0 */
  CHKERRQ(STSetUp(st));
  CHKERRQ(STGetType(st,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  CHKERRQ(PCGetOperators(pc,NULL,&Pmat));
  CHKERRQ(MatView(Pmat,NULL));
  CHKERRQ(STMatSolve(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* sigma=0.1 */
  sigma = 0.1;
  CHKERRQ(STSetShift(st,sigma));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  CHKERRQ(PCGetOperators(pc,NULL,&Pmat));
  CHKERRQ(MatView(Pmat,NULL));
  CHKERRQ(STMatSolve(st,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STDestroy(&st));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatDestroy(&Pa));
  CHKERRQ(MatDestroy(&Pb));
  CHKERRQ(MatDestroy(&Pc));
  CHKERRQ(MatDestroy(&Pd));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -st_type {{shift sinvert}separate output} -st_pc_type jacobi
      requires: !single

TEST*/
