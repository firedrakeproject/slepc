/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTest ST with four matrices, n=%" PetscInt_FMT "\n\n",n));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&D));
  PetscCall(MatSetSizes(D,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(D));
  PetscCall(MatSetUp(D));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    PetscCall(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (i>0) {
      PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
      PetscCall(MatSetValue(B,i,i,(PetscScalar)i,INSERT_VALUES));
    } else PetscCall(MatSetValue(B,i,i,-1.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(C,i,n-i-1,1.0,INSERT_VALUES));
    PetscCall(MatSetValue(D,i,i,i*.1,INSERT_VALUES));
    if (i==0) PetscCall(MatSetValue(D,0,n-1,1.0,INSERT_VALUES));
    if (i==n-1) PetscCall(MatSetValue(D,n-1,0,1.0,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A,&v,&w));
  PetscCall(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the split preconditioner matrices (four diagonals)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Pa));
  PetscCall(MatSetSizes(Pa,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(Pa));
  PetscCall(MatSetUp(Pa));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Pb));
  PetscCall(MatSetSizes(Pb,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(Pb));
  PetscCall(MatSetUp(Pb));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Pc));
  PetscCall(MatSetSizes(Pc,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(Pc));
  PetscCall(MatSetUp(Pc));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Pd));
  PetscCall(MatSetSizes(Pd,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(Pd));
  PetscCall(MatSetUp(Pd));

  PetscCall(MatGetOwnershipRange(Pa,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    PetscCall(MatSetValue(Pa,i,i,2.0,INSERT_VALUES));
    if (i>0) PetscCall(MatSetValue(Pb,i,i,(PetscScalar)i,INSERT_VALUES));
    else PetscCall(MatSetValue(Pb,i,i,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(Pd,i,i,i*.1,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(Pa,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pa,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Pb,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pb,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Pc,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pc,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Pd,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pd,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  mat[1] = B;
  mat[2] = C;
  mat[3] = D;
  PetscCall(STSetMatrices(st,4,mat));
  mat[0] = Pa;
  mat[1] = Pb;
  mat[2] = Pc;
  mat[3] = Pd;
  PetscCall(STSetSplitPreconditioner(st,4,mat,SUBSET_NONZERO_PATTERN));
  PetscCall(STGetKSP(st,&ksp));
  PetscCall(KSPSetTolerances(ksp,100*PETSC_MACHINE_EPSILON,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(STSetTransform(st,PETSC_TRUE));
  PetscCall(STSetFromOptions(st));
  PetscCall(STGetKSP(st,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Apply the operator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* sigma=0.0 */
  PetscCall(STSetUp(st));
  PetscCall(STGetType(st,&type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  PetscCall(PCGetOperators(pc,NULL,&Pmat));
  PetscCall(MatView(Pmat,NULL));
  PetscCall(STMatSolve(st,v,w));
  PetscCall(VecView(w,NULL));

  /* sigma=0.1 */
  sigma = 0.1;
  PetscCall(STSetShift(st,sigma));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  PetscCall(PCGetOperators(pc,NULL,&Pmat));
  PetscCall(MatView(Pmat,NULL));
  PetscCall(STMatSolve(st,v,w));
  PetscCall(VecView(w,NULL));

  PetscCall(STDestroy(&st));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&Pa));
  PetscCall(MatDestroy(&Pb));
  PetscCall(MatDestroy(&Pc));
  PetscCall(MatDestroy(&Pd));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -st_type {{shift sinvert}separate output} -st_pc_type jacobi
      requires: !single

TEST*/
