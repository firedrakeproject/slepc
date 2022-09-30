/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test RSVD on a low-rank matrix.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat            A,Ur,Vr;
  SVD            svd;
  PetscInt       m=80,n=40,rank=3,Istart,Iend,i,j;
  PetscScalar    *u;
  PetscReal      tol=PETSC_SMALL;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-rank",&rank,NULL));
  PetscCheck(rank>0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"The rank must be >=1");
  PetscCheck(rank<PetscMin(m,n),PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"The rank must be <min(m,n)");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSVD of low-rank matrix, size %" PetscInt_FMT "x%" PetscInt_FMT ", rank %" PetscInt_FMT "\n\n",m,n,rank));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create a low-rank matrix A = Ur*Vr'
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Ur));
  PetscCall(MatSetSizes(Ur,PETSC_DECIDE,PETSC_DECIDE,m,rank));
  PetscCall(MatSetType(Ur,MATDENSE));
  PetscCall(MatSetUp(Ur));
  PetscCall(MatGetOwnershipRange(Ur,&Istart,&Iend));
  PetscCall(MatDenseGetArray(Ur,&u));
  for (i=Istart;i<Iend;i++) {
    if (i<m/2) u[i-Istart] = 1.0;
    if (i==0) u[i+Iend-2*Istart] = -2.0;
    if (i==1) u[i+Iend-2*Istart] = -1.0;
    if (i==2) u[i+Iend-2*Istart] = -1.0;
    for (j=2;j<rank;j++) if (i==j) u[j+j*Iend-(j+1)*Istart] = j;
  }
  PetscCall(MatDenseRestoreArray(Ur,&u));
  PetscCall(MatAssemblyBegin(Ur,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Ur,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&Vr));
  PetscCall(MatSetSizes(Vr,PETSC_DECIDE,PETSC_DECIDE,n,rank));
  PetscCall(MatSetType(Vr,MATDENSE));
  PetscCall(MatSetUp(Vr));
  PetscCall(MatGetOwnershipRange(Vr,&Istart,&Iend));
  PetscCall(MatDenseGetArray(Vr,&u));
  for (i=Istart;i<Iend;i++) {
    if (i>n/2) u[i-Istart] = 1.0;
    if (i==0) u[i+Iend-2*Istart] = 1.0;
    if (i==1) u[i+Iend-2*Istart] = 2.0;
    if (i==2) u[i+Iend-2*Istart] = 2.0;
    for (j=2;j<rank;j++) if (i==j) u[j+j*Iend-(j+1)*Istart] = j;
  }
  PetscCall(MatDenseRestoreArray(Vr,&u));
  PetscCall(MatAssemblyBegin(Vr,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Vr,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateLRC(NULL,Ur,NULL,Vr,&A));
  PetscCall(MatDestroy(&Ur));
  PetscCall(MatDestroy(&Vr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Create the SVD solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));
  PetscCall(SVDSetOperators(svd,A,NULL));
  PetscCall(SVDSetType(svd,SVDRANDOMIZED));
  PetscCall(SVDSetDimensions(svd,2,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(SVDSetConvergenceTest(svd,SVD_CONV_MAXIT));
  PetscCall(SVDSetTolerances(svd,tol,1));   /* maxit=1 to disable outer iteration */
  PetscCall(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Compute the singular triplets and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDSolve(svd));
  PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      args: -bv_orthog_block tsqr    # currently fail with other block orthogonalization methods
      requires: !single

TEST*/
