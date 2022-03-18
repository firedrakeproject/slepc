/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "SVD via the cross-product matrix with a user-provided EPS.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = matrix rows.\n"
  "  -n <n>, where <n> = matrix columns (defaults to m+2).\n\n";

#include <slepcsvd.h>

/*
   This example computes the singular values of a rectangular bidiagonal matrix

              |  1  2                     |
              |     1  2                  |
              |        1  2               |
          A = |          .  .             |
              |             .  .          |
              |                1  2       |
              |                   1  2    |
 */

int main(int argc,char **argv)
{
  Mat            A;
  SVD            svd;
  EPS            eps;
  ST             st;
  KSP            ksp;
  PC             pc;
  PetscInt       m=20,n,Istart,Iend,i,col[2];
  PetscScalar    value[] = { 1, 2 };
  PetscBool      flg,expmat;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n=m+2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nRectangular bidiagonal matrix, m=%" PetscInt_FMT " n=%" PetscInt_FMT "\n\n",m,n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) {
      CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    } else if (i==n-1) {
      CHKERRQ(MatSetValue(A,i,col[0],value[0],INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create a standalone EPS with appropriate settings
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STSetType(st,STSINVERT));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPBCGS));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCJACOBI));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A,NULL));
  CHKERRQ(SVDSetType(svd,SVDCROSS));
  CHKERRQ(SVDCrossSetEPS(svd,eps));
  CHKERRQ(SVDSetWhichSingularTriplets(svd,SVD_SMALLEST));
  CHKERRQ(SVDSetFromOptions(svd));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)svd,SVDCROSS,&flg));
  if (flg) {
    CHKERRQ(SVDCrossGetExplicitMatrix(svd,&expmat));
    if (expmat) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Using explicit matrix with cross solver\n"));
    }
  }
  CHKERRQ(SVDSolve(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test6_1.out
      test:
         suffix: 1_subspace
         args: -eps_type subspace
      test:
         suffix: 1_lobpcg
         args: -eps_type lobpcg -st_type precond
      test:
         suffix: 2_cuda
         args: -eps_type subspace -mat_type aijcusparse
         requires: cuda

TEST*/
