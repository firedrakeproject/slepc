/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to EPSSolve with the same matrix.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;           /* problem matrix */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;
  PetscReal      tol=PetscMax(1000*PETSC_MACHINE_EPSILON,1e-9);
  PetscInt       n=30,i,Istart,Iend;
  PetscBool      flg;
  PetscErrorCode ierr;
  EPSLanczosReorthogType reorth;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Create the eigensolver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  CHKERRQ(EPSSetFromOptions(eps));

  /* illustrate how to extract parameters from specific solver types */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps,EPSLANCZOS,&flg));
  if (flg) {
    CHKERRQ(EPSLanczosGetReorthog(eps,&reorth));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Reorthogonalization type used in Lanczos: %s\n",EPSLanczosReorthogTypes[reorth]));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Solve for largest eigenvalues
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL));
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," - - - Largest eigenvalues - - -\n"));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Solve for smallest eigenvalues
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," - - - Smallest eigenvalues - - -\n"));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Solve for interior eigenvalues (target=2.1)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE));
  CHKERRQ(EPSSetTarget(eps,2.1));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps,EPSLANCZOS,&flg));
  if (flg) {
    CHKERRQ(EPSGetST(eps,&st));
    CHKERRQ(STSetType(st,STSINVERT));
  } else {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)eps,EPSKRYLOVSCHUR,&flg));
    if (!flg) {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)eps,EPSARNOLDI,&flg));
    }
    if (flg) {
      CHKERRQ(EPSSetExtraction(eps,EPS_HARMONIC));
    }
  }
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," - - - Interior eigenvalues - - -\n"));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -eps_nev 4
      requires: !single
      output_file: output/test2_1.out
      test:
         suffix: 1
         args: -eps_type {{arnoldi gd jd lapack}}
      test:
         suffix: 1_gd2
         args: -eps_type gd -eps_gd_double_expansion
         timeoutfactor: 2
      test:
         suffix: 1_krylovschur
         args: -eps_type krylovschur -eps_krylovschur_locking {{0 1}}
      test:
         suffix: 1_scalapack
         requires: scalapack
         args: -eps_type scalapack
      test:
         suffix: 1_elpa
         requires: elpa
         args: -eps_type elpa
      test:
         suffix: 1_elemental
         requires: elemental
         args: -eps_type elemental

   testset:
      args: -eps_type lanczos -eps_nev 4
      requires: !single
      filter: grep -v "Lanczos"
      output_file: output/test2_1.out
      test:
         suffix: 2
         args: -eps_lanczos_reorthog {{local full selective periodic partial}}

   testset:
      args: -n 32 -eps_nev 4
      requires: !single
      output_file: output/test2_3.out
      test:
         nsize: 2
         suffix: 3
         args: -eps_type {{krylovschur lapack}}
      test:
         nsize: 2
         suffix: 3_gd
         args: -eps_type gd -eps_gd_krylov_start
         timeoutfactor: 2
      test:
         suffix: 3_jd
         args: -eps_type jd -eps_jd_krylov_start -eps_ncv 18

   testset:
      args: -eps_nev 4 -mat_type aijcusparse
      requires: cuda !single
      output_file: output/test2_1.out
      test:
         suffix: 4_cuda
         args: -eps_type {{krylovschur arnoldi gd jd}}

TEST*/
