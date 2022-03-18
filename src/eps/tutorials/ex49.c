/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "User-defined split preconditioner when solving a generalized eigenproblem.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A,B,A0,B0,mats[2]; /* problem matrices and sparser approximations */
  EPS            eps;               /* eigenproblem solver context */
  ST             st;
  PetscInt       N,n=24,m,Istart,Iend,II,i,j;
  PetscBool      flag,terse;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGHEP with split preconditioner, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Compute the problem matrices A and B
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-0.2,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,-0.2,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-3.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-3.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,7.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,II,II,2.0,INSERT_VALUES));
  }
  if (Istart==0) {
    CHKERRQ(MatSetValue(B,0,0,6.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,0,1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,1,0,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,1,1,1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Compute sparser approximations A0 and B0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A0));
  CHKERRQ(MatSetSizes(A0,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A0));
  CHKERRQ(MatSetUp(A0));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B0));
  CHKERRQ(MatSetSizes(B0,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(B0));
  CHKERRQ(MatSetUp(B0));

  CHKERRQ(MatGetOwnershipRange(A0,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (j>0) CHKERRQ(MatSetValue(A0,II,II-1,-3.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A0,II,II+1,-3.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A0,II,II,7.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B0,II,II,2.0,INSERT_VALUES));
  }
  if (Istart==0) {
    CHKERRQ(MatSetValue(B0,0,0,6.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B0,1,1,1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A0,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A0,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B0,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B0,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,B));
  CHKERRQ(EPSSetProblemType(eps,EPS_GHEP));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STSetType(st,STSINVERT));
  mats[0] = A0; mats[1] = B0;
  CHKERRQ(STSetSplitPreconditioner(st,2,mats,SUBSET_NONZERO_PATTERN));
  CHKERRQ(EPSSetTarget(eps,0.0));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  } else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A0));
  CHKERRQ(MatDestroy(&B0));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -eps_nev 4 -terse
      output_file: output/ex49_1.out
      requires: !single
      test:
         suffix: 1
      test:
         suffix: 1_jd
         args: -eps_type jd -st_type precond
      test:
         suffix: 1_lobpcg
         args: -eps_type lobpcg -st_type precond -eps_smallest_real -st_shift 0.2

   testset:
      args: -eps_type ciss -eps_all -rg_type ellipse -rg_ellipse_center 0 -rg_ellipse_radius 0.34 -rg_ellipse_vscale .2 -terse
      output_file: output/ex49_2.out
      test:
         suffix: 2
      test:
         suffix: 2_nost
         args: -eps_ciss_usest 0
         requires: !single
      test:
         suffix: 2_par
         nsize: 2
         args: -eps_ciss_partitions 2
         requires: !single

TEST*/
