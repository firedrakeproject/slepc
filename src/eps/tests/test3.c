/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to EPSSolve with different matrix.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A1,A2;       /* problem matrices */
  EPS            eps;         /* eigenproblem solver context */
  PetscReal      tol=PETSC_SMALL,v;
  Vec            d;
  PetscInt       n=30,i,Istart,Iend;
  PetscRandom    myrand;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nTridiagonal with random diagonal, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Create matrix tridiag([-1 0 -1])
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A1));
  CHKERRQ(MatSetSizes(A1,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A1));
  CHKERRQ(MatSetUp(A1));

  CHKERRQ(MatGetOwnershipRange(A1,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A1,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A1,i,i+1,-1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A1,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A1,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create two matrices by filling the diagonal with rand values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDuplicate(A1,MAT_COPY_VALUES,&A2));
  CHKERRQ(MatCreateVecs(A1,NULL,&d));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&myrand));
  CHKERRQ(PetscRandomSetFromOptions(myrand));
  CHKERRQ(PetscRandomSetInterval(myrand,0.0,1.0));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(PetscRandomGetValueReal(myrand,&v));
    CHKERRQ(VecSetValue(d,i,v,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(d));
  CHKERRQ(VecAssemblyEnd(d));
  CHKERRQ(MatDiagonalSet(A1,d,INSERT_VALUES));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(PetscRandomGetValueReal(myrand,&v));
    CHKERRQ(VecSetValue(d,i,v,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(d));
  CHKERRQ(VecAssemblyEnd(d));
  CHKERRQ(MatDiagonalSet(A2,d,INSERT_VALUES));
  CHKERRQ(VecDestroy(&d));
  CHKERRQ(PetscRandomDestroy(&myrand));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Create the eigensolver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  CHKERRQ(EPSSetOperators(eps,A1,NULL));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve first eigenproblem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," - - - First matrix - - -\n"));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve second eigenproblem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSSetOperators(eps,A2,NULL));
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," - - - Second matrix - - -\n"));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A1));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -eps_nev 4
      requires: !single
      output_file: output/test3_1.out
      test:
         suffix: 1
         args: -eps_type {{krylovschur subspace arnoldi lapack}}
      test:
         suffix: 1_lanczos
         args: -eps_type lanczos -eps_lanczos_reorthog local
      test:
         suffix: 1_power
         args: -eps_type power -eps_max_it 20000
      test:
         suffix: 1_jd
         args: -eps_type jd -eps_jd_initial_size 7
      test:
         suffix: 1_gd
         args: -eps_type gd -eps_gd_initial_size 7
      test:
         suffix: 1_gd2
         args: -eps_type gd -eps_gd_double_expansion
      test:
         suffix: 1_arpack
         args: -eps_type arpack
         requires: arpack
      test:
         suffix: 1_primme
         args: -eps_type primme -eps_conv_abs -eps_primme_blocksize 4
         requires: primme
      test:
         suffix: 1_trlan
         args: -eps_type trlan
         requires: trlan
      test:
         suffix: 1_scalapack
         args: -eps_type scalapack
         requires: scalapack
      test:
         suffix: 1_elpa
         args: -eps_type elpa
         requires: elpa
      test:
         suffix: 1_elemental
         args: -eps_type elemental
         requires: elemental

   testset:
      args: -eps_nev 4 -eps_smallest_real -eps_max_it 500
      output_file: output/test3_2.out
      test:
         suffix: 2_rqcg
         args: -eps_type rqcg -eps_rqcg_reset 5 -eps_ncv 32
      test:
         suffix: 2_lobpcg
         args: -eps_type lobpcg -eps_lobpcg_blocksize 5 -st_pc_type none
      test:
         suffix: 2_lanczos
         args: -eps_type lanczos -eps_lanczos_reorthog local
         requires: !single
      test:
         suffix: 2_lanczos_delayed
         args: -eps_type lanczos -eps_lanczos_reorthog delayed -eps_tol 1e-8
         requires: !single
      test:
         suffix: 2_trlan
         args: -eps_type trlan
         requires: trlan
      test:
         suffix: 2_blopex
         args: -eps_type blopex -eps_conv_abs -st_shift -2
         requires: blopex

TEST*/
