/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests a GHEP problem with symmetric matrices.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A,B;        /* matrices */
  EPS            eps;        /* eigenproblem solver context */
  ST             st;
  KSP            ksp;
  PC             pc;
  PCType         pctype;
  PetscInt       N,n=45,m,Istart,Iend,II,i,j;
  PetscBool      flag;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized Symmetric Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
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
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,4.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,II,II,2.0/PetscLogScalar(II+2),INSERT_VALUES));
  }
  CHKERRQ(MatSetValue(B,0,1,0.4,INSERT_VALUES));
  CHKERRQ(MatSetValue(B,1,0,0.4,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,B));
  CHKERRQ(EPSSetProblemType(eps,EPS_GHEP));
  CHKERRQ(EPSSetFromOptions(eps));
  CHKERRQ(EPSSetUp(eps));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCGetType(pc,&pctype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Using %s for the PC\n",pctype));
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_BACKWARD,NULL));

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -n 18 -eps_nev 3 -st_type sinvert -eps_target 1.02

   test:
      suffix: 2
      args: -n 18 -eps_type ciss -rg_interval_endpoints 1.0,1.2
      requires: !single

   testset:
      nsize: {{1 4}}
      args: -n 8 -eps_nev 60 -st_pc_type redundant
      filter: grep -v Using
      requires: !single !valgrind
      output_file: output/test32_3.out
      test:
         suffix: 3
      test:
         suffix: 3_gnhep
         args: -eps_gen_non_hermitian

   testset:
      nsize: {{1 4}}
      args: -n 8 -eps_nev 64 -st_pc_type redundant
      filter: grep -v Using
      requires: !single
      output_file: output/test32_4.out
      test:
         suffix: 4
      test:
         suffix: 4_gnhep
         args: -eps_gen_non_hermitian

   testset:
      requires: !single
      args: -eps_tol 1e-10 -st_type sinvert -st_ksp_type preonly -st_pc_type cholesky -eps_interval .8,1.1 -eps_krylovschur_partitions 2
      output_file: output/test32_5.out
      nsize: 3
      filter: grep -v Using
      test:
         suffix: 5_redundant
         args: -st_pc_type redundant -st_redundant_pc_type cholesky
      test:
         suffix: 5_mumps
         requires: mumps !complex
         args: -st_pc_factor_mat_solver_type mumps -st_mat_mumps_icntl_13 1
      test:
         suffix: 5_superlu
         requires: superlu_dist
         args: -st_pc_factor_mat_solver_type superlu_dist -st_mat_superlu_dist_rowperm NOROWPERM
         timeoutfactor: 10

TEST*/
