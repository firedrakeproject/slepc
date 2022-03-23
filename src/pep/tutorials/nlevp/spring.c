/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This example implements one of the problems found at
       NLEVP: A Collection of Nonlinear Eigenvalue Problems,
       The University of Manchester.
   The details of the collection can be found at:
       [1] T. Betcke et al., "NLEVP: A Collection of Nonlinear Eigenvalue
           Problems", ACM Trans. Math. Software 39(2), Article 7, 2013.

   The spring problem is a QEP from the finite element model of a damped
   mass-spring system. This implementation supports only scalar parameters,
   that is all masses, dampers and springs have the same constants.
   Furthermore, this implementation does not consider different constants
   for dampers and springs connecting adjacent masses or masses to the ground.
*/

static char help[] = "FEM model of a damped mass-spring system.\n\n"
  "The command line options are:\n"
  "  -n <n> ... dimension of the matrices.\n"
  "  -mu <value> ... mass (default 1).\n"
  "  -tau <value> ... damping constant of the dampers (default 10).\n"
  "  -kappa <value> ... damping constant of the springs (default 5).\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            M,C,K,A[3];      /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PetscInt       n=5,Istart,Iend,i;
  PetscReal      mu=1.0,tau=10.0,kappa=5.0;
  PetscBool      terse;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mu",&mu,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nDamped mass-spring system, n=%" PetscInt_FMT " mu=%g tau=%g kappa=%g\n\n",n,(double)mu,(double)tau,(double)kappa));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is a tridiagonal */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&K));
  CHKERRQ(MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(K));
  CHKERRQ(MatSetUp(K));

  CHKERRQ(MatGetOwnershipRange(K,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(K,i,i-1,-kappa,INSERT_VALUES));
    CHKERRQ(MatSetValue(K,i,i,kappa*3.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(K,i,i+1,-kappa,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY));

  /* C is a tridiagonal */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  CHKERRQ(MatGetOwnershipRange(C,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(C,i,i-1,-tau,INSERT_VALUES));
    CHKERRQ(MatSetValue(C,i,i,tau*3.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(C,i,i+1,-tau,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* M is a diagonal matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&M));
  CHKERRQ(MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(M));
  CHKERRQ(MatSetUp(M));
  CHKERRQ(MatGetOwnershipRange(M,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(M,i,i,mu,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  CHKERRQ(PEPSetOperators(pep,3,A));
  CHKERRQ(PEPSetFromOptions(pep));
  CHKERRQ(PEPSolve(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(PEPDestroy(&pep));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&K));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -pep_nev 4 -n 24 -pep_ncv 18 -pep_target -.5 -st_type sinvert -pep_scale diagonal -terse
      output_file: output/spring_1.out
      filter: sed -e "s/[+-]0\.0*i//g"
      test:
         suffix: 1
         args: -pep_type {{toar linear}} -pep_conv_norm
      test:
         suffix: 1_stoar
         args: -pep_type stoar -pep_hermitian -pep_conv_rel
      test:
         suffix: 1_qarnoldi
         args: -pep_type qarnoldi -pep_conv_rel
      test:
         suffix: 1_cuda
         args: -mat_type aijcusparse
         requires: cuda

   test:
      suffix: 2
      args: -pep_type jd -pep_jd_minimality_index 1 -pep_nev 4 -n 24 -pep_ncv 18 -pep_target -50 -terse
      requires: !single
      filter: sed -e "s/[+-]0\.0*i//g"

   test:
      suffix: 3
      args: -n 300 -pep_hermitian -pep_interval -10.1,-9.5 -pep_type stoar -st_type sinvert -st_pc_type cholesky -terse
      filter: sed -e "s/52565/52566/" | sed -e "s/90758/90759/"

   test:
      suffix: 4
      args: -n 300 -pep_hyperbolic -pep_interval -9.6,-.527 -pep_type stoar -st_type sinvert -st_pc_type cholesky -terse
      requires: !single
      timeoutfactor: 2

   test:
      suffix: 5
      args: -n 300 -pep_hyperbolic -pep_interval -.506,-.3 -pep_type stoar -st_type sinvert -st_pc_type cholesky -pep_stoar_nev 11 -terse
      requires: !single

   test:
      suffix: 6
      args: -n 24 -pep_ncv 18 -pep_target -.5 -terse -pep_type jd -pep_jd_restart .6 -pep_jd_fix .001
      requires: !single

TEST*/
