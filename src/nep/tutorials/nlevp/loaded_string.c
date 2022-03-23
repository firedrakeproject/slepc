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

   The loaded_string problem is a rational eigenvalue problem for the
   finite element model of a loaded vibrating string.
*/

static char help[] = "Finite element model of a loaded vibrating string.\n\n"
  "The command line options are:\n"
  "  -n <n>, dimension of the matrices.\n"
  "  -kappa <kappa>, stiffness of elastic spring.\n"
  "  -mass <m>, mass of the attached load.\n\n";

#include <slepcnep.h>

#define NMAT 3

int main(int argc,char **argv)
{
  Mat            A[NMAT];         /* problem matrices */
  FN             f[NMAT];         /* functions to define the nonlinear operator */
  NEP            nep;             /* nonlinear eigensolver context */
  PetscInt       n=100,Istart,Iend,i;
  PetscReal      kappa=1.0,m=1.0;
  PetscScalar    sigma,numer[2],denom[2];
  PetscBool      terse;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mass",&m,NULL));
  sigma = kappa/m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Loaded vibrating string, n=%" PetscInt_FMT " kappa=%g m=%g\n\n",n,(double)kappa,(double)m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Build the problem matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* initialize matrices */
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[i]));
    CHKERRQ(MatSetSizes(A[i],PETSC_DECIDE,PETSC_DECIDE,n,n));
    CHKERRQ(MatSetFromOptions(A[i]));
    CHKERRQ(MatSetUp(A[i]));
  }
  CHKERRQ(MatGetOwnershipRange(A[0],&Istart,&Iend));

  /* A0 */
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[0],i,i,(i==n-1)?1.0*n:2.0*n,INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[0],i,i-1,-1.0*n,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A[0],i,i+1,-1.0*n,INSERT_VALUES));
  }

  /* A1 */
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[1],i,i,(i==n-1)?2.0/(6.0*n):4.0/(6.0*n),INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[1],i,i-1,1.0/(6.0*n),INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A[1],i,i+1,1.0/(6.0*n),INSERT_VALUES));
  }

  /* A2 */
  if (Istart<=n-1 && n-1<Iend) CHKERRQ(MatSetValue(A[2],n-1,n-1,kappa,INSERT_VALUES));

  /* assemble matrices */
  for (i=0;i<NMAT;i++) CHKERRQ(MatAssemblyBegin(A[i],MAT_FINAL_ASSEMBLY));
  for (i=0;i<NMAT;i++) CHKERRQ(MatAssemblyEnd(A[i],MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Create the problem functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* f1=1 */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[0]));
  CHKERRQ(FNSetType(f[0],FNRATIONAL));
  numer[0] = 1.0;
  CHKERRQ(FNRationalSetNumerator(f[0],1,numer));

  /* f2=-lambda */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[1]));
  CHKERRQ(FNSetType(f[1],FNRATIONAL));
  numer[0] = -1.0; numer[1] = 0.0;
  CHKERRQ(FNRationalSetNumerator(f[1],2,numer));

  /* f3=lambda/(lambda-sigma) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[2]));
  CHKERRQ(FNSetType(f[2],FNRATIONAL));
  numer[0] = 1.0; numer[1] = 0.0;
  denom[0] = 1.0; denom[1] = -sigma;
  CHKERRQ(FNRationalSetNumerator(f[2],2,numer));
  CHKERRQ(FNRationalSetDenominator(f[2],2,denom));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));
  CHKERRQ(NEPSetSplitOperator(nep,3,A,f,SUBSET_NONZERO_PATTERN));
  CHKERRQ(NEPSetProblemType(nep,NEP_RATIONAL));
  CHKERRQ(NEPSetFromOptions(nep));
  CHKERRQ(NEPSolve(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(NEPConvergedReasonView(nep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(NEPDestroy(&nep));
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatDestroy(&A[i]));
    CHKERRQ(FNDestroy(&f[i]));
  }
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -nep_type rii -nep_target 4 -terse
      requires: !single
      filter: sed -e "s/[+-]0\.0*i//g"

   testset:
      args: -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_interpolation_degree 12 -nep_refine simple -terse
      requires: !single
      output_file: output/loaded_string_2.out
      test:
         suffix: 2
         args: -nep_refine_scheme {{schur explicit}}
      test:
         suffix: 2_mbe
         args: -nep_refine_scheme mbe -nep_refine_ksp_type preonly -nep_refine_pc_type lu

   testset:
      nsize: 2
      args: -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_interpolation_degree 12 -nep_refine simple -nep_refine_partitions 2 -nep_interpol_st_ksp_type bcgs -nep_interpol_st_pc_type bjacobi -terse
      requires: !single
      output_file: output/loaded_string_2.out
      timeoutfactor: 2
      test:
         suffix: 3_explicit
         args: -nep_refine_scheme explicit
      test:
         suffix: 3_mbe
         args: -nep_refine_scheme mbe -nep_refine_ksp_type preonly -nep_refine_pc_type cholesky

   test:
      suffix: 4
      nsize: 4
      args: -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_interpolation_degree 10 -nep_refine simple -nep_refine_partitions 2 -nep_refine_scheme explicit -nep_interpol_st_ksp_type bcgs -nep_interpol_st_pc_type bjacobi -terse -log_exclude nep,pep,fn
      requires: !single
      output_file: output/loaded_string_2.out
      timeoutfactor: 4

   test:
      suffix: 5
      args: -nep_type nleigs -rg_type interval -rg_interval_endpoints 4,700,-.1,.1 -nep_nev 8 -nep_target 5 -terse
      filter: sed -e "s/[+-]0\.0*i//g"
      requires: !single

   test:
      suffix: 6
      args: -nep_type nleigs -rg_type interval -rg_interval_endpoints 100,700 -nep_nev 5 -nep_tol 1e-9 -nep_target 140 -nep_nleigs_interpolation_degree 15 -nep_general -terse
      requires: !complex !single

   test:
      suffix: 6_complex
      args: -nep_type nleigs -rg_type interval -rg_interval_endpoints 100,700,-.1,.1 -nep_nev 5 -nep_tol 1e-9 -nep_target 140 -nep_nleigs_interpolation_degree 15 -nep_general -terse
      filter: sed -e "s/[+-]0\.0*i//g"
      requires: complex !single
      output_file: output/loaded_string_6.out

   test:
      suffix: 7
      args: -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700 -nep_nev 5 -nep_target 100 -nep_interpol_interpolation_degree 20 -nep_ncv 20 -n 20 -nep_refine simple -nep_refine_its 1 -terse
      requires: !complex double

   test:
      suffix: 7_complex
      args: -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 5 -nep_target 100 -nep_interpol_interpolation_degree 20 -nep_ncv 20 -n 20 -nep_refine simple -nep_refine_its 1 -terse
      requires: complex double
      output_file: output/loaded_string_7.out

   testset:
      args: -nep_target 10 -nep_nev 3 -nep_tol 5e-10 -terse
      requires: !single
      output_file: output/loaded_string_8.out
      filter: sed -e "s/[+-]0\.0*i//g"
      test:
         suffix: 8
         args: -nep_type {{rii slp narnoldi}}
      test:
         suffix: 8_rii_thres
         args: -nep_type rii -nep_rii_deflation_threshold 5e-10
      test:
         suffix: 8_slp_thres
         args: -nep_type slp -nep_slp_deflation_threshold 5e-10

      test:
         suffix: 8_slp_two_thres
         args: -nep_type slp -nep_slp_deflation_threshold 5e-10 -nep_two_sided

   test:
      suffix: 9
      args: -nep_type ciss -rg_type ellipse -rg_ellipse_center 500 -rg_ellipse_radius 500 -rg_ellipse_vscale .1 -nep_ciss_moments 4 -nep_ciss_blocksize 5 -terse
      requires: complex double

TEST*/
