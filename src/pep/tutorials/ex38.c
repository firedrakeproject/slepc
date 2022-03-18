/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Spectrum slicing on quadratic symmetric eigenproblem.\n\n"
  "The command line options are:\n"
  "  -n <n> ... dimension of the matrices.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            M,C,K,A[3]; /* problem matrices */
  PEP            pep;        /* polynomial eigenproblem solver context */
  ST             st;         /* spectral transformation context */
  KSP            ksp;
  PC             pc;
  PEPType        type;
  PetscBool      show=PETSC_FALSE,terse;
  PetscInt       n=100,Istart,Iend,nev,i,*inertias,ns;
  PetscReal      mu=1,tau=10,kappa=5,inta,intb,*shifts;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-show_inertias",&show,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum slicing on PEP, n=%" PetscInt_FMT "\n\n",n));

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
    if (i>0) {
      CHKERRQ(MatSetValue(K,i,i-1,-kappa,INSERT_VALUES));
    }
    CHKERRQ(MatSetValue(K,i,i,kappa*3.0,INSERT_VALUES));
    if (i<n-1) {
      CHKERRQ(MatSetValue(K,i,i+1,-kappa,INSERT_VALUES));
    }
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
    if (i>0) {
      CHKERRQ(MatSetValue(C,i,i-1,-tau,INSERT_VALUES));
    }
    CHKERRQ(MatSetValue(C,i,i,tau*3.0,INSERT_VALUES));
    if (i<n-1) {
      CHKERRQ(MatSetValue(C,i,i+1,-tau,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* M is a diagonal matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&M));
  CHKERRQ(MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(M));
  CHKERRQ(MatSetUp(M));
  CHKERRQ(MatGetOwnershipRange(M,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(M,i,i,mu,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));

  /*
     Set operators and set problem type
  */
  A[0] = K; A[1] = C; A[2] = M;
  CHKERRQ(PEPSetOperators(pep,3,A));
  CHKERRQ(PEPSetProblemType(pep,PEP_HYPERBOLIC));

  /*
     Set interval for spectrum slicing
  */
  inta = -11.3;
  intb = -9.5;
  CHKERRQ(PEPSetInterval(pep,inta,intb));
  CHKERRQ(PEPSetWhichEigenpairs(pep,PEP_ALL));

  /*
     Spectrum slicing requires STOAR
  */
  CHKERRQ(PEPSetType(pep,PEPSTOAR));

  /*
     Set shift-and-invert with Cholesky; select MUMPS if available
  */
  CHKERRQ(PEPGetST(pep,&st));
  CHKERRQ(STSetType(st,STSINVERT));

  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCCHOLESKY));

  /*
     Use MUMPS if available.
     Note that in complex scalars we cannot use MUMPS for spectrum slicing,
     because MatGetInertia() is not available in that case.
  */
#if defined(PETSC_HAVE_MUMPS) && !defined(PETSC_USE_COMPLEX)
  CHKERRQ(PEPSTOARSetDetectZeros(pep,PETSC_TRUE));  /* enforce zero detection */
  CHKERRQ(PCFactorSetMatSolverType(pc,MATSOLVERMUMPS));
  /*
     Add several MUMPS options (see ex43.c for a better way of setting them in program):
     '-st_mat_mumps_icntl_13 1': turn off ScaLAPACK for matrix inertia
     '-st_mat_mumps_icntl_24 1': detect null pivots in factorization (for the case that a shift is equal to an eigenvalue)
     '-st_mat_mumps_cntl_3 <tol>': a tolerance used for null pivot detection (must be larger than machine epsilon)

     Note: depending on the interval, it may be necessary also to increase the workspace:
     '-st_mat_mumps_icntl_14 <percentage>': increase workspace with a percentage (50, 100 or more)
  */
  CHKERRQ(PetscOptionsInsertString(NULL,"-st_mat_mumps_icntl_13 1 -st_mat_mumps_icntl_24 1 -st_mat_mumps_cntl_3 1e-12"));
#endif

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PEPSetUp(pep));
  if (show) {
    CHKERRQ(PEPSTOARGetInertias(pep,&ns,&shifts,&inertias));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Subintervals (after setup):\n"));
    for (i=0;i<ns;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Shift %g  Inertia %" PetscInt_FMT " \n",(double)shifts[i],inertias[i]));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    CHKERRQ(PetscFree(shifts));
    CHKERRQ(PetscFree(inertias));
  }
  CHKERRQ(PEPSolve(pep));
  if (show && !terse) {
    CHKERRQ(PEPSTOARGetInertias(pep,&ns,&shifts,&inertias));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"All shifts (after solve):\n"));
    for (i=0;i<ns;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Shift %g  Inertia %" PetscInt_FMT " \n",(double)shifts[i],inertias[i]));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    CHKERRQ(PetscFree(shifts));
    CHKERRQ(PetscFree(inertias));
  }

  /*
     Show eigenvalues in interval and print solution
  */
  CHKERRQ(PEPGetType(pep,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(PEPGetDimensions(pep,&nev,NULL,NULL));
  CHKERRQ(PEPGetInterval(pep,&inta,&intb));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT " eigenvalues found in [%g, %g]\n",nev,(double)inta,(double)intb));

  /*
     Show detailed info unless -terse option is given by user
   */
  if (terse) {
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  } else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PEPDestroy(&pep));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&K));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      requires: !single
      args: -show_inertias -terse
      output_file: output/ex38_1.out
      test:
         suffix: 1
         requires: !complex
      test:
         suffix: 1_complex
         requires: complex !mumps

   test:
      suffix: 2
      args: -pep_interval 0,2

TEST*/
