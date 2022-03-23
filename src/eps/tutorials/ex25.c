/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Spectrum slicing on generalized symmetric eigenproblem.\n\n"
  "The problem is similar to ex13.c.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A,B;         /* matrices */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  KSP            ksp;
  PC             pc;
  EPSType        type;
  PetscInt       N,n=10,m,Istart,Iend,II,nev,i,j,*inertias,ns;
  PetscReal      inta,intb,*shifts;
  PetscBool      flag,show=PETSC_FALSE,terse;
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_MUMPS) && !defined(PETSC_USE_COMPLEX)
  Mat            F;
#endif

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-show_inertias",&show,NULL));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum slicing on GHEP, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

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
    CHKERRQ(MatSetValue(B,II,II,4.0,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
     Set operators and set problem type
  */
  CHKERRQ(EPSSetOperators(eps,A,B));
  CHKERRQ(EPSSetProblemType(eps,EPS_GHEP));

  /*
     Set interval for spectrum slicing
  */
  inta = 0.1;
  intb = 0.2;
  CHKERRQ(EPSSetInterval(eps,inta,intb));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_ALL));

  /*
     Spectrum slicing requires Krylov-Schur
  */
  CHKERRQ(EPSSetType(eps,EPSKRYLOVSCHUR));

  /*
     Set shift-and-invert with Cholesky; select MUMPS if available
  */

  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STSetType(st,STSINVERT));
  CHKERRQ(EPSKrylovSchurGetKSP(eps,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCCHOLESKY));

  /*
     Use MUMPS if available.
     Note that in complex scalars we cannot use MUMPS for spectrum slicing,
     because MatGetInertia() is not available in that case.
  */
#if defined(PETSC_HAVE_MUMPS) && !defined(PETSC_USE_COMPLEX)
  CHKERRQ(EPSKrylovSchurSetDetectZeros(eps,PETSC_TRUE));  /* enforce zero detection */
  CHKERRQ(PCFactorSetMatSolverType(pc,MATSOLVERMUMPS));
  CHKERRQ(PCFactorSetUpMatSolverType(pc));
  /*
     Set several MUMPS options, the corresponding command-line options are:
     '-st_mat_mumps_icntl_13 1': turn off ScaLAPACK for matrix inertia
     '-st_mat_mumps_icntl_24 1': detect null pivots in factorization (for the case that a shift is equal to an eigenvalue)
     '-st_mat_mumps_cntl_3 <tol>': a tolerance used for null pivot detection (must be larger than machine epsilon)

     Note: depending on the interval, it may be necessary also to increase the workspace:
     '-st_mat_mumps_icntl_14 <percentage>': increase workspace with a percentage (50, 100 or more)
  */
  CHKERRQ(PCFactorGetMatrix(pc,&F));
  CHKERRQ(MatMumpsSetIcntl(F,13,1));
  CHKERRQ(MatMumpsSetIcntl(F,24,1));
  CHKERRQ(MatMumpsSetCntl(F,3,1e-12));
#endif

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSSetUp(eps));
  if (show) {
    CHKERRQ(EPSKrylovSchurGetInertias(eps,&ns,&shifts,&inertias));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Subintervals (after setup):\n"));
    for (i=0;i<ns;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Shift %g  Inertia %" PetscInt_FMT " \n",(double)shifts[i],inertias[i]));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    CHKERRQ(PetscFree(shifts));
    CHKERRQ(PetscFree(inertias));
  }
  CHKERRQ(EPSSolve(eps));
  if (show) {
    CHKERRQ(EPSKrylovSchurGetInertias(eps,&ns,&shifts,&inertias));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"All shifts (after solve):\n"));
    for (i=0;i<ns;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Shift %g  Inertia %" PetscInt_FMT " \n",(double)shifts[i],inertias[i]));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    CHKERRQ(PetscFree(shifts));
    CHKERRQ(PetscFree(inertias));
  }

  /*
     Show eigenvalues in interval and print solution
  */
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(EPSGetInterval(eps,&inta,&intb));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT " eigenvalues found in [%g, %g]\n",nev,(double)inta,(double)intb));

  /*
     Show detailed info unless -terse option is given by user
   */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -terse
      test:
         requires: !mumps
      test:
         requires: mumps !complex

TEST*/
