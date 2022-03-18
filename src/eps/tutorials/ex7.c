/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a generalized eigensystem Ax=kBx with matrices loaded from a file.\n"
  "The command line options are:\n"
  "  -f1 <filename> -f2 <filename>, PETSc binary files containing A and B.\n"
  "  -evecs <filename>, output file to save computed eigenvectors.\n"
  "  -ninitial <nini>, number of user-provided initial guesses.\n"
  "  -finitial <filename>, binary file containing <nini> vectors.\n"
  "  -nconstr <ncon>, number of user-provided constraints.\n"
  "  -fconstr <filename>, binary file containing <ncon> vectors.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A,B;             /* matrices */
  EPS            eps;             /* eigenproblem solver context */
  ST             st;
  KSP            ksp;
  EPSType        type;
  PetscReal      tol;
  Vec            xr,xi,*Iv,*Cv;
  PetscInt       nev,maxit,i,its,lits,nconv,nini=0,ncon=0;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,evecs,ishermitian,terse;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized eigenproblem stored in file.\n\n"));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f1",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n"));
#else
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n"));
#endif
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f2",filename,sizeof(filename),&flg));
  if (flg) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatLoad(B,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=I\n\n"));
    B = NULL;
  }

  CHKERRQ(MatCreateVecs(A,NULL,&xr));
  CHKERRQ(MatCreateVecs(A,NULL,&xi));

  /*
     Read user constraints if available
  */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nconstr",&ncon,&flg));
  if (flg) {
    PetscCheck(ncon>0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"The number of constraints must be >0");
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fconstr",filename,sizeof(filename),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must specify the name of the file storing the constraints");
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(VecDuplicateVecs(xr,ncon,&Cv));
    for (i=0;i<ncon;i++) {
      CHKERRQ(VecLoad(Cv[i],viewer));
    }
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  /*
     Read initial guesses if available
  */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ninitial",&nini,&flg));
  if (flg) {
    PetscCheck(nini>0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"The number of initial vectors must be >0");
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-finitial",filename,sizeof(filename),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must specify the name of the file containing the initial vectors");
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(VecDuplicateVecs(xr,nini,&Iv));
    for (i=0;i<nini;i++) {
      CHKERRQ(VecLoad(Iv[i],viewer));
    }
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
     Set operators. In this case, it is a generalized eigenvalue problem
  */
  CHKERRQ(EPSSetOperators(eps,A,B));

  /*
     If the user provided initial guesses or constraints, pass them here
  */
  CHKERRQ(EPSSetInitialSpace(eps,nini,Iv));
  CHKERRQ(EPSSetDeflationSpace(eps,ncon,Cv));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(EPSGetIterationNumber(eps,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPGetTotalIterations(ksp,&lits));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of linear iterations of the method: %" PetscInt_FMT "\n",lits));
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));
  CHKERRQ(EPSGetTolerances(eps,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Show detailed info unless -terse option is given by user
   */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  } else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  /*
     Save eigenvectors, if requested
  */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-evecs",filename,sizeof(filename),&evecs));
  CHKERRQ(EPSGetConverged(eps,&nconv));
  if (nconv>0 && evecs) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer));
    CHKERRQ(EPSIsHermitian(eps,&ishermitian));
    for (i=0;i<nconv;i++) {
      CHKERRQ(EPSGetEigenvector(eps,i,xr,xi));
      CHKERRQ(VecView(xr,viewer));
#if !defined(PETSC_USE_COMPLEX)
      if (!ishermitian) CHKERRQ(VecView(xi,viewer));
#endif
    }
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  /*
     Free work space
  */
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&xr));
  CHKERRQ(VecDestroy(&xi));
  if (nini > 0) {
    CHKERRQ(VecDestroyVecs(nini,&Iv));
  }
  if (ncon > 0) {
    CHKERRQ(VecDestroyVecs(ncon,&Cv));
  }
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -f1 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -f2 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62b.petsc -eps_nev 4 -terse
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: ciss_1
      args: -f1 ${DATAFILESPATH}/matrices/complex/mhd1280a.petsc -f2 ${DATAFILESPATH}/matrices/complex/mhd1280b.petsc -eps_type ciss -eps_ciss_usest 0 -eps_ciss_quadrule chebyshev -rg_type ring -rg_ring_center 0 -rg_ring_radius .49 -rg_ring_width 0.2 -rg_ring_startangle .25 -rg_ring_endangle .49 -terse -eps_max_it 1
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      timeoutfactor: 2

   test:
      suffix: 3  # test problem (A,A)
      args: -f1 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -f2 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -eps_nev 4 -terse
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
