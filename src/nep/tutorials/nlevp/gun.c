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

   The gun problem arises from model of a radio-frequency gun cavity, with
   the complex nonlinear function
   T(lambda) = K-lambda*M+i*lambda^(1/2)*W1+i*(lambda-108.8774^2)^(1/2)*W2

   Data files can be downloaded from https://slepc.upv.es/datafiles
*/

static char help[] = "Radio-frequency gun cavity.\n\n"
  "The command line options are:\n"
  "-K <filename1> -M <filename2> -W1 <filename3> -W2 <filename4>, where filename1,..,filename4 are files containing the matrices in PETSc binary form defining the GUN problem.\n\n";

#include <slepcnep.h>

#define NMAT 4
#define SIGMA 108.8774

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Mat            A[NMAT];         /* problem matrices */
  FN             f[NMAT];         /* functions to define the nonlinear operator */
  FN             ff[2];           /* auxiliary functions to define the nonlinear operator */
  NEP            nep;             /* nonlinear eigensolver context */
  PetscBool      terse,flg;
  const char*    string[NMAT]={"-K","-M","-W1","-W2"};
  char           filename[PETSC_MAX_PATH_LEN];
  PetscScalar    numer[2],sigma;
  PetscInt       i;
  PetscViewer    viewer;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"GUN problem\n\n"));
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example requires complex scalars!");
#endif

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Load the problem matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  for (i=0;i<NMAT;i++) {
    CHKERRQ(PetscOptionsGetString(NULL,NULL,string[i],filename,sizeof(filename),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a filename with the %s option",string[i]);
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[i]));
    CHKERRQ(MatSetFromOptions(A[i]));
    CHKERRQ(MatLoad(A[i],viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

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

  /* f3=i*sqrt(lambda) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[2]));
  CHKERRQ(FNSetType(f[2],FNSQRT));
  CHKERRQ(FNSetScale(f[2],1.0,PETSC_i));

  /* f4=i*sqrt(lambda-sigma^2) */
  sigma = SIGMA*SIGMA;
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&ff[0]));
  CHKERRQ(FNSetType(ff[0],FNSQRT));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&ff[1]));
  CHKERRQ(FNSetType(ff[1],FNRATIONAL));
  numer[0] = 1.0; numer[1] = -sigma;
  CHKERRQ(FNRationalSetNumerator(ff[1],2,numer));
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[3]));
  CHKERRQ(FNSetType(f[3],FNCOMBINE));
  CHKERRQ(FNCombineSetChildren(f[3],FN_COMBINE_COMPOSE,ff[1],ff[0]));
  CHKERRQ(FNSetScale(f[3],1.0,PETSC_i));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));
  CHKERRQ(NEPSetSplitOperator(nep,4,A,f,UNKNOWN_NONZERO_PATTERN));
  CHKERRQ(NEPSetFromOptions(nep));

  CHKERRQ(NEPSolve(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));
  } else {
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
  for (i=0;i<2;i++) {
    CHKERRQ(FNDestroy(&ff[i]));
  }
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   build:
      requires: complex

   test:
      suffix: 1
      args: -K ${DATAFILESPATH}/matrices/complex/gun_K.petsc -M ${DATAFILESPATH}/matrices/complex/gun_M.petsc -W1 ${DATAFILESPATH}/matrices/complex/gun_W1.petsc -W2 ${DATAFILESPATH}/matrices/complex/gun_W2.petsc -nep_type nleigs -rg_type polygon -rg_polygon_vertices 12500-1i,120500-1i,120500+30000i,70000+30000i -nep_target 65000 -nep_nev 24 -terse
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      timeoutfactor: 10

TEST*/
