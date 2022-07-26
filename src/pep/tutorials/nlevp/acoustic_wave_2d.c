/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

   The acoustic_wave_2d problem is a 2-D version of acoustic_wave_1d, also
   scaled for real arithmetic.
*/

static char help[] = "Quadratic eigenproblem from an acoustics application (2-D).\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = grid size, the matrices have dimension m*(m-1).\n"
  "  -z <z>, where <z> = impedance (default 1.0).\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            M,C,K,A[3];      /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PetscInt       m=6,n,II,Istart,Iend,i,j;
  PetscScalar    z=1.0;
  PetscReal      h;
  char           str[50];
  PetscBool      terse;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCheck(m>1,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"m must be at least 2");
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-z",&z,NULL));
  h = 1.0/m;
  n = m*(m-1);
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),z,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nAcoustic wave 2-D, n=%" PetscInt_FMT " (m=%" PetscInt_FMT "), z=%s\n\n",n,m,str));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K has a pattern similar to the 2D Laplacian */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&K));
  PetscCall(MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(K));
  PetscCall(MatSetUp(K));

  PetscCall(MatGetOwnershipRange(K,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    if (i>0) PetscCall(MatSetValue(K,II,II-m,(j==m-1)?-0.5:-1.0,INSERT_VALUES));
    if (i<m-2) PetscCall(MatSetValue(K,II,II+m,(j==m-1)?-0.5:-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(K,II,II-1,-1.0,INSERT_VALUES));
    if (j<m-1) PetscCall(MatSetValue(K,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(K,II,II,(j==m-1)?2.0:4.0,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY));

  /* C is the zero matrix except for a few nonzero elements on the diagonal */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  PetscCall(MatGetOwnershipRange(C,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i%m==m-1) PetscCall(MatSetValue(C,i,i,-2*PETSC_PI*h/z,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* M is a diagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&M));
  PetscCall(MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(M));
  PetscCall(MatSetUp(M));

  PetscCall(MatGetOwnershipRange(M,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i%m==m-1) PetscCall(MatSetValue(M,i,i,2*PETSC_PI*PETSC_PI*h*h,INSERT_VALUES));
    else PetscCall(MatSetValue(M,i,i,4*PETSC_PI*PETSC_PI*h*h,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  PetscCall(PEPSetOperators(pep,3,A));
  PetscCall(PEPSetFromOptions(pep));
  PetscCall(PEPSolve(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(PEPDestroy(&pep));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&K));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -pep_nev 2 -pep_ncv 18 -terse
      output_file: output/acoustic_wave_2d_1.out
      filter: sed -e "s/2.60936i/2.60937i/g" | sed -e "s/2.60938i/2.60937i/g"
      test:
         suffix: 1
         args: -pep_type {{qarnoldi linear}}
      test:
         suffix: 1_toar
         args: -pep_type toar -pep_toar_locking 0

   testset:
      args: -pep_nev 2 -pep_ncv 18 -pep_type stoar -pep_hermitian -pep_scale scalar -st_type sinvert -terse
      output_file: output/acoustic_wave_2d_2.out
      test:
         suffix: 2
      test:
         suffix: 2_lin_b
         args: -pep_stoar_linearization 0,1
      test:
         suffix: 2_lin_ab
         args: -pep_stoar_linearization 0.1,0.9

TEST*/
