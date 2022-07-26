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

   The butterfly problem is a quartic PEP with T-even structure.
*/

static char help[] = "Quartic polynomial eigenproblem with T-even structure.\n\n"
  "The command line options are:\n"
  "  -m <m>, grid size, the dimension of the matrices is n=m*m.\n"
  "  -c <array>, problem parameters, must be 10 comma-separated real values.\n\n";

#include <slepcpep.h>

#define NMAT 5

int main(int argc,char **argv)
{
  Mat            A[NMAT];         /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PetscInt       n,m=8,k,II,Istart,Iend,i,j;
  PetscReal      c[10] = { 0.6, 1.3, 1.3, 0.1, 0.1, 1.2, 1.0, 1.0, 1.2, 1.0 };
  PetscBool      flg;
  PetscBool      terse;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n = m*m;
  k = 10;
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-c",c,&k,&flg));
  PetscCheck(!flg || k==10,PETSC_COMM_WORLD,PETSC_ERR_USER,"The number of parameters -c should be 10, you provided %" PetscInt_FMT,k);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nButterfly problem, n=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n\n",n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Compute the polynomial matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* initialize matrices */
  for (i=0;i<NMAT;i++) {
    PetscCall(MatCreate(PETSC_COMM_WORLD,&A[i]));
    PetscCall(MatSetSizes(A[i],PETSC_DECIDE,PETSC_DECIDE,n,n));
    PetscCall(MatSetFromOptions(A[i]));
    PetscCall(MatSetUp(A[i]));
  }
  PetscCall(MatGetOwnershipRange(A[0],&Istart,&Iend));

  /* A0 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    PetscCall(MatSetValue(A[0],II,II,4.0*c[0]/6.0+4.0*c[1]/6.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A[0],II,II-1,c[0]/6.0,INSERT_VALUES));
    if (j<m-1) PetscCall(MatSetValue(A[0],II,II+1,c[0]/6.0,INSERT_VALUES));
    if (i>0) PetscCall(MatSetValue(A[0],II,II-m,c[1]/6.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A[0],II,II+m,c[1]/6.0,INSERT_VALUES));
  }

  /* A1 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    if (j>0) PetscCall(MatSetValue(A[1],II,II-1,c[2],INSERT_VALUES));
    if (j<m-1) PetscCall(MatSetValue(A[1],II,II+1,-c[2],INSERT_VALUES));
    if (i>0) PetscCall(MatSetValue(A[1],II,II-m,c[3],INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A[1],II,II+m,-c[3],INSERT_VALUES));
  }

  /* A2 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    PetscCall(MatSetValue(A[2],II,II,-2.0*c[4]-2.0*c[5],INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A[2],II,II-1,c[4],INSERT_VALUES));
    if (j<m-1) PetscCall(MatSetValue(A[2],II,II+1,c[4],INSERT_VALUES));
    if (i>0) PetscCall(MatSetValue(A[2],II,II-m,c[5],INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A[2],II,II+m,c[5],INSERT_VALUES));
  }

  /* A3 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    if (j>0) PetscCall(MatSetValue(A[3],II,II-1,c[6],INSERT_VALUES));
    if (j<m-1) PetscCall(MatSetValue(A[3],II,II+1,-c[6],INSERT_VALUES));
    if (i>0) PetscCall(MatSetValue(A[3],II,II-m,c[7],INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A[3],II,II+m,-c[7],INSERT_VALUES));
  }

  /* A4 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    PetscCall(MatSetValue(A[4],II,II,2.0*c[8]+2.0*c[9],INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A[4],II,II-1,-c[8],INSERT_VALUES));
    if (j<m-1) PetscCall(MatSetValue(A[4],II,II+1,-c[8],INSERT_VALUES));
    if (i>0) PetscCall(MatSetValue(A[4],II,II-m,-c[9],INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A[4],II,II+m,-c[9],INSERT_VALUES));
  }

  /* assemble matrices */
  for (i=0;i<NMAT;i++) PetscCall(MatAssemblyBegin(A[i],MAT_FINAL_ASSEMBLY));
  for (i=0;i<NMAT;i++) PetscCall(MatAssemblyEnd(A[i],MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  PetscCall(PEPSetOperators(pep,NMAT,A));
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
  for (i=0;i<NMAT;i++) PetscCall(MatDestroy(&A[i]));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -pep_nev 4 -st_type sinvert -pep_target 0.01 -terse
      output_file: output/butterfly_1.out
      test:
         suffix: 1_toar
         args: -pep_type toar -pep_toar_restart 0.3
      test:
         suffix: 1_linear
         args: -pep_type linear

   test:
      suffix: 2
      args: -pep_type {{toar linear}} -pep_nev 4 -terse
      requires: double

   testset:
      args: -pep_type ciss -rg_type ellipse -rg_ellipse_center 1+.5i -rg_ellipse_radius .15 -terse
      requires: complex
      filter: sed -e "s/95386/95385/" | sed -e "s/91010/91009/" | sed -e "s/93092/93091/" | sed -e "s/96723/96724/" | sed -e "s/43015/43016/" | sed -e "s/53513/53514/"
      output_file: output/butterfly_ciss.out
      timeoutfactor: 2
      test:
         suffix: ciss_hankel
         args: -pep_ciss_extraction hankel -pep_ciss_integration_points 40
         requires: !single
      test:
         suffix: ciss_ritz
         args: -pep_ciss_extraction ritz
      test:
         suffix: ciss_caa
         args: -pep_ciss_extraction caa -pep_ciss_moments 4
      test:
         suffix: ciss_part
         nsize: 2
         args: -pep_ciss_partitions 2
      test:
         suffix: ciss_refine
         args: -pep_ciss_refine_inner 1 -pep_ciss_refine_blocksize 1

   testset:
      args: -pep_type ciss -rg_type ellipse -rg_ellipse_center .5+.5i -rg_ellipse_radius .25 -pep_ciss_moments 4 -pep_ciss_blocksize 5 -pep_ciss_refine_blocksize 2 -terse
      requires: complex double
      filter: sed -e "s/46483/46484/" | sed -e "s/54946/54945/" | sed -e "s/48456/48457/" | sed -e "s/74117/74116/" | sed -e "s/37240/37241/"
      output_file: output/butterfly_4.out
      test:
         suffix: 4
      test:
         suffix: 4_hankel
         args: -pep_ciss_extraction hankel

TEST*/
