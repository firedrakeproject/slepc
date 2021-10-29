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
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  n = m*m;
  k = 10;
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-c",c,&k,&flg);CHKERRQ(ierr);
  if (flg && k!=10) SETERRQ1(PETSC_COMM_WORLD,1,"The number of parameters -c should be 10, you provided %D",k);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nButterfly problem, n=%D (m=%D)\n\n",n,m);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Compute the polynomial matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* initialize matrices */
  for (i=0;i<NMAT;i++) {
    ierr = MatCreate(PETSC_COMM_WORLD,&A[i]);CHKERRQ(ierr);
    ierr = MatSetSizes(A[i],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A[i]);CHKERRQ(ierr);
    ierr = MatSetUp(A[i]);CHKERRQ(ierr);
  }
  ierr = MatGetOwnershipRange(A[0],&Istart,&Iend);CHKERRQ(ierr);

  /* A0 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    ierr = MatSetValue(A[0],II,II,4.0*c[0]/6.0+4.0*c[1]/6.0,INSERT_VALUES);CHKERRQ(ierr);
    if (j>0) { ierr = MatSetValue(A[0],II,II-1,c[0]/6.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<m-1) { ierr = MatSetValue(A[0],II,II+1,c[0]/6.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i>0) { ierr = MatSetValue(A[0],II,II-m,c[1]/6.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A[0],II,II+m,c[1]/6.0,INSERT_VALUES);CHKERRQ(ierr); }
  }

  /* A1 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    if (j>0) { ierr = MatSetValue(A[1],II,II-1,c[2],INSERT_VALUES);CHKERRQ(ierr); }
    if (j<m-1) { ierr = MatSetValue(A[1],II,II+1,-c[2],INSERT_VALUES);CHKERRQ(ierr); }
    if (i>0) { ierr = MatSetValue(A[1],II,II-m,c[3],INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A[1],II,II+m,-c[3],INSERT_VALUES);CHKERRQ(ierr); }
  }

  /* A2 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    ierr = MatSetValue(A[2],II,II,-2.0*c[4]-2.0*c[5],INSERT_VALUES);CHKERRQ(ierr);
    if (j>0) { ierr = MatSetValue(A[2],II,II-1,c[4],INSERT_VALUES);CHKERRQ(ierr); }
    if (j<m-1) { ierr = MatSetValue(A[2],II,II+1,c[4],INSERT_VALUES);CHKERRQ(ierr); }
    if (i>0) { ierr = MatSetValue(A[2],II,II-m,c[5],INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A[2],II,II+m,c[5],INSERT_VALUES);CHKERRQ(ierr); }
  }

  /* A3 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    if (j>0) { ierr = MatSetValue(A[3],II,II-1,c[6],INSERT_VALUES);CHKERRQ(ierr); }
    if (j<m-1) { ierr = MatSetValue(A[3],II,II+1,-c[6],INSERT_VALUES);CHKERRQ(ierr); }
    if (i>0) { ierr = MatSetValue(A[3],II,II-m,c[7],INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A[3],II,II+m,-c[7],INSERT_VALUES);CHKERRQ(ierr); }
  }

  /* A4 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    ierr = MatSetValue(A[4],II,II,2.0*c[8]+2.0*c[9],INSERT_VALUES);CHKERRQ(ierr);
    if (j>0) { ierr = MatSetValue(A[4],II,II-1,-c[8],INSERT_VALUES);CHKERRQ(ierr); }
    if (j<m-1) { ierr = MatSetValue(A[4],II,II+1,-c[8],INSERT_VALUES);CHKERRQ(ierr); }
    if (i>0) { ierr = MatSetValue(A[4],II,II-m,-c[9],INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A[4],II,II+m,-c[9],INSERT_VALUES);CHKERRQ(ierr); }
  }

  /* assemble matrices */
  for (i=0;i<NMAT;i++) {
    ierr = MatAssemblyBegin(A[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  for (i=0;i<NMAT;i++) {
    ierr = MatAssemblyEnd(A[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PEPCreate(PETSC_COMM_WORLD,&pep);CHKERRQ(ierr);
  ierr = PEPSetOperators(pep,NMAT,A);CHKERRQ(ierr);
  ierr = PEPSetFromOptions(pep);CHKERRQ(ierr);
  ierr = PEPSolve(pep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PEPDestroy(&pep);CHKERRQ(ierr);
  for (i=0;i<NMAT;i++) {
    ierr = MatDestroy(&A[i]);CHKERRQ(ierr);
  }
  ierr = SlepcFinalize();
  return ierr;
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
      filter: sed -e "s/95386/95385/" | sed -e "s/91010/91009/"
      output_file: output/butterfly_ciss.out
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

   test:
      suffix: 4
      args: -pep_type ciss -rg_type ellipse -rg_ellipse_center .5+.5i -rg_ellipse_radius .25 -pep_ciss_moments 4 -pep_ciss_blocksize 5 -pep_ciss_refine_blocksize 2 -terse
      requires: complex
      filter: sed -e "s/46483/46484/" | sed -e "s/54946/54945/" | sed -e "s/48456/48457/" | sed -e "s/74117/74116/" | sed -e "s/37240/37241/"

TEST*/
