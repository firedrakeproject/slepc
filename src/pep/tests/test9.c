/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates use of PEPSetEigenvalueComparison().\n\n"
  "Based on butterfly.c. The command line options are:\n"
  "  -m <m>, grid size, the dimension of the matrices is n=m*m.\n"
  "  -c <array>, problem parameters, must be 10 comma-separated real values.\n\n";

#include <slepcpep.h>

#define NMAT 5

/*
    Function for user-defined eigenvalue ordering criterion.

    Given two eigenvalues ar+i*ai and br+i*bi, the subroutine must choose
    one of them as the preferred one according to the criterion.
    In this example, eigenvalues are sorted by magnitude but those with
    positive real part are preferred.
*/
PetscErrorCode MyEigenSort(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  PetscReal rea,reb;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_COMPLEX)
  rea = PetscRealPart(ar); reb = PetscRealPart(br);
#else
  rea = ar; reb = br;
#endif
  *r = rea<0.0? 1: (reb<0.0? -1: PetscSign(SlepcAbsEigenvalue(br,bi)-SlepcAbsEigenvalue(ar,ai)));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A[NMAT];         /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PetscInt       n,m=8,k,II,Istart,Iend,i,j;
  PetscReal      c[10] = { 0.6, 1.3, 1.3, 0.1, 0.1, 1.2, 1.0, 1.0, 1.2, 1.0 };
  PetscBool      flg;
  PetscBool      terse;
  const char     *prefix;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n = m*m;
  k = 10;
  CHKERRQ(PetscOptionsGetRealArray(NULL,NULL,"-c",c,&k,&flg));
  PetscCheck(!flg || k==10,PETSC_COMM_WORLD,PETSC_ERR_USER,"The number of parameters -c should be 10, you provided %" PetscInt_FMT,k);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nButterfly problem, n=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n\n",n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Compute the polynomial matrices
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
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    CHKERRQ(MatSetValue(A[0],II,II,4.0*c[0]/6.0+4.0*c[1]/6.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A[0],II,II-1,c[0]/6.0,INSERT_VALUES));
    if (j<m-1) CHKERRQ(MatSetValue(A[0],II,II+1,c[0]/6.0,INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[0],II,II-m,c[1]/6.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A[0],II,II+m,c[1]/6.0,INSERT_VALUES));
  }

  /* A1 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    if (j>0) CHKERRQ(MatSetValue(A[1],II,II-1,c[2],INSERT_VALUES));
    if (j<m-1) CHKERRQ(MatSetValue(A[1],II,II+1,-c[2],INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[1],II,II-m,c[3],INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A[1],II,II+m,-c[3],INSERT_VALUES));
  }

  /* A2 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    CHKERRQ(MatSetValue(A[2],II,II,-2.0*c[4]-2.0*c[5],INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A[2],II,II-1,c[4],INSERT_VALUES));
    if (j<m-1) CHKERRQ(MatSetValue(A[2],II,II+1,c[4],INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[2],II,II-m,c[5],INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A[2],II,II+m,c[5],INSERT_VALUES));
  }

  /* A3 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    if (j>0) CHKERRQ(MatSetValue(A[3],II,II-1,c[6],INSERT_VALUES));
    if (j<m-1) CHKERRQ(MatSetValue(A[3],II,II+1,-c[6],INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[3],II,II-m,c[7],INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A[3],II,II+m,-c[7],INSERT_VALUES));
  }

  /* A4 */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    CHKERRQ(MatSetValue(A[4],II,II,2.0*c[8]+2.0*c[9],INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A[4],II,II-1,-c[8],INSERT_VALUES));
    if (j<m-1) CHKERRQ(MatSetValue(A[4],II,II+1,-c[8],INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[4],II,II-m,-c[9],INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A[4],II,II+m,-c[9],INSERT_VALUES));
  }

  /* assemble matrices */
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatAssemblyBegin(A[i],MAT_FINAL_ASSEMBLY));
  }
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatAssemblyEnd(A[i],MAT_FINAL_ASSEMBLY));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));
  CHKERRQ(PEPSetOptionsPrefix(pep,"check_"));
  CHKERRQ(PEPAppendOptionsPrefix(pep,"myprefix_"));
  CHKERRQ(PEPGetOptionsPrefix(pep,&prefix));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"PEP prefix is currently: %s\n\n",prefix));

  CHKERRQ(PEPSetOperators(pep,NMAT,A));
  CHKERRQ(PEPSetEigenvalueComparison(pep,MyEigenSort,NULL));
  CHKERRQ(PEPSetFromOptions(pep));
  CHKERRQ(PEPSolve(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  } else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(PEPDestroy(&pep));
  for (i=0;i<NMAT;i++) {
    CHKERRQ(MatDestroy(&A[i]));
  }
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      args: -check_myprefix_pep_nev 4 -terse
      requires: double

TEST*/
