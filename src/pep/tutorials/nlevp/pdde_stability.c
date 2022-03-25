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

   The pdde_stability problem is a complex-symmetric QEP from the stability
   analysis of a discretized partial delay-differential equation. It requires
   complex scalars.
*/

static char help[] = "Stability analysis of a discretized partial delay-differential equation.\n\n"
  "The command line options are:\n"
  "  -m <m>, grid size, the matrices have dimension n=m*m.\n"
  "  -c <a0,b0,a1,b1,a2,b2,phi1>, comma-separated list of 7 real parameters.\n\n";

#include <slepcpep.h>

#define NMAT 3

/*
    Function for user-defined eigenvalue ordering criterion.

    Given two eigenvalues ar+i*ai and br+i*bi, the subroutine must choose
    one of them as the preferred one according to the criterion.
    In this example, the preferred value is the one with absolute value closest to 1.
*/
PetscErrorCode MyEigenSort(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  PetscReal aa,ab;

  PetscFunctionBeginUser;
  aa = PetscAbsReal(SlepcAbsEigenvalue(ar,ai)-PetscRealConstant(1.0));
  ab = PetscAbsReal(SlepcAbsEigenvalue(br,bi)-PetscRealConstant(1.0));
  *r = aa > ab ? 1 : (aa < ab ? -1 : 0);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A[NMAT];         /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PetscInt       m=15,n,II,Istart,Iend,i,j,k;
  PetscReal      h,xi,xj,c[7] = { 2, .3, -2, .2, -2, -.3, -PETSC_PI/2 };
  PetscScalar    alpha,beta,gamma;
  PetscBool      flg,terse;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example requires complex scalars");
#endif

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n = m*m;
  h = PETSC_PI/(m+1);
  gamma = PetscExpScalar(PETSC_i*c[6]);
  gamma = gamma/PetscAbsScalar(gamma);
  k = 7;
  CHKERRQ(PetscOptionsGetRealArray(NULL,NULL,"-c",c,&k,&flg));
  PetscCheck(!flg || k==7,PETSC_COMM_WORLD,PETSC_ERR_USER,"The number of parameters -c should be 7, you provided %" PetscInt_FMT,k);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nPDDE stability, n=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n\n",n,m));

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

  /* A[1] has a pattern similar to the 2D Laplacian */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    xi = (i+1)*h; xj = (j+1)*h;
    alpha = c[0]+c[1]*PetscSinReal(xi)+gamma*(c[2]+c[3]*xi*(1.0-PetscExpReal(xi-PETSC_PI)));
    beta = c[0]+c[1]*PetscSinReal(xj)-gamma*(c[2]+c[3]*xj*(1.0-PetscExpReal(xj-PETSC_PI)));
    CHKERRQ(MatSetValue(A[1],II,II,alpha+beta-4.0/(h*h),INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A[1],II,II-1,1.0/(h*h),INSERT_VALUES));
    if (j<m-1) CHKERRQ(MatSetValue(A[1],II,II+1,1.0/(h*h),INSERT_VALUES));
    if (i>0) CHKERRQ(MatSetValue(A[1],II,II-m,1.0/(h*h),INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A[1],II,II+m,1.0/(h*h),INSERT_VALUES));
  }

  /* A[0] and A[2] are diagonal */
  for (II=Istart;II<Iend;II++) {
    i = II/m; j = II-i*m;
    xi = (i+1)*h; xj = (j+1)*h;
    alpha = c[4]+c[5]*xi*(PETSC_PI-xi);
    beta = c[4]+c[5]*xj*(PETSC_PI-xj);
    CHKERRQ(MatSetValue(A[0],II,II,alpha,INSERT_VALUES));
    CHKERRQ(MatSetValue(A[2],II,II,beta,INSERT_VALUES));
  }

  /* assemble matrices */
  for (i=0;i<NMAT;i++) CHKERRQ(MatAssemblyBegin(A[i],MAT_FINAL_ASSEMBLY));
  for (i=0;i<NMAT;i++) CHKERRQ(MatAssemblyEnd(A[i],MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));
  CHKERRQ(PEPSetOperators(pep,NMAT,A));
  CHKERRQ(PEPSetEigenvalueComparison(pep,MyEigenSort,NULL));
  CHKERRQ(PEPSetDimensions(pep,4,PETSC_DEFAULT,PETSC_DEFAULT));
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
  for (i=0;i<NMAT;i++) CHKERRQ(MatDestroy(&A[i]));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   build:
      requires: complex

   test:
      suffix: 1
      args: -pep_type {{toar qarnoldi linear}} -pep_ncv 25 -terse
      requires: complex !single

TEST*/
