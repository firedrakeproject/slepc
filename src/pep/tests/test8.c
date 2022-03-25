/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test interface functions of polynomial JD.\n\n"
  "This is based on ex16.c. The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat             M,C,K,A[3];      /* problem matrices */
  PEP             pep;             /* polynomial eigenproblem solver context */
  PetscInt        N,n=10,m,Istart,Iend,II,i,j,midx;
  PetscReal       restart,fix;
  PetscBool       flag,reuse;
  PEPJDProjection proj;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nQuadratic Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is the 2-D Laplacian */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&K));
  CHKERRQ(MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(K));
  CHKERRQ(MatSetUp(K));
  CHKERRQ(MatGetOwnershipRange(K,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(K,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(K,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(K,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(K,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(K,II,II,4.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY));

  /* C is the 1-D Laplacian on horizontal lines */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatGetOwnershipRange(C,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (j>0) CHKERRQ(MatSetValue(C,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(C,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(C,II,II,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* M is a diagonal matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&M));
  CHKERRQ(MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(M));
  CHKERRQ(MatSetUp(M));
  CHKERRQ(MatGetOwnershipRange(M,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) CHKERRQ(MatSetValue(M,II,II,(PetscReal)(II+1),INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  CHKERRQ(PEPSetOperators(pep,3,A));
  CHKERRQ(PEPSetType(pep,PEPJD));

  /*
     Test interface functions of JD solver
  */
  CHKERRQ(PEPJDGetRestart(pep,&restart));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Restart parameter before changing = %g",(double)restart));
  CHKERRQ(PEPJDSetRestart(pep,0.3));
  CHKERRQ(PEPJDGetRestart(pep,&restart));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %g\n",(double)restart));

  CHKERRQ(PEPJDGetFix(pep,&fix));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Fix parameter before changing = %g",(double)fix));
  CHKERRQ(PEPJDSetFix(pep,0.001));
  CHKERRQ(PEPJDGetFix(pep,&fix));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %g\n",(double)fix));

  CHKERRQ(PEPJDGetReusePreconditioner(pep,&reuse));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reuse preconditioner flag before changing = %d",(int)reuse));
  CHKERRQ(PEPJDSetReusePreconditioner(pep,PETSC_TRUE));
  CHKERRQ(PEPJDGetReusePreconditioner(pep,&reuse));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)reuse));

  CHKERRQ(PEPJDGetProjection(pep,&proj));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Projection type before changing = %d",(int)proj));
  CHKERRQ(PEPJDSetProjection(pep,PEP_JD_PROJECTION_ORTHOGONAL));
  CHKERRQ(PEPJDGetProjection(pep,&proj));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)proj));

  CHKERRQ(PEPJDGetMinimalityIndex(pep,&midx));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Minimality index before changing = %" PetscInt_FMT,midx));
  CHKERRQ(PEPJDSetMinimalityIndex(pep,2));
  CHKERRQ(PEPJDGetMinimalityIndex(pep,&midx));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %" PetscInt_FMT "\n",midx));

  CHKERRQ(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPSolve(pep));
  CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  CHKERRQ(PEPDestroy(&pep));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&K));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      args: -n 12 -pep_nev 2 -pep_ncv 21 -pep_conv_abs

TEST*/
