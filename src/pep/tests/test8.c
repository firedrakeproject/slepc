/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nQuadratic Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is the 2-D Laplacian */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&K));
  PetscCall(MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(K));
  PetscCall(MatSetUp(K));
  PetscCall(MatGetOwnershipRange(K,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(K,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(K,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(K,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(K,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(K,II,II,4.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY));

  /* C is the 1-D Laplacian on horizontal lines */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatGetOwnershipRange(C,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (j>0) PetscCall(MatSetValue(C,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(C,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(C,II,II,2.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* M is a diagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&M));
  PetscCall(MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(M));
  PetscCall(MatSetUp(M));
  PetscCall(MatGetOwnershipRange(M,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) PetscCall(MatSetValue(M,II,II,(PetscReal)(II+1),INSERT_VALUES));
  PetscCall(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  PetscCall(PEPSetOperators(pep,3,A));
  PetscCall(PEPSetType(pep,PEPJD));

  /*
     Test interface functions of JD solver
  */
  PetscCall(PEPJDGetRestart(pep,&restart));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Restart parameter before changing = %g",(double)restart));
  PetscCall(PEPJDSetRestart(pep,0.3));
  PetscCall(PEPJDGetRestart(pep,&restart));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %g\n",(double)restart));

  PetscCall(PEPJDGetFix(pep,&fix));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Fix parameter before changing = %g",(double)fix));
  PetscCall(PEPJDSetFix(pep,0.001));
  PetscCall(PEPJDGetFix(pep,&fix));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %g\n",(double)fix));

  PetscCall(PEPJDGetReusePreconditioner(pep,&reuse));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Reuse preconditioner flag before changing = %d",(int)reuse));
  PetscCall(PEPJDSetReusePreconditioner(pep,PETSC_TRUE));
  PetscCall(PEPJDGetReusePreconditioner(pep,&reuse));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)reuse));

  PetscCall(PEPJDGetProjection(pep,&proj));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Projection type before changing = %d",(int)proj));
  PetscCall(PEPJDSetProjection(pep,PEP_JD_PROJECTION_ORTHOGONAL));
  PetscCall(PEPJDGetProjection(pep,&proj));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)proj));

  PetscCall(PEPJDGetMinimalityIndex(pep,&midx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Minimality index before changing = %" PetscInt_FMT,midx));
  PetscCall(PEPJDSetMinimalityIndex(pep,2));
  PetscCall(PEPJDGetMinimalityIndex(pep,&midx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %" PetscInt_FMT "\n",midx));

  PetscCall(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPSolve(pep));
  PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  PetscCall(PEPDestroy(&pep));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&K));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      args: -n 12 -pep_nev 2 -pep_ncv 21 -pep_conv_abs

TEST*/
