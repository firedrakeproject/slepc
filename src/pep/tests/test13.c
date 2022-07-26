/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solve a quadratic problem with CISS.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat               M,C,K,A[3];
  PEP               pep;
  RG                rg;
  KSP               *ksp;
  PC                pc;
  PEPCISSExtraction ext;
  PetscInt          N,n=10,m,Istart,Iend,II,i,j,nsolve;
  PetscBool         flg;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flg));
  if (!flg) m=n;
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
             Create the eigensolver and solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  PetscCall(PEPSetOperators(pep,3,A));
  PetscCall(PEPSetProblemType(pep,PEP_GENERAL));

  /* customize polynomial eigensolver; set runtime options */
  PetscCall(PEPSetType(pep,PEPCISS));
  PetscCall(PEPGetRG(pep,&rg));
  PetscCall(RGSetType(rg,RGELLIPSE));
  PetscCall(RGEllipseSetParameters(rg,PetscCMPLX(-0.1,0.3),0.1,0.25));
  PetscCall(PEPCISSSetSizes(pep,24,PETSC_DEFAULT,PETSC_DEFAULT,1,PETSC_DEFAULT,PETSC_TRUE));
  PetscCall(PEPCISSGetKSPs(pep,&nsolve,&ksp));
  for (i=0;i<nsolve;i++) {
    PetscCall(KSPSetTolerances(ksp[i],1e-12,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
    PetscCall(KSPSetType(ksp[i],KSPPREONLY));
    PetscCall(KSPGetPC(ksp[i],&pc));
    PetscCall(PCSetType(pc,PCLU));
  }
  PetscCall(PEPSetFromOptions(pep));

  /* solve */
  PetscCall(PetscObjectTypeCompare((PetscObject)pep,PEPCISS,&flg));
  if (flg) {
    PetscCall(PEPCISSGetExtraction(pep,&ext));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Running CISS with %" PetscInt_FMT " KSP solvers (%s extraction)\n",nsolve,PEPCISSExtractions[ext]));
  }
  PetscCall(PEPSolve(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  PetscCall(PEPDestroy(&pep));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&K));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   build:
      requires: complex

   test:
      suffix: 1
      requires: complex

TEST*/
