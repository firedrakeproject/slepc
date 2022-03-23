/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test the solution of a PEP without calling PEPSetFromOptions (based on ex16.c).\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n"
  "  -type <pep_type> = pep type to test.\n"
  "  -epstype <eps_type> = eps type to test (for linear).\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            M,C,K,A[3];      /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PetscInt       N,n=10,m,Istart,Iend,II,nev,i,j;
  PetscReal      keep;
  PetscBool      flag,isgd2,epsgiven,lock;
  char           peptype[30] = "linear",epstype[30] = "";
  EPS            eps;
  ST             st;
  KSP            ksp;
  PC             pc;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-type",peptype,sizeof(peptype),NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-epstype",epstype,sizeof(epstype),&epsgiven));
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
  CHKERRQ(PEPSetProblemType(pep,PEP_GENERAL));
  CHKERRQ(PEPSetDimensions(pep,4,20,PETSC_DEFAULT));
  CHKERRQ(PEPSetTolerances(pep,PETSC_SMALL,PETSC_DEFAULT));

  /*
     Set solver type at runtime
  */
  CHKERRQ(PEPSetType(pep,peptype));
  if (epsgiven) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&flag));
    if (flag) {
      CHKERRQ(PEPLinearGetEPS(pep,&eps));
      CHKERRQ(PetscStrcmp(epstype,"gd2",&isgd2));
      if (isgd2) {
        CHKERRQ(EPSSetType(eps,EPSGD));
        CHKERRQ(EPSGDSetDoubleExpansion(eps,PETSC_TRUE));
      } else CHKERRQ(EPSSetType(eps,epstype));
      CHKERRQ(EPSGetST(eps,&st));
      CHKERRQ(STGetKSP(st,&ksp));
      CHKERRQ(KSPGetPC(ksp,&pc));
      CHKERRQ(PCSetType(pc,PCJACOBI));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)eps,EPSGD,&flag));
    }
    CHKERRQ(PEPLinearSetExplicitMatrix(pep,PETSC_TRUE));
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pep,PEPQARNOLDI,&flag));
  if (flag) {
    CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
    CHKERRQ(STSetTransform(st,PETSC_TRUE));
    CHKERRQ(PEPSetST(pep,st));
    CHKERRQ(STDestroy(&st));
    CHKERRQ(PEPQArnoldiGetRestart(pep,&keep));
    CHKERRQ(PEPQArnoldiGetLocking(pep,&lock));
    if (!lock && keep<0.6) CHKERRQ(PEPQArnoldiSetRestart(pep,0.6));
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pep,PEPTOAR,&flag));
  if (flag) {
    CHKERRQ(PEPTOARGetRestart(pep,&keep));
    CHKERRQ(PEPTOARGetLocking(pep,&lock));
    if (!lock && keep<0.6) CHKERRQ(PEPTOARSetRestart(pep,0.6));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPSolve(pep));
  CHKERRQ(PEPGetDimensions(pep,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  CHKERRQ(PEPDestroy(&pep));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&K));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -m 11
      output_file: output/test1_1.out
      filter: sed -e "s/1.16403/1.16404/g" | sed -e "s/1.65362i/1.65363i/g" | sed -e "s/-1.16404-1.65363i, -1.16404+1.65363i/-1.16404+1.65363i, -1.16404-1.65363i/" | sed -e "s/-0.51784-1.31039i, -0.51784+1.31039i/-0.51784+1.31039i, -0.51784-1.31039i/"
      requires: !single
      test:
         suffix: 1
         args: -type {{toar qarnoldi linear}}
      test:
         suffix: 1_linear_gd
         args: -type linear -epstype gd

TEST*/
