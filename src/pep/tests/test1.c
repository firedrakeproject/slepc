/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscOptionsGetString(NULL,NULL,"-type",peptype,sizeof(peptype),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-epstype",epstype,sizeof(epstype),&epsgiven));
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
  PetscCall(PEPSetProblemType(pep,PEP_GENERAL));
  PetscCall(PEPSetDimensions(pep,4,20,PETSC_DEFAULT));
  PetscCall(PEPSetTolerances(pep,PETSC_SMALL,PETSC_DEFAULT));

  /*
     Set solver type at runtime
  */
  PetscCall(PEPSetType(pep,peptype));
  if (epsgiven) {
    PetscCall(PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&flag));
    if (flag) {
      PetscCall(PEPLinearGetEPS(pep,&eps));
      PetscCall(PetscStrcmp(epstype,"gd2",&isgd2));
      if (isgd2) {
        PetscCall(EPSSetType(eps,EPSGD));
        PetscCall(EPSGDSetDoubleExpansion(eps,PETSC_TRUE));
      } else PetscCall(EPSSetType(eps,epstype));
      PetscCall(EPSGetST(eps,&st));
      PetscCall(STGetKSP(st,&ksp));
      PetscCall(KSPGetPC(ksp,&pc));
      PetscCall(PCSetType(pc,PCJACOBI));
      PetscCall(PetscObjectTypeCompare((PetscObject)eps,EPSGD,&flag));
    }
    PetscCall(PEPLinearSetExplicitMatrix(pep,PETSC_TRUE));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)pep,PEPQARNOLDI,&flag));
  if (flag) {
    PetscCall(STCreate(PETSC_COMM_WORLD,&st));
    PetscCall(STSetTransform(st,PETSC_TRUE));
    PetscCall(PEPSetST(pep,st));
    PetscCall(STDestroy(&st));
    PetscCall(PEPQArnoldiGetRestart(pep,&keep));
    PetscCall(PEPQArnoldiGetLocking(pep,&lock));
    if (!lock && keep<0.6) PetscCall(PEPQArnoldiSetRestart(pep,0.6));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)pep,PEPTOAR,&flag));
  if (flag) {
    PetscCall(PEPTOARGetRestart(pep,&keep));
    PetscCall(PEPTOARGetLocking(pep,&lock));
    if (!lock && keep<0.6) PetscCall(PEPTOARSetRestart(pep,0.6));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPSolve(pep));
  PetscCall(PEPGetDimensions(pep,&nev,NULL,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

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
         requires: !__float128

TEST*/
