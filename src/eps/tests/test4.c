/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test the solution of a HEP without calling EPSSetFromOptions (based on ex1.c).\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n"
  "  -type <eps_type> = eps type to test.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;           /* problem matrix */
  EPS            eps;         /* eigenproblem solver context */
  PetscReal      tol=1000*PETSC_MACHINE_EPSILON;
  PetscInt       n=30,i,Istart,Iend,nev;
  PetscBool      flg,gd2;
  char           epstype[30] = "krylovschur";

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-type",epstype,sizeof(epstype),NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create eigensolver context
  */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
     Set operators. In this case, it is a standard eigenvalue problem
  */
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetDimensions(eps,4,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(PetscStrcmp(epstype,"gd2",&flg));
  if (flg) {
    CHKERRQ(EPSSetType(eps,EPSGD));
    CHKERRQ(EPSGDSetDoubleExpansion(eps,PETSC_TRUE));
    CHKERRQ(EPSGDGetDoubleExpansion(eps,&gd2));  /* not used */
  } else CHKERRQ(EPSSetType(eps,epstype));
  CHKERRQ(PetscStrcmp(epstype,"jd",&flg));
  if (flg) {
    CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE));
    CHKERRQ(EPSSetTarget(eps,4.0));
  }
  CHKERRQ(PetscStrcmp(epstype,"lanczos",&flg));
  if (flg) CHKERRQ(EPSLanczosSetReorthog(eps,EPS_LANCZOS_REORTHOG_LOCAL));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)eps,&flg,EPSRQCG,EPSLOBPCG,""));
  if (flg) CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/test4_1.out
      test:
         suffix: 1
         args: -type {{krylovschur subspace arnoldi lanczos gd jd gd2 lapack}}
      test:
         suffix: 1_arpack
         args: -type arpack
         requires: arpack
      test:
         suffix: 1_primme
         args: -type primme
         requires: primme
      test:
         suffix: 1_trlan
         args: -type trlan
         requires: trlan

   test:
      suffix: 2
      args: -type {{rqcg lobpcg}}

TEST*/
