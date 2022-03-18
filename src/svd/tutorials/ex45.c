/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Computes a partial generalized singular value decomposition (GSVD).\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of rows of A.\n"
  "  -n <n>, where <n> = number of columns of A.\n"
  "  -p <p>, where <p> = number of rows of B.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat            A,B;             /* operator matrices */
  Vec            u,v,x;           /* singular vectors */
  SVD            svd;             /* singular value problem solver context */
  SVDType        type;
  Vec            uv,aux[2],*U,*V;
  PetscReal      error,tol,sigma,lev1=0.0,lev2=0.0;
  PetscInt       m=100,n,p=14,i,j,d,Istart,Iend,nsv,maxit,its,nconv;
  PetscBool      flg,skiporth=PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n = m;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-p",&p,&flg));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value decomposition, (%" PetscInt_FMT "+%" PetscInt_FMT ")x%" PetscInt_FMT "\n\n",m,p,n));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-skiporth",&skiporth,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          Build the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0 && i-1<n) CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i+1<n) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    if (i<n) CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (i>n) CHKERRQ(MatSetValue(A,i,n-1,1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,p,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  d = PetscMax(0,n-p);
  for (i=Istart;i<Iend;i++) {
    for (j=0;j<=PetscMin(i,n-1);j++) {
      CHKERRQ(MatSetValue(B,i,j+d,1.0,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));

  /*
     Set operators and problem type
  */
  CHKERRQ(SVDSetOperators(svd,A,B));
  CHKERRQ(SVDSetProblemType(svd,SVD_GENERALIZED));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDGetIterationNumber(svd,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(SVDGetType(svd,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(SVDGetDimensions(svd,&nsv,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %" PetscInt_FMT "\n",nsv));
  CHKERRQ(SVDGetTolerances(svd,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Get number of converged singular triplets
  */
  CHKERRQ(SVDGetConverged(svd,&nconv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate singular triplets: %" PetscInt_FMT "\n\n",nconv));

  if (nconv>0) {
    /*
       Create vectors. The interface returns u and v as stacked on top of each other
       [u;v] so need to create a special vector (VecNest) to extract them
    */
    CHKERRQ(MatCreateVecs(A,&x,&u));
    CHKERRQ(MatCreateVecs(B,NULL,&v));
    aux[0] = u;
    aux[1] = v;
    CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,aux,&uv));

    CHKERRQ(VecDuplicateVecs(u,nconv,&U));
    CHKERRQ(VecDuplicateVecs(v,nconv,&V));

    /*
       Display singular values and errors relative to the norms
    */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,
         "          sigma           ||r||/||[A;B]||\n"
         "  --------------------- ------------------\n"));
    for (i=0;i<nconv;i++) {
      /*
         Get converged singular triplets: i-th singular value is stored in sigma
      */
      CHKERRQ(SVDGetSingularTriplet(svd,i,&sigma,uv,x));

      /* at this point, u and v can be used normally as individual vectors */
      CHKERRQ(VecCopy(u,U[i]));
      CHKERRQ(VecCopy(v,V[i]));

      /*
         Compute the error associated to each singular triplet
      */
      CHKERRQ(SVDComputeError(svd,i,SVD_ERROR_NORM,&error));

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"       % 6f      ",(double)sigma));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"   % 12g\n",(double)error));
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));

    if (!skiporth) {
      CHKERRQ(VecCheckOrthonormality(U,nconv,NULL,nconv,NULL,NULL,&lev1));
      CHKERRQ(VecCheckOrthonormality(V,nconv,NULL,nconv,NULL,NULL,&lev2));
    }
    if (lev1+lev2<20*tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below the tolerance\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g (U) %g (V)\n",(double)lev1,(double)lev2));
    }
    CHKERRQ(VecDestroyVecs(nconv,&U));
    CHKERRQ(VecDestroyVecs(nconv,&V));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&u));
    CHKERRQ(VecDestroy(&v));
    CHKERRQ(VecDestroy(&uv));
  }

  /*
     Free work space
  */
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      filter: grep -v "Solution method" | grep -v "Number of iterations" | sed -e "s/, maxit=1[0]*$//" | sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double
      test:
         args: -svd_type lapack -m 20 -n 10 -p 6
         suffix: 1
      test:
         args: -svd_type lapack -m 15 -n 20 -p 10 -svd_smallest
         suffix: 2
      test:
         args: -svd_type lapack -m 15 -n 20 -p 21
         suffix: 3
      test:
         args: -svd_type lapack -m 20 -n 15 -p 21
         suffix: 4

   testset:
      args: -m 25 -n 20 -p 21 -svd_smallest -svd_nsv 4
      filter: grep -v "Solution method" | grep -v "Number of iterations" | sed -e "s/, maxit=1[0]*$//" | sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double
      output_file: output/ex45_5.out
      test:
         args: -svd_type trlanczos -svd_ncv 8 -svd_trlanczos_gbidiag {{upper lower}}
         suffix: 5
      test:
         args: -svd_type cross -svd_ncv 10 -svd_cross_explicitmatrix {{0 1}}
         suffix: 5_cross
      test:
         args: -svd_type cyclic -svd_ncv 12 -svd_cyclic_explicitmatrix {{0 1}}
         suffix: 5_cyclic
         requires: !complex

   testset:
      args: -m 15 -n 20 -p 21 -svd_nsv 4 -svd_ncv 9
      filter: grep -v "Solution method" | grep -v "Number of iterations" | sed -e "s/7.884967/7.884968/" | sed -e "s/, maxit=1[0]*$//" | sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double
      output_file: output/ex45_6.out
      test:
         args: -svd_type trlanczos -svd_trlanczos_gbidiag {{single upper lower}} -svd_trlanczos_locking {{0 1}}
         suffix: 6
      test:
         args: -svd_type cross -svd_cross_explicitmatrix {{0 1}}
         suffix: 6_cross
      test:
         args: -svd_type cyclic -svd_cyclic_explicitmatrix {{0 1}}
         suffix: 6_cyclic

   testset:
      args: -m 20 -n 15 -p 21 -svd_nsv 4 -svd_ncv 9
      filter: grep -v "Solution method" | grep -v "Number of iterations" | sed -e "s/, maxit=1[0]*$//" | sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double
      output_file: output/ex45_7.out
      test:
         args: -svd_type trlanczos -svd_trlanczos_gbidiag {{single upper lower}} -svd_trlanczos_restart 0.4
         suffix: 7
      test:
         args: -svd_type cross -svd_cross_explicitmatrix {{0 1}}
         suffix: 7_cross
      test:
         args: -svd_type cyclic -svd_cyclic_explicitmatrix {{0 1}}
         suffix: 7_cyclic

TEST*/
