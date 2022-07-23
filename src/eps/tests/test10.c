/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Computes the smallest nonzero eigenvalue of the Laplacian of a graph.\n\n"
  "This example illustrates EPSSetDeflationSpace(). The example graph corresponds to a "
  "2-D regular mesh. The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

int main (int argc,char **argv)
{
  EPS            eps;             /* eigenproblem solver context */
  Mat            A;               /* operator matrix */
  Vec            x;
  PetscInt       N,n=10,m,i,j,II,Istart,Iend,nev;
  PetscScalar    w;
  PetscBool      flag;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nFiedler vector of a 2-D regular mesh, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     In this example, A = L(G), where L is the Laplacian of graph G, i.e.
     Lii = degree of node i, Lij = -1 if edge (i,j) exists in G
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    w = 0.0;
    if (i>0) { PetscCall(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES)); w=w+1.0; }
    if (i<m-1) { PetscCall(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES)); w=w+1.0; }
    if (j>0) { PetscCall(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES)); w=w+1.0; }
    if (j<n-1) { PetscCall(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES)); w=w+1.0; }
    PetscCall(MatSetValue(A,II,II,w,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
     Set operators. In this case, it is a standard eigenvalue problem
  */
  PetscCall(EPSSetOperators(eps,A,NULL));
  PetscCall(EPSSetProblemType(eps,EPS_HEP));

  /*
     Select portion of spectrum
  */
  PetscCall(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));

  /*
     Set solver parameters at runtime
  */
  PetscCall(EPSSetFromOptions(eps));

  /*
     Attach deflation space: in this case, the matrix has a constant
     nullspace, [1 1 ... 1]^T is the eigenvector of the zero eigenvalue
  */
  PetscCall(MatCreateVecs(A,&x,NULL));
  PetscCall(VecSet(x,1.0));
  PetscCall(EPSSetDeflationSpace(eps,1,&x));
  PetscCall(VecDestroy(&x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetDimensions(eps,&nev,NULL,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -eps_nev 4 -m 11 -eps_max_it 500
      output_file: output/test10_1.out
      test:
         suffix: 1
         args: -eps_type {{krylovschur arnoldi gd jd rqcg}}
      test:
         suffix: 1_lobpcg
         args: -eps_type lobpcg -eps_lobpcg_blocksize 6
      test:
         suffix: 1_lanczos
         args: -eps_type lanczos -eps_lanczos_reorthog local
         requires: !single
      test:
         suffix: 1_gd2
         args: -eps_type gd -eps_gd_double_expansion

TEST*/
