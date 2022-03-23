/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Shows how to recover symmetry when solving a GHEP as non-symmetric.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

/*
   User context for shell matrix
*/
typedef struct {
  KSP       ksp;
  Mat       B;
  Vec       w;
} CTX_SHELL;

/*
    Matrix-vector product function for user matrix
       y <-- A^{-1}*B*x
    The matrix A^{-1}*B*x is not symmetric, but it is self-adjoint with respect
    to the B-inner product. Here we assume A is symmetric and B is SPD.
 */
PetscErrorCode MatMult_Sinvert0(Mat S,Vec x,Vec y)
{
  CTX_SHELL      *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(S,&ctx));
  CHKERRQ(MatMult(ctx->B,x,ctx->w));
  CHKERRQ(KSPSolve(ctx->ksp,ctx->w,y));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat               A,B,S;      /* matrices */
  EPS               eps;        /* eigenproblem solver context */
  BV                bv;
  Vec               *X,v;
  PetscReal         lev=0.0,tol=1000*PETSC_MACHINE_EPSILON;
  PetscInt          N,n=45,m,Istart,Iend,II,i,j,nconv;
  PetscBool         flag;
  CTX_SHELL         *ctx;
  PetscErrorCode    ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized Symmetric Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,4.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,II,II,2.0/PetscLogScalar(II+2),INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(B,&v,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create a shell matrix S = A^{-1}*B
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscNew(&ctx));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ctx->ksp));
  CHKERRQ(KSPSetOperators(ctx->ksp,A,A));
  CHKERRQ(KSPSetTolerances(ctx->ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(KSPSetFromOptions(ctx->ksp));
  ctx->B = B;
  CHKERRQ(MatCreateVecs(A,&ctx->w,NULL));
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,(void*)ctx,&S));
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT,(void(*)(void))MatMult_Sinvert0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,S,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));  /* even though S is not symmetric */
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSetUp(eps));   /* explicitly call setup */
  CHKERRQ(EPSGetBV(eps,&bv));
  CHKERRQ(BVSetMatrix(bv,B,PETSC_FALSE));  /* set inner product matrix to recover symmetry */
  CHKERRQ(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Display solution and check B-orthogonality
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSGetTolerances(eps,&tol,NULL));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  CHKERRQ(EPSGetConverged(eps,&nconv));
  if (nconv>1) {
    CHKERRQ(VecDuplicateVecs(v,nconv,&X));
    for (i=0;i<nconv;i++) CHKERRQ(EPSGetEigenvector(eps,i,X[i],NULL));
    CHKERRQ(VecCheckOrthonormality(X,nconv,NULL,nconv,B,NULL,&lev));
    if (lev<10*tol) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below the tolerance\n"));
    else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)lev));
    CHKERRQ(VecDestroyVecs(nconv,&X));
  }

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(KSPDestroy(&ctx->ksp));
  CHKERRQ(VecDestroy(&ctx->w));
  CHKERRQ(PetscFree(ctx));
  CHKERRQ(MatDestroy(&S));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      args: -n 18 -eps_nev 4 -eps_max_it 1500
      requires: !single

TEST*/
