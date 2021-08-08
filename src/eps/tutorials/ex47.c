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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(S,&ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->B,x,ctx->w);CHKERRQ(ierr);
  ierr = KSPSolve(ctx->ksp,ctx->w,y);CHKERRQ(ierr);
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if (!flag) m=n;
  N = n*m;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized Symmetric Eigenproblem, N=%D (%Dx%D grid)\n\n",N,n,m);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) { ierr = MatSetValue(A,II,II-n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A,II,II+n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j>0) { ierr = MatSetValue(A,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<n-1) { ierr = MatSetValue(A,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(A,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(B,II,II,2.0/PetscLogScalar(II+2),INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&v,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create a shell matrix S = A^{-1}*B
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ctx->ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ctx->ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
  ctx->B = B;
  ierr = MatCreateVecs(A,&ctx->w,NULL);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,(void*)ctx,&S);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT,(void(*)(void))MatMult_Sinvert0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,S,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);  /* even though S is not symmetric */
  ierr = EPSSetTolerances(eps,tol,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSetUp(eps);CHKERRQ(ierr);   /* explicitly call setup */
  ierr = EPSGetBV(eps,&bv);CHKERRQ(ierr);
  ierr = BVSetMatrix(bv,B,PETSC_FALSE);CHKERRQ(ierr);  /* set inner product matrix to recover symmetry */
  ierr = EPSSolve(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Display solution and check B-orthogonality
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSGetTolerances(eps,&tol,NULL);CHKERRQ(ierr);
  ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  if (nconv>1) {
    ierr = VecDuplicateVecs(v,nconv,&X);CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      ierr = EPSGetEigenvector(eps,i,X[i],NULL);CHKERRQ(ierr);
    }
    ierr = VecCheckOrthonormality(X,nconv,NULL,nconv,B,NULL,&lev);CHKERRQ(ierr);
    if (lev<10*tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below the tolerance\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)lev);CHKERRQ(ierr);
    }
    ierr = VecDestroyVecs(nconv,&X);CHKERRQ(ierr);
  }

  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      args: -n 18 -eps_nev 4 -eps_max_it 1500
      requires: !single

TEST*/
