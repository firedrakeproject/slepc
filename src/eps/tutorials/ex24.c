/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Spectrum folding for a standard symmetric eigenproblem.\n\n"
  "The problem matrix is the 2-D Laplacian.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n";

#include <slepceps.h>

/*
   User context for spectrum folding
*/
typedef struct {
  Mat       A;
  Vec       w;
  PetscReal target;
} CTX_FOLD;

/*
   Auxiliary routines
*/
PetscErrorCode MatMult_Fold(Mat,Vec,Vec);
PetscErrorCode RayleighQuotient(Mat,Vec,PetscScalar*);
PetscErrorCode ComputeResidualNorm(Mat,PetscScalar,Vec,PetscReal*);

int main(int argc,char **argv)
{
  Mat            A,M,P;       /* problem matrix, shell matrix and preconditioner */
  Vec            x;           /* eigenvector */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation */
  KSP            ksp;
  PC             pc;
  EPSType        type;
  CTX_FOLD       *ctx;
  PetscInt       nconv,N,n=10,m,nloc,mloc,Istart,Iend,II,i,j;
  PetscReal      *error,*evals,target=0.0,tol;
  PetscScalar    lambda;
  PetscBool      flag,terse,errok,hasmat;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-target",&target,NULL));
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum Folding, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid) target=%f\n\n",N,n,m,(double)target));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the 5-point stencil Laplacian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,4.0,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&x,NULL));
  CHKERRQ(MatGetLocalSize(A,&nloc,&mloc));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create shell matrix to perform spectrum folding
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscNew(&ctx));
  ctx->A = A;
  ctx->target = target;
  CHKERRQ(VecDuplicate(x,&ctx->w));

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,nloc,mloc,N,N,ctx,&M));
  CHKERRQ(MatShellSetOperation(M,MATOP_MULT,(void(*)(void))MatMult_Fold));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,M,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  CHKERRQ(EPSSetFromOptions(eps));

  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)eps,&flag,EPSGD,EPSJD,EPSBLOPEX,EPSLOBPCG,EPSRQCG,""));
  if (flag) {
    /*
       Build preconditioner specific for this application (diagonal of A^2)
    */
    CHKERRQ(MatGetRowSum(A,x));
    CHKERRQ(VecScale(x,-1.0));
    CHKERRQ(VecShift(x,20.0));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&P));
    CHKERRQ(MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,N,N));
    CHKERRQ(MatSetFromOptions(P));
    CHKERRQ(MatSetUp(P));
    CHKERRQ(MatDiagonalSet(P,x,INSERT_VALUES));
    /*
       Set diagonal preconditioner
    */
    CHKERRQ(EPSGetST(eps,&st));
    CHKERRQ(STSetType(st,STPRECOND));
    CHKERRQ(STSetPreconditionerMat(st,P));
    CHKERRQ(MatDestroy(&P));
    CHKERRQ(STGetKSP(st,&ksp));
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCJACOBI));
    CHKERRQ(STPrecondGetKSPHasMat(st,&hasmat));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Preconditioned solver, hasmat=%s\n",hasmat?"true":"false"));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(EPSGetTolerances(eps,&tol,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSGetConverged(eps,&nconv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %" PetscInt_FMT "\n\n",nconv));
  if (nconv>0) {
    CHKERRQ(PetscMalloc2(nconv,&evals,nconv,&error));
    for (i=0;i<nconv;i++) {
      /*  Get i-th eigenvector, compute eigenvalue approximation from
          Rayleigh quotient and compute residual norm */
      CHKERRQ(EPSGetEigenpair(eps,i,NULL,NULL,x,NULL));
      CHKERRQ(RayleighQuotient(A,x,&lambda));
      CHKERRQ(ComputeResidualNorm(A,lambda,x,&error[i]));
#if defined(PETSC_USE_COMPLEX)
      evals[i] = PetscRealPart(lambda);
#else
      evals[i] = lambda;
#endif
    }
    CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
    if (!terse) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,
           "           k              ||Ax-kx||\n"
           "   ----------------- ------------------\n"));
      for (i=0;i<nconv;i++) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12.2g\n",(double)evals[i],(double)error[i]));
      }
    } else {
      errok = PETSC_TRUE;
      for (i=0;i<nconv;i++) errok = (errok && error[i]<5.0*tol)? PETSC_TRUE: PETSC_FALSE;
      if (!errok) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Problem: some of the first %" PetscInt_FMT " relative errors are higher than the tolerance\n\n",nconv));
      } else {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," nconv=%" PetscInt_FMT " eigenvalues computed up to the required tolerance:",nconv));
        for (i=0;i<nconv;i++) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %.5f",(double)evals[i]));
        }
      }
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    CHKERRQ(PetscFree2(evals,error));
  }

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(VecDestroy(&ctx->w));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(PetscFree(ctx));
  ierr = SlepcFinalize();
  return ierr;
}

/*
    Matrix-vector product subroutine for the spectrum folding.
       y <-- (A-t*I)^2*x
 */
PetscErrorCode MatMult_Fold(Mat M,Vec x,Vec y)
{
  CTX_FOLD       *ctx;
  PetscScalar    sigma;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(M,&ctx));
  sigma = -ctx->target;
  CHKERRQ(MatMult(ctx->A,x,ctx->w));
  CHKERRQ(VecAXPY(ctx->w,sigma,x));
  CHKERRQ(MatMult(ctx->A,ctx->w,y));
  CHKERRQ(VecAXPY(y,sigma,ctx->w));
  PetscFunctionReturn(0);
}

/*
    Computes the Rayleigh quotient of a vector x
       r <-- x^T*A*x       (assumes x has unit norm)
 */
PetscErrorCode RayleighQuotient(Mat A,Vec x,PetscScalar *r)
{
  Vec            Ax;

  PetscFunctionBeginUser;
  CHKERRQ(VecDuplicate(x,&Ax));
  CHKERRQ(MatMult(A,x,Ax));
  CHKERRQ(VecDot(Ax,x,r));
  CHKERRQ(VecDestroy(&Ax));
  PetscFunctionReturn(0);
}

/*
    Computes the residual norm of an approximate eigenvector x, |A*x-lambda*x|
 */
PetscErrorCode ComputeResidualNorm(Mat A,PetscScalar lambda,Vec x,PetscReal *r)
{
  Vec            Ax;

  PetscFunctionBeginUser;
  CHKERRQ(VecDuplicate(x,&Ax));
  CHKERRQ(MatMult(A,x,Ax));
  CHKERRQ(VecAXPY(Ax,-lambda,x));
  CHKERRQ(VecNorm(Ax,NORM_2,r));
  CHKERRQ(VecDestroy(&Ax));
  PetscFunctionReturn(0);
}

/*TEST

   testset:
      args: -n 15 -eps_nev 1 -eps_ncv 12 -eps_max_it 1000 -eps_tol 1e-5 -terse
      filter: grep -v Solution
      test:
         suffix: 1
      test:
         suffix: 1_lobpcg
         args: -eps_type lobpcg
         requires: !single
      test:
         suffix: 1_gd
         args: -eps_type gd
         requires: !single

TEST*/
