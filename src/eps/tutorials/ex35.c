/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Shell spectral transformations with a non-injective mapping. "
  "Implements spectrum folding for the 2-D Laplacian, as in ex24.c.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n";

#include <slepceps.h>

/* Context for spectrum folding spectral transformation */
typedef struct {
  Mat         A;
  Vec         w;
  PetscScalar target;
} FoldShellST;

/* Routines for shell spectral transformation */
PetscErrorCode STCreate_Fold(Mat,PetscScalar,FoldShellST**);
PetscErrorCode STApply_Fold(ST,Vec,Vec);
PetscErrorCode STDestroy_Fold(FoldShellST*);

int main (int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  EPS            eps;             /* eigenproblem solver context */
  ST             st;              /* spectral transformation context */
  FoldShellST    *fold;           /* user-defined spectral transform context */
  EPSType        type;
  PetscInt       N,n=10,m,i,j,II,Istart,Iend,nev;
  PetscBool      isShell,terse,flag;
  PetscScalar    target=1.1;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m = n;
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-target",&target,NULL));
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum Folding via shell ST, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid) target=%3.2f\n\n",N,n,m,(double)PetscRealPart(target)));

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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetTarget(eps,target));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STSetType(st,STSHELL));
  CHKERRQ(EPSSetFromOptions(eps));

  /*
     Initialize shell spectral transformation
  */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)st,STSHELL,&isShell));
  if (isShell) {
    /* Change sorting criterion since this shell ST computes eigenvalues
       of the transformed operator closest to 0 */
    CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));

    /* Create the context for the user-defined spectral transform */
    CHKERRQ(STCreate_Fold(A,target,&fold));
    CHKERRQ(STShellSetContext(st,fold));

    /* Set callback function for applying the operator (in this case we do not
       provide a back-transformation callback since the mapping is not one-to-one) */
    CHKERRQ(STShellSetApply(st,STApply_Fold));
    CHKERRQ(PetscObjectSetName((PetscObject)st,"STFOLD"));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  } else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  if (isShell) {
    CHKERRQ(STDestroy_Fold(fold));
  }
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*
   STCreate_Fold - Creates the spectrum folding ST context.

   Input Parameter:
+  A - problem matrix
-  target - target value

   Output Parameter:
.  fold - user-defined spectral transformation context
*/
PetscErrorCode STCreate_Fold(Mat A,PetscScalar target,FoldShellST **fold)
{
  FoldShellST    *newctx;

  PetscFunctionBeginUser;
  CHKERRQ(PetscNew(&newctx));
  newctx->A = A;
  CHKERRQ(PetscObjectReference((PetscObject)A));
  newctx->target = target;
  CHKERRQ(MatCreateVecs(A,&newctx->w,NULL));
  *fold = newctx;
  PetscFunctionReturn(0);
}

/*
   STApply_Fold - Applies the operator (A-target*I)^2 to a given vector.

   Input Parameters:
+  st - spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector
*/
PetscErrorCode STApply_Fold(ST st,Vec x,Vec y)
{
  FoldShellST    *fold;
  PetscScalar    sigma;

  PetscFunctionBeginUser;
  CHKERRQ(STShellGetContext(st,&fold));
  sigma = -fold->target;
  CHKERRQ(MatMult(fold->A,x,fold->w));
  CHKERRQ(VecAXPY(fold->w,sigma,x));
  CHKERRQ(MatMult(fold->A,fold->w,y));
  CHKERRQ(VecAXPY(y,sigma,fold->w));
  PetscFunctionReturn(0);
}

/*
   STDestroy_Fold - This routine destroys the shell ST context.

   Input Parameter:
.  fold - user-defined spectral transformation context
*/
PetscErrorCode STDestroy_Fold(FoldShellST *fold)
{
  PetscFunctionBeginUser;
  CHKERRQ(MatDestroy(&fold->A));
  CHKERRQ(VecDestroy(&fold->w));
  CHKERRQ(PetscFree(fold));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -m 11 -eps_nev 4 -terse
      suffix: 1
      requires: !single

TEST*/
