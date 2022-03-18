/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test ST with shell matrices.\n\n";

#include <slepcst.h>

static PetscErrorCode MatGetDiagonal_Shell(Mat S,Vec diag);
static PetscErrorCode MatMultTranspose_Shell(Mat S,Vec x,Vec y);
static PetscErrorCode MatMult_Shell(Mat S,Vec x,Vec y);
static PetscErrorCode MatDuplicate_Shell(Mat S,MatDuplicateOption op,Mat *M);

static PetscErrorCode MyShellMatCreate(Mat *A,Mat *M)
{
  MPI_Comm       comm;
  PetscInt       n;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(*A,&n,NULL));
  CHKERRQ(PetscObjectGetComm((PetscObject)*A,&comm));
  CHKERRQ(MatCreateShell(comm,PETSC_DECIDE,PETSC_DECIDE,n,n,A,M));
  CHKERRQ(MatShellSetOperation(*M,MATOP_MULT,(void(*)(void))MatMult_Shell));
  CHKERRQ(MatShellSetOperation(*M,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Shell));
  CHKERRQ(MatShellSetOperation(*M,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Shell));
  CHKERRQ(MatShellSetOperation(*M,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_Shell));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A,S,mat[1];
  ST             st;
  Vec            v,w;
  STType         type;
  KSP            ksp;
  PC             pc;
  PetscScalar    sigma;
  PetscInt       n=10,i,Istart,Iend;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian with shell matrices, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix for the 1-D Laplacian
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

  /* create the shell version of A */
  CHKERRQ(MyShellMatCreate(&A,&S));

  /* work vectors */
  CHKERRQ(MatCreateVecs(A,&v,&w));
  CHKERRQ(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = S;
  CHKERRQ(STSetMatrices(st,1,mat));
  CHKERRQ(STSetTransform(st,PETSC_TRUE));
  CHKERRQ(STSetFromOptions(st));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Apply the transformed operator for several ST's
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* shift, sigma=0.0 */
  CHKERRQ(STSetUp(st));
  CHKERRQ(STGetType(st,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));
  CHKERRQ(STApplyTranspose(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* shift, sigma=0.1 */
  sigma = 0.1;
  CHKERRQ(STSetShift(st,sigma));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* sinvert, sigma=0.1 */
  CHKERRQ(STPostSolve(st));   /* undo changes if inplace */
  CHKERRQ(STSetType(st,STSINVERT));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPGMRES));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCJACOBI));
  CHKERRQ(STGetType(st,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* sinvert, sigma=-0.5 */
  sigma = -0.5;
  CHKERRQ(STSetShift(st,sigma));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STDestroy(&st));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&S));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  ierr = SlepcFinalize();
  return ierr;
}

static PetscErrorCode MatMult_Shell(Mat S,Vec x,Vec y)
{
  Mat               *A;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatMult(*A,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Shell(Mat S,Vec x,Vec y)
{
  Mat               *A;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatMultTranspose(*A,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_Shell(Mat S,Vec diag)
{
  Mat               *A;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MatGetDiagonal(*A,diag));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_Shell(Mat S,MatDuplicateOption op,Mat *M)
{
  Mat            *A;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(S,&A));
  CHKERRQ(MyShellMatCreate(A,M));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      args: -st_matmode {{inplace shell}}
      output_file: output/test1_1.out
      requires: !single

TEST*/
