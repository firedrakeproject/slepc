/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test ST with two matrices.\n\n";

#include <slepcst.h>

int main(int argc,char **argv)
{
  Mat            A,B,M,mat[2];
  ST             st;
  Vec            v,w;
  STType         type;
  PetscScalar    sigma,tau;
  PetscInt       n=10,i,Istart,Iend;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian plus diagonal, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix for the 1-D Laplacian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (i>0) {
      CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
      CHKERRQ(MatSetValue(B,i,i,(PetscScalar)i,INSERT_VALUES));
    } else CHKERRQ(MatSetValue(B,i,i,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&v,&w));
  CHKERRQ(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  mat[1] = B;
  CHKERRQ(STSetMatrices(st,2,mat));
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

  /* cayley, sigma=-0.5, tau=-0.5 (equal to sigma by default) */
  CHKERRQ(STPostSolve(st));   /* undo changes if inplace */
  CHKERRQ(STSetType(st,STCAYLEY));
  CHKERRQ(STSetUp(st));
  CHKERRQ(STGetType(st,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(STCayleyGetAntishift(st,&tau));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g, antishift=%g\n",(double)PetscRealPart(sigma),(double)PetscRealPart(tau)));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* cayley, sigma=1.1, tau=1.1 (still equal to sigma) */
  sigma = 1.1;
  CHKERRQ(STSetShift(st,sigma));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(STCayleyGetAntishift(st,&tau));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g, antishift=%g\n",(double)PetscRealPart(sigma),(double)PetscRealPart(tau)));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* cayley, sigma=1.1, tau=-1.0 */
  tau = -1.0;
  CHKERRQ(STCayleySetAntishift(st,tau));
  CHKERRQ(STGetShift(st,&sigma));
  CHKERRQ(STCayleyGetAntishift(st,&tau));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g, antishift=%g\n",(double)PetscRealPart(sigma),(double)PetscRealPart(tau)));
  CHKERRQ(STApply(st,v,w));
  CHKERRQ(VecView(w,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Check inner product matrix in Cayley
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(STGetBilinearForm(st,&M));
  CHKERRQ(MatMult(M,v,w));
  CHKERRQ(VecView(w,NULL));

  CHKERRQ(STDestroy(&st));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -st_matmode {{copy inplace shell}}
      output_file: output/test3_1.out
      requires: !single

TEST*/
