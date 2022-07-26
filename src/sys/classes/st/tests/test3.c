/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian plus diagonal, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix for the 1-D Laplacian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    PetscCall(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (i>0) {
      PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
      PetscCall(MatSetValue(B,i,i,(PetscScalar)i,INSERT_VALUES));
    } else PetscCall(MatSetValue(B,i,i,-1.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A,&v,&w));
  PetscCall(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  mat[1] = B;
  PetscCall(STSetMatrices(st,2,mat));
  PetscCall(STSetTransform(st,PETSC_TRUE));
  PetscCall(STSetFromOptions(st));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Apply the transformed operator for several ST's
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* shift, sigma=0.0 */
  PetscCall(STSetUp(st));
  PetscCall(STGetType(st,&type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* shift, sigma=0.1 */
  sigma = 0.1;
  PetscCall(STSetShift(st,sigma));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* sinvert, sigma=0.1 */
  PetscCall(STPostSolve(st));   /* undo changes if inplace */
  PetscCall(STSetType(st,STSINVERT));
  PetscCall(STGetType(st,&type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* sinvert, sigma=-0.5 */
  sigma = -0.5;
  PetscCall(STSetShift(st,sigma));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* cayley, sigma=-0.5, tau=-0.5 (equal to sigma by default) */
  PetscCall(STPostSolve(st));   /* undo changes if inplace */
  PetscCall(STSetType(st,STCAYLEY));
  PetscCall(STSetUp(st));
  PetscCall(STGetType(st,&type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(STCayleyGetAntishift(st,&tau));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g, antishift=%g\n",(double)PetscRealPart(sigma),(double)PetscRealPart(tau)));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* cayley, sigma=1.1, tau=1.1 (still equal to sigma) */
  sigma = 1.1;
  PetscCall(STSetShift(st,sigma));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(STCayleyGetAntishift(st,&tau));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g, antishift=%g\n",(double)PetscRealPart(sigma),(double)PetscRealPart(tau)));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* cayley, sigma=1.1, tau=-1.0 */
  tau = -1.0;
  PetscCall(STCayleySetAntishift(st,tau));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(STCayleyGetAntishift(st,&tau));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g, antishift=%g\n",(double)PetscRealPart(sigma),(double)PetscRealPart(tau)));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Check inner product matrix in Cayley
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(STGetBilinearForm(st,&M));
  PetscCall(MatMult(M,v,w));
  PetscCall(VecView(w,NULL));

  PetscCall(STDestroy(&st));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&M));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -st_matmode {{copy inplace shell}}
      output_file: output/test3_1.out
      requires: !single

TEST*/
