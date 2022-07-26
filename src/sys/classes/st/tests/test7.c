/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test ST with one matrix and split preconditioner.\n\n";

#include <slepcst.h>

int main(int argc,char **argv)
{
  Mat            A,Pa,Pmat,mat[1];
  ST             st;
  KSP            ksp;
  PC             pc;
  Vec            v,w;
  STType         type;
  PetscBool      flg;
  PetscScalar    sigma;
  PetscInt       n=10,i,Istart,Iend;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix for the 1-D Laplacian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,i,i,2.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A,&v,&w));
  PetscCall(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the split preconditioner matrix (one diagonal)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Pa));
  PetscCall(MatSetSizes(Pa,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(Pa));
  PetscCall(MatSetUp(Pa));

  PetscCall(MatGetOwnershipRange(Pa,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(Pa,i,i,2.0,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Pa,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pa,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(STCreate(PETSC_COMM_WORLD,&st));
  mat[0] = A;
  PetscCall(STSetMatrices(st,1,mat));
  mat[0] = Pa;
  PetscCall(STSetSplitPreconditioner(st,1,mat,SAME_NONZERO_PATTERN));
  PetscCall(STSetTransform(st,PETSC_TRUE));
  PetscCall(STSetFromOptions(st));
  PetscCall(STCayleySetAntishift(st,-0.3));   /* only relevant for cayley */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Form the preconditioner matrix and print it
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscObjectTypeCompareAny((PetscObject)st,&flg,STSINVERT,STCAYLEY,""));
  if (flg) {
    PetscCall(STGetKSP(st,&ksp));
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(STGetOperator(st,NULL));
    PetscCall(PCGetOperators(pc,NULL,&Pmat));
    PetscCall(MatView(Pmat,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Apply the operator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* sigma=0.0 */
  PetscCall(STSetUp(st));
  PetscCall(STGetType(st,&type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type));
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  /* sigma=0.1 */
  sigma = 0.1;
  PetscCall(STSetShift(st,sigma));
  PetscCall(STGetShift(st,&sigma));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma)));
  if (flg) {
    PetscCall(STGetOperator(st,NULL));
    PetscCall(PCGetOperators(pc,NULL,&Pmat));
    PetscCall(MatView(Pmat,NULL));
  }
  PetscCall(STApply(st,v,w));
  PetscCall(VecView(w,NULL));

  PetscCall(STDestroy(&st));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Pa));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -st_type {{cayley shift sinvert}separate output}
      requires: !single

TEST*/
