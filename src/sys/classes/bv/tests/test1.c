/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV operations.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  Vec               t,v;
  Mat               Q=NULL,M=NULL;
  BV                X,Y;
  PetscInt          i,j,n=10,k=5,l=3,nloc;
  PetscMPIInt       rank;
  PetscScalar       *q,*z;
  const PetscScalar *pX;
  PetscReal         nrm;
  PetscViewer       view;
  PetscBool         verbose,matcuda;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-matcuda",&matcuda));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV with %" PetscInt_FMT " columns of dimension %" PetscInt_FMT ".\n",k,n));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));
  CHKERRQ(VecGetLocalSize(t,&nloc));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetFromOptions(X));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(BVView(X,view));
  CHKERRQ(PetscViewerPopFormat(view));

  /* Fill X entries */
  for (j=0;j<k;j++) {
    CHKERRQ(BVGetColumn(X,j,&v));
    CHKERRQ(VecSet(v,0.0));
    for (i=0;i<4;i++) {
      if (i+j<n) CHKERRQ(VecSetValue(v,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(X,j,&v));
  }
  if (verbose) CHKERRQ(BVView(X,view));

  /* Create BV object Y */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&Y));
  CHKERRQ(PetscObjectSetName((PetscObject)Y,"Y"));
  CHKERRQ(BVSetSizesFromVec(Y,t,l));
  CHKERRQ(BVSetFromOptions(Y));

  /* Fill Y entries */
  for (j=0;j<l;j++) {
    CHKERRQ(BVGetColumn(Y,j,&v));
    CHKERRQ(VecSet(v,(PetscScalar)(j+1)/4.0));
    CHKERRQ(BVRestoreColumn(Y,j,&v));
  }
  if (verbose) CHKERRQ(BVView(Y,view));

  /* Create Mat */
  if (matcuda) {
#if defined(PETSC_HAVE_CUDA)
    CHKERRQ(MatCreateSeqDenseCUDA(PETSC_COMM_SELF,k,l,NULL,&Q));
#endif
  } else CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,l,NULL,&Q));
  CHKERRQ(PetscObjectSetName((PetscObject)Q,"Q"));
  CHKERRQ(MatDenseGetArray(Q,&q));
  for (i=0;i<k;i++)
    for (j=0;j<l;j++)
      q[i+j*k] = (i<j)? 2.0: -0.5;
  CHKERRQ(MatDenseRestoreArray(Q,&q));
  if (verbose) CHKERRQ(MatView(Q,NULL));

  /* Test BVMult */
  CHKERRQ(BVMult(Y,2.0,1.0,X,Q));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVMult - - - - - - - - -\n"));
    CHKERRQ(BVView(Y,view));
  }

  /* Test BVMultVec */
  CHKERRQ(BVGetColumn(Y,0,&v));
  CHKERRQ(PetscMalloc1(k,&z));
  z[0] = 2.0;
  for (i=1;i<k;i++) z[i] = -0.5*z[i-1];
  CHKERRQ(BVMultVec(X,-1.0,1.0,v,z));
  CHKERRQ(PetscFree(z));
  CHKERRQ(BVRestoreColumn(Y,0,&v));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVMultVec - - - - - - -\n"));
    CHKERRQ(BVView(Y,view));
  }

  /* Test BVDot */
  if (matcuda) {
#if defined(PETSC_HAVE_CUDA)
    CHKERRQ(MatCreateSeqDenseCUDA(PETSC_COMM_SELF,l,k,NULL,&M));
#endif
  } else CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,l,k,NULL,&M));
  CHKERRQ(PetscObjectSetName((PetscObject)M,"M"));
  CHKERRQ(BVDot(X,Y,M));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVDot - - - - - - - - -\n"));
    CHKERRQ(MatView(M,NULL));
  }

  /* Test BVDotVec */
  CHKERRQ(BVGetColumn(Y,0,&v));
  CHKERRQ(PetscMalloc1(k,&z));
  CHKERRQ(BVDotVec(X,v,z));
  CHKERRQ(BVRestoreColumn(Y,0,&v));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVDotVec - - - - - - -\n"));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,k,z,&v));
    CHKERRQ(PetscObjectSetName((PetscObject)v,"z"));
    CHKERRQ(VecView(v,view));
    CHKERRQ(VecDestroy(&v));
  }
  CHKERRQ(PetscFree(z));

  /* Test BVMultInPlace and BVScale */
  CHKERRQ(BVMultInPlace(X,Q,1,l));
  CHKERRQ(BVScale(X,2.0));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVMultInPlace - - - - -\n"));
    CHKERRQ(BVView(X,view));
  }

  /* Test BVNorm */
  CHKERRQ(BVNormColumn(X,0,NORM_2,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"2-Norm of X[0] = %g\n",(double)nrm));
  CHKERRQ(BVNorm(X,NORM_FROBENIUS,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Frobenius Norm of X = %g\n",(double)nrm));

  /* Test BVGetArrayRead */
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (!rank) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"First row of X =\n"));
    CHKERRQ(BVGetArrayRead(X,&pX));
    for (i=0;i<k;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%g ",(double)PetscRealPart(pX[i*nloc])));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    CHKERRQ(BVRestoreArrayRead(X,&pX));
  }

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(MatDestroy(&Q));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -bv_type {{vecs contiguous svec mat}separate output} -verbose

   testset:
      args: -bv_type svec -vec_type cuda -verbose
      requires: cuda
      output_file: output/test1_1_cuda.out
      test:
         suffix: 1_cuda
      test:
         suffix: 1_cuda_mat
         args: -matcuda
         filter: sed -e "s/seqdensecuda/seqdense/"

TEST*/
