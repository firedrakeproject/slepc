/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test tensor BV.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  Vec               t,v;
  Mat               S,M,Q;
  BV                U,V,UU;
  PetscInt          i,ii,j,jj,n=10,k=6,l=3,d=3,deg,id,lds;
  PetscScalar       *pS,*q;
  PetscReal         norm;
  PetscViewer       view;
  PetscBool         verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-d",&d,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test tensor BV of degree %" PetscInt_FMT " with %" PetscInt_FMT " columns of dimension %" PetscInt_FMT "*d.\n",d,k,n));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));

  /* Create BV object U */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&U));
  CHKERRQ(PetscObjectSetName((PetscObject)U,"U"));
  CHKERRQ(BVSetSizesFromVec(U,t,k+d-1));
  CHKERRQ(BVSetFromOptions(U));
  CHKERRQ(PetscObjectSetName((PetscObject)U,"U"));

  /* Fill first d columns of U */
  for (j=0;j<d;j++) {
    CHKERRQ(BVGetColumn(U,j,&v));
    CHKERRQ(VecSet(v,0.0));
    for (i=0;i<4;i++) {
      if (i+j<n) CHKERRQ(VecSetValue(v,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(U,j,&v));
  }

  /* Create tensor BV */
  CHKERRQ(BVCreateTensor(U,d,&V));
  CHKERRQ(PetscObjectSetName((PetscObject)V,"V"));
  CHKERRQ(BVTensorGetDegree(V,&deg));
  PetscCheck(deg==d,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Wrong degree");

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(BVView(V,view));
  CHKERRQ(PetscViewerPopFormat(view));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(BVView(V,view));
  }

  /* Build first column from previously introduced coefficients */
  CHKERRQ(BVTensorBuildFirstColumn(V,d));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After building the first column - - - - -\n"));
    CHKERRQ(BVView(V,view));
  }

  /* Test orthogonalization */
  CHKERRQ(BVTensorGetFactors(V,&UU,&S));
  CHKERRQ(BVGetActiveColumns(UU,NULL,&j));
  CHKERRQ(BVGetSizes(UU,NULL,NULL,&id));
  PetscCheck(id==k+d-1,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Wrong dimensions");
  lds = id*d;
  for (jj=1;jj<k;jj++) {
    /* set new orthogonal column in U */
    CHKERRQ(BVGetColumn(UU,j,&v));
    CHKERRQ(VecSet(v,0.0));
    for (i=0;i<4;i++) {
      if (i+j<n) CHKERRQ(VecSetValue(v,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(UU,j,&v));
    CHKERRQ(BVOrthonormalizeColumn(UU,j,PETSC_TRUE,NULL,NULL));
    j++;
    CHKERRQ(BVSetActiveColumns(UU,0,j));
    /* set new column of S */
    CHKERRQ(MatDenseGetArray(S,&pS));
    for (ii=0;ii<d;ii++) {
      for (i=0;i<ii+jj+1;i++) {
        pS[i+ii*id+jj*lds] = (PetscScalar)(2*ii+i+0.5*jj);
      }
    }
    CHKERRQ(MatDenseRestoreArray(S,&pS));
    CHKERRQ(BVOrthonormalizeColumn(V,jj,PETSC_TRUE,NULL,NULL));
  }
  CHKERRQ(BVTensorRestoreFactors(V,&UU,&S));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After orthogonalization - - - - -\n"));
    CHKERRQ(BVView(V,view));
  }

  /* Check orthogonality */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));
  CHKERRQ(BVDot(V,V,M));
  CHKERRQ(MatShift(M,-1.0));
  CHKERRQ(MatNorm(M,NORM_1,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)norm));

  /* Test BVTensorCompress */
  CHKERRQ(BVSetActiveColumns(V,0,l));
  CHKERRQ(BVTensorCompress(V,0));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVTensorCompress - - - - -\n"));
    CHKERRQ(BVView(V,view));
  }

  /* Create Mat */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,l,NULL,&Q));
  CHKERRQ(PetscObjectSetName((PetscObject)Q,"Q"));
  CHKERRQ(MatDenseGetArray(Q,&q));
  for (i=0;i<k;i++)
    for (j=0;j<l;j++)
      q[i+j*k] = (i<j)? 2.0: -0.5;
  CHKERRQ(MatDenseRestoreArray(Q,&q));
  if (verbose) CHKERRQ(MatView(Q,NULL));

  /* Test BVMultInPlace */
  CHKERRQ(BVMultInPlace(V,Q,1,l));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVMultInPlace - - - - -\n"));
    CHKERRQ(BVView(V,view));
  }

  /* Test BVNorm */
  CHKERRQ(BVNorm(V,NORM_1,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm: %g\n",(double)norm));

  CHKERRQ(BVDestroy(&U));
  CHKERRQ(BVDestroy(&V));
  CHKERRQ(MatDestroy(&Q));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      args: -bv_type {{vecs contiguous svec mat}shared output}
      output_file: output/test16_1.out
      filter: grep -v "doing matmult"

TEST*/
