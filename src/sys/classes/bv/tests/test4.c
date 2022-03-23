/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV operations, changing the number of active columns.\n\n";

#include <slepcbv.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,v;
  Mat            Q,M;
  BV             X,Y;
  PetscInt       i,j,n=10,kx=6,lx=3,ky=5,ly=2;
  PetscScalar    *q,*z;
  PetscReal      nrm;
  PetscViewer    view;
  PetscBool      verbose,trans;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-kx",&kx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-lx",&lx,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ky",&ky,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ly",&ly,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"First BV with %" PetscInt_FMT " active columns (%" PetscInt_FMT " leading columns) of dimension %" PetscInt_FMT ".\n",kx,lx,n));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Second BV with %" PetscInt_FMT " active columns (%" PetscInt_FMT " leading columns) of dimension %" PetscInt_FMT ".\n",ky,ly,n));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,kx+2));  /* two extra columns to test active columns */
  CHKERRQ(BVSetFromOptions(X));
  CHKERRQ(BVSetActiveColumns(X,lx,kx));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill X entries */
  for (j=0;j<kx+2;j++) {
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
  CHKERRQ(BVSetSizesFromVec(Y,t,ky+1));
  CHKERRQ(BVSetFromOptions(Y));
  CHKERRQ(BVSetActiveColumns(Y,ly,ky));

  /* Fill Y entries */
  for (j=0;j<ky+1;j++) {
    CHKERRQ(BVGetColumn(Y,j,&v));
    CHKERRQ(VecSet(v,(PetscScalar)(j+1)/4.0));
    CHKERRQ(BVRestoreColumn(Y,j,&v));
  }
  if (verbose) CHKERRQ(BVView(Y,view));

  /* Create Mat */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,kx,ky,NULL,&Q));
  CHKERRQ(PetscObjectSetName((PetscObject)Q,"Q"));
  CHKERRQ(MatDenseGetArray(Q,&q));
  for (i=0;i<kx;i++)
    for (j=0;j<ky;j++)
      q[i+j*kx] = (i<j)? 2.0: -0.5;
  CHKERRQ(MatDenseRestoreArray(Q,&q));
  if (verbose) CHKERRQ(MatView(Q,NULL));

  /* Test BVResize */
  CHKERRQ(BVResize(X,kx+4,PETSC_TRUE));

  /* Test BVMult */
  CHKERRQ(BVMult(Y,2.0,0.5,X,Q));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVMult - - - - - - - - -\n"));
    CHKERRQ(BVView(Y,view));
  }

  /* Test BVMultVec */
  CHKERRQ(BVGetColumn(Y,0,&v));
  CHKERRQ(PetscMalloc1(kx-lx,&z));
  z[0] = 2.0;
  for (i=1;i<kx-lx;i++) z[i] = -0.5*z[i-1];
  CHKERRQ(BVMultVec(X,-1.0,1.0,v,z));
  CHKERRQ(PetscFree(z));
  CHKERRQ(BVRestoreColumn(Y,0,&v));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVMultVec - - - - - - -\n"));
    CHKERRQ(BVView(Y,view));
  }

  /* Test BVDot */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,ky,kx,NULL,&M));
  CHKERRQ(PetscObjectSetName((PetscObject)M,"M"));
  CHKERRQ(BVDot(X,Y,M));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVDot - - - - - - - - -\n"));
    CHKERRQ(MatView(M,NULL));
  }

  /* Test BVDotVec */
  CHKERRQ(BVGetColumn(Y,0,&v));
  CHKERRQ(PetscMalloc1(kx-lx,&z));
  CHKERRQ(BVDotVec(X,v,z));
  CHKERRQ(BVRestoreColumn(Y,0,&v));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVDotVec - - - - - - -\n"));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,kx-lx,z,&v));
    CHKERRQ(PetscObjectSetName((PetscObject)v,"z"));
    CHKERRQ(VecView(v,view));
    CHKERRQ(VecDestroy(&v));
  }
  CHKERRQ(PetscFree(z));

  /* Test BVMultInPlace and BVScale */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-trans",&trans));
  if (trans) {
    Mat Qt;
    CHKERRQ(MatTranspose(Q,MAT_INITIAL_MATRIX,&Qt));
    CHKERRQ(BVMultInPlaceHermitianTranspose(X,Qt,lx+1,ky));
    CHKERRQ(MatDestroy(&Qt));
  } else CHKERRQ(BVMultInPlace(X,Q,lx+1,ky));
  CHKERRQ(BVScale(X,2.0));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After BVMultInPlace - - - - -\n"));
    CHKERRQ(BVView(X,view));
  }

  /* Test BVNorm */
  CHKERRQ(BVNormColumn(X,lx,NORM_2,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"2-Norm of X[%" PetscInt_FMT "] = %g\n",lx,(double)nrm));
  CHKERRQ(BVNorm(X,NORM_FROBENIUS,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Frobenius Norm of X = %g\n",(double)nrm));

  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(MatDestroy(&Q));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test4_1.out
      args: -n 18 -kx 12 -ky 8
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_vecs_vmip
         args: -bv_type vecs -bv_vecs_vmip 1
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda
      test:
         suffix: 2
         args: -bv_type {{vecs contiguous svec mat}shared output} -trans

TEST*/
