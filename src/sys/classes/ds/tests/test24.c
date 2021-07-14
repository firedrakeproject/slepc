/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGSVD with compact storage and rectangular matrix A.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  Mat            X;
  Vec            x0;
  SlepcSC        sc;
  PetscReal      *T,*D,sigma,rnorm,aux;
  PetscScalar    *U,*V,*w,d;
  PetscInt       i,n=10,m,l=0,k=0,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GSVD with compact storage - dimension %Dx%D.\n",n+1,n);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  if (l>n || k>n || l>k) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Wrong value of dimensions");
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow);CHKERRQ(ierr);
  m = n+1;

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSGSVD);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);
  ld = n+2;  /* test leading dimension larger than n */
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,m,l,k);CHKERRQ(ierr);
  ierr = DSGSVDSetDimensions(ds,n,n);CHKERRQ(ierr);
  ierr = DSSetCompact(ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSSetExtraRow(ds,extrarow);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = DSView(ds,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);

  /* Fill A and B with lower/upper arrow-bidiagonal matrices
     verifying that [A;B] has orthonormal columns */
  ierr = DSGetArrayReal(ds,DS_MAT_T,&T);CHKERRQ(ierr);
  ierr = DSGetArrayReal(ds,DS_MAT_D,&D);CHKERRQ(ierr);
  for (i=0;i<n;i++) T[i] = (PetscReal)(i+1)/(n+1); /* diagonal of matrix A */
  for (i=0;i<k;i++) D[i] = PetscSqrtReal(1.0-T[i]*T[i]);
  for (i=l;i<k;i++) {
    T[i+ld] = PetscSqrtReal((1.0-T[k]*T[k])/(1.0+T[i]*T[i]/(D[i]*D[i])))*0.5*(1.0/k); /* upper diagonal of matrix A */
    T[i+2*ld] = -T[i+ld]*T[i]/D[i]; /* upper diagonal of matrix B */
  }
  aux = 1.0-T[k]*T[k];
  for (i=l;i<k;i++) aux -= T[i+ld]*T[i+ld]+T[i+2*ld]*T[i+2*ld];
  T[k+ld] = PetscSqrtReal((1.0-aux)*.1);
  aux -= T[k+ld]*T[k+ld];
  D[k] = PetscSqrtReal(aux);
  for (i=k+1;i<n;i++) {
    T[i-1+2*ld] = -T[i-1+ld]*T[i]/D[i-1]; /* upper diagonal of matrix B */
    aux = 1.0-T[i]*T[i]-T[2*ld+i-1]*T[2*ld+i-1];
    T[i+ld] = PetscSqrtReal(aux)*.1; /* upper diagonal of matrix A */
    D[i] = PetscSqrtReal(aux-T[i+ld]*T[i+ld]);
  }
  if (extrarow) { T[n]=-1.0; T[n-1+2*ld]=1.0; }
  /* Fill locked eigenvalues */
  ierr = PetscMalloc1(n,&w);CHKERRQ(ierr);
  for (i=0;i<l;i++) w[i] = T[i]/D[i];
  ierr = DSRestoreArrayReal(ds,DS_MAT_T,&T);CHKERRQ(ierr);
  ierr = DSRestoreArrayReal(ds,DS_MAT_D,&D);CHKERRQ(ierr);
  if (l==0 && k==0) {
    ierr = DSSetState(ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
  } else {
    ierr = DSSetState(ds,DS_STATE_RAW);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Solve */
  ierr = DSGetSlepcSC(ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSSolve(ds,w,NULL);CHKERRQ(ierr);
  ierr = DSSort(ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (extrarow) { ierr = DSUpdateExtraRow(ds);CHKERRQ(ierr); }
  ierr = DSSynchronize(ds,w,NULL);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Print singular values */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed singular values =\n");CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    sigma = PetscRealPart(w[i]);
    ierr = PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)sigma);CHKERRQ(ierr);
  }

  if (extrarow) {
    /* Check that extra row is correct */
    ierr = DSGetArrayReal(ds,DS_MAT_T,&T);CHKERRQ(ierr);
    ierr = DSGetArray(ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = DSGetArray(ds,DS_MAT_V,&V);CHKERRQ(ierr);
    d = 0.0;
    for (i=0;i<n;i++) d += T[i+ld]+U[n+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in A's extra row of %g\n",(double)PetscAbsScalar(d));CHKERRQ(ierr);
    }
    d = 0.0;
    for (i=0;i<n;i++) d += T[i+2*ld]-V[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in B's extra row of %g\n",(double)PetscAbsScalar(d));CHKERRQ(ierr);
    }
    ierr = DSRestoreArrayReal(ds,DS_MAT_T,&T);CHKERRQ(ierr);
    ierr = DSRestoreArray(ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = DSRestoreArray(ds,DS_MAT_V,&V);CHKERRQ(ierr);
  }

  /* Singular vectors */
  ierr = DSVectors(ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);  /* all singular vectors */
  ierr = DSGetMat(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = MatCreateVecs(X,NULL,&x0);CHKERRQ(ierr);
  ierr = MatGetColumnVector(X,x0,0);CHKERRQ(ierr);
  ierr = VecNorm(x0,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st X vector = %.3f\n",(double)rnorm);CHKERRQ(ierr);

  ierr = DSGetMat(ds,DS_MAT_U,&X);CHKERRQ(ierr);
  ierr = MatCreateVecs(X,NULL,&x0);CHKERRQ(ierr);
  ierr = MatGetColumnVector(X,x0,0);CHKERRQ(ierr);
  ierr = VecNorm(x0,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  if (PetscAbs(rnorm-1.0)>10*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: the 1st U vector has norm %g\n",(double)rnorm);CHKERRQ(ierr);
  }

  ierr = DSGetMat(ds,DS_MAT_V,&X);CHKERRQ(ierr);
  ierr = MatCreateVecs(X,NULL,&x0);CHKERRQ(ierr);
  ierr = MatGetColumnVector(X,x0,0);CHKERRQ(ierr);
  ierr = VecNorm(x0,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  if (PetscAbs(rnorm-1.0)>10*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: the 1st V vector has norm %g\n",(double)rnorm);CHKERRQ(ierr);
  }

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      requires: double
      output_file: output/test24_1.out
      test:
         suffix: 1
         args: -l 1 -k 4
      test:
         suffix: 1_extrarow
         filter: sed -e "s/extrarow//"
         args: -l 1 -k 4 -extrarow

TEST*/
