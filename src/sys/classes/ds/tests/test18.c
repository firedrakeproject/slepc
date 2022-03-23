/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSSynchronize() on a NHEP.\n\n";

#include <slepcds.h>

PetscErrorCode CheckArray(PetscScalar *A,const char *label,PetscInt k)
{
  PetscInt       j;
  PetscMPIInt    p,size,rank;
  PetscScalar    dif,*buf;
  PetscReal      error;

  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank) CHKERRMPI(MPI_Send(A,k,MPIU_SCALAR,0,111,PETSC_COMM_WORLD));
  else {
    CHKERRQ(PetscMalloc1(k,&buf));
    for (p=1;p<size;p++) {
      CHKERRMPI(MPI_Recv(buf,k,MPIU_SCALAR,p,111,PETSC_COMM_WORLD,MPI_STATUS_IGNORE));
      dif = 0.0;
      for (j=0;j<k;j++) dif += A[j]-buf[j];
      error = PetscAbsScalar(dif);
      if (error>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Array %s differs in proc %d: %g\n",label,(int)p,(double)error));
    }
    CHKERRQ(PetscFree(buf));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  PetscScalar    *A,*Q,*wr,*wi;
  PetscReal      re,im;
  PetscInt       i,j,n=10;
  PetscMPIInt    size;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type NHEP - dimension %" PetscInt_FMT ".\n",n));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSNHEP));
  CHKERRQ(DSSetFromOptions(ds));
  CHKERRQ(DSAllocate(ds,n));
  CHKERRQ(DSSetDimensions(ds,n,0,0));

  /* Fill with Grcar matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  for (i=1;i<n;i++) A[i+(i-1)*n]=-1.0;
  for (j=0;j<4;j++) {
    for (i=0;i<n-j;i++) A[i+(i+j)*n]=1.0;
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
  CHKERRQ(DSSetState(ds,DS_STATE_INTERMEDIATE));

  /* Solve */
  CHKERRQ(PetscMalloc2(n,&wr,n,&wi));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,wr,wi));
  CHKERRQ(DSSort(ds,wr,wi,NULL,NULL,NULL));

  /* Print eigenvalues */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(wr[i]);
    im = PetscImaginaryPart(wr[i]);
#else
    re = wr[i];
    im = wi[i];
#endif
    if (PetscAbs(im)<1e-10) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  %.5f\n",(double)re));
    else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  %.5f%+.5fi\n",(double)re,(double)im));
  }

  /* Synchronize data and check */
  CHKERRQ(DSSynchronize(ds,wr,wi));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  if (size>1) {
    CHKERRQ(CheckArray(wr,"wr",n));
#if !defined(PETSC_USE_COMPLEX)
    CHKERRQ(CheckArray(wi,"wi",n));
#endif
    CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
    CHKERRQ(CheckArray(A,"A",n*n));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));
    CHKERRQ(DSGetArray(ds,DS_MAT_Q,&Q));
    CHKERRQ(CheckArray(Q,"Q",n*n));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_Q,&Q));
  }

  CHKERRQ(PetscFree2(wr,wi));
  CHKERRQ(DSDestroy(&ds));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: {{1 2 3}}
      filter: sed -e "s/[+-]\([0-9]\.[0-9]*i\)/+-\\1/" | sed -e "s/1.58254/1.58255/" | sed -e "s/1.75989/1.75988/"
      output_file: output/test18_1.out
      test:
         suffix: 1
         args: -ds_parallel redundant
      test:
         suffix: 2
         args: -ds_parallel synchronized

TEST*/
