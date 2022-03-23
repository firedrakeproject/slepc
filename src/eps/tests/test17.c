/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test interface functions of spectrum-slicing Krylov-Schur.\n\n"
  "This is based on ex12.c. The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A,B;         /* matrices */
  Mat            As,Bs;       /* matrices distributed in subcommunicators */
  Mat            Au;          /* matrix used to modify A on subcommunicators */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  KSP            ksp;
  PC             pc;
  Vec            v;
  PetscMPIInt    size,rank;
  PetscInt       N,n=35,m,Istart,Iend,II,nev,ncv,mpd,i,j,k,*inertias,npart,nval,nloc,nlocs,mlocs;
  PetscBool      flag,showinertia=PETSC_TRUE,lock,detect;
  PetscReal      int0,int1,*shifts,keep,*subint,*evals;
  PetscScalar    lambda;
  char           vlist[4000];
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-showinertia",&showinertia,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum-slicing test, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,4.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,II,II,2.0,INSERT_VALUES));
  }
  if (Istart==0) {
    CHKERRQ(MatSetValue(B,0,0,6.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,0,1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,1,0,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,1,1,1.0,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,B));
  CHKERRQ(EPSSetProblemType(eps,EPS_GHEP));
  CHKERRQ(EPSSetType(eps,EPSKRYLOVSCHUR));

  /*
     Set interval and other settings for spectrum slicing
  */
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_ALL));
  int0 = 1.1; int1 = 1.3;
  CHKERRQ(EPSSetInterval(eps,int0,int1));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STSetType(st,STSINVERT));
  if (size>1) CHKERRQ(EPSKrylovSchurSetPartitions(eps,size));
  CHKERRQ(EPSKrylovSchurGetKSP(eps,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(PCSetType(pc,PCCHOLESKY));

  /*
     Test interface functions of Krylov-Schur solver
  */
  CHKERRQ(EPSKrylovSchurGetRestart(eps,&keep));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Restart parameter before changing = %g",(double)keep));
  CHKERRQ(EPSKrylovSchurSetRestart(eps,0.4));
  CHKERRQ(EPSKrylovSchurGetRestart(eps,&keep));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %g\n",(double)keep));

  CHKERRQ(EPSKrylovSchurGetDetectZeros(eps,&detect));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Detect zeros before changing = %d",(int)detect));
  CHKERRQ(EPSKrylovSchurSetDetectZeros(eps,PETSC_TRUE));
  CHKERRQ(EPSKrylovSchurGetDetectZeros(eps,&detect));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)detect));

  CHKERRQ(EPSKrylovSchurGetLocking(eps,&lock));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Locking flag before changing = %d",(int)lock));
  CHKERRQ(EPSKrylovSchurSetLocking(eps,PETSC_FALSE));
  CHKERRQ(EPSKrylovSchurGetLocking(eps,&lock));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)lock));

  CHKERRQ(EPSKrylovSchurGetDimensions(eps,&nev,&ncv,&mpd));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Sub-solve dimensions before changing = [%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "]",nev,ncv,mpd));
  CHKERRQ(EPSKrylovSchurSetDimensions(eps,30,60,60));
  CHKERRQ(EPSKrylovSchurGetDimensions(eps,&nev,&ncv,&mpd));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to [%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "]\n",nev,ncv,mpd));

  if (size>1) {
    CHKERRQ(EPSKrylovSchurGetPartitions(eps,&npart));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Using %" PetscInt_FMT " partitions\n",npart));

    CHKERRQ(PetscMalloc1(npart+1,&subint));
    subint[0] = int0;
    subint[npart] = int1;
    for (i=1;i<npart;i++) subint[i] = int0+i*(int1-int0)/npart;
    CHKERRQ(EPSKrylovSchurSetSubintervals(eps,subint));
    CHKERRQ(PetscFree(subint));
    CHKERRQ(EPSKrylovSchurGetSubintervals(eps,&subint));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Using sub-interval separations = "));
    for (i=1;i<npart;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %g",(double)subint[i]));
    CHKERRQ(PetscFree(subint));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  }

  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute all eigenvalues in interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSetUp(eps));
  CHKERRQ(EPSKrylovSchurGetInertias(eps,&k,&shifts,&inertias));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Inertias after EPSSetUp:\n"));
  for (i=0;i<k;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
  CHKERRQ(PetscFree(shifts));
  CHKERRQ(PetscFree(inertias));

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(EPSGetInterval(eps,&int0,&int1));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Found %" PetscInt_FMT " eigenvalues in interval [%g,%g]\n",nev,(double)int0,(double)int1));

  if (showinertia) {
    CHKERRQ(EPSKrylovSchurGetInertias(eps,&k,&shifts,&inertias));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Used %" PetscInt_FMT " shifts (inertia):\n",k));
    for (i=0;i<k;i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
    CHKERRQ(PetscFree(shifts));
    CHKERRQ(PetscFree(inertias));
  }

  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  if (size>1) {
    CHKERRQ(EPSKrylovSchurGetSubcommInfo(eps,&k,&nval,&v));
    CHKERRQ(PetscMalloc1(nval,&evals));
    for (i=0;i<nval;i++) {
      CHKERRQ(EPSKrylovSchurGetSubcommPairs(eps,i,&lambda,v));
      evals[i] = PetscRealPart(lambda);
    }
    CHKERRQ(PetscFormatRealArray(vlist,sizeof(vlist),"%f",nval,evals));
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD," Process %d has worked in sub-interval %" PetscInt_FMT ", containing %" PetscInt_FMT " eigenvalues: %s\n",(int)rank,k,nval,vlist));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    CHKERRQ(VecDestroy(&v));
    CHKERRQ(PetscFree(evals));

    CHKERRQ(EPSKrylovSchurGetSubcommMats(eps,&As,&Bs));
    CHKERRQ(MatGetLocalSize(A,&nloc,NULL));
    CHKERRQ(MatGetLocalSize(As,&nlocs,&mlocs));
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD," Process %d owns %" PetscInt_FMT " rows of the global matrices, and %" PetscInt_FMT " rows in the subcommunicator\n",(int)rank,nloc,nlocs));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

    /* modify A on subcommunicators */
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)As),&Au));
    CHKERRQ(MatSetSizes(Au,nlocs,mlocs,N,N));
    CHKERRQ(MatSetFromOptions(Au));
    CHKERRQ(MatSetUp(Au));
    CHKERRQ(MatGetOwnershipRange(Au,&Istart,&Iend));
    for (II=Istart;II<Iend;II++) CHKERRQ(MatSetValue(Au,II,II,0.5,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(Au,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Au,MAT_FINAL_ASSEMBLY));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Updating internal matrices\n"));
    CHKERRQ(EPSKrylovSchurUpdateSubcommMats(eps,1.1,-5.0,Au,1.0,0.0,NULL,DIFFERENT_NONZERO_PATTERN,PETSC_TRUE));
    CHKERRQ(MatDestroy(&Au));
    CHKERRQ(EPSSolve(eps));
    CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
    CHKERRQ(EPSGetInterval(eps,&int0,&int1));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," After update, found %" PetscInt_FMT " eigenvalues in interval [%g,%g]\n",nev,(double)int0,(double)int1));
  }
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      args: -showinertia 0 -log_exclude eps,st,rg,bv,ds
      requires: !single

   test:
      suffix: 2
      nsize: 1
      args: -showinertia 0 -log_exclude eps,st,rg,bv,ds
      requires: !single

TEST*/
