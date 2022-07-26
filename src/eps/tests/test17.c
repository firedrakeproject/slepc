/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-showinertia",&showinertia,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum-slicing test, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,II,II,4.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,II,II,2.0,INSERT_VALUES));
  }
  if (Istart==0) {
    PetscCall(MatSetValue(B,0,0,6.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,0,1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,1,0,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,1,1,1.0,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOperators(eps,A,B));
  PetscCall(EPSSetProblemType(eps,EPS_GHEP));
  PetscCall(EPSSetType(eps,EPSKRYLOVSCHUR));

  /*
     Set interval and other settings for spectrum slicing
  */
  PetscCall(EPSSetWhichEigenpairs(eps,EPS_ALL));
  int0 = 1.1; int1 = 1.3;
  PetscCall(EPSSetInterval(eps,int0,int1));
  PetscCall(EPSGetST(eps,&st));
  PetscCall(STSetType(st,STSINVERT));
  if (size>1) PetscCall(EPSKrylovSchurSetPartitions(eps,size));
  PetscCall(EPSKrylovSchurGetKSP(eps,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(KSPSetType(ksp,KSPPREONLY));
  PetscCall(PCSetType(pc,PCCHOLESKY));

  /*
     Test interface functions of Krylov-Schur solver
  */
  PetscCall(EPSKrylovSchurGetRestart(eps,&keep));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Restart parameter before changing = %g",(double)keep));
  PetscCall(EPSKrylovSchurSetRestart(eps,0.4));
  PetscCall(EPSKrylovSchurGetRestart(eps,&keep));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %g\n",(double)keep));

  PetscCall(EPSKrylovSchurGetDetectZeros(eps,&detect));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Detect zeros before changing = %d",(int)detect));
  PetscCall(EPSKrylovSchurSetDetectZeros(eps,PETSC_TRUE));
  PetscCall(EPSKrylovSchurGetDetectZeros(eps,&detect));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)detect));

  PetscCall(EPSKrylovSchurGetLocking(eps,&lock));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Locking flag before changing = %d",(int)lock));
  PetscCall(EPSKrylovSchurSetLocking(eps,PETSC_FALSE));
  PetscCall(EPSKrylovSchurGetLocking(eps,&lock));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)lock));

  PetscCall(EPSKrylovSchurGetDimensions(eps,&nev,&ncv,&mpd));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Sub-solve dimensions before changing = [%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "]",nev,ncv,mpd));
  PetscCall(EPSKrylovSchurSetDimensions(eps,30,60,60));
  PetscCall(EPSKrylovSchurGetDimensions(eps,&nev,&ncv,&mpd));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," ... changed to [%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "]\n",nev,ncv,mpd));

  if (size>1) {
    PetscCall(EPSKrylovSchurGetPartitions(eps,&npart));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Using %" PetscInt_FMT " partitions\n",npart));

    PetscCall(PetscMalloc1(npart+1,&subint));
    subint[0] = int0;
    subint[npart] = int1;
    for (i=1;i<npart;i++) subint[i] = int0+i*(int1-int0)/npart;
    PetscCall(EPSKrylovSchurSetSubintervals(eps,subint));
    PetscCall(PetscFree(subint));
    PetscCall(EPSKrylovSchurGetSubintervals(eps,&subint));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Using sub-interval separations = "));
    for (i=1;i<npart;i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD," %g",(double)subint[i]));
    PetscCall(PetscFree(subint));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  }

  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute all eigenvalues in interval and display info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSSetUp(eps));
  PetscCall(EPSKrylovSchurGetInertias(eps,&k,&shifts,&inertias));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Inertias after EPSSetUp:\n"));
  for (i=0;i<k;i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
  PetscCall(PetscFree(shifts));
  PetscCall(PetscFree(inertias));

  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetDimensions(eps,&nev,NULL,NULL));
  PetscCall(EPSGetInterval(eps,&int0,&int1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Found %" PetscInt_FMT " eigenvalues in interval [%g,%g]\n",nev,(double)int0,(double)int1));

  if (showinertia) {
    PetscCall(EPSKrylovSchurGetInertias(eps,&k,&shifts,&inertias));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Used %" PetscInt_FMT " shifts (inertia):\n",k));
    for (i=0;i<k;i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD," .. %g (%" PetscInt_FMT ")\n",(double)shifts[i],inertias[i]));
    PetscCall(PetscFree(shifts));
    PetscCall(PetscFree(inertias));
  }

  PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  if (size>1) {
    PetscCall(EPSKrylovSchurGetSubcommInfo(eps,&k,&nval,&v));
    PetscCall(PetscMalloc1(nval,&evals));
    for (i=0;i<nval;i++) {
      PetscCall(EPSKrylovSchurGetSubcommPairs(eps,i,&lambda,v));
      evals[i] = PetscRealPart(lambda);
    }
    PetscCall(PetscFormatRealArray(vlist,sizeof(vlist),"%f",nval,evals));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD," Process %d has worked in sub-interval %" PetscInt_FMT ", containing %" PetscInt_FMT " eigenvalues: %s\n",(int)rank,k,nval,vlist));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    PetscCall(VecDestroy(&v));
    PetscCall(PetscFree(evals));

    PetscCall(EPSKrylovSchurGetSubcommMats(eps,&As,&Bs));
    PetscCall(MatGetLocalSize(A,&nloc,NULL));
    PetscCall(MatGetLocalSize(As,&nlocs,&mlocs));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD," Process %d owns %" PetscInt_FMT " rows of the global matrices, and %" PetscInt_FMT " rows in the subcommunicator\n",(int)rank,nloc,nlocs));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

    /* modify A on subcommunicators */
    PetscCall(MatCreate(PetscObjectComm((PetscObject)As),&Au));
    PetscCall(MatSetSizes(Au,nlocs,mlocs,N,N));
    PetscCall(MatSetFromOptions(Au));
    PetscCall(MatSetUp(Au));
    PetscCall(MatGetOwnershipRange(Au,&Istart,&Iend));
    for (II=Istart;II<Iend;II++) PetscCall(MatSetValue(Au,II,II,0.5,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Au,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Au,MAT_FINAL_ASSEMBLY));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Updating internal matrices\n"));
    PetscCall(EPSKrylovSchurUpdateSubcommMats(eps,1.1,-5.0,Au,1.0,0.0,NULL,DIFFERENT_NONZERO_PATTERN,PETSC_TRUE));
    PetscCall(MatDestroy(&Au));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetDimensions(eps,&nev,NULL,NULL));
    PetscCall(EPSGetInterval(eps,&int0,&int1));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," After update, found %" PetscInt_FMT " eigenvalues in interval [%g,%g]\n",nev,(double)int0,(double)int1));
  }
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(SlepcFinalize());
  return 0;
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
