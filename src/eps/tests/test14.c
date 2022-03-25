/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test EPS interface functions.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat                A,B;         /* problem matrix */
  EPS                eps;         /* eigenproblem solver context */
  ST                 st;
  KSP                ksp;
  DS                 ds;
  PetscReal          cut,tol;
  PetscScalar        target;
  PetscInt           n=20,i,its,nev,ncv,mpd,Istart,Iend;
  PetscBool          flg,pur,track;
  EPSConvergedReason reason;
  EPSType            type;
  EPSExtraction      extr;
  EPSBalance         bal;
  EPSWhich           which;
  EPSConv            conv;
  EPSStop            stop;
  EPSProblemType     ptype;
  PetscViewerAndFormat *vf;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nDiagonal Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(A,i,i,i+1,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create eigensolver and test interface functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSGetOperators(eps,&B,NULL));
  CHKERRQ(MatView(B,NULL));

  CHKERRQ(EPSSetType(eps,EPSKRYLOVSCHUR));
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Type set to %s\n",type));

  CHKERRQ(EPSGetProblemType(eps,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Problem type before changing = %d",(int)ptype));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSGetProblemType(eps,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d.",(int)ptype));
  CHKERRQ(EPSIsGeneralized(eps,&flg));
  if (flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," generalized"));
  CHKERRQ(EPSIsHermitian(eps,&flg));
  if (flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," hermitian"));
  CHKERRQ(EPSIsPositive(eps,&flg));
  if (flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," positive"));

  CHKERRQ(EPSGetExtraction(eps,&extr));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n Extraction before changing = %d",(int)extr));
  CHKERRQ(EPSSetExtraction(eps,EPS_HARMONIC));
  CHKERRQ(EPSGetExtraction(eps,&extr));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)extr));

  CHKERRQ(EPSSetBalance(eps,EPS_BALANCE_ONESIDE,8,1e-6));
  CHKERRQ(EPSGetBalance(eps,&bal,&its,&cut));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Balance: %s, its=%" PetscInt_FMT ", cutoff=%g\n",EPSBalanceTypes[bal],its,(double)cut));

  CHKERRQ(EPSSetPurify(eps,PETSC_FALSE));
  CHKERRQ(EPSGetPurify(eps,&pur));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Eigenvector purification: %s\n",pur?"on":"off"));
  CHKERRQ(EPSGetTrackAll(eps,&track));

  CHKERRQ(EPSSetTarget(eps,4.8));
  CHKERRQ(EPSGetTarget(eps,&target));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE));
  CHKERRQ(EPSGetWhichEigenpairs(eps,&which));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Which = %d, target = %g\n",(int)which,(double)PetscRealPart(target)));

  CHKERRQ(EPSSetDimensions(eps,4,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(EPSGetDimensions(eps,&nev,&ncv,&mpd));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Dimensions: nev=%" PetscInt_FMT ", ncv=%" PetscInt_FMT ", mpd=%" PetscInt_FMT "\n",nev,ncv,mpd));

  CHKERRQ(EPSSetTolerances(eps,2.2e-4,200));
  CHKERRQ(EPSGetTolerances(eps,&tol,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Tolerance = %.5f, max_its = %" PetscInt_FMT "\n",(double)tol,its));

  CHKERRQ(EPSSetConvergenceTest(eps,EPS_CONV_ABS));
  CHKERRQ(EPSGetConvergenceTest(eps,&conv));
  CHKERRQ(EPSSetStoppingTest(eps,EPS_STOP_BASIC));
  CHKERRQ(EPSGetStoppingTest(eps,&stop));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Convergence test = %d, stopping test = %d\n",(int)conv,(int)stop));

  CHKERRQ(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf));
  CHKERRQ(EPSMonitorSet(eps,(PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))EPSMonitorFirst,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  CHKERRQ(EPSMonitorCancel(eps));

  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetTolerances(ksp,1e-8,1e-35,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(STView(st,NULL));
  CHKERRQ(EPSGetDS(eps,&ds));
  CHKERRQ(DSView(ds,NULL));

  CHKERRQ(EPSSetFromOptions(eps));
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSGetConvergedReason(eps,&reason));
  CHKERRQ(EPSGetIterationNumber(eps,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Finished - converged reason = %d, its=%" PetscInt_FMT "\n",(int)reason,its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -eps_ncv 14
      filter: sed -e "s/00001/00000/" | sed -e "s/4.99999/5.00000/" | sed -e "s/5.99999/6.00000/"

TEST*/
