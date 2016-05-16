/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test EPS interface functions.\n\n";

#include <slepceps.h>

#undef __FUNCT__
#define __FUNCT__ "main"
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
  PetscBool          flg;
  EPSConvergedReason reason;
  EPSType            type;
  EPSExtraction      extr;
  EPSBalance         bal;
  EPSWhich           which;
  EPSConv            conv;
  EPSProblemType     ptype;
  PetscErrorCode     ierr;
  PetscViewerAndFormat *vf;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nDiagonal Eigenproblem, n=%D\n\n",n);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(A,i,i,i+1,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create eigensolver and test interface functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSGetOperators(eps,&B,NULL);CHKERRQ(ierr);
  ierr = MatView(B,NULL);CHKERRQ(ierr);

  ierr = EPSSetType(eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Type set to %s\n",type);CHKERRQ(ierr);

  ierr = EPSGetProblemType(eps,&ptype);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Problem type before changing = %d",(int)ptype);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSGetProblemType(eps,&ptype);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," ... changed to %d.",(int)ptype);CHKERRQ(ierr);
  ierr = EPSIsGeneralized(eps,&flg);CHKERRQ(ierr);
  if (flg) { ierr = PetscPrintf(PETSC_COMM_WORLD," generalized");CHKERRQ(ierr); }
  ierr = EPSIsHermitian(eps,&flg);CHKERRQ(ierr);
  if (flg) { ierr = PetscPrintf(PETSC_COMM_WORLD," hermitian");CHKERRQ(ierr); }
  ierr = EPSIsPositive(eps,&flg);CHKERRQ(ierr);
  if (flg) { ierr = PetscPrintf(PETSC_COMM_WORLD," positive");CHKERRQ(ierr); }

  ierr = EPSGetExtraction(eps,&extr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Extraction before changing = %d",(int)extr);CHKERRQ(ierr);
  ierr = EPSSetExtraction(eps,EPS_HARMONIC);CHKERRQ(ierr);
  ierr = EPSGetExtraction(eps,&extr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)extr);CHKERRQ(ierr);

  ierr = EPSSetBalance(eps,EPS_BALANCE_ONESIDE,8,1e-6);CHKERRQ(ierr);
  ierr = EPSGetBalance(eps,&bal,&its,&cut);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Balance: %s, its=%D, cutoff=%g\n",EPSBalanceTypes[bal],its,(double)cut);CHKERRQ(ierr);

  ierr = EPSSetTarget(eps,4.8);CHKERRQ(ierr);
  ierr = EPSGetTarget(eps,&target);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
  ierr = EPSGetWhichEigenpairs(eps,&which);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Which = %d, target = %g\n",(int)which,(double)PetscRealPart(target));CHKERRQ(ierr);

  ierr = EPSSetDimensions(eps,4,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,&ncv,&mpd);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Dimensions: nev=%D, ncv=%D, mpd=%D\n",nev,ncv,mpd);CHKERRQ(ierr);

  ierr = EPSSetTolerances(eps,2.2e-4,200);CHKERRQ(ierr);
  ierr = EPSGetTolerances(eps,&tol,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Tolerance = %.5f, max_its = %D\n",(double)tol,its);CHKERRQ(ierr);

  ierr = EPSSetConvergenceTest(eps,EPS_CONV_ABS);CHKERRQ(ierr);
  ierr = EPSGetConvergenceTest(eps,&conv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Convergence test = %d\n",(int)conv);CHKERRQ(ierr);

  ierr = PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf);CHKERRQ(ierr);
  ierr = EPSMonitorSet(eps,(PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))EPSMonitorFirst,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);
  ierr = EPSMonitorCancel(eps);CHKERRQ(ierr);

  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-8,1e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = STView(st,NULL);CHKERRQ(ierr);
  ierr = EPSGetDS(eps,&ds);CHKERRQ(ierr);
  ierr = DSView(ds,NULL);CHKERRQ(ierr);

  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(eps,&reason);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Finished - converged reason = %d, its=%D\n",(int)reason,its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
