/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test NEP interface functions.\n\n";

#include <slepcnep.h>

int main(int argc,char **argv)
{
  Mat                  A[3],B;      /* problem matrices */
  FN                   f[3],g;      /* problem functions */
  NEP                  nep;         /* eigenproblem solver context */
  DS                   ds;
  RG                   rg;
  PetscReal            tol;
  PetscScalar          coeffs[2],target;
  PetscInt             n=20,i,its,nev,ncv,mpd,Istart,Iend,nterm;
  PetscBool            twoside;
  NEPWhich             which;
  NEPConvergedReason   reason;
  NEPType              type;
  NEPRefine            refine;
  NEPRefineScheme      rscheme;
  NEPConv              conv;
  NEPStop              stop;
  NEPProblemType       ptype;
  MatStructure         mstr;
  PetscErrorCode       ierr;
  PetscViewerAndFormat *vf;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nDiagonal Nonlinear Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  /*
     Matrices
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[0]));
  CHKERRQ(MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[0]));
  CHKERRQ(MatSetUp(A[0]));
  CHKERRQ(MatGetOwnershipRange(A[0],&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[0],i,i,i+1,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[1]));
  CHKERRQ(MatSetSizes(A[1],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[1]));
  CHKERRQ(MatSetUp(A[1]));
  CHKERRQ(MatGetOwnershipRange(A[1],&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[1],i,i,1.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[2]));
  CHKERRQ(MatSetSizes(A[2],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[2]));
  CHKERRQ(MatSetUp(A[2]));
  CHKERRQ(MatGetOwnershipRange(A[1],&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[2],i,i,n/(PetscReal)(i+1),INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[2],MAT_FINAL_ASSEMBLY));

  /*
     Functions: f0=-lambda, f1=1.0, f2=sqrt(lambda)
  */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[0]));
  CHKERRQ(FNSetType(f[0],FNRATIONAL));
  coeffs[0] = -1.0; coeffs[1] = 0.0;
  CHKERRQ(FNRationalSetNumerator(f[0],2,coeffs));

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[1]));
  CHKERRQ(FNSetType(f[1],FNRATIONAL));
  coeffs[0] = 1.0;
  CHKERRQ(FNRationalSetNumerator(f[1],1,coeffs));

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[2]));
  CHKERRQ(FNSetType(f[2],FNSQRT));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create eigensolver and test interface functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));
  CHKERRQ(NEPSetSplitOperator(nep,3,A,f,SAME_NONZERO_PATTERN));
  CHKERRQ(NEPGetSplitOperatorInfo(nep,&nterm,&mstr));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Nonlinear function with %" PetscInt_FMT " terms, with %s nonzero pattern\n",nterm,MatStructures[mstr]));
  CHKERRQ(NEPGetSplitOperatorTerm(nep,0,&B,&g));
  CHKERRQ(MatView(B,NULL));
  CHKERRQ(FNView(g,NULL));

  CHKERRQ(NEPSetType(nep,NEPRII));
  CHKERRQ(NEPGetType(nep,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Type set to %s\n",type));
  CHKERRQ(NEPGetTwoSided(nep,&twoside));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Two-sided flag = %s\n",twoside?"true":"false"));

  CHKERRQ(NEPGetProblemType(nep,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Problem type before changing = %d",(int)ptype));
  CHKERRQ(NEPSetProblemType(nep,NEP_RATIONAL));
  CHKERRQ(NEPGetProblemType(nep,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d.\n",(int)ptype));

  CHKERRQ(NEPSetRefine(nep,NEP_REFINE_SIMPLE,1,1e-9,2,NEP_REFINE_SCHEME_EXPLICIT));
  CHKERRQ(NEPGetRefine(nep,&refine,NULL,&tol,&its,&rscheme));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Refinement: %s, tol=%g, its=%" PetscInt_FMT ", scheme=%s\n",NEPRefineTypes[refine],(double)tol,its,NEPRefineSchemes[rscheme]));

  CHKERRQ(NEPSetTarget(nep,1.1));
  CHKERRQ(NEPGetTarget(nep,&target));
  CHKERRQ(NEPSetWhichEigenpairs(nep,NEP_TARGET_MAGNITUDE));
  CHKERRQ(NEPGetWhichEigenpairs(nep,&which));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Which = %d, target = %g\n",(int)which,(double)PetscRealPart(target)));

  CHKERRQ(NEPSetDimensions(nep,1,12,PETSC_DEFAULT));
  CHKERRQ(NEPGetDimensions(nep,&nev,&ncv,&mpd));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Dimensions: nev=%" PetscInt_FMT ", ncv=%" PetscInt_FMT ", mpd=%" PetscInt_FMT "\n",nev,ncv,mpd));

  CHKERRQ(NEPSetTolerances(nep,1.0e-6,200));
  CHKERRQ(NEPGetTolerances(nep,&tol,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Tolerance = %.6f, max_its = %" PetscInt_FMT "\n",(double)tol,its));

  CHKERRQ(NEPSetConvergenceTest(nep,NEP_CONV_ABS));
  CHKERRQ(NEPGetConvergenceTest(nep,&conv));
  CHKERRQ(NEPSetStoppingTest(nep,NEP_STOP_BASIC));
  CHKERRQ(NEPGetStoppingTest(nep,&stop));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Convergence test = %d, stopping test = %d\n",(int)conv,(int)stop));

  CHKERRQ(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf));
  CHKERRQ(NEPMonitorSet(nep,(PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))NEPMonitorFirst,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  CHKERRQ(NEPMonitorCancel(nep));

  CHKERRQ(NEPGetDS(nep,&ds));
  CHKERRQ(DSView(ds,NULL));
  CHKERRQ(NEPSetFromOptions(nep));

  CHKERRQ(NEPGetRG(nep,&rg));
  CHKERRQ(RGView(rg,NULL));

  CHKERRQ(NEPSolve(nep));
  CHKERRQ(NEPGetConvergedReason(nep,&reason));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Finished - converged reason = %d\n",(int)reason));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));
  CHKERRQ(NEPDestroy(&nep));
  CHKERRQ(MatDestroy(&A[0]));
  CHKERRQ(MatDestroy(&A[1]));
  CHKERRQ(MatDestroy(&A[2]));
  CHKERRQ(FNDestroy(&f[0]));
  CHKERRQ(FNDestroy(&f[1]));
  CHKERRQ(FNDestroy(&f[2]));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1

TEST*/
