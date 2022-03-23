/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test PEP interface functions.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat                A[3],B;      /* problem matrices */
  PEP                pep;         /* eigenproblem solver context */
  ST                 st;
  KSP                ksp;
  DS                 ds;
  PetscReal          tol,alpha;
  PetscScalar        target;
  PetscInt           n=20,i,its,nev,ncv,mpd,Istart,Iend,nmat;
  PEPWhich           which;
  PEPConvergedReason reason;
  PEPType            type;
  PEPExtract         extr;
  PEPBasis           basis;
  PEPScale           scale;
  PEPRefine          refine;
  PEPRefineScheme    rscheme;
  PEPConv            conv;
  PEPStop            stop;
  PEPProblemType     ptype;
  PetscErrorCode     ierr;
  PetscViewerAndFormat *vf;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nDiagonal Quadratic Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[0]));
  CHKERRQ(MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[0]));
  CHKERRQ(MatSetUp(A[0]));
  CHKERRQ(MatGetOwnershipRange(A[0],&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(A[0],i,i,i+1,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[1]));
  CHKERRQ(MatSetSizes(A[1],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[1]));
  CHKERRQ(MatSetUp(A[1]));
  CHKERRQ(MatGetOwnershipRange(A[1],&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(A[1],i,i,1.0,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[2]));
  CHKERRQ(MatSetSizes(A[2],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[2]));
  CHKERRQ(MatSetUp(A[2]));
  CHKERRQ(MatGetOwnershipRange(A[1],&Istart,&Iend));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(A[2],i,i,n/(PetscReal)(i+1),INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[2],MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create eigensolver and test interface functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));
  CHKERRQ(PEPSetOperators(pep,3,A));
  CHKERRQ(PEPGetNumMatrices(pep,&nmat));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Polynomial of degree %" PetscInt_FMT "\n",nmat-1));
  CHKERRQ(PEPGetOperators(pep,0,&B));
  CHKERRQ(MatView(B,NULL));

  CHKERRQ(PEPSetType(pep,PEPTOAR));
  CHKERRQ(PEPGetType(pep,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Type set to %s\n",type));

  CHKERRQ(PEPGetProblemType(pep,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Problem type before changing = %d",(int)ptype));
  CHKERRQ(PEPSetProblemType(pep,PEP_HERMITIAN));
  CHKERRQ(PEPGetProblemType(pep,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d.\n",(int)ptype));

  CHKERRQ(PEPGetExtract(pep,&extr));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Extraction before changing = %d",(int)extr));
  CHKERRQ(PEPSetExtract(pep,PEP_EXTRACT_STRUCTURED));
  CHKERRQ(PEPGetExtract(pep,&extr));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," ... changed to %d\n",(int)extr));

  CHKERRQ(PEPSetScale(pep,PEP_SCALE_SCALAR,.1,NULL,NULL,5,1.0));
  CHKERRQ(PEPGetScale(pep,&scale,&alpha,NULL,NULL,&its,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Scaling: %s, alpha=%g, its=%" PetscInt_FMT "\n",PEPScaleTypes[scale],(double)alpha,its));

  CHKERRQ(PEPSetBasis(pep,PEP_BASIS_CHEBYSHEV1));
  CHKERRQ(PEPGetBasis(pep,&basis));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Polynomial basis: %s\n",PEPBasisTypes[basis]));

  CHKERRQ(PEPSetRefine(pep,PEP_REFINE_SIMPLE,1,1e-9,2,PEP_REFINE_SCHEME_SCHUR));
  CHKERRQ(PEPGetRefine(pep,&refine,NULL,&tol,&its,&rscheme));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Refinement: %s, tol=%g, its=%" PetscInt_FMT ", scheme=%s\n",PEPRefineTypes[refine],(double)tol,its,PEPRefineSchemes[rscheme]));

  CHKERRQ(PEPSetTarget(pep,4.8));
  CHKERRQ(PEPGetTarget(pep,&target));
  CHKERRQ(PEPSetWhichEigenpairs(pep,PEP_TARGET_MAGNITUDE));
  CHKERRQ(PEPGetWhichEigenpairs(pep,&which));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Which = %d, target = %g\n",(int)which,(double)PetscRealPart(target)));

  CHKERRQ(PEPSetDimensions(pep,4,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(PEPGetDimensions(pep,&nev,&ncv,&mpd));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Dimensions: nev=%" PetscInt_FMT ", ncv=%" PetscInt_FMT ", mpd=%" PetscInt_FMT "\n",nev,ncv,mpd));

  CHKERRQ(PEPSetTolerances(pep,2.2e-4,200));
  CHKERRQ(PEPGetTolerances(pep,&tol,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Tolerance = %.5f, max_its = %" PetscInt_FMT "\n",(double)tol,its));

  CHKERRQ(PEPSetConvergenceTest(pep,PEP_CONV_ABS));
  CHKERRQ(PEPGetConvergenceTest(pep,&conv));
  CHKERRQ(PEPSetStoppingTest(pep,PEP_STOP_BASIC));
  CHKERRQ(PEPGetStoppingTest(pep,&stop));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Convergence test = %d, stopping test = %d\n",(int)conv,(int)stop));

  CHKERRQ(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf));
  CHKERRQ(PEPMonitorSet(pep,(PetscErrorCode (*)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))PEPMonitorFirst,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  CHKERRQ(PEPMonitorCancel(pep));

  CHKERRQ(PEPGetST(pep,&st));
  CHKERRQ(STGetKSP(st,&ksp));
  CHKERRQ(KSPSetTolerances(ksp,1e-8,1e-35,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(STView(st,NULL));
  CHKERRQ(PEPGetDS(pep,&ds));
  CHKERRQ(DSView(ds,NULL));

  CHKERRQ(PEPSetFromOptions(pep));
  CHKERRQ(PEPSolve(pep));
  CHKERRQ(PEPGetConvergedReason(pep,&reason));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Finished - converged reason = %d\n",(int)reason));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PEPErrorView(pep,PEP_ERROR_RELATIVE,NULL));
  CHKERRQ(PEPDestroy(&pep));
  CHKERRQ(MatDestroy(&A[0]));
  CHKERRQ(MatDestroy(&A[1]));
  CHKERRQ(MatDestroy(&A[2]));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -pep_tol 1e-6 -pep_ncv 22
      filter: sed -e "s/[+-]0\.0*i//g"

TEST*/
