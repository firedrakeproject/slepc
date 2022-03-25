/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test NEP view and monitor functionality.\n\n";

#include <slepcnep.h>

int main(int argc,char **argv)
{
  Mat                A[3];
  FN                 f[3];
  NEP                nep;
  Vec                xr,xi;
  PetscScalar        kr,ki,coeffs[3];
  PetscComplex       *eigs,eval;
  PetscInt           n=6,i,Istart,Iend,nconv,its;
  PetscReal          errest;
  PetscBool          checkfile;
  char               filename[PETSC_MAX_PATH_LEN];
  PetscViewer        viewer;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nDiagonal Nonlinear Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

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
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(A[1],i,i,-1.5,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[2]));
  CHKERRQ(MatSetSizes(A[2],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[2]));
  CHKERRQ(MatSetUp(A[2]));
  for (i=Istart;i<Iend;i++) CHKERRQ(MatSetValue(A[2],i,i,-1.0/(i+1),INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[2],MAT_FINAL_ASSEMBLY));

  /*
     Functions: f0=1.0, f1=lambda, f2=lambda^2
  */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[0]));
  CHKERRQ(FNSetType(f[0],FNRATIONAL));
  coeffs[0] = 1.0;
  CHKERRQ(FNRationalSetNumerator(f[0],1,coeffs));

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[1]));
  CHKERRQ(FNSetType(f[1],FNRATIONAL));
  coeffs[0] = 1.0; coeffs[1] = 0.0;
  CHKERRQ(FNRationalSetNumerator(f[1],2,coeffs));

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[2]));
  CHKERRQ(FNSetType(f[2],FNRATIONAL));
  coeffs[0] = 1.0; coeffs[1] = 0.0; coeffs[2] = 0.0;
  CHKERRQ(FNRationalSetNumerator(f[2],3,coeffs));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Create the NEP solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));
  CHKERRQ(PetscObjectSetName((PetscObject)nep,"nep"));
  CHKERRQ(NEPSetSplitOperator(nep,3,A,f,SAME_NONZERO_PATTERN));
  CHKERRQ(NEPSetTarget(nep,1.1));
  CHKERRQ(NEPSetWhichEigenpairs(nep,NEP_TARGET_MAGNITUDE));
  CHKERRQ(NEPSetFromOptions(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(NEPSolve(nep));
  CHKERRQ(NEPGetConverged(nep,&nconv));
  CHKERRQ(NEPGetIterationNumber(nep,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT " converged eigenpairs after %" PetscInt_FMT " iterations\n",nconv,its));
  if (nconv>0) {
    CHKERRQ(MatCreateVecs(A[0],&xr,&xi));
    CHKERRQ(NEPGetEigenpair(nep,0,&kr,&ki,xr,xi));
    CHKERRQ(VecDestroy(&xr));
    CHKERRQ(VecDestroy(&xi));
    CHKERRQ(NEPGetErrorEstimate(nep,0,&errest));
  }
  CHKERRQ(NEPErrorView(nep,NEP_ERROR_BACKWARD,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Check file containing the eigenvalues
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-checkfile",filename,sizeof(filename),&checkfile));
  if (checkfile) {
    CHKERRQ(PetscMalloc1(nconv,&eigs));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(PetscViewerBinaryRead(viewer,eigs,nconv,NULL,PETSC_COMPLEX));
    CHKERRQ(PetscViewerDestroy(&viewer));
    for (i=0;i<nconv;i++) {
      CHKERRQ(NEPGetEigenpair(nep,i,&kr,&ki,NULL,NULL));
#if defined(PETSC_USE_COMPLEX)
      eval = kr;
#else
      eval = PetscCMPLX(kr,ki);
#endif
      PetscCheck(eval==eigs[i],PETSC_COMM_WORLD,PETSC_ERR_FILE_UNEXPECTED,"Eigenvalues in the file do not match");
    }
    CHKERRQ(PetscFree(eigs));
  }

  CHKERRQ(NEPDestroy(&nep));
  CHKERRQ(MatDestroy(&A[0]));
  CHKERRQ(MatDestroy(&A[1]));
  CHKERRQ(MatDestroy(&A[2]));
  CHKERRQ(FNDestroy(&f[0]));
  CHKERRQ(FNDestroy(&f[1]));
  CHKERRQ(FNDestroy(&f[2]));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -nep_type slp -nep_target -.5 -nep_error_backward ::ascii_info_detail -nep_view_values -nep_error_absolute ::ascii_matlab -nep_monitor_all -nep_converged_reason -nep_view
      filter: grep -v "tolerance" | grep -v "problem type" | sed -e "s/[+-]0\.0*i//g" -e "s/+0i//" -e "s/[+-][0-9]\.[0-9]*e-[0-9]*i//g" -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double

   test:
      suffix: 2
      args: -nep_type rii -nep_target -.5 -nep_rii_hermitian -nep_monitor -nep_view_values ::ascii_matlab
      filter: sed -e "s/[+-][0-9]\.[0-9]*e-[0-9]*i//" -e "s/([0-9]\.[0-9]*e[+-]\([0-9]*\))/(removed)/g"
      requires: double

   test:
      suffix: 3
      args: -nep_type slp -nep_nev 4 -nep_view_values binary:myvalues.bin -checkfile myvalues.bin -nep_error_relative ::ascii_matlab
      filter: sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double

   test:
      suffix: 4
      args: -nep_type slp -nep_nev 4 -nep_monitor draw::draw_lg -nep_monitor_all draw::draw_lg -nep_view_values draw -draw_save myeigen.ppm -draw_virtual
      requires: double

TEST*/
