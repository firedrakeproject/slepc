/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test PEP view and monitor functionality.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            A[3];
  PEP            pep;
  Vec            xr,xi;
  PetscScalar    kr,ki;
  PetscComplex   *eigs,eval;
  PetscInt       n=6,Istart,Iend,i,nconv,its;
  PetscReal      errest;
  PetscBool      checkfile;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nPEP of diagonal problem, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[1],i,i,-1.5,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[2]));
  CHKERRQ(MatSetSizes(A[2],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A[2]));
  CHKERRQ(MatSetUp(A[2]));
  for (i=Istart;i<Iend;i++) {
    CHKERRQ(MatSetValue(A[2],i,i,-1.0/(i+1),INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A[2],MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Create the PEP solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PEPCreate(PETSC_COMM_WORLD,&pep));
  CHKERRQ(PetscObjectSetName((PetscObject)pep,"pep"));
  CHKERRQ(PEPSetOperators(pep,3,A));
  CHKERRQ(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PEPSolve(pep));
  CHKERRQ(PEPGetConverged(pep,&nconv));
  CHKERRQ(PEPGetIterationNumber(pep,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT " converged eigenpairs after %" PetscInt_FMT " iterations\n",nconv,its));
  if (nconv>0) {
    CHKERRQ(MatCreateVecs(A[0],&xr,&xi));
    CHKERRQ(PEPGetEigenpair(pep,0,&kr,&ki,xr,xi));
    CHKERRQ(VecDestroy(&xr));
    CHKERRQ(VecDestroy(&xi));
    CHKERRQ(PEPGetErrorEstimate(pep,0,&errest));
  }
  CHKERRQ(PEPErrorView(pep,PEP_ERROR_RELATIVE,NULL));

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
      CHKERRQ(PEPGetEigenpair(pep,i,&kr,&ki,NULL,NULL));
#if defined(PETSC_USE_COMPLEX)
      eval = kr;
#else
      eval = PetscCMPLX(kr,ki);
#endif
      PetscCheck(eval==eigs[i],PETSC_COMM_WORLD,PETSC_ERR_FILE_UNEXPECTED,"Eigenvalues in the file do not match");
    }
    CHKERRQ(PetscFree(eigs));
  }

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
      args: -pep_error_backward ::ascii_info_detail -pep_largest_real -pep_view_values -pep_monitor_conv -pep_error_absolute ::ascii_matlab -pep_monitor_all -pep_converged_reason -pep_view
      requires: !single
      filter: grep -v "tolerance" | grep -v "problem type" | sed -e "s/[+-]0\.0*i//g" -e "s/\([0-9]\.[5]*\)[+-][0-9]\.[0-9]*e-[0-9]*i/\\1/g" -e "s/[0-9]\.[0-9]*e-\([0-9]*\)/removed/g"

   test:
      suffix: 2
      args: -n 12 -pep_largest_real -pep_monitor -pep_view_values ::ascii_matlab
      requires: double
      filter: sed -e "s/[+-][0-9]\.[0-9]*e-[0-9]*i//" -e "s/[0-9]\.[0-9]*e-\([0-9]*\)/removed/g" -e "s/5\.\([49]\)999999[0-9]*e+00/5.\\1999999999999999e+00/"

   test:
      suffix: 3
      args: -pep_nev 4 -pep_view_values binary:myvalues.bin -checkfile myvalues.bin
      requires: double

   test:
      suffix: 4
      args: -pep_nev 4 -pep_ncv 10 -pep_refine -pep_conv_norm -pep_extract none -pep_scale scalar -pep_view -pep_monitor -pep_error_relative ::ascii_info_detail
      requires: double !complex
      filter: grep -v "tolerance" | sed -e "s/[0-9]\.[0-9]*e-\([0-9]*\)/removed/g"

   test:
      suffix: 5
      args: -n 12 -pep_largest_real -pep_monitor draw::draw_lg -pep_monitor_all draw::draw_lg -pep_view_values draw -draw_save myeigen.ppm -draw_virtual
      requires: double

TEST*/
