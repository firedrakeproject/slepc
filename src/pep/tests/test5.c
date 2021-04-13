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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nPEP of diagonal problem, n=%D\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A[0]);CHKERRQ(ierr);
  ierr = MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A[0]);CHKERRQ(ierr);
  ierr = MatSetUp(A[0]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A[0],&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(A[0],i,i,i+1,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A[1]);CHKERRQ(ierr);
  ierr = MatSetSizes(A[1],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A[1]);CHKERRQ(ierr);
  ierr = MatSetUp(A[1]);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(A[1],i,i,-1.5,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A[2]);CHKERRQ(ierr);
  ierr = MatSetSizes(A[2],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A[2]);CHKERRQ(ierr);
  ierr = MatSetUp(A[2]);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(A[2],i,i,-1.0/(i+1),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Create the PEP solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PEPCreate(PETSC_COMM_WORLD,&pep);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)pep,"pep");CHKERRQ(ierr);
  ierr = PEPSetOperators(pep,3,A);CHKERRQ(ierr);
  ierr = PEPSetFromOptions(pep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PEPSolve(pep);CHKERRQ(ierr);
  ierr = PEPGetConverged(pep,&nconv);CHKERRQ(ierr);
  ierr = PEPGetIterationNumber(pep,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," %D converged eigenpairs after %D iterations\n",nconv,its);CHKERRQ(ierr);
  if (nconv>0) {
    ierr = MatCreateVecs(A[0],&xr,&xi);CHKERRQ(ierr);
    ierr = PEPGetEigenpair(pep,0,&kr,&ki,xr,xi);CHKERRQ(ierr);
    ierr = VecDestroy(&xr);CHKERRQ(ierr);
    ierr = VecDestroy(&xi);CHKERRQ(ierr);
    ierr = PEPGetErrorEstimate(pep,0,&errest);CHKERRQ(ierr);
  }
  ierr = PEPErrorView(pep,PEP_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Check file containing the eigenvalues
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsGetString(NULL,NULL,"-checkfile",filename,sizeof(filename),&checkfile);CHKERRQ(ierr);
  if (checkfile) {
    ierr = PetscMalloc1(nconv,&eigs);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer,eigs,nconv,NULL,PETSC_COMPLEX);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      ierr = PEPGetEigenpair(pep,i,&kr,&ki,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      eval = kr;
#else
      eval = PetscCMPLX(kr,ki);
#endif
      if (eval!=eigs[i]) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FILE_UNEXPECTED,"Eigenvalues in the file do not match");
    }
    ierr = PetscFree(eigs);CHKERRQ(ierr);
  }

  ierr = PEPDestroy(&pep);CHKERRQ(ierr);
  ierr = MatDestroy(&A[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&A[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&A[2]);CHKERRQ(ierr);
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
