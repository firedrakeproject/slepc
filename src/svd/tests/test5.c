/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test SVD view and monitor functionality.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat            A;
  SVD            svd;
  PetscReal      *sigma,sval;
  PetscInt       n=6,Istart,Iend,i,nconv;
  PetscBool      checkfile;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSVD of diagonal matrix, n=%D\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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
                     Create the SVD solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)svd,"svd");CHKERRQ(ierr);
  ierr = SVDSetOperators(svd,A,NULL);CHKERRQ(ierr);
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Compute the singular triplets and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Check file containing the singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsGetString(NULL,NULL,"-checkfile",filename,sizeof(filename),&checkfile);CHKERRQ(ierr);
  if (checkfile) {
    ierr = SVDGetConverged(svd,&nconv);CHKERRQ(ierr);
    ierr = PetscMalloc1(nconv,&sigma);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer,sigma,nconv,NULL,PETSC_REAL);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      ierr = SVDGetSingularTriplet(svd,i,&sval,NULL,NULL);CHKERRQ(ierr);
      if (sval!=sigma[i]) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FILE_UNEXPECTED,"Singular values in the file do not match");
    }
    ierr = PetscFree(sigma);CHKERRQ(ierr);
  }

  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -svd_error_relative ::ascii_info_detail -svd_view_values -svd_monitor_conv -svd_error_absolute ::ascii_matlab -svd_monitor_all -svd_converged_reason -svd_view
      filter: grep -v "tolerance" | grep -v "problem type" | sed -e "s/1.999999/2.000000/" | sed -e "s/2.000001/2.000000/" | sed -e "s/[0-9]\.[0-9]*e-\([0-9]*\)/removed/g"
      requires: !single

   test:
      suffix: 2
      args: -svd_nsv 4 -svd_view_values binary:myvalues.bin -checkfile myvalues.bin
      requires: double

   test:
      suffix: 3
      args: -svd_type trlanczos -svd_error_relative ::ascii_info_detail -svd_view_values -svd_monitor_conv -svd_error_absolute ::ascii_matlab -svd_monitor_all -svd_converged_reason -svd_view
      filter: grep -v "tolerance" | grep -v "problem type" | sed -e "s/1.999999/2.000000/" | sed -e "s/2.000001/2.000000/" | sed -e "s/[0-9]\.[0-9]*e-\([0-9]*\)/removed/g"
      requires: !single

   test:
      suffix: 4
      args: -svd_monitor draw::draw_lg -svd_monitor_all draw::draw_lg -svd_view_values draw -draw_save mysingu.ppm -draw_virtual
      requires: !single

TEST*/
