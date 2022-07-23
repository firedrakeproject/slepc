/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSVD of diagonal matrix, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(A,i,i,i+1,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Create the SVD solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));
  PetscCall(PetscObjectSetName((PetscObject)svd,"svd"));
  PetscCall(SVDSetOperators(svd,A,NULL));
  PetscCall(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Compute the singular triplets and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SVDSolve(svd));
  PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Check file containing the singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-checkfile",filename,sizeof(filename),&checkfile));
  if (checkfile) {
    PetscCall(SVDGetConverged(svd,&nconv));
    PetscCall(PetscMalloc1(nconv,&sigma));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    PetscCall(PetscViewerBinaryRead(viewer,sigma,nconv,NULL,PETSC_REAL));
    PetscCall(PetscViewerDestroy(&viewer));
    for (i=0;i<nconv;i++) {
      PetscCall(SVDGetSingularTriplet(svd,i,&sval,NULL,NULL));
      PetscCheck(sval==sigma[i],PETSC_COMM_WORLD,PETSC_ERR_FILE_UNEXPECTED,"Singular values in the file do not match");
    }
    PetscCall(PetscFree(sigma));
  }

  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
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
      requires: x !single

TEST*/
