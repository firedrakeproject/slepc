/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test EPS view and monitor functionality.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;
  EPS            eps;
  PetscInt       n=6,Istart,Iend,i;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nEPS of diagonal problem, n=%" PetscInt_FMT "\n\n",n));

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
                     Create the EPS solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(PetscObjectSetName((PetscObject)eps,"eps"));
  PetscCall(EPSSetOperators(eps,A,NULL));
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSSolve(eps));
  PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -eps_error_backward ::ascii_info_detail -eps_largest_real -eps_balance oneside -eps_view_values -eps_monitor_conv -eps_error_absolute ::ascii_matlab -eps_monitor_all -eps_converged_reason -eps_view
      requires: !single
      filter: grep -v "tolerance" | sed -e "s/hermitian/symmetric/" -e "s/[+-]0\.0*i//g" -e "s/\([1-6]\.\)[+-][0-9]\.[0-9]*e-[0-9]*i/\\1/g" -e "s/[0-9]\.[0-9]*e-\([0-9]*\)/removed/g"

   test:
      suffix: 2
      args: -n 20 -eps_largest_real -eps_monitor -eps_view_values ::ascii_matlab
      requires: double
      filter: sed -e "s/[+-][0-9]\.[0-9]*e-[0-9]*i//" -e "s/[0-9]\.[0-9]*e-\([0-9]*\)/removed/g" -e "s/2\.[0-9]*e+01/2.0000000000000000e+01/" -e "s/1\.9999999999[0-9]*e+01/2.0000000000000000e+01/"

   test:
      suffix: 3
      args: -n 20 -eps_largest_real -eps_monitor draw::draw_lg -eps_monitor_all draw::draw_lg -eps_view_values draw -draw_save myeigen.ppm -draw_virtual
      requires: x double

TEST*/
