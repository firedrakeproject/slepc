/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test MFN interface functions.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n\n";

#include <slepcmfn.h>

int main(int argc,char **argv)
{
  Mat                  A,B;
  MFN                  mfn;
  FN                   f;
  MFNConvergedReason   reason;
  MFNType              type;
  PetscReal            norm,tol;
  Vec                  v,y;
  PetscInt             N,n=4,Istart,Iend,i,j,II,ncv,its,maxit;
  PetscBool            flg,testprefix=PETSC_FALSE;
  const char           *prefix;
  PetscViewerAndFormat *vf;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  N = n*n;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSquare root of Laplacian y=sqrt(A)*e_1, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,n));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_prefix",&testprefix,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Compute the discrete 2-D Laplacian, A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,4.0,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));

  CHKERRQ(MatCreateVecs(A,NULL,&v));
  CHKERRQ(VecSetValue(v,0,1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  CHKERRQ(VecDuplicate(v,&y));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the solver, set the matrix and the function
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfn));
  CHKERRQ(MFNSetOperator(mfn,A));
  CHKERRQ(MFNGetFN(mfn,&f));
  CHKERRQ(FNSetType(f,FNSQRT));

  CHKERRQ(MFNSetType(mfn,MFNKRYLOV));
  CHKERRQ(MFNGetType(mfn,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Type set to %s\n",type));

  /* test prefix usage */
  if (testprefix) {
    CHKERRQ(MFNSetOptionsPrefix(mfn,"check_"));
    CHKERRQ(MFNAppendOptionsPrefix(mfn,"myprefix_"));
    CHKERRQ(MFNGetOptionsPrefix(mfn,&prefix));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MFN prefix is currently: %s\n",prefix));
  }

  /* test some interface functions */
  CHKERRQ(MFNGetOperator(mfn,&B));
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MFNSetTolerances(mfn,1e-4,500));
  CHKERRQ(MFNSetDimensions(mfn,6));
  CHKERRQ(MFNSetErrorIfNotConverged(mfn,PETSC_TRUE));
  /* test monitors */
  CHKERRQ(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf));
  CHKERRQ(MFNMonitorSet(mfn,(PetscErrorCode (*)(MFN,PetscInt,PetscReal,void*))MFNMonitorDefault,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  /* CHKERRQ(MFNMonitorCancel(mfn)); */
  CHKERRQ(MFNSetFromOptions(mfn));

  /* query properties and print them */
  CHKERRQ(MFNGetTolerances(mfn,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Tolerance: %g, max iterations: %" PetscInt_FMT "\n",(double)tol,maxit));
  CHKERRQ(MFNGetDimensions(mfn,&ncv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Subspace dimension: %" PetscInt_FMT "\n",ncv));
  CHKERRQ(MFNGetErrorIfNotConverged(mfn,&flg));
  if (flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Erroring out if convergence fails\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                           Solve  y=sqrt(A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MFNSolve(mfn,v,y));
  CHKERRQ(MFNGetConvergedReason(mfn,&reason));
  CHKERRQ(MFNGetIterationNumber(mfn,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Finished - converged reason = %d\n",(int)reason));
  /* CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," its = %" PetscInt_FMT "\n",its)); */
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," sqrt(A)*v has norm %g\n",(double)norm));

  /*
     Free work space
  */
  CHKERRQ(MFNDestroy(&mfn));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -mfn_monitor_cancel -mfn_converged_reason -mfn_view -log_exclude mfn,bv,fn -mfn_monitor draw::draw_lg -draw_virtual

   test:
      suffix: 2
      args: -test_prefix -check_myprefix_mfn_monitor
      filter: sed -e "s/estimate [0-9]\.[0-9]*e[+-]\([0-9]*\)/estimate (removed)/g" | sed -e "s/4.0[0-9]*e-10/4.03e-10/"

TEST*/
