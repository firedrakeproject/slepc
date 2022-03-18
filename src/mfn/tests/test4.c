/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates use of MFNSetBV().\n\n"
  "The command line options are:\n"
  "  -t <sval>, where <sval> = scalar value that multiplies the argument.\n"
  "  -file <filename>, where <filename> = matrix file in PETSc binary form.\n\n";

#include <slepcmfn.h>

int main(int argc,char **argv)
{
  Mat                A;           /* problem matrix */
  MFN                mfn;
  FN                 f;
  BV                 V;
  PetscInt           k=8;
  PetscReal          norm;
  PetscScalar        t=2.0;
  Vec                v,y;
  PetscErrorCode     ierr;
  PetscViewer        viewer;
  PetscBool          flg;
  char               filename[PETSC_MAX_PATH_LEN];
  MFNConvergedReason reason;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-t",&t,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMatrix exponential y=exp(t*A)*e, loaded from file\n\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Load matrix A from binary file
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-file",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name with the -file option");

#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n"));
#else
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n"));
#endif
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  /* set v = ones(n,1) */
  CHKERRQ(MatCreateVecs(A,NULL,&y));
  CHKERRQ(MatCreateVecs(A,NULL,&v));
  CHKERRQ(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Create the BV object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&V));
  CHKERRQ(PetscObjectSetName((PetscObject)V,"V"));
  CHKERRQ(BVSetSizesFromVec(V,v,k));
  CHKERRQ(BVSetFromOptions(V));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfn));
  CHKERRQ(MFNSetOperator(mfn,A));
  CHKERRQ(MFNSetBV(mfn,V));
  CHKERRQ(MFNGetFN(mfn,&f));
  CHKERRQ(FNSetType(f,FNEXP));
  CHKERRQ(FNSetScale(f,t,1.0));
  CHKERRQ(MFNSetDimensions(mfn,k));
  CHKERRQ(MFNSetFromOptions(mfn));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the problem, y=exp(t*A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MFNSolve(mfn,v,y));
  CHKERRQ(MFNGetConvergedReason(mfn,&reason));
  PetscCheck(reason>=0,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Solver did not converge");
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computed vector at time t=%.4g has norm %g\n\n",(double)PetscRealPart(t),(double)norm));

  /*
     Free work space
  */
  CHKERRQ(BVDestroy(&V));
  CHKERRQ(MFNDestroy(&mfn));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&y));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -file ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62b.petsc -k 12
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: 2
      args: -file ${DATAFILESPATH}/matrices/complex/qc324.petsc -k 12
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)

TEST*/
