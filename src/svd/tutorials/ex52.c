/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Partial hyperbolic singular value decomposition (HSVD) from a file.\n"
  "The command line options are:\n"
  "  -file <filename>, PETSc binary file containing matrix A.\n"
  "  -p <p>, where <p> = number of -1's in signature.\n"
  "  -transpose, to transpose the matrix before doing the computation.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat            A,Omega;         /* operator matrix, signature matrix */
  SVD            svd;             /* singular value problem solver context */
  Mat            At;
  Vec            u,v,vomega,*U,*V;
  MatType        Atype;
  PetscReal      tol,lev1=0.0,lev2=0.0;
  PetscInt       M,N,p=0,i,Istart,Iend,nconv;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,terse,skiporth=PETSC_FALSE,transpose=PETSC_FALSE;

  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load matrix that defines the hyperbolic singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nHyperbolic singular value problem stored in file.\n\n"));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-file",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -file option");

#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n"));
#else
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n"));
#endif
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* transpose the matrix if requested */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-transpose",&transpose,NULL));
  if (transpose) {
    PetscCall(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&At));
    PetscCall(MatDestroy(&A));
    A = At;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          Create the signature
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-p",&p,&flg));
  PetscCheck(p>=0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Parameter p cannot be negative");
  PetscCheck(p<M,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Parameter p cannot be larger than the number of rows of A");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n Matrix dimensions: %" PetscInt_FMT "x%" PetscInt_FMT,M,N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,", using signature Omega=blkdiag(-I_%" PetscInt_FMT ",I_%" PetscInt_FMT ")\n\n",p,M-p));

  PetscCall(MatCreateVecs(A,NULL,&vomega));
  PetscCall(VecSet(vomega,1.0));
  PetscCall(VecGetOwnershipRange(vomega,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i<p) PetscCall(VecSetValue(vomega,i,-1.0,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(vomega));
  PetscCall(VecAssemblyEnd(vomega));

  PetscCall(MatGetType(A,&Atype));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&Omega));
  PetscCall(MatSetSizes(Omega,PETSC_DECIDE,PETSC_DECIDE,M,M));
  PetscCall(MatSetType(Omega,Atype));
  PetscCall(MatSetUp(Omega));
  PetscCall(MatDiagonalSet(Omega,vomega,INSERT_VALUES));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));

  PetscCall(SVDSetOperators(svd,A,NULL));
  PetscCall(SVDSetSignature(svd,vomega));
  PetscCall(SVDSetProblemType(svd,SVD_HYPERBOLIC));

  PetscCall(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Solve the problem, display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDSolve(svd));

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(SVDErrorView(svd,SVD_ERROR_NORM,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(SVDConvergedReasonView(svd,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(SVDErrorView(svd,SVD_ERROR_NORM,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  /* check orthogonality */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-skiporth",&skiporth,NULL));
  PetscCall(SVDGetConverged(svd,&nconv));
  if (nconv>0 && !skiporth) {
    PetscCall(SVDGetTolerances(svd,&tol,NULL));
    PetscCall(MatCreateVecs(A,&v,&u));
    PetscCall(VecDuplicateVecs(u,nconv,&U));
    PetscCall(VecDuplicateVecs(v,nconv,&V));
    for (i=0;i<nconv;i++) PetscCall(SVDGetSingularTriplet(svd,i,NULL,U[i],V[i]));
    PetscCall(VecCheckOrthonormality(U,nconv,NULL,nconv,Omega,NULL,&lev1));
    PetscCall(VecCheckOrthonormality(V,nconv,NULL,nconv,NULL,NULL,&lev2));
    if (lev1+lev2<20*tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below the tolerance\n"));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g (U) %g (V)\n",(double)lev1,(double)lev2));
    PetscCall(VecDestroyVecs(nconv,&U));
    PetscCall(VecDestroyVecs(nconv,&V));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&v));
  }

  /* free work space */
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Omega));
  PetscCall(VecDestroy(&vomega));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -file ${DATAFILESPATH}/matrices/real/illc1033.petsc -svd_nsv 62 -p 40 -terse
      requires: double !complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex52_1.out
      test:
         args: -svd_type cross -svd_cross_explicitmatrix {{0 1}} -svd_implicittranspose {{0 1}}
         suffix: 1_cross

   testset:
      args: -file ${DATAFILESPATH}/matrices/real/illc1033.petsc -transpose -svd_nsv 6 -p 130 -terse
      requires: double !complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex52_2.out
      test:
         args: -svd_type cross -svd_cross_explicitmatrix {{0 1}} -svd_implicittranspose {{0 1}}
         suffix: 2_cross

   testset:
      args: -file ${DATAFILESPATH}/matrices/complex/qc324.petsc -svd_nsv 12 -p 24 -terse
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex52_3.out
      test:
         args: -svd_type cross -svd_cross_explicitmatrix {{0 1}}
         suffix: 3_cross

TEST*/
