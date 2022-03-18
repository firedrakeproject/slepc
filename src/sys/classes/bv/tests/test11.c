/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test BV block orthogonalization.\n\n";

#include <slepcbv.h>

/*
   Compute the Frobenius norm ||A(l:k,l:k)-diag||_F
 */
PetscErrorCode MyMatNorm(Mat A,PetscInt lda,PetscInt l,PetscInt k,PetscScalar diag,PetscReal *norm)
{
  PetscInt          i,j;
  const PetscScalar *pA;
  PetscReal         s,val;

  PetscFunctionBeginUser;
  CHKERRQ(MatDenseGetArrayRead(A,&pA));
  s = 0.0;
  for (i=l;i<k;i++) {
    for (j=l;j<k;j++) {
      val = (i==j)? PetscAbsScalar(pA[i+j*lda]-diag): PetscAbsScalar(pA[i+j*lda]);
      s += val*val;
    }
  }
  *norm = PetscSqrtReal(s);
  CHKERRQ(MatDenseRestoreArrayRead(A,&pA));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  BV             X,Y,Z,cached;
  Mat            B=NULL,M,R=NULL;
  Vec            v,t;
  PetscInt       i,j,n=20,l=2,k=8,Istart,Iend;
  PetscViewer    view;
  PetscBool      withb,resid,rand,verbose;
  PetscReal      norm;
  PetscScalar    alpha;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-withb",&withb));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-resid",&resid));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-rand",&rand));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test BV block orthogonalization (length %" PetscInt_FMT ", l=%" PetscInt_FMT ", k=%" PetscInt_FMT ")%s.\n",n,l,k,withb?" with non-standard inner product":""));

  /* Create template vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&t));
  CHKERRQ(VecSetSizes(t,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(t));

  /* Create BV object X */
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(BVSetSizesFromVec(X,t,k));
  CHKERRQ(BVSetFromOptions(X));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&view));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Fill X entries */
  if (rand) {
    CHKERRQ(BVSetRandom(X));
  } else {
    for (j=0;j<k;j++) {
      CHKERRQ(BVGetColumn(X,j,&v));
      CHKERRQ(VecSet(v,0.0));
      for (i=0;i<=n/2;i++) {
        if (i+j<n) {
          alpha = (3.0*i+j-2)/(2*(i+j+1));
          CHKERRQ(VecSetValue(v,i+j,alpha,INSERT_VALUES));
        }
      }
      CHKERRQ(VecAssemblyBegin(v));
      CHKERRQ(VecAssemblyEnd(v));
      CHKERRQ(BVRestoreColumn(X,j,&v));
    }
  }
  if (verbose) {
    CHKERRQ(BVView(X,view));
  }

  if (withb) {
    /* Create inner product matrix */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatSetUp(B));
    CHKERRQ(PetscObjectSetName((PetscObject)B,"B"));

    CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) {
      if (i>0) CHKERRQ(MatSetValue(B,i,i-1,-1.0,INSERT_VALUES));
      if (i<n-1) CHKERRQ(MatSetValue(B,i,i+1,-1.0,INSERT_VALUES));
      CHKERRQ(MatSetValue(B,i,i,2.0,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
    if (verbose) {
      CHKERRQ(MatView(B,view));
    }
    CHKERRQ(BVSetMatrix(X,B,PETSC_FALSE));
  }

  /* Create copy on Y */
  CHKERRQ(BVDuplicate(X,&Y));
  CHKERRQ(PetscObjectSetName((PetscObject)Y,"Y"));
  CHKERRQ(BVCopy(X,Y));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M));

  if (resid) {
    /* Create matrix R to store triangular factor */
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&R));
    CHKERRQ(PetscObjectSetName((PetscObject)R,"R"));
  }

  if (l>0) {
    /* First orthogonalize leading columns */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Orthogonalizing leading columns\n"));
    CHKERRQ(BVSetActiveColumns(Y,0,l));
    CHKERRQ(BVSetActiveColumns(X,0,l));
    CHKERRQ(BVOrthogonalize(Y,R));
    if (verbose) {
      CHKERRQ(BVView(Y,view));
      if (resid) CHKERRQ(MatView(R,view));
    }

    if (withb) {
      /* Extract cached BV and check it is equal to B*X */
      CHKERRQ(BVGetCachedBV(Y,&cached));
      CHKERRQ(BVDuplicate(X,&Z));
      CHKERRQ(BVSetMatrix(Z,NULL,PETSC_FALSE));
      CHKERRQ(BVSetActiveColumns(Z,0,l));
      CHKERRQ(BVCopy(X,Z));
      CHKERRQ(BVMatMult(X,B,Z));
      CHKERRQ(BVMult(Z,-1.0,1.0,cached,NULL));
      CHKERRQ(BVNorm(Z,NORM_FROBENIUS,&norm));
      if (norm<100*PETSC_MACHINE_EPSILON) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Difference ||cached-BX|| < 100*eps\n"));
      } else {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Difference ||cached-BX||: %g\n",(double)norm));
      }
      CHKERRQ(BVDestroy(&Z));
    }

    /* Check orthogonality */
    CHKERRQ(BVDot(Y,Y,M));
    CHKERRQ(MyMatNorm(M,k,0,l,1.0,&norm));
    if (norm<100*PETSC_MACHINE_EPSILON) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of orthogonality of Q1 < 100*eps\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of orthogonality of Q1: %g\n",(double)norm));
    }

    if (resid) {
      /* Check residual */
      CHKERRQ(BVDuplicate(X,&Z));
      CHKERRQ(BVSetMatrix(Z,NULL,PETSC_FALSE));
      CHKERRQ(BVSetActiveColumns(Z,0,l));
      CHKERRQ(BVCopy(X,Z));
      CHKERRQ(BVMult(Z,-1.0,1.0,Y,R));
      CHKERRQ(BVNorm(Z,NORM_FROBENIUS,&norm));
      if (norm<100*PETSC_MACHINE_EPSILON) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Residual ||X1-Q1*R11|| < 100*eps\n"));
      } else {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Residual ||X1-Q1*R11||: %g\n",(double)norm));
      }
      CHKERRQ(BVDestroy(&Z));
    }

  }

  /* Now orthogonalize the rest of columns */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Orthogonalizing active columns\n"));
  CHKERRQ(BVSetActiveColumns(Y,l,k));
  CHKERRQ(BVSetActiveColumns(X,l,k));
  CHKERRQ(BVOrthogonalize(Y,R));
  if (verbose) {
    CHKERRQ(BVView(Y,view));
    if (resid) CHKERRQ(MatView(R,view));
  }

  if (l>0) {
    /* Check orthogonality */
    CHKERRQ(BVDot(Y,Y,M));
    CHKERRQ(MyMatNorm(M,k,l,k,1.0,&norm));
    if (norm<100*PETSC_MACHINE_EPSILON) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of orthogonality of Q2 < 100*eps\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of orthogonality of Q2: %g\n",(double)norm));
    }
  }

  /* Check the complete decomposition */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Overall decomposition\n"));
  CHKERRQ(BVSetActiveColumns(Y,0,k));
  CHKERRQ(BVSetActiveColumns(X,0,k));

  /* Check orthogonality */
  CHKERRQ(BVDot(Y,Y,M));
  CHKERRQ(MyMatNorm(M,k,0,k,1.0,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of orthogonality of Q < 100*eps\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of orthogonality of Q: %g\n",(double)norm));
  }

  if (resid) {
    /* Check residual */
    CHKERRQ(BVMult(X,-1.0,1.0,Y,R));
    CHKERRQ(BVSetMatrix(X,NULL,PETSC_FALSE));
    CHKERRQ(BVNorm(X,NORM_FROBENIUS,&norm));
    if (norm<100*PETSC_MACHINE_EPSILON) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Residual ||X-Q*R|| < 100*eps\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Residual ||X-Q*R||: %g\n",(double)norm));
    }
    CHKERRQ(MatDestroy(&R));
  }

  if (B) CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&M));
  CHKERRQ(BVDestroy(&X));
  CHKERRQ(BVDestroy(&Y));
  CHKERRQ(VecDestroy(&t));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -bv_orthog_block {{gs chol tsqr tsqrchol svqb}}
      nsize: 2
      output_file: output/test11_1.out
      test:
         suffix: 1
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 1_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda

   testset:
      args: -withb -bv_orthog_block {{gs chol svqb}}
      nsize: 2
      output_file: output/test11_4.out
      test:
         suffix: 4
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 4_cuda
         args: -bv_type svec -vec_type cuda -mat_type aijcusparse
         requires: cuda

   testset:
      args: -resid -bv_orthog_block {{gs chol tsqr tsqrchol svqb}}
      nsize: 2
      output_file: output/test11_6.out
      test:
         suffix: 6
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 6_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda

   testset:
      args: -resid -withb -bv_orthog_block {{gs chol svqb}}
      nsize: 2
      output_file: output/test11_9.out
      test:
         suffix: 9
         args: -bv_type {{vecs contiguous svec mat}shared output}
      test:
         suffix: 9_cuda
         args: -bv_type svec -vec_type cuda -mat_type aijcusparse
         requires: cuda

   testset:
      args: -bv_orthog_block tsqr
      nsize: 7
      output_file: output/test11_1.out
      test:
         suffix: 11
         args: -bv_type {{vecs contiguous svec mat}shared output}
         requires: !valgrind
      test:
         suffix: 11_cuda
         args: -bv_type svec -vec_type cuda
         requires: cuda !valgrind

   testset:
      args: -resid -n 180 -l 0 -k 7 -bv_orthog_block tsqr
      nsize: 9
      output_file: output/test11_12.out
      test:
         suffix: 12
         args: -bv_type {{vecs contiguous svec mat}shared output}
         requires: !single !valgrind
      test:
         suffix: 12_cuda
         args: -bv_type svec -vec_type cuda
         requires: !single !valgrind cuda

TEST*/
