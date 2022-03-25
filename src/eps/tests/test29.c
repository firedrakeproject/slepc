/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates the computation of left eigenvectors for generalized eigenproblems.\n\n"
  "The command line options are:\n"
  "  -f1 <filename> -f2 <filename>, PETSc binary files containing A and B\n\n";

#include <slepceps.h>

/*
   User-defined routines
*/
PetscErrorCode ComputeResidualNorm(Mat,Mat,PetscBool,PetscScalar,PetscScalar,Vec,Vec,Vec*,PetscReal*);

int main(int argc,char **argv)
{
  Mat            A,B;
  EPS            eps;
  EPSType        type;
  PetscInt       i,nconv;
  PetscBool      twosided,flg;
  PetscReal      nrmr,nrml=0.0,re,im,lev;
  PetscScalar    *kr,*ki;
  Vec            t,*xr,*xi,*yr,*yi,*z;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized eigenproblem stored in file.\n\n"));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f1",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n"));
#else
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n"));
#endif
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f2",filename,sizeof(filename),&flg));
  if (flg) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatLoad(B,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=I\n\n"));
    B = NULL;
  }
  CHKERRQ(MatCreateVecs(A,NULL,&t));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,B));

  /* use a two-sided algorithm to compute left eigenvectors as well */
  CHKERRQ(EPSSetTwoSided(eps,PETSC_TRUE));

  /* allow user to change settings at run time */
  CHKERRQ(EPSSetFromOptions(eps));
  CHKERRQ(EPSGetTwoSided(eps,&twosided));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Get number of converged approximate eigenpairs
  */
  CHKERRQ(EPSGetConverged(eps,&nconv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %" PetscInt_FMT "\n\n",nconv));
  CHKERRQ(PetscMalloc2(nconv,&kr,nconv,&ki));
  CHKERRQ(VecDuplicateVecs(t,3,&z));
  CHKERRQ(VecDuplicateVecs(t,nconv,&xr));
  CHKERRQ(VecDuplicateVecs(t,nconv,&xi));
  if (twosided) {
    CHKERRQ(VecDuplicateVecs(t,nconv,&yr));
    CHKERRQ(VecDuplicateVecs(t,nconv,&yi));
  }

  if (nconv>0) {
    /*
       Display eigenvalues and relative errors
    */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,
         "           k            ||Ax-kBx||         ||y'A-y'Bk||\n"
         "   ---------------- ------------------ ------------------\n"));

    for (i=0;i<nconv;i++) {
      /*
        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
        ki (imaginary part)
      */
      CHKERRQ(EPSGetEigenpair(eps,i,&kr[i],&ki[i],xr[i],xi[i]));
      if (twosided) CHKERRQ(EPSGetLeftEigenvector(eps,i,yr[i],yi[i]));
      /*
         Compute the residual norms associated to each eigenpair
      */
      CHKERRQ(ComputeResidualNorm(A,B,PETSC_FALSE,kr[i],ki[i],xr[i],xi[i],z,&nrmr));
      if (twosided) CHKERRQ(ComputeResidualNorm(A,B,PETSC_TRUE,kr[i],ki[i],yr[i],yi[i],z,&nrml));

#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr[i]);
      im = PetscImaginaryPart(kr[i]);
#else
      re = kr[i];
      im = ki[i];
#endif
      if (im!=0.0) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %8f%+8fi %12g       %12g\n",(double)re,(double)im,(double)nrmr,(double)nrml));
      else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12g       %12g\n",(double)re,(double)nrmr,(double)nrml));
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    /*
       Check bi-orthogonality of eigenvectors
    */
    if (twosided) {
      CHKERRQ(VecCheckOrthogonality(xr,nconv,yr,nconv,B,NULL,&lev));
      if (lev<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of bi-orthogonality of eigenvectors < 100*eps\n\n"));
      else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of bi-orthogonality of eigenvectors: %g\n\n",(double)lev));
    }
  }

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(PetscFree2(kr,ki));
  CHKERRQ(VecDestroyVecs(3,&z));
  CHKERRQ(VecDestroyVecs(nconv,&xr));
  CHKERRQ(VecDestroyVecs(nconv,&xi));
  if (twosided) {
    CHKERRQ(VecDestroyVecs(nconv,&yr));
    CHKERRQ(VecDestroyVecs(nconv,&yi));
  }
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*
   ComputeResidualNorm - Computes the norm of the residual vector
   associated with an eigenpair.

   Input Parameters:
     trans - whether A' must be used instead of A
     kr,ki - eigenvalue
     xr,xi - eigenvector
     z     - three work vectors (the second one not referenced in complex scalars)
*/
PetscErrorCode ComputeResidualNorm(Mat A,Mat B,PetscBool trans,PetscScalar kr,PetscScalar ki,Vec xr,Vec xi,Vec *z,PetscReal *norm)
{
  Vec            u,w=NULL;
  PetscScalar    alpha;
#if !defined(PETSC_USE_COMPLEX)
  Vec            v;
  PetscReal      ni,nr;
#endif
  PetscErrorCode (*matmult)(Mat,Vec,Vec) = trans? MatMultHermitianTranspose: MatMult;

  PetscFunctionBegin;
  u = z[0];
  if (B) w = z[2];

#if !defined(PETSC_USE_COMPLEX)
  v = z[1];
  if (ki == 0 || PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    CHKERRQ((*matmult)(A,xr,u));                          /* u=A*x */
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      if (B) CHKERRQ((*matmult)(B,xr,w));             /* w=B*x */
      else w = xr;
      alpha = trans? -PetscConj(kr): -kr;
      CHKERRQ(VecAXPY(u,alpha,w));                        /* u=A*x-k*B*x */
    }
    CHKERRQ(VecNorm(u,NORM_2,norm));
#if !defined(PETSC_USE_COMPLEX)
  } else {
    CHKERRQ((*matmult)(A,xr,u));                          /* u=A*xr */
    if (SlepcAbsEigenvalue(kr,ki) > PETSC_MACHINE_EPSILON) {
      if (B) CHKERRQ((*matmult)(B,xr,v));             /* v=B*xr */
      else CHKERRQ(VecCopy(xr,v));
      CHKERRQ(VecAXPY(u,-kr,v));                          /* u=A*xr-kr*B*xr */
      if (B) CHKERRQ((*matmult)(B,xi,w));             /* w=B*xi */
      else w = xi;
      CHKERRQ(VecAXPY(u,trans?-ki:ki,w));                 /* u=A*xr-kr*B*xr+ki*B*xi */
    }
    CHKERRQ(VecNorm(u,NORM_2,&nr));
    CHKERRQ((*matmult)(A,xi,u));                          /* u=A*xi */
    if (SlepcAbsEigenvalue(kr,ki) > PETSC_MACHINE_EPSILON) {
      CHKERRQ(VecAXPY(u,-kr,w));                          /* u=A*xi-kr*B*xi */
      CHKERRQ(VecAXPY(u,trans?ki:-ki,v));                 /* u=A*xi-kr*B*xi-ki*B*xr */
    }
    CHKERRQ(VecNorm(u,NORM_2,&ni));
    *norm = SlepcAbsEigenvalue(nr,ni);
  }
#endif
  PetscFunctionReturn(0);
}

/*TEST

   testset:
      args: -f1 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -f2 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62b.petsc -eps_nev 4 -st_type sinvert -eps_target -190000
      filter: grep -v "method" | sed -e "s/[+-]0\.0*i//g" | sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      test:
         suffix: 1
      test:
         suffix: 1_rqi
         args: -eps_type power -eps_power_shift_type rayleigh -eps_nev 2 -eps_target -2000
      test:
         suffix: 1_rqi_singular
         args: -eps_type power -eps_power_shift_type rayleigh -eps_nev 1 -eps_target -195500

   test:
      suffix: 2
      args: -f1 ${DATAFILESPATH}/matrices/complex/mhd1280a.petsc -f2 ${DATAFILESPATH}/matrices/complex/mhd1280b.petsc -eps_nev 6 -eps_tol 1e-11
      filter: sed -e "s/-892/+892/" | sed -e "s/-759/+759/" | sed -e "s/-674/+674/" | sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      timeoutfactor: 2

TEST*/
