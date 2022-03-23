/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates the computation of left eigenvectors.\n\n"
  "The problem is the Markov model as in ex5.c.\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of grid subdivisions in each dimension.\n\n";

#include <slepceps.h>

/*
   User-defined routines
*/
PetscErrorCode MatMarkovModel(PetscInt,Mat);
PetscErrorCode ComputeResidualNorm(Mat,PetscBool,PetscScalar,PetscScalar,Vec,Vec,Vec,PetscReal*);

int main(int argc,char **argv)
{
  Vec            v0,w0;           /* initial vectors */
  Mat            A;               /* operator matrix */
  EPS            eps;             /* eigenproblem solver context */
  EPSType        type;
  PetscInt       i,N,m=15,nconv;
  PetscBool      twosided;
  PetscReal      nrmr,nrml=0.0,re,im,lev;
  PetscScalar    *kr,*ki;
  Vec            t,*xr,*xi,*yr,*yi;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  N = m*(m+1)/2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMarkov Model, N=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n\n",N,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatMarkovModel(m,A));
  CHKERRQ(MatCreateVecs(A,NULL,&t));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));

  /* use a two-sided algorithm to compute left eigenvectors as well */
  CHKERRQ(EPSSetTwoSided(eps,PETSC_TRUE));

  /* allow user to change settings at run time */
  CHKERRQ(EPSSetFromOptions(eps));
  CHKERRQ(EPSGetTwoSided(eps,&twosided));

  /*
     Set the initial vectors. This is optional, if not done the initial
     vectors are set to random values
  */
  CHKERRQ(MatCreateVecs(A,&v0,&w0));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (!rank) {
    CHKERRQ(VecSetValue(v0,0,1.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(v0,1,1.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(v0,2,1.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(w0,0,2.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(w0,2,0.5,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(v0));
  CHKERRQ(VecAssemblyBegin(w0));
  CHKERRQ(VecAssemblyEnd(v0));
  CHKERRQ(VecAssemblyEnd(w0));
  CHKERRQ(EPSSetInitialSpace(eps,1,&v0));
  CHKERRQ(EPSSetLeftInitialSpace(eps,1,&w0));

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
         "           k            ||Ax-kx||         ||y'A-y'k||\n"
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
      CHKERRQ(ComputeResidualNorm(A,PETSC_FALSE,kr[i],ki[i],xr[i],xi[i],t,&nrmr));
      if (twosided) CHKERRQ(ComputeResidualNorm(A,PETSC_TRUE,kr[i],ki[i],yr[i],yi[i],t,&nrml));

#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr[i]);
      im = PetscImaginaryPart(kr[i]);
#else
      re = kr[i];
      im = ki[i];
#endif
      if (im!=0.0) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %8f%+8fi %12g %12g\n",(double)re,(double)im,(double)nrmr,(double)nrml));
      else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12g       %12g\n",(double)re,(double)nrmr,(double)nrml));
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    /*
       Check bi-orthogonality of eigenvectors
    */
    if (twosided) {
      CHKERRQ(VecCheckOrthogonality(xr,nconv,yr,nconv,NULL,NULL,&lev));
      if (lev<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of bi-orthogonality of eigenvectors < 100*eps\n\n"));
      else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Level of bi-orthogonality of eigenvectors: %g\n\n",(double)lev));
    }
  }

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v0));
  CHKERRQ(VecDestroy(&w0));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(PetscFree2(kr,ki));
  CHKERRQ(VecDestroyVecs(nconv,&xr));
  CHKERRQ(VecDestroyVecs(nconv,&xi));
  if (twosided) {
    CHKERRQ(VecDestroyVecs(nconv,&yr));
    CHKERRQ(VecDestroyVecs(nconv,&yi));
  }
  ierr = SlepcFinalize();
  return ierr;
}

/*
    Matrix generator for a Markov model of a random walk on a triangular grid.

    This subroutine generates a test matrix that models a random walk on a
    triangular grid. This test example was used by G. W. Stewart ["{SRRIT} - a
    FORTRAN subroutine to calculate the dominant invariant subspaces of a real
    matrix", Tech. report. TR-514, University of Maryland (1978).] and in a few
    papers on eigenvalue problems by Y. Saad [see e.g. LAA, vol. 34, pp. 269-295
    (1980) ]. These matrices provide reasonably easy test problems for eigenvalue
    algorithms. The transpose of the matrix  is stochastic and so it is known
    that one is an exact eigenvalue. One seeks the eigenvector of the transpose
    associated with the eigenvalue unity. The problem is to calculate the steady
    state probability distribution of the system, which is the eigevector
    associated with the eigenvalue one and scaled in such a way that the sum all
    the components is equal to one.

    Note: the code will actually compute the transpose of the stochastic matrix
    that contains the transition probabilities.
*/
PetscErrorCode MatMarkovModel(PetscInt m,Mat A)
{
  const PetscReal cst = 0.5/(PetscReal)(m-1);
  PetscReal       pd,pu;
  PetscInt        Istart,Iend,i,j,jmax,ix=0;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=1;i<=m;i++) {
    jmax = m-i+1;
    for (j=1;j<=jmax;j++) {
      ix = ix + 1;
      if (ix-1<Istart || ix>Iend) continue;  /* compute only owned rows */
      if (j!=jmax) {
        pd = cst*(PetscReal)(i+j-1);
        /* north */
        if (i==1) CHKERRQ(MatSetValue(A,ix-1,ix,2*pd,INSERT_VALUES));
        else CHKERRQ(MatSetValue(A,ix-1,ix,pd,INSERT_VALUES));
        /* east */
        if (j==1) CHKERRQ(MatSetValue(A,ix-1,ix+jmax-1,2*pd,INSERT_VALUES));
        else CHKERRQ(MatSetValue(A,ix-1,ix+jmax-1,pd,INSERT_VALUES));
      }
      /* south */
      pu = 0.5 - cst*(PetscReal)(i+j-3);
      if (j>1) CHKERRQ(MatSetValue(A,ix-1,ix-2,pu,INSERT_VALUES));
      /* west */
      if (i>1) CHKERRQ(MatSetValue(A,ix-1,ix-jmax-2,pu,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
   ComputeResidualNorm - Computes the norm of the residual vector
   associated with an eigenpair.

   Input Parameters:
     trans - whether A' must be used instead of A
     kr,ki - eigenvalue
     xr,xi - eigenvector
     u     - work vector
*/
PetscErrorCode ComputeResidualNorm(Mat A,PetscBool trans,PetscScalar kr,PetscScalar ki,Vec xr,Vec xi,Vec u,PetscReal *norm)
{
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      ni,nr;
#endif
  PetscErrorCode (*matmult)(Mat,Vec,Vec) = trans? MatMultTranspose: MatMult;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  if (ki == 0 || PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    CHKERRQ((*matmult)(A,xr,u));
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) CHKERRQ(VecAXPY(u,-kr,xr));
    CHKERRQ(VecNorm(u,NORM_2,norm));
#if !defined(PETSC_USE_COMPLEX)
  } else {
    CHKERRQ((*matmult)(A,xr,u));
    if (SlepcAbsEigenvalue(kr,ki) > PETSC_MACHINE_EPSILON) {
      CHKERRQ(VecAXPY(u,-kr,xr));
      CHKERRQ(VecAXPY(u,ki,xi));
    }
    CHKERRQ(VecNorm(u,NORM_2,&nr));
    CHKERRQ((*matmult)(A,xi,u));
    if (SlepcAbsEigenvalue(kr,ki) > PETSC_MACHINE_EPSILON) {
      CHKERRQ(VecAXPY(u,-kr,xi));
      CHKERRQ(VecAXPY(u,-ki,xr));
    }
    CHKERRQ(VecNorm(u,NORM_2,&ni));
    *norm = SlepcAbsEigenvalue(nr,ni);
  }
#endif
  PetscFunctionReturn(0);
}

/*TEST

   testset:
      args: -st_type sinvert -eps_target 1.1 -eps_nev 4
      filter: grep -v method | sed -e "s/[+-]0\.0*i//g" | sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: !single
      output_file: output/ex41_1.out
      test:
         suffix: 1
         args: -eps_type {{power krylovschur}}
      test:
         suffix: 1_balance
         args: -eps_balance {{oneside twoside}} -eps_ncv 18 -eps_krylovschur_locking 0

TEST*/
