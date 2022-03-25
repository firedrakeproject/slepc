/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates region filtering. "
  "Based on ex5.\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of grid subdivisions in each dimension.\n\n";

#include <slepceps.h>

/*
   User-defined routines
*/
PetscErrorCode MatMarkovModel(PetscInt m,Mat A);

int main(int argc,char **argv)
{
  Mat            A;
  EPS            eps;
  ST             st;
  RG             rg;
  PetscReal      radius,tol=PETSC_SMALL;
  PetscScalar    target=0.5,kr,ki;
  PetscComplex   *eigs,eval;
  PetscInt       N,m=15,nev,i,nconv;
  PetscBool      checkfile;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  char           str[50];

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
#if defined(PETSC_HAVE_COMPLEX)
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  N = m*(m+1)/2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMarkov Model, N=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n",N,m));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-target",&target,NULL));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),target,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Searching closest eigenvalues to the right of %s.\n\n",str));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatMarkovModel(m,A));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create a standalone spectral transformation
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(STCreate(PETSC_COMM_WORLD,&st));
  CHKERRQ(STSetType(st,STSINVERT));
  CHKERRQ(STSetShift(st,target));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Create a region for filtering
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(RGCreate(PETSC_COMM_WORLD,&rg));
  CHKERRQ(RGSetType(rg,RGELLIPSE));
  radius = (1.0-PetscRealPart(target))/2.0;
  CHKERRQ(RGEllipseSetParameters(rg,target+radius,radius,0.1));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetST(eps,st));
  CHKERRQ(EPSSetRG(eps,rg));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  CHKERRQ(EPSSetTarget(eps,target));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Check file containing the eigenvalues
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-checkfile",filename,sizeof(filename),&checkfile));
  if (checkfile) {
    CHKERRQ(EPSGetConverged(eps,&nconv));
    CHKERRQ(PetscMalloc1(nconv,&eigs));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(PetscViewerBinaryRead(viewer,eigs,nconv,NULL,PETSC_COMPLEX));
    CHKERRQ(PetscViewerDestroy(&viewer));
    for (i=0;i<nconv;i++) {
      CHKERRQ(EPSGetEigenpair(eps,i,&kr,&ki,NULL,NULL));
#if defined(PETSC_USE_COMPLEX)
      eval = kr;
#else
      eval = PetscCMPLX(kr,ki);
#endif
      PetscCheck(eval==eigs[i],PETSC_COMM_WORLD,PETSC_ERR_FILE_UNEXPECTED,"Eigenvalues in the file do not match");
    }
    CHKERRQ(PetscFree(eigs));
  }

  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(STDestroy(&st));
  CHKERRQ(RGDestroy(&rg));
  CHKERRQ(MatDestroy(&A));
#else
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example requires C99 complex numbers");
#endif
  CHKERRQ(SlepcFinalize());
  return 0;
}

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

/*TEST

   test:
      suffix: 1
      args: -eps_nev 4 -eps_ncv 20 -eps_view_values binary:myvalues.bin -checkfile myvalues.bin
      output_file: output/test11_1.out
      requires: !single c99_complex

TEST*/
