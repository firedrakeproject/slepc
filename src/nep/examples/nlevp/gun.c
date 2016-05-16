/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This example implements one of the problems found at
       NLEVP: A Collection of Nonlinear Eigenvalue Problems,
       The University of Manchester.
   The details of the collection can be found at:
       [1] T. Betcke et al., "NLEVP: A Collection of Nonlinear Eigenvalue
           Problems", ACM Trans. Math. Software 39(2), Article 7, 2013.

   The gun problem arises from model of a radio-frequency gun cavity, with
   the complex nonlinear function
   T(lambda) = K-lambda*M+i*lambda^(1/2)*W1+i*(lambda-108.8774^2)^(1/2)*W2

   Data files can be downloaded from http://slepc.upv.es/datafiles
*/

static char help[] = "Radio-frequency gun cavity.\n\n"
  "The command line options are:\n"
  "-K <filename1> -M <filename2> -W1 <filename3> -W2 <filename4>, where filename1,..,filename4 are files containing the matrices in PETSc binary form defining the GUN problem.\n\n";

#include <slepcnep.h>

#define NMAT 4
#define SIGMA 108.8774

PetscErrorCode ComputeSingularities(NEP,PetscInt*,PetscScalar*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Mat            A[NMAT];         /* problem matrices */
  FN             f[NMAT];         /* functions to define the nonlinear operator */
  FN             ff[2];           /* auxiliary functions to define the nonlinear operator */
  NEP            nep;             /* nonlinear eigensolver context */
  PetscBool      terse,flg;
  const char*    string[NMAT]={"-K","-M","-W1","-W2"};
  char           filename[PETSC_MAX_PATH_LEN];
  PetscScalar    numer[2],sigma;
  PetscInt       i;
  PetscViewer    viewer;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"GUN problem\n\n");CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_SELF,1,"This example requires complex scalars!");
#endif

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                       Load the problem matrices 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  for (i=0;i<NMAT;i++) {
    ierr = PetscOptionsGetString(NULL,NULL,string[i],filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ1(PETSC_COMM_WORLD,1,"Must indicate a filename with the %s option",string[i]);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A[i]);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A[i]);CHKERRQ(ierr);
    ierr = MatLoad(A[i],viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                       Create the problem functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* f1=1 */
  ierr = FNCreate(PETSC_COMM_WORLD,&f[0]);CHKERRQ(ierr);
  ierr = FNSetType(f[0],FNRATIONAL);CHKERRQ(ierr);
  numer[0] = 1.0;
  ierr = FNRationalSetNumerator(f[0],1,numer);CHKERRQ(ierr);

  /* f2=-lambda */
  ierr = FNCreate(PETSC_COMM_WORLD,&f[1]);CHKERRQ(ierr);
  ierr = FNSetType(f[1],FNRATIONAL);CHKERRQ(ierr);
  numer[0] = -1.0; numer[1] = 0.0;
  ierr = FNRationalSetNumerator(f[1],2,numer);CHKERRQ(ierr);

  /* f3=i*sqrt(lambda) */
  ierr = FNCreate(PETSC_COMM_WORLD,&f[2]);CHKERRQ(ierr);
  ierr = FNSetType(f[2],FNSQRT);CHKERRQ(ierr);
  ierr = FNSetScale(f[2],1.0,PETSC_i);CHKERRQ(ierr);

  /* f4=i*sqrt(lambda-sigma^2) */
  sigma = SIGMA*SIGMA;
  ierr = FNCreate(PETSC_COMM_WORLD,&ff[0]);CHKERRQ(ierr);
  ierr = FNSetType(ff[0],FNSQRT);CHKERRQ(ierr);
  ierr = FNCreate(PETSC_COMM_WORLD,&ff[1]);CHKERRQ(ierr);
  ierr = FNSetType(ff[1],FNRATIONAL);CHKERRQ(ierr);
  numer[0] = 1.0; numer[1] = -sigma;
  ierr = FNRationalSetNumerator(ff[1],2,numer);CHKERRQ(ierr);
  ierr = FNCreate(PETSC_COMM_WORLD,&f[3]);CHKERRQ(ierr);
  ierr = FNSetType(f[3],FNCOMBINE);CHKERRQ(ierr);
  ierr = FNCombineSetChildren(f[3],FN_COMBINE_COMPOSE,ff[1],ff[0]);CHKERRQ(ierr);
  ierr = FNSetScale(f[3],1.0,PETSC_i);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPCreate(PETSC_COMM_WORLD,&nep);CHKERRQ(ierr);
  ierr = NEPSetSplitOperator(nep,4,A,f,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = NEPSetFromOptions(nep);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)nep,NEPNLEIGS,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = NEPNLEIGSSetSingularitiesFunction(nep,ComputeSingularities,NULL);CHKERRQ(ierr);
  }

  ierr = NEPSolve(nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = NEPReasonView(nep,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = NEPDestroy(&nep);CHKERRQ(ierr);
  for (i=0;i<NMAT;i++) {
    ierr = MatDestroy(&A[i]);CHKERRQ(ierr);
    ierr = FNDestroy(&f[i]);CHKERRQ(ierr);
  }
  for (i=0;i<2;i++) {
    ierr = FNDestroy(&ff[i]);CHKERRQ(ierr);
  }
  ierr = SlepcFinalize();
  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeSingularities"
/*
   ComputeSingularities - Computes maxnp points (at most) in the complex plane where
   the function T(.) is not analytic.

   In this case, we discretize the singularity region (-inf,108.8774^2)~(-10e+12,-10e-12+108.8774^2) 
*/
PetscErrorCode ComputeSingularities(NEP nep,PetscInt *maxnp,PetscScalar *xi,void *pt)
{
  PetscReal h;
  PetscInt  i;
  PetscReal   sigma,end;

  PetscFunctionBeginUser;
  sigma = SIGMA*SIGMA;
  end = PetscLogReal(sigma);  
  h = (12.0+end)/(*maxnp-1);
  xi[0] = sigma;
  for (i=1;i<*maxnp;i++) xi[i] = -PetscPowReal(10,h*i)+sigma;
  PetscFunctionReturn(0);
}
