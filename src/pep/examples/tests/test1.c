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

static char help[] = "Test the solution of a PEP without calling PEPSetFromOptions (based on ex16.c).\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n"
  "  -type <pep_type> = pep type to test.\n"
  "  -epstype <eps_type> = eps type to test (for linear).\n\n";

#include <slepcpep.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            M,C,K,A[3];      /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  PEPType        type;
  PetscInt       N,n=10,m,Istart,Iend,II,nev,maxit,i,j;
  PetscBool      flag,isgd2,epsgiven;
  char           peptype[30] = "linear",epstype[30] = "";
  EPS            eps;
  ST             st;
  KSP            ksp;
  PC             pc;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if (!flag) m=n;
  N = n*m;
  ierr = PetscOptionsGetString(NULL,NULL,"-type",peptype,30,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-epstype",epstype,30,&epsgiven);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nQuadratic Eigenproblem, N=%D (%Dx%D grid)",N,n,m);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nPEP type: %s",peptype);CHKERRQ(ierr);
  if (epsgiven) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nEPS type: %s",epstype);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is the 2-D Laplacian */
  ierr = MatCreate(PETSC_COMM_WORLD,&K);CHKERRQ(ierr);
  ierr = MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(K);CHKERRQ(ierr);
  ierr = MatSetUp(K);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(K,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) { ierr = MatSetValue(K,II,II-n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(K,II,II+n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j>0) { ierr = MatSetValue(K,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<n-1) { ierr = MatSetValue(K,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(K,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* C is the 1-D Laplacian on horizontal lines */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (j>0) { ierr = MatSetValue(C,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<n-1) { ierr = MatSetValue(C,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(C,II,II,2.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* M is a diagonal matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(M);CHKERRQ(ierr);
  ierr = MatSetUp(M);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(M,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    ierr = MatSetValue(M,II,II,(PetscReal)(II+1),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PEPCreate(PETSC_COMM_WORLD,&pep);CHKERRQ(ierr);
  A[0] = K; A[1] = C; A[2] = M;
  ierr = PEPSetOperators(pep,3,A);CHKERRQ(ierr);
  ierr = PEPSetProblemType(pep,PEP_GENERAL);CHKERRQ(ierr);
  ierr = PEPSetDimensions(pep,4,20,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PEPSetTolerances(pep,PETSC_SMALL,PETSC_DEFAULT);CHKERRQ(ierr);

  /*
     Set solver type at runtime
  */
  ierr = PEPSetType(pep,peptype);CHKERRQ(ierr);
  if (epsgiven) {
    ierr = PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = PEPLinearGetEPS(pep,&eps);CHKERRQ(ierr);
      ierr = PetscStrcmp(epstype,"gd2",&isgd2);CHKERRQ(ierr);
      if (isgd2) {
        ierr = EPSSetType(eps,EPSGD);CHKERRQ(ierr);
        ierr = EPSGDSetDoubleExpansion(eps,PETSC_TRUE);CHKERRQ(ierr);
      } else {
        ierr = EPSSetType(eps,epstype);CHKERRQ(ierr);
      }
      ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
      ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)eps,EPSGD,&flag);CHKERRQ(ierr);
    }
    ierr = PEPLinearSetExplicitMatrix(pep,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep,PEPQARNOLDI,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PEPGetST(pep,&st);CHKERRQ(ierr);
    ierr = STSetTransform(st,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PEPSolve(pep);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = PEPGetType(pep,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = PEPGetDimensions(pep,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
  ierr = PEPGetTolerances(pep,NULL,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: maxit=%D\n",maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL);CHKERRQ(ierr);
  ierr = PEPDestroy(&pep);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

