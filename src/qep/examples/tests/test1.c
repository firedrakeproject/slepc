/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Test the solution of a QEP without calling QEPSetFromOptions (based on ex16.c).\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n"
  "  -type <qep_type> = qep type to test.\n"
  "  -epstype <eps_type> = eps type to test (for linear).\n\n";

#include <slepcqep.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            M,C,K;           /* problem matrices */
  QEP            qep;             /* quadratic eigenproblem solver context */
  QEPType        type;
  PetscInt       N,n=10,m,Istart,Iend,II,nev,maxit,i,j;
  PetscBool      flag,isgd2;
  char           qeptype[30] = "linear",epstype[30] = "";
  EPS            eps;
  ST             st;
  KSP            ksp;
  PC             pc;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if (!flag) m=n;
  N = n*m;
  ierr = PetscOptionsGetString(NULL,"-type",qeptype,30,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,"-epstype",epstype,30,&flag);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nQuadratic Eigenproblem, N=%D (%Dx%D grid)",N,n,m);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nQEP type: %s",qeptype);CHKERRQ(ierr);
  if (flag) {
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

  /* C is the zero matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* M is the identity matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(M);CHKERRQ(ierr);
  ierr = MatSetUp(M);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(M,1.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  ierr = QEPCreate(PETSC_COMM_WORLD,&qep);CHKERRQ(ierr);

  /*
     Set matrices and problem type
  */
  ierr = QEPSetOperators(qep,M,C,K);CHKERRQ(ierr);
  ierr = QEPSetProblemType(qep,QEP_GENERAL);CHKERRQ(ierr);
  ierr = QEPSetDimensions(qep,4,20,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = QEPSetTolerances(qep,PETSC_SMALL,PETSC_DEFAULT);CHKERRQ(ierr);

  /*
     Set solver type at runtime
  */
  ierr = QEPSetType(qep,qeptype);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscObjectTypeCompare((PetscObject)qep,QEPLINEAR,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = QEPLinearGetEPS(qep,&eps);CHKERRQ(ierr);
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
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = QEPSolve(qep);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = QEPGetType(qep,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = QEPGetDimensions(qep,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
  ierr = QEPGetTolerances(qep,NULL,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: maxit=%D\n",maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = QEPPrintSolution(qep,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = QEPDestroy(&qep);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
}

