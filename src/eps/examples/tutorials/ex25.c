/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Spectrum slicing on generalized symmetric eigenproblem.\n\n"
  "The problem is similar to ex13.c.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n";

#include <slepceps.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,B;         /* matrices */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  KSP            ksp;
  PC             pc;
  EPSType        type;
  PetscInt       N,n=10,m,Istart,Iend,II,nev,i,j;
  PetscReal      inta,intb;
  PetscBool      flag;
  PetscErrorCode ierr;
#if !defined(PETSC_HAVE_MUMPS)
  PetscMPIInt    size;
#endif

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if (!flag) m=n;
  N = n*m;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSpectrum slicing on GHEP, N=%D (%Dx%D grid)\n\n",N,n,m);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) { ierr = MatSetValue(A,II,II-n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<m-1) { ierr = MatSetValue(A,II,II+n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j>0) { ierr = MatSetValue(A,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<n-1) { ierr = MatSetValue(A,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(A,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(B,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /*
     Set operators and set problem type
  */
  ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);

  /*
     Set interval for spectrum slicing
  */
  inta = 0.1;
  intb = 0.2;
  ierr = EPSSetInterval(eps,inta,intb);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_ALL);CHKERRQ(ierr);

  /*
     Set shift-and-invert with Cholesky; select MUMPS if available
  */

  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
  
  ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
    
#if defined(PETSC_HAVE_MUMPS)
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Spectrum slicing with MUMPS is not available for complex scalars");
#endif
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr); 
  /*
     must use runtime option '-mat_mumps_icntl_13 1' (turn off scaLAPACK for
     matrix inertia), currently there is no better way of setting this in program
  */
  ierr = PetscOptionsInsertString("-mat_mumps_icntl_13 1");CHKERRQ(ierr); 
#else
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size>1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Configure with MUMPS if you want to run this example in parallel");
#endif

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSSolve(eps);CHKERRQ(ierr);

  /*
     Show eigenvalues in interval and print solution
  */
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = EPSGetInterval(eps,&inta,&intb);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," %D eigenvalues found in [%g, %g]\n",nev,(double)inta,(double)intb);CHKERRQ(ierr);
  ierr = EPSPrintSolution(eps,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
}

