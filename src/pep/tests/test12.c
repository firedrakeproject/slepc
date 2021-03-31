/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates region filtering in PEP (based on spring).\n"
  "The command line options are:\n"
  "  -n <n> ... number of grid subdivisions.\n"
  "  -mu <value> ... mass (default 1).\n"
  "  -tau <value> ... damping constant of the dampers (default 10).\n"
  "  -kappa <value> ... damping constant of the springs (default 5).\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            M,C,K,A[3];      /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  RG             rg;
  PetscInt       n=30,Istart,Iend,i;
  PetscReal      mu=1.0,tau=10.0,kappa=5.0;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nDamped mass-spring system, n=%D mu=%g tau=%g kappa=%g\n\n",n,(double)mu,(double)tau,(double)kappa);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is a tridiagonal */
  ierr = MatCreate(PETSC_COMM_WORLD,&K);CHKERRQ(ierr);
  ierr = MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(K);CHKERRQ(ierr);
  ierr = MatSetUp(K);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(K,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) {
      ierr = MatSetValue(K,i,i-1,-kappa,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatSetValue(K,i,i,kappa*3.0,INSERT_VALUES);CHKERRQ(ierr);
    if (i<n-1) {
      ierr = MatSetValue(K,i,i+1,-kappa,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* C is a tridiagonal */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) {
      ierr = MatSetValue(C,i,i-1,-tau,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatSetValue(C,i,i,tau*3.0,INSERT_VALUES);CHKERRQ(ierr);
    if (i<n-1) {
      ierr = MatSetValue(C,i,i+1,-tau,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* M is a diagonal matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(M);CHKERRQ(ierr);
  ierr = MatSetUp(M);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(M,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(M,i,i,mu,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Create a region for filtering
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = RGCreate(PETSC_COMM_WORLD,&rg);CHKERRQ(ierr);
  ierr = RGSetType(rg,RGINTERVAL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = RGIntervalSetEndpoints(rg,-47.0,-35.0,-0.001,0.001);CHKERRQ(ierr);
#else
  ierr = RGIntervalSetEndpoints(rg,-47.0,-35.0,-0.0,0.0);CHKERRQ(ierr);
#endif

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PEPCreate(PETSC_COMM_WORLD,&pep);CHKERRQ(ierr);
  ierr = PEPSetRG(pep,rg);CHKERRQ(ierr);
  A[0] = K; A[1] = C; A[2] = M;
  ierr = PEPSetOperators(pep,3,A);CHKERRQ(ierr);
  ierr = PEPSetTolerances(pep,PETSC_SMALL,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PEPSetFromOptions(pep);CHKERRQ(ierr);
  ierr = PEPSolve(pep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL);CHKERRQ(ierr);
  ierr = PEPDestroy(&pep);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = RGDestroy(&rg);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      args: -pep_nev 8 -pep_type {{toar linear qarnoldi}}
      suffix: 1
      requires: !single
      output_file: output/test12_1.out

TEST*/
