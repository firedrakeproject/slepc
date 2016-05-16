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

static char help[] = "Test the solution of a SVD without calling SVDSetFromOptions (based on ex8.c).\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n"
  "  -type <svd_type> = svd type to test.\n\n";

#include <slepcsvd.h>

/*
   This example computes the singular values of an nxn Grcar matrix,
   which is a nonsymmetric Toeplitz matrix:

              |  1  1  1  1               |
              | -1  1  1  1  1            |
              |    -1  1  1  1  1         |
              |       .  .  .  .  .       |
          A = |          .  .  .  .  .    |
              |            -1  1  1  1  1 |
              |               -1  1  1  1 |
              |                  -1  1  1 |
              |                     -1  1 |

 */

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A;               /* Grcar matrix */
  SVD            svd;             /* singular value solver context */
  PetscInt       N=30,Istart,Iend,i,col[5],nconv1,nconv2;
  PetscScalar    value[] = { -1, 1, 1, 1, 1 };
  PetscReal      sigma_1,sigma_n;
  char           svdtype[30] = "cross",epstype[30] = "";
  PetscBool      flg;
  EPS            eps;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-type",svdtype,30,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-epstype",epstype,30,&flg);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nEstimate the condition number of a Grcar matrix, n=%D",N);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSVD type: %s",svdtype);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nEPS type: %s",epstype);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) {
      ierr = MatSetValues(A,1,&i,4,col+1,value+1,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatSetValues(A,1,&i,PetscMin(5,N-i+1),col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the singular value solver and set the solution method
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value context
  */
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);

  /*
     Set operator
  */
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = SVDSetType(svd,svdtype);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscObjectTypeCompare((PetscObject)svd,SVDCROSS,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SVDCrossGetEPS(svd,&eps);CHKERRQ(ierr);
      ierr = EPSSetType(eps,epstype);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompare((PetscObject)svd,SVDCYCLIC,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SVDCyclicGetEPS(svd,&eps);CHKERRQ(ierr);
      ierr = EPSSetType(eps,epstype);CHKERRQ(ierr);
    }
  }
  ierr = SVDSetDimensions(svd,1,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = SVDSetTolerances(svd,1e-6,1000);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     First request an eigenvalue from one end of the spectrum
  */
  ierr = SVDSetWhichSingularTriplets(svd,SVD_LARGEST);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  /*
     Get number of converged singular values
  */
  ierr = SVDGetConverged(svd,&nconv1);CHKERRQ(ierr);
  /*
     Get converged singular values: largest singular value is stored in sigma_1.
     In this example, we are not interested in the singular vectors
  */
  if (nconv1 > 0) {
    ierr = SVDGetSingularTriplet(svd,0,&sigma_1,NULL,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Unable to compute large singular value!\n\n");CHKERRQ(ierr);
  }

  /*
     Request an eigenvalue from the other end of the spectrum
  */
  ierr = SVDSetWhichSingularTriplets(svd,SVD_SMALLEST);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  /*
     Get number of converged eigenpairs
  */
  ierr = SVDGetConverged(svd,&nconv2);CHKERRQ(ierr);
  /*
     Get converged singular values: smallest singular value is stored in sigma_n.
     As before, we are not interested in the singular vectors
  */
  if (nconv2 > 0) {
    ierr = SVDGetSingularTriplet(svd,0,&sigma_n,NULL,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Unable to compute small singular value!\n\n");CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (nconv1 > 0 && nconv2 > 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Computed singular values: sigma_1=%6f, sigma_n=%6f\n",(double)sigma_1,(double)sigma_n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Estimated condition number: sigma_1/sigma_n=%6f\n\n",(double)(sigma_1/sigma_n));CHKERRQ(ierr);
  }

  /*
     Free work space
  */
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

