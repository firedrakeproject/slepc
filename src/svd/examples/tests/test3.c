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

static char help[] = "Test SVD with user-provided initial vectors.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = row dimension.\n"
  "  -m <m>, where <m> = column dimension.\n\n";

#include <slepcsvd.h>

/*
   This example computes the singular values of a rectangular nxm Grcar matrix:

              |  1  1  1  1               |
              | -1  1  1  1  1            |
              |    -1  1  1  1  1         |
          A = |       .  .  .  .  .       |
              |          .  .  .  .  .    |
              |            -1  1  1  1  1 |
              |               -1  1  1  1 |

 */

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A;               /* Grcar matrix */
  SVD            svd;             /* singular value solver context */
  Vec            v0,w0;           /* initial vectors */
  PetscInt       N=35,M=30,Istart,Iend,i,col[5];
  PetscScalar    value[] = { -1, 1, 1, 1, 1 };
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&M,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSVD of a rectangular Grcar matrix, %Dx%D\n\n",N,M);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,M);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) {
      ierr = MatSetValues(A,1,&i,PetscMin(4,M-i+1),col+1,value+1,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatSetValues(A,1,&i,PetscMin(5,M-i+1),col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the SVD context and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /*
     Set the initial vectors. This is optional, if not done the initial
     vectors are set to random values
  */
  ierr = MatCreateVecs(A,&v0,&w0);CHKERRQ(ierr);
  ierr = VecSet(v0,1.0);CHKERRQ(ierr);
  ierr = VecSet(w0,1.0);CHKERRQ(ierr);
  ierr = SVDSetInitialSpace(svd,1,&v0);CHKERRQ(ierr);
  ierr = SVDSetInitialSpaceLeft(svd,1,&w0);CHKERRQ(ierr);

  /*
     Compute solution
  */
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  /*
     Free work space
  */
  ierr = VecDestroy(&v0);CHKERRQ(ierr);
  ierr = VecDestroy(&w0);CHKERRQ(ierr);
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

