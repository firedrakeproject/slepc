
static char help[] = "This example estimates the 2-norm condition number of a"
  "matrix A, that is, the ratio of the largest singular value of A to the "
  "smallest. The matrix is a Grcar matrix.\n\n"
  "The command line options are:\n\n"
  "  -n <n>, where <n> = matrix dimension.\n\n";

#include "slepceps.h"

/*
   This example computes the singular values of A by computing the eigenvalues
   of A^T*A, where A^T denotes the transpose of A. 

   An nxn Grcar matrix is a nonsymmetric Toeplitz matrix:

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


/* 
   Matrix multiply routine
*/
#undef __FUNCT__
#define __FUNCT__ "MatSVD_Mult"
int MatSVD_Mult(Mat H,Vec x,Vec y)
{
  Mat      A;
  Vec      w;
  int      n, m, N, M, ierr;
  MPI_Comm comm;

  ierr = MatShellGetContext(H,(void**)&A);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&n,&m);CHKERRQ(ierr);
  ierr = MatGetSize(A,&N,&M);CHKERRQ(ierr);
  ierr = VecCreate(comm,&w);CHKERRQ(ierr);
  ierr = VecSetSizes(w,n,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(w);CHKERRQ(ierr);
  ierr = MatMult(A,x,w);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,w,y);CHKERRQ(ierr);
  ierr = VecDestroy(w);CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  Mat         A;               /* Grcar matrix */
  Mat         H;               /* eigenvalue problem matrix, H=A^T*A */
  EPS         eps;             /* eigenproblem solver context */
  int         N=30, n, ierr, i, its, nconv, col[5], Istart, Iend;
  PetscScalar *kr, sigma_1, sigma_n, value[] = { -1, 1, 1, 1, 1 };

  SlepcInitialize(&argc,&argv,(char*)0,help);

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(1,"This example does not work with complex numbers!");
#endif
#if !defined(SLEPC_HAVE_ARPACK)
  SETERRQ(1,"This example requires that ARPACK is installed!");
#endif

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nEstimate de condition number of a Grcar matrix, n=%d\n\n",N);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Generate the matrix 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for( i=Istart; i<Iend; i++ ) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) {
      ierr = MatSetValues(A,1,&i,4,col+1,value+1,INSERT_VALUES);CHKERRQ(ierr);
    }
    else {
      ierr = MatSetValues(A,1,&i,PetscMin(5,N-i+1),col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* 
     Now create a symmetric shell matrix H=A^T*A
  */
  ierr = MatGetLocalSize(A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD,n,n,PETSC_DETERMINE,PETSC_DETERMINE,(void*)A,&H);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H,MATOP_MULT,(void(*)())MatSVD_Mult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H,MATOP_MULT_TRANSPOSE,(void(*)())MatSVD_Mult);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
             Create the eigensolver and set the solution method
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /* 
     Set operators. In this case, it is a standard symmetric eigenvalue problem
  */
  ierr = EPSSetOperators(eps,H,PETSC_NULL);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /*
     Set the solution method. Two eigenvalues are requested, one from each end
     of the spectrum
  */
  ierr = EPSSetType(eps,EPSARPACK);CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps,2,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_BOTH_ENDS);CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps,PETSC_DEFAULT,1000);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(eps, &its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %d\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Get number of converged eigenpairs
  */
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);

  if (nconv==2) {
    /* 
       Get converged eigenpairs: i-th eigenvalue is stored in kr[i]. In this
       example, we are not interested in the eigenvectors
    */
    ierr = EPSGetSolution(eps,&kr,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    /* 
       The singular values of A are the square roots of the eigenvalues of H
    */
    sigma_1 = PetscSqrtScalar(PetscMax(kr[0],kr[1]));
    sigma_n = PetscSqrtScalar(PetscMin(kr[0],kr[1]));

    ierr = PetscPrintf(PETSC_COMM_WORLD," Computed singular values: sigma_1=%6f, sigma_n=%6f\n",sigma_1,sigma_n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Estimated condition number: sigma_1/sigma_n=%6f\n\n",sigma_1/sigma_n);CHKERRQ(ierr);

  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Process did not converge!\n\n");CHKERRQ(ierr);
  }
  
  /* 
     Free work space
  */
  ierr = EPSDestroy(eps);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(H);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

