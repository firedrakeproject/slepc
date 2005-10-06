
static char help[] = "Estimates the 2-norm condition number of a matrix A, that is, the ratio of the largest  to the smallest singular values of A. "
  "The matrix is a Grcar matrix.\n\n"
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

   Note that working with A^T*A can lead to poor accuracy of the computed
   singular values when A is ill-conditioned (which is not the case here).
   Another alternative would be to compute the eigenvalues of

              |  0   A |
              | A^T  0 |

   but this significantly increases the cost of the solution process.

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
  int      ierr;

  ierr = MatShellGetContext(H,(void**)&A);CHKERRQ(ierr);
  ierr = MatGetVecs(A,PETSC_NULL,&w);CHKERRQ(ierr);
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
  PetscInt    N=30, n, Istart, Iend, i, col[5];
  int         ierr, nconv1, nconv2;
  PetscScalar kl, ks, sigma_1, sigma_n, value[] = { -1, 1, 1, 1, 1 };

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nEstimate the condition number of a Grcar matrix, n=%d\n\n",N);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Generate the matrix 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
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
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
#ifdef SLEPC_HAVE_ARPACK
  ierr = EPSSetType(eps, EPSARPACK);CHKERRQ(ierr);
#else
  ierr = EPSSetType(eps, EPSLAPACK);CHKERRQ(ierr);
#endif
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps,1,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps,PETSC_DEFAULT,1000);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     First request an eigenvalue from one end of the spectrum
  */
  ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
  ierr = EPSSolve(eps);CHKERRQ(ierr);
  /* 
     Get number of converged eigenpairs
  */
  ierr = EPSGetConverged(eps,&nconv1);CHKERRQ(ierr);
    /* 
       Get converged eigenpairs: largest eigenvalue is stored in kl. In this
       example, we are not interested in the eigenvectors
    */
  if (nconv1 > 0) {
    ierr = EPSGetEigenpair(eps,0,&kl,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }

  /*
     Request an eigenvalue from the other end of the spectrum
  */
  ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
  ierr = EPSSolve(eps);CHKERRQ(ierr);
  /* 
     Get number of converged eigenpairs
  */
  ierr = EPSGetConverged(eps,&nconv2);CHKERRQ(ierr);
  /* 
     Get converged eigenpairs: smallest eigenvalue is stored in ks. In this
     example, we are not interested in the eigenvectors
  */
  if (nconv2 > 0) {
    ierr = EPSGetEigenpair(eps,0,&ks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (nconv1 > 0 && nconv2 > 0) {
    sigma_1 = PetscSqrtScalar(kl);
    sigma_n = PetscSqrtScalar(ks);

    ierr = PetscPrintf(PETSC_COMM_WORLD," Computed singular values: sigma_1=%6f, sigma_n=%6f\n",sigma_1,sigma_n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Estimated condition number: sigma_1/sigma_n=%6f\n\n",sigma_1/sigma_n);CHKERRQ(ierr);
  } else {
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

