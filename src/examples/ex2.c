
static char help[] = "Standard symmetric eigenproblem corresponding to the Laplacian operator in 2 dimensions.\n\n"
  "The command line options are:\n\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include "slepceps.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  Mat         A;               /* operator matrix */
  EPS         eps;             /* eigenproblem solver context */
  EPSType     type;
  PetscReal   error, tol, re, im;
  PetscScalar kr, ki;
  int         N, n=10, m, nev, ierr, maxit, i, j, I, J, its, nconv, Istart, Iend;
  PetscScalar v;
  PetscTruth  flag;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if( flag==PETSC_FALSE ) m=n;
  N = n*m;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n2-D Laplacian Eigenproblem, N=%d (%dx%d grid)\n\n",N,n,m);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I-i*n;  
    if(i>0) { J=I-n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
    if(i<m-1) { J=I+n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
    if(j>0) { J=I-1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
    if(j<n-1) { J=I+1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
    v=4.0; MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /* 
     Set operators. In this case, it is a standard eigenvalue problem
  */
  ierr = EPSSetOperators(eps,A,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(eps, &its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %d\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %d\n",nev);CHKERRQ(ierr);
  ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%d\n",tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Get number of converged approximate eigenpairs
  */
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate eigenpairs: %d\n\n",nconv);
         CHKERRQ(ierr);

  if (nconv>0) {
    /*
       Display eigenvalues and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k          ||Ax-kx||/||kx||\n"
         "   ----------------- ------------------\n" );CHKERRQ(ierr);

    for( i=0; i<nconv; i++ ) {
      /* 
        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
        ki (imaginary part)
      */
      ierr = EPSGetEigenpair(eps,i,&kr,&ki,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      /*
         Compute the relative error associated to each eigenpair
      */
      ierr = EPSComputeRelativeError(eps,i,&error);CHKERRQ(ierr);

#ifdef PETSC_USE_COMPLEX
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
#else
      re = kr;
      im = ki;
#endif 
      if (im!=0.0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9f j %12f\n",re,im,error);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12f\n",re,error);CHKERRQ(ierr); 
      }
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n" );CHKERRQ(ierr);
  }
  
  /* 
     Free work space
  */
  ierr = EPSDestroy(eps);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

