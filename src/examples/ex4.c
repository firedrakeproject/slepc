
static char help[] = "Solves an eigensystem loaded from a file.\n\n"
  "This example works for both real and complex numbers.\n\n"
  "The command line options are:\n\n"
  "  -file <filename>, where <filename> = matrix file in PETSc binary form.\n\n";

#include "slepceps.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  Vec         *x;              /* basis vectors */
  Mat         A;               /* operator matrix */
  EPS         eps;             /* eigenproblem solver context */
  EPSType     type;
  PetscReal   *error, tol, re, im;
  PetscScalar *kr, *ki;
  int         nev, ierr, maxit, i, its, nconv;
  char        filename[256];
  PetscViewer viewer;
  PetscTruth  flg;


  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Load the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nEigenproblem stored in file.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-file",filename,256,&flg);CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(1,"Must indicate a file name with the -file option.");
  }

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n");CHKERRQ(ierr);
#endif
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,PETSC_FILE_RDONLY,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(viewer,MATMPIAIJ,&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

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

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps,&its);CHKERRQ(ierr); 
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
     Get number of converged eigenpairs
  */
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate eigenpairs: %d\n\n",nconv);CHKERRQ(ierr);

  if (nconv>0) {
    /* 
       Get converged eigenpairs: i-th eigenvalue is stored in kr[i] (real part) and
       ki[i] (imaginary part), and the corresponding eigenvector is stored in x[i]
    */
    ierr = EPSGetSolution(eps,&kr,&ki,&x);CHKERRQ(ierr);

    /*
       Compute the relative error associated to each eigenpair
    */
    ierr = PetscMalloc(nconv*sizeof(PetscReal),&error);CHKERRQ(ierr);
    ierr = EPSComputeError(eps,error);CHKERRQ(ierr);

    /*
       Display eigenvalues and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k              ||Ax-kx||/|k|\n"
         "  --------------------- -----------------\n" );CHKERRQ(ierr);
    for( i=0; i<nconv; i++ ) {
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr[i]);
      im = PetscImaginaryPart(kr[i]);
#else
      re = kr[i];
      im = ki[i];
#endif
      if( im>0.0 ) ierr = PetscPrintf(PETSC_COMM_WORLD," % 6f + %6f i",re,im);
      else if( im<0.0 ) ierr = PetscPrintf(PETSC_COMM_WORLD," % 6f - %6f i",re,-im);
      else ierr = PetscPrintf(PETSC_COMM_WORLD,"       % 6f      ",re);
      CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD," % 12f\n",error[i]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n" );CHKERRQ(ierr);
    ierr = PetscFree(error);CHKERRQ(ierr);
  }
  
  /* 
     Free work space
  */
  ierr = EPSDestroy(eps);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

