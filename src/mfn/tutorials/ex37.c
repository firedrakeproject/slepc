/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Computes exp(t*A)*v for an advection diffusion operator with Peclet number.\n\n"
  "The command line options are:\n"
  "  -n <idim>, where <idim> = number of subdivisions of the mesh in each spatial direction.\n"
  "  -t <sval>, where <sval> = scalar value that multiplies the argument.\n"
  "  -peclet <sval>, where <sval> = Peclet value.\n"
  "  -steps <ival>, where <ival> = number of time steps.\n\n";

#include <slepcmfn.h>

int main(int argc,char **argv)
{
  Mat                A;           /* problem matrix */
  MFN                mfn;
  FN                 f;
  PetscInt           i,j,Istart,Iend,II,m,n=10,N,steps=5,its,totits=0,ncv,maxit;
  PetscReal          tol,norm,h,h2,peclet=0.5,epsilon=1.0,c,i1h,j1h;
  PetscScalar        t=1e-4,sone=1.0,value,upper,diag,lower;
  Vec                v;
  MFNConvergedReason reason;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-t",&t,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-peclet",&peclet,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-steps",&steps,NULL));
  m = n;
  N = m*n;
  /* interval [0,1], homogeneous Dirichlet boundary conditions */
  h = 1.0/(n+1.0);
  h2 = h*h;
  c = 2.0*epsilon*peclet/h;
  upper = epsilon/h2+c/(2.0*h);
  diag = 2.0*(-2.0*epsilon/h2);
  lower = epsilon/h2-c/(2.0*h);

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nAdvection diffusion via y=exp(%g*A), n=%" PetscInt_FMT ", steps=%" PetscInt_FMT ", Peclet=%g\n\n",(double)PetscRealPart(t),n,steps,(double)peclet));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Generate matrix A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,lower,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,upper,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,lower,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,upper,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,diag,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,NULL,&v));

  /*
     Set initial condition v = 256*i^2*(1-i)^2*j^2*(1-j)^2
  */
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    i1h = (i+1)*h; j1h = (j+1)*h;
    value = 256.0*i1h*i1h*(1.0-i1h)*(1.0-i1h)*(j1h*j1h)*(1.0-j1h)*(1.0-j1h);
    CHKERRQ(VecSetValue(v,i+j*n,value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfn));
  CHKERRQ(MFNSetOperator(mfn,A));
  CHKERRQ(MFNGetFN(mfn,&f));
  CHKERRQ(FNSetType(f,FNEXP));
  CHKERRQ(FNSetScale(f,t,sone));
  CHKERRQ(MFNSetFromOptions(mfn));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the problem, y=exp(t*A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  for (i=0;i<steps;i++) {
    CHKERRQ(MFNSolve(mfn,v,v));
    CHKERRQ(MFNGetConvergedReason(mfn,&reason));
    PetscCheck(reason>=0,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Solver did not converge");
    CHKERRQ(MFNGetIterationNumber(mfn,&its));
    totits += its;
  }

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",totits));
  CHKERRQ(MFNGetDimensions(mfn,&ncv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Subspace dimension: %" PetscInt_FMT "\n",ncv));
  CHKERRQ(MFNGetTolerances(mfn,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));
  CHKERRQ(VecNorm(v,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computed vector at time t=%.4g has norm %g\n",(double)PetscRealPart(t)*steps,(double)norm));

  /*
     Free work space
  */
  CHKERRQ(MFNDestroy(&mfn));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -mfn_tol 1e-6

TEST*/
