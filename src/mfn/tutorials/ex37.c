/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-t",&t,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-peclet",&peclet,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-steps",&steps,NULL));
  m = n;
  N = m*n;
  /* interval [0,1], homogeneous Dirichlet boundary conditions */
  h = 1.0/(n+1.0);
  h2 = h*h;
  c = 2.0*epsilon*peclet/h;
  upper = epsilon/h2+c/(2.0*h);
  diag = 2.0*(-2.0*epsilon/h2);
  lower = epsilon/h2-c/(2.0*h);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nAdvection diffusion via y=exp(%g*A), n=%" PetscInt_FMT ", steps=%" PetscInt_FMT ", Peclet=%g\n\n",(double)PetscRealPart(t),n,steps,(double)peclet));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Generate matrix A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(A,II,II-n,lower,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A,II,II+n,upper,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A,II,II-1,lower,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A,II,II+1,upper,INSERT_VALUES));
    PetscCall(MatSetValue(A,II,II,diag,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A,NULL,&v));

  /*
     Set initial condition v = 256*i^2*(1-i)^2*j^2*(1-j)^2
  */
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    i1h = (i+1)*h; j1h = (j+1)*h;
    value = 256.0*i1h*i1h*(1.0-i1h)*(1.0-i1h)*(j1h*j1h)*(1.0-j1h)*(1.0-j1h);
    PetscCall(VecSetValue(v,i+j*n,value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MFNCreate(PETSC_COMM_WORLD,&mfn));
  PetscCall(MFNSetOperator(mfn,A));
  PetscCall(MFNGetFN(mfn,&f));
  PetscCall(FNSetType(f,FNEXP));
  PetscCall(FNSetScale(f,t,sone));
  PetscCall(MFNSetFromOptions(mfn));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the problem, y=exp(t*A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  for (i=0;i<steps;i++) {
    PetscCall(MFNSolve(mfn,v,v));
    PetscCall(MFNGetConvergedReason(mfn,&reason));
    PetscCheck(reason>=0,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Solver did not converge");
    PetscCall(MFNGetIterationNumber(mfn,&its));
    totits += its;
  }

  /*
     Optional: Get some information from the solver and display it
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",totits));
  PetscCall(MFNGetDimensions(mfn,&ncv));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Subspace dimension: %" PetscInt_FMT "\n",ncv));
  PetscCall(MFNGetTolerances(mfn,&tol,&maxit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));
  PetscCall(VecNorm(v,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Computed vector at time t=%.4g has norm %g\n",(double)PetscRealPart(t)*steps,(double)norm));

  /*
     Free work space
  */
  PetscCall(MFNDestroy(&mfn));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&v));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -mfn_tol 1e-6

TEST*/
