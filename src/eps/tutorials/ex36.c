/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Use the matrix exponential to compute rightmost eigenvalues.\n\n"
  "Same problem as ex9.c but with explicitly created matrix. The command line options are:\n"
  "  -n <n>, where <n> = block dimension of the 2x2 block matrix.\n"
  "  -L <L>, where <L> = bifurcation parameter.\n"
  "  -alpha <alpha>, -beta <beta>, -delta1 <delta1>,  -delta2 <delta2>,\n"
  "       where <alpha> <beta> <delta1> <delta2> = model parameters.\n\n";

#include <slepceps.h>
#include <slepcmfn.h>

/*
   This example computes the eigenvalues with largest real part of the
   following matrix

        A = [ tau1*T+(beta-1)*I     alpha^2*I
                  -beta*I        tau2*T-alpha^2*I ],

   where

        T = tridiag{1,-2,1}
        h = 1/(n+1)
        tau1 = delta1/(h*L)^2
        tau2 = delta2/(h*L)^2

   but it builds A explicitly, as opposed to ex9.c
*/

/* Routines for shell spectral transformation */
PetscErrorCode STApply_Exp(ST,Vec,Vec);
PetscErrorCode STBackTransform_Exp(ST,PetscInt,PetscScalar*,PetscScalar*);

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  EPS            eps;             /* eigenproblem solver context */
  ST             st;              /* spectral transformation context */
  MFN            mfn;             /* matrix function solver object to compute exp(A)*v */
  FN             f;
  EPSType        type;
  PetscScalar    alpha,beta,tau1,tau2,delta1,delta2,L,h;
  PetscInt       n=30,i,Istart,Iend,nev;
  PetscBool      isShell,terse;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
#if defined(PETSC_HAVE_COMPLEX)
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nBrusselator wave model with matrix exponential, n=%" PetscInt_FMT "\n\n",n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  alpha  = 2.0;
  beta   = 5.45;
  delta1 = 0.008;
  delta2 = 0.004;
  L      = 0.51302;

  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-L",&L,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-alpha",&alpha,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-beta",&beta,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-delta1",&delta1,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-delta2",&delta2,NULL));

  h = 1.0 / (PetscReal)(n+1);
  tau1 = delta1 / ((h*L)*(h*L));
  tau2 = delta2 / ((h*L)*(h*L));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2*n,2*n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i<n) {  /* upper blocks */
      if (i>0) CHKERRQ(MatSetValue(A,i,i-1,tau1,INSERT_VALUES));
      if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,tau1,INSERT_VALUES));
      CHKERRQ(MatSetValue(A,i,i,-2.0*tau1+beta-1.0,INSERT_VALUES));
      CHKERRQ(MatSetValue(A,i,i+n,alpha*alpha,INSERT_VALUES));
    } else {  /* lower blocks */
      if (i>n) CHKERRQ(MatSetValue(A,i,i-1,tau2,INSERT_VALUES));
      if (i<2*n-1) CHKERRQ(MatSetValue(A,i,i+1,tau2,INSERT_VALUES));
      CHKERRQ(MatSetValue(A,i,i,-2.0*tau2-alpha*alpha,INSERT_VALUES));
      CHKERRQ(MatSetValue(A,i,i-n,-beta,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL));
  CHKERRQ(EPSGetST(eps,&st));
  CHKERRQ(STSetType(st,STSHELL));
  CHKERRQ(EPSSetFromOptions(eps));

  /*
     Initialize shell spectral transformation
  */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)st,STSHELL,&isShell));
  if (isShell) {

    /* Create the MFN object to be used by the spectral transform */
    CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfn));
    CHKERRQ(MFNSetOperator(mfn,A));
    CHKERRQ(MFNGetFN(mfn,&f));
    CHKERRQ(FNSetType(f,FNEXP));
    CHKERRQ(FNSetScale(f,0.03,1.0));  /* this can be set with -fn_scale */
    CHKERRQ(MFNSetFromOptions(mfn));

    /* Set callback functions */
    CHKERRQ(STShellSetApply(st,STApply_Exp));
    CHKERRQ(STShellSetBackTransform(st,STBackTransform_Exp));
    CHKERRQ(STShellSetContext(st,mfn));
    CHKERRQ(PetscObjectSetName((PetscObject)st,"STEXP"));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  if (isShell) CHKERRQ(MFNDestroy(&mfn));
#else
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example requires C99 complex numbers");
#endif
  CHKERRQ(SlepcFinalize());
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   STBackTransform_Exp - Undoes the exp(A) transformation by taking logarithms.

   Input Parameters:
+  st - spectral transformation context
-  n  - number of eigenvalues to transform

   Input/Output Parameters:
+  eigr - pointer to real part of eigenvalues
-  eigi - pointer to imaginary part of eigenvalues
*/
PetscErrorCode STBackTransform_Exp(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt       j;
  MFN            mfn;
  FN             fn;
  PetscScalar    tau,eta;
#if !defined(PETSC_USE_COMPLEX)
  PetscComplex   theta,lambda;
#endif

  PetscFunctionBeginUser;
  CHKERRQ(STShellGetContext(st,&mfn));
  CHKERRQ(MFNGetFN(mfn,&fn));
  CHKERRQ(FNGetScale(fn,&tau,&eta));
  for (j=0;j<n;j++) {
#if defined(PETSC_USE_COMPLEX)
    eigr[j] = PetscLogComplex(eigr[j]/eta)/tau;
#else
    theta   = PetscCMPLX(eigr[j],eigi[j])/eta;
    lambda  = PetscLogComplex(theta)/tau;
    eigr[j] = PetscRealPartComplex(lambda);
    eigi[j] = PetscImaginaryPartComplex(lambda);
#endif
  }
  PetscFunctionReturn(0);
#else
  return 0;
#endif
}

/*
   STApply_Exp - Applies the operator exp(tau*A) to a given vector using an MFN object.

   Input Parameters:
+  st - spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector
*/
PetscErrorCode STApply_Exp(ST st,Vec x,Vec y)
{
  MFN            mfn;

  PetscFunctionBeginUser;
  CHKERRQ(STShellGetContext(st,&mfn));
  CHKERRQ(MFNSolve(mfn,x,y));
  PetscFunctionReturn(0);
}

/*TEST

   testset:
      args: -eps_nev 4 -mfn_ncv 16 -terse
      requires: c99_complex !single
      filter: sed -e "s/-2/+2/g"
      output_file: output/ex36_1.out
      test:
         suffix: 1
         requires: !__float128
      test:
         suffix: 1_quad
         args: -eps_tol 1e-14
         requires: __float128

   test:
      suffix: 2
      args: -n 56 -eps_nev 2 -st_type sinvert -eps_target -390 -eps_target_magnitude -eps_type power
      args: -eps_power_shift_type {{constant rayleigh}} -eps_two_sided {{0 1}} -eps_tol 1e-14 -terse
      requires: c99_complex !single
      filter: sed -e "s/[+-]0\.0*i//g"

   test:
      suffix: 3
      args: -n 100 -st_type sinvert -eps_type ciss -rg_type ellipse -rg_ellipse_center 0 -rg_ellipse_radius 6 -eps_all -eps_tol 1e-6 -terse
      requires: c99_complex !single
      filter: sed -e "s/-3.37036-3.55528i, -3.37036+3.55528i/-3.37036+3.55528i, -3.37036-3.55528i/" -e "s/-1.79853-3.03216i, -1.79853+3.03216i/-1.79853+3.03216i, -1.79853-3.03216i/" -e "s/-0.67471-2.52856i, -0.67471+2.52856i/-0.67471+2.52856i, -0.67471-2.52856i/" -e "s/0.00002-2.13950i, 0.00002+2.13950i/0.00002+2.13950i, 0.00002-2.13950i/"

TEST*/
