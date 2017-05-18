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
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
#if (defined(__cplusplus) && defined(PETSC_HAVE_CXX_COMPLEX)) || (!defined(__cplusplus) && defined(PETSC_HAVE_C99_COMPLEX))
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nBrusselator wave model with matrix exponential, n=%D\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  alpha  = 2.0;
  beta   = 5.45;
  delta1 = 0.008;
  delta2 = 0.004;
  L      = 0.51302;

  ierr = PetscOptionsGetScalar(NULL,NULL,"-L",&L,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-alpha",&alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-beta",&beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-delta1",&delta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-delta2",&delta2,NULL);CHKERRQ(ierr);

  h = 1.0 / (PetscReal)(n+1);
  tau1 = delta1 / ((h*L)*(h*L));
  tau2 = delta2 / ((h*L)*(h*L));

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2*n,2*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i<n) {  /* upper blocks */
      if (i>0) { ierr = MatSetValue(A,i,i-1,tau1,INSERT_VALUES);CHKERRQ(ierr); }
      if (i<n-1) { ierr = MatSetValue(A,i,i+1,tau1,INSERT_VALUES);CHKERRQ(ierr); }
      ierr = MatSetValue(A,i,i,-2.0*tau1+beta-1.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(A,i,i+n,alpha*alpha,INSERT_VALUES);CHKERRQ(ierr);
    } else {  /* lower blocks */
      if (i>n) { ierr = MatSetValue(A,i,i-1,tau2,INSERT_VALUES);CHKERRQ(ierr); }
      if (i<2*n-1) { ierr = MatSetValue(A,i,i+1,tau2,INSERT_VALUES);CHKERRQ(ierr); }
      ierr = MatSetValue(A,i,i,-2.0*tau2-alpha*alpha,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(A,i,i-n,-beta,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STSetType(st,STSHELL);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /*
     Initialize shell spectral transformation
  */
  ierr = PetscObjectTypeCompare((PetscObject)st,STSHELL,&isShell);CHKERRQ(ierr);
  if (isShell) {

    /* Create the MFN object to be used by the spectral transform */
    ierr = MFNCreate(PETSC_COMM_WORLD,&mfn);CHKERRQ(ierr);
    ierr = MFNSetOperator(mfn,A);CHKERRQ(ierr);
    ierr = MFNGetFN(mfn,&f);CHKERRQ(ierr);
    ierr = FNSetType(f,FNEXP);CHKERRQ(ierr);
    ierr = FNSetScale(f,0.03,1.0);CHKERRQ(ierr);  /* this can be set with -fn_scale */
    ierr = MFNSetFromOptions(mfn);CHKERRQ(ierr);

    /* Set callback functions */
    ierr = STShellSetApply(st,STApply_Exp);CHKERRQ(ierr);
    ierr = STShellSetBackTransform(st,STBackTransform_Exp);CHKERRQ(ierr);
    ierr = STShellSetContext(st,mfn);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)st,"STEXP");CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MFNDestroy(&mfn);CHKERRQ(ierr);
#else
  SETERRQ(PETSC_COMM_SELF,1,"This examples requires C99 complex numbers");
#endif
  ierr = SlepcFinalize();
  return ierr;
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
#if (defined(__cplusplus) && defined(PETSC_HAVE_CXX_COMPLEX)) || (!defined(__cplusplus) && defined(PETSC_HAVE_C99_COMPLEX))
  PetscErrorCode ierr;
  PetscInt       j;
  MFN            mfn;
  FN             fn;
  PetscScalar    tau,eta;
#if !defined(PETSC_USE_COMPLEX)
  PetscComplex theta,lambda;
#endif

  PetscFunctionBeginUser;
  ierr = STShellGetContext(st,(void**)&mfn);CHKERRQ(ierr);
  ierr = MFNGetFN(mfn,&fn);CHKERRQ(ierr);
  ierr = FNGetScale(fn,&tau,&eta);CHKERRQ(ierr);
  for (j=0;j<n;j++) {
#if defined(PETSC_USE_COMPLEX)
    eigr[j] = PetscLogComplex(eigr[j]/eta)/tau;
#else
    theta   = (eigr[j]+eigi[j]*PETSC_i)/eta;
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = STShellGetContext(st,(void**)&mfn);CHKERRQ(ierr);
  ierr = MFNSolve(mfn,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

