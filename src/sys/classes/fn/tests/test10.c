/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test Phi functions.\n\n";

#include <slepcfn.h>

/*
   Evaluates phi_k function on a scalar and on a matrix
 */
PetscErrorCode TestPhiFunction(FN fn,PetscScalar x,Mat A,PetscBool verbose)
{
  PetscScalar    y,yp;
  char           strx[50],str[50];
  Vec            v,f;
  PetscReal      nrm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  CHKERRQ(FNView(fn,NULL));
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(fn,x,&y));
  CHKERRQ(FNEvaluateDerivative(fn,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nf(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"f'(%s)=%s\n",strx,str));
  /* compute phi_k(A)*e_1 */
  CHKERRQ(MatCreateVecs(A,&v,&f));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(FNEvaluateFunctionMatVec(fn,A,f));  /* reference result by diagonalization */
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunctionMatVec(fn,A,v));
  CHKERRQ(VecAXPY(v,-1.0,f));
  CHKERRQ(VecNorm(v,NORM_2,&nrm));
  if (nrm>100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of f(A)*e_1-ref is %g\n",(double)nrm));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"f(A)*e_1 =\n"));
    CHKERRQ(VecView(v,NULL));
  }
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&f));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  FN             phi0,phi1,phik,phicopy;
  Mat            A;
  PetscInt       i,j,n=8,k;
  PetscScalar    tau,eta,*As;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test Phi functions, n=%" PetscInt_FMT ".\n",n));

  /* Create matrix, fill it with 1-D Laplacian */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));
  CHKERRQ(MatDenseGetArray(A,&As));
  for (i=0;i<n;i++) As[i+i*n]=2.0;
  j=1;
  for (i=0;i<n-j;i++) { As[i+(i+j)*n]=-1.0; As[(i+j)+i*n]=-1.0; }
  CHKERRQ(MatDenseRestoreArray(A,&As));

  /* phi_0(x) = exp(x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&phi0));
  CHKERRQ(FNSetType(phi0,FNPHI));
  CHKERRQ(FNPhiSetIndex(phi0,0));
  CHKERRQ(TestPhiFunction(phi0,2.2,A,verbose));

  /* phi_1(x) = (exp(x)-1)/x with scaling factors eta*phi_1(tau*x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&phi1));
  CHKERRQ(FNSetType(phi1,FNPHI));  /* default index should be 1 */
  tau = 0.2;
  eta = 1.3;
  CHKERRQ(FNSetScale(phi1,tau,eta));
  CHKERRQ(TestPhiFunction(phi1,2.2,A,verbose));

  /* phi_k(x) with index set from command-line arguments */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&phik));
  CHKERRQ(FNSetType(phik,FNPHI));
  CHKERRQ(FNSetFromOptions(phik));

  CHKERRQ(FNDuplicate(phik,PETSC_COMM_WORLD,&phicopy));
  CHKERRQ(FNPhiGetIndex(phicopy,&k));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Index of phi function is %" PetscInt_FMT "\n",k));
  CHKERRQ(TestPhiFunction(phicopy,2.2,A,verbose));

  CHKERRQ(FNDestroy(&phi0));
  CHKERRQ(FNDestroy(&phi1));
  CHKERRQ(FNDestroy(&phik));
  CHKERRQ(FNDestroy(&phicopy));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -fn_phi_index 3

TEST*/
