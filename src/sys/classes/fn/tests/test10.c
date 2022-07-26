/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  PetscCall(FNView(fn,NULL));
  PetscCall(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  PetscCall(FNEvaluateFunction(fn,x,&y));
  PetscCall(FNEvaluateDerivative(fn,x,&yp));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nf(%s)=%s\n",strx,str));
  PetscCall(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"f'(%s)=%s\n",strx,str));
  /* compute phi_k(A)*e_1 */
  PetscCall(MatCreateVecs(A,&v,&f));
  PetscCall(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));
  PetscCall(FNEvaluateFunctionMatVec(fn,A,f));  /* reference result by diagonalization */
  PetscCall(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  PetscCall(FNEvaluateFunctionMatVec(fn,A,v));
  PetscCall(VecAXPY(v,-1.0,f));
  PetscCall(VecNorm(v,NORM_2,&nrm));
  if (nrm>100*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of f(A)*e_1-ref is %g\n",(double)nrm));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"f(A)*e_1 =\n"));
    PetscCall(VecView(v,NULL));
  }
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&f));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  FN             phi0,phi1,phik,phicopy;
  Mat            A;
  PetscInt       i,j,n=8,k;
  PetscScalar    tau,eta,*As;
  PetscBool      verbose;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test Phi functions, n=%" PetscInt_FMT ".\n",n));

  /* Create matrix, fill it with 1-D Laplacian */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  PetscCall(PetscObjectSetName((PetscObject)A,"A"));
  PetscCall(MatDenseGetArray(A,&As));
  for (i=0;i<n;i++) As[i+i*n]=2.0;
  j=1;
  for (i=0;i<n-j;i++) { As[i+(i+j)*n]=-1.0; As[(i+j)+i*n]=-1.0; }
  PetscCall(MatDenseRestoreArray(A,&As));

  /* phi_0(x) = exp(x) */
  PetscCall(FNCreate(PETSC_COMM_WORLD,&phi0));
  PetscCall(FNSetType(phi0,FNPHI));
  PetscCall(FNPhiSetIndex(phi0,0));
  PetscCall(TestPhiFunction(phi0,2.2,A,verbose));

  /* phi_1(x) = (exp(x)-1)/x with scaling factors eta*phi_1(tau*x) */
  PetscCall(FNCreate(PETSC_COMM_WORLD,&phi1));
  PetscCall(FNSetType(phi1,FNPHI));  /* default index should be 1 */
  tau = 0.2;
  eta = 1.3;
  PetscCall(FNSetScale(phi1,tau,eta));
  PetscCall(TestPhiFunction(phi1,2.2,A,verbose));

  /* phi_k(x) with index set from command-line arguments */
  PetscCall(FNCreate(PETSC_COMM_WORLD,&phik));
  PetscCall(FNSetType(phik,FNPHI));
  PetscCall(FNSetFromOptions(phik));

  PetscCall(FNDuplicate(phik,PETSC_COMM_WORLD,&phicopy));
  PetscCall(FNPhiGetIndex(phicopy,&k));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Index of phi function is %" PetscInt_FMT "\n",k));
  PetscCall(TestPhiFunction(phicopy,2.2,A,verbose));

  PetscCall(FNDestroy(&phi0));
  PetscCall(FNDestroy(&phi1));
  PetscCall(FNDestroy(&phik));
  PetscCall(FNDestroy(&phicopy));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -fn_phi_index 3
      requires: !single

TEST*/
