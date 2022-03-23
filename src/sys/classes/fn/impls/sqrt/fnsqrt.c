/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Square root function  sqrt(x)
*/

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode FNEvaluateFunction_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  PetscCheck(x>=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Function not defined in the requested value");
#endif
  *y = PetscSqrtScalar(x);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateDerivative_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  PetscCheck(x!=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
#if !defined(PETSC_USE_COMPLEX)
  PetscCheck(x>0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
#endif
  *y = 1.0/(2.0*PetscSqrtScalar(x));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Schur(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) CHKERRQ(MatCopy(A,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDenseGetArray(B,&T));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  CHKERRQ(FNSqrtmSchur(fn,n,T,n,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArray(B,&T));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMatVec_Sqrt_Schur(FN fn,Mat A,Vec v)
{
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;
  Mat            B;

  PetscFunctionBegin;
  CHKERRQ(FN_AllocateWorkMat(fn,A,&B));
  CHKERRQ(MatDenseGetArray(B,&T));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  CHKERRQ(FNSqrtmSchur(fn,n,T,n,PETSC_TRUE));
  CHKERRQ(MatDenseRestoreArray(B,&T));
  CHKERRQ(MatGetColumnVector(B,v,0));
  CHKERRQ(FN_FreeWorkMat(fn,&B));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_DBP(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) CHKERRQ(MatCopy(A,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDenseGetArray(B,&T));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  CHKERRQ(FNSqrtmDenmanBeavers(fn,n,T,n,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArray(B,&T));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_NS(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) CHKERRQ(MatCopy(A,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDenseGetArray(B,&Ba));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  CHKERRQ(FNSqrtmNewtonSchulz(fn,n,Ba,n,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}

#define MAXIT 50

/*
   Computes the principal square root of the matrix A using the
   Sadeghi iteration. A is overwritten with sqrtm(A).
 */
PetscErrorCode FNSqrtmSadeghi(FN fn,PetscBLASInt n,PetscScalar *A,PetscBLASInt ld)
{
  PetscScalar    *M,*M2,*G,*X=A,*work,work1,sqrtnrm;
  PetscScalar    szero=0.0,sone=1.0,smfive=-5.0,s1d16=1.0/16.0;
  PetscReal      tol,Mres=0.0,nrm,rwork[1],done=1.0;
  PetscInt       i,it;
  PetscBLASInt   N,*piv=NULL,info,lwork=0,query=-1,one=1,zero=0;
  PetscBool      converged=PETSC_FALSE;
  unsigned int   ftz;

  PetscFunctionBegin;
  N = n*n;
  tol = PetscSqrtReal((PetscReal)n)*PETSC_MACHINE_EPSILON/2;
  CHKERRQ(SlepcSetFlushToZero(&ftz));

  /* query work size */
  PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,A,&ld,piv,&work1,&query,&info));
  CHKERRQ(PetscBLASIntCast((PetscInt)PetscRealPart(work1),&lwork));

  CHKERRQ(PetscMalloc5(N,&M,N,&M2,N,&G,lwork,&work,n,&piv));
  CHKERRQ(PetscArraycpy(M,A,N));

  /* scale M */
  nrm = LAPACKlange_("fro",&n,&n,M,&n,rwork);
  if (nrm>1.0) {
    sqrtnrm = PetscSqrtReal(nrm);
    PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&nrm,&done,&N,&one,M,&N,&info));
    SlepcCheckLapackInfo("lascl",info);
    tol *= nrm;
  }
  CHKERRQ(PetscInfo(fn,"||A||_F = %g, new tol: %g\n",(double)nrm,(double)tol));

  /* X = I */
  CHKERRQ(PetscArrayzero(X,N));
  for (i=0;i<n;i++) X[i+i*ld] = 1.0;

  for (it=0;it<MAXIT && !converged;it++) {

    /* G = (5/16)*I + (1/16)*M*(15*I-5*M+M*M) */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,M,&ld,M,&ld,&szero,M2,&ld));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&N,&smfive,M,&one,M2,&one));
    for (i=0;i<n;i++) M2[i+i*ld] += 15.0;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&s1d16,M,&ld,M2,&ld,&szero,G,&ld));
    for (i=0;i<n;i++) G[i+i*ld] += 5.0/16.0;

    /* X = X*G */
    CHKERRQ(PetscArraycpy(M2,X,N));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,M2,&ld,G,&ld,&szero,X,&ld));

    /* M = M*inv(G*G) */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,G,&ld,G,&ld,&szero,M2,&ld));
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n,&n,M2,&ld,piv,&info));
    SlepcCheckLapackInfo("getrf",info);
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,M2,&ld,piv,work,&lwork,&info));
    SlepcCheckLapackInfo("getri",info);

    CHKERRQ(PetscArraycpy(G,M,N));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,G,&ld,M2,&ld,&szero,M,&ld));

    /* check ||I-M|| */
    CHKERRQ(PetscArraycpy(M2,M,N));
    for (i=0;i<n;i++) M2[i+i*ld] -= 1.0;
    Mres = LAPACKlange_("fro",&n,&n,M2,&n,rwork);
    PetscCheck(!PetscIsNanReal(Mres),PETSC_COMM_SELF,PETSC_ERR_FP,"The computed norm is not-a-number");
    if (Mres<=tol) converged = PETSC_TRUE;
    CHKERRQ(PetscInfo(fn,"it: %" PetscInt_FMT " res: %g\n",it,(double)Mres));
    CHKERRQ(PetscLogFlops(8.0*n*n*n+2.0*n*n+2.0*n*n*n/3.0+4.0*n*n*n/3.0+2.0*n*n*n+2.0*n*n));
  }

  PetscCheck(Mres<=tol,PETSC_COMM_SELF,PETSC_ERR_LIB,"SQRTM not converged after %d iterations",MAXIT);

  /* undo scaling */
  if (nrm>1.0) PetscStackCallBLAS("BLASscal",BLASscal_(&N,&sqrtnrm,A,&one));

  CHKERRQ(PetscFree5(M,M2,G,work,piv));
  CHKERRQ(SlepcResetFlushToZero(&ftz));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
#include "../src/sys/classes/fn/impls/cuda/fnutilcuda.h"
#include <slepccublas.h>

#if defined(PETSC_HAVE_MAGMA)
#include <slepcmagma.h>

/*
 * Matrix square root by Sadeghi iteration. CUDA version.
 * Computes the principal square root of the matrix T using the
 * Sadeghi iteration. T is overwritten with sqrtm(T).
 */
PetscErrorCode FNSqrtmSadeghi_CUDAm(FN fn,PetscBLASInt n,PetscScalar *A,PetscBLASInt ld)
{
  PetscScalar        *d_X,*d_M,*d_M2,*d_G,*d_work,alpha;
  const PetscScalar  szero=0.0,sone=1.0,smfive=-5.0,s15=15.0,s1d16=1.0/16.0;
  PetscReal          tol,Mres=0.0,nrm,sqrtnrm;
  PetscInt           it,nb,lwork;
  PetscBLASInt       info,*piv,N;
  const PetscBLASInt one=1,zero=0;
  PetscBool          converged=PETSC_FALSE;
  cublasHandle_t     cublasv2handle;

  PetscFunctionBegin;
  CHKERRQ(PetscDeviceInitialize(PETSC_DEVICE_CUDA)); /* For CUDA event timers */
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  magma_init();
  N = n*n;
  tol = PetscSqrtReal((PetscReal)n)*PETSC_MACHINE_EPSILON/2;

  CHKERRQ(PetscMalloc1(n,&piv));
  CHKERRCUDA(cudaMalloc((void **)&d_X,sizeof(PetscScalar)*N));
  CHKERRCUDA(cudaMalloc((void **)&d_M,sizeof(PetscScalar)*N));
  CHKERRCUDA(cudaMalloc((void **)&d_M2,sizeof(PetscScalar)*N));
  CHKERRCUDA(cudaMalloc((void **)&d_G,sizeof(PetscScalar)*N));

  nb = magma_get_xgetri_nb(n);
  lwork = nb*n;
  CHKERRCUDA(cudaMalloc((void **)&d_work,sizeof(PetscScalar)*lwork));
  CHKERRQ(PetscLogGpuTimeBegin());

  /* M = A */
  CHKERRCUDA(cudaMemcpy(d_M,A,sizeof(PetscScalar)*N,cudaMemcpyHostToDevice));

  /* scale M */
  CHKERRCUBLAS(cublasXnrm2(cublasv2handle,N,d_M,one,&nrm));
  if (nrm>1.0) {
    sqrtnrm = PetscSqrtReal(nrm);
    alpha = 1.0/nrm;
    CHKERRCUBLAS(cublasXscal(cublasv2handle,N,&alpha,d_M,one));
    tol *= nrm;
  }
  CHKERRQ(PetscInfo(fn,"||A||_F = %g, new tol: %g\n",(double)nrm,(double)tol));

  /* X = I */
  CHKERRCUDA(cudaMemset(d_X,zero,sizeof(PetscScalar)*N));
  CHKERRQ(set_diagonal(n,d_X,ld,sone));

  for (it=0;it<MAXIT && !converged;it++) {

    /* G = (5/16)*I + (1/16)*M*(15*I-5*M+M*M) */
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_M,ld,d_M,ld,&szero,d_M2,ld));
    CHKERRCUBLAS(cublasXaxpy(cublasv2handle,N,&smfive,d_M,one,d_M2,one));
    CHKERRQ(shift_diagonal(n,d_M2,ld,s15));
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&s1d16,d_M,ld,d_M2,ld,&szero,d_G,ld));
    CHKERRQ(shift_diagonal(n,d_G,ld,5.0/16.0));

    /* X = X*G */
    CHKERRCUDA(cudaMemcpy(d_M2,d_X,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice));
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_M2,ld,d_G,ld,&szero,d_X,ld));

    /* M = M*inv(G*G) */
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_G,ld,d_G,ld,&szero,d_M2,ld));
    /* magma */
    CHKERRMAGMA(magma_xgetrf_gpu(n,n,d_M2,ld,piv,&info));
    PetscCheck(info>=0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKgetrf: Illegal value on argument %" PetscBLASInt_FMT,PetscAbsInt(info));
    PetscCheck(info<=0,PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"LAPACKgetrf: Matrix is singular. U(%" PetscBLASInt_FMT ",%" PetscBLASInt_FMT ") is zero",info,info);
    CHKERRMAGMA(magma_xgetri_gpu(n,d_M2,ld,piv,d_work,lwork,&info));
    PetscCheck(info>=0,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKgetri: Illegal value on argument %" PetscBLASInt_FMT,PetscAbsInt(info));
    PetscCheck(info<=0,PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"LAPACKgetri: Matrix is singular. U(%" PetscBLASInt_FMT ",%" PetscBLASInt_FMT ") is zero",info,info);
    /* magma */
    CHKERRCUDA(cudaMemcpy(d_G,d_M,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice));
    CHKERRCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_G,ld,d_M2,ld,&szero,d_M,ld));

    /* check ||I-M|| */
    CHKERRCUDA(cudaMemcpy(d_M2,d_M,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice));
    CHKERRQ(shift_diagonal(n,d_M2,ld,-1.0));
    CHKERRCUBLAS(cublasXnrm2(cublasv2handle,N,d_M2,one,&Mres));
    PetscCheck(!PetscIsNanReal(Mres),PETSC_COMM_SELF,PETSC_ERR_FP,"The computed norm is not-a-number");
    if (Mres<=tol) converged = PETSC_TRUE;
    CHKERRQ(PetscInfo(fn,"it: %" PetscInt_FMT " res: %g\n",it,(double)Mres));
    CHKERRQ(PetscLogGpuFlops(8.0*n*n*n+2.0*n*n+2.0*n*n*n/3.0+4.0*n*n*n/3.0+2.0*n*n*n+2.0*n*n));
  }

  PetscCheck(Mres<=tol,PETSC_COMM_SELF,PETSC_ERR_LIB,"SQRTM not converged after %d iterations", MAXIT);

  if (nrm>1.0) CHKERRCUBLAS(cublasXscal(cublasv2handle,N,&sqrtnrm,d_X,one));
  CHKERRCUDA(cudaMemcpy(A,d_X,sizeof(PetscScalar)*N,cudaMemcpyDeviceToHost));
  CHKERRQ(PetscLogGpuTimeEnd());

  CHKERRCUDA(cudaFree(d_X));
  CHKERRCUDA(cudaFree(d_M));
  CHKERRCUDA(cudaFree(d_M2));
  CHKERRCUDA(cudaFree(d_G));
  CHKERRCUDA(cudaFree(d_work));
  CHKERRQ(PetscFree(piv));

  magma_finalize();
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MAGMA */
#endif /* PETSC_HAVE_CUDA */

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Sadeghi(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) CHKERRQ(MatCopy(A,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDenseGetArray(B,&Ba));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  CHKERRQ(FNSqrtmSadeghi(fn,n,Ba,n));
  CHKERRQ(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
PetscErrorCode FNEvaluateFunctionMat_Sqrt_NS_CUDA(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) CHKERRQ(MatCopy(A,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDenseGetArray(B,&Ba));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  CHKERRQ(FNSqrtmNewtonSchulz_CUDA(fn,n,Ba,n,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MAGMA)
PetscErrorCode FNEvaluateFunctionMat_Sqrt_DBP_CUDAm(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) CHKERRQ(MatCopy(A,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDenseGetArray(B,&T));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  CHKERRQ(FNSqrtmDenmanBeavers_CUDAm(fn,n,T,n,PETSC_FALSE));
  CHKERRQ(MatDenseRestoreArray(B,&T));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Sadeghi_CUDAm(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) CHKERRQ(MatCopy(A,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDenseGetArray(B,&Ba));
  CHKERRQ(MatGetSize(A,&m,NULL));
  CHKERRQ(PetscBLASIntCast(m,&n));
  CHKERRQ(FNSqrtmSadeghi_CUDAm(fn,n,Ba,n));
  CHKERRQ(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MAGMA */
#endif /* PETSC_HAVE_CUDA */

PetscErrorCode FNView_Sqrt(FN fn,PetscViewer viewer)
{
  PetscBool      isascii;
  char           str[50];
  const char     *methodname[] = {
                  "Schur method for the square root",
                  "Denman-Beavers (product form)",
                  "Newton-Schulz iteration",
                  "Sadeghi iteration"
#if defined(PETSC_HAVE_CUDA)
                 ,"Newton-Schulz iteration CUDA"
#if defined(PETSC_HAVE_MAGMA)
                 ,"Denman-Beavers (product form) CUDA/MAGMA",
                  "Sadeghi iteration CUDA/MAGMA"
#endif
#endif
  };
  const int      nmeth=sizeof(methodname)/sizeof(methodname[0]);

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (fn->beta==(PetscScalar)1.0) {
      if (fn->alpha==(PetscScalar)1.0) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Square root: sqrt(x)\n"));
      else {
        CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),fn->alpha,PETSC_TRUE));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Square root: sqrt(%s*x)\n",str));
      }
    } else {
      CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),fn->beta,PETSC_TRUE));
      if (fn->alpha==(PetscScalar)1.0) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Square root: %s*sqrt(x)\n",str));
      else {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Square root: %s",str));
        CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
        CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),fn->alpha,PETSC_TRUE));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"*sqrt(%s*x)\n",str));
        CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      }
    }
    if (fn->method<nmeth) CHKERRQ(PetscViewerASCIIPrintf(viewer,"  computing matrix functions with: %s\n",methodname[fn->method]));
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode FNCreate_Sqrt(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction          = FNEvaluateFunction_Sqrt;
  fn->ops->evaluatederivative        = FNEvaluateDerivative_Sqrt;
  fn->ops->evaluatefunctionmat[0]    = FNEvaluateFunctionMat_Sqrt_Schur;
  fn->ops->evaluatefunctionmat[1]    = FNEvaluateFunctionMat_Sqrt_DBP;
  fn->ops->evaluatefunctionmat[2]    = FNEvaluateFunctionMat_Sqrt_NS;
  fn->ops->evaluatefunctionmat[3]    = FNEvaluateFunctionMat_Sqrt_Sadeghi;
#if defined(PETSC_HAVE_CUDA)
  fn->ops->evaluatefunctionmat[4]    = FNEvaluateFunctionMat_Sqrt_NS_CUDA;
#if defined(PETSC_HAVE_MAGMA)
  fn->ops->evaluatefunctionmat[5]    = FNEvaluateFunctionMat_Sqrt_DBP_CUDAm;
  fn->ops->evaluatefunctionmat[6]    = FNEvaluateFunctionMat_Sqrt_Sadeghi_CUDAm;
#endif /* PETSC_HAVE_MAGMA */
#endif /* PETSC_HAVE_CUDA */
  fn->ops->evaluatefunctionmatvec[0] = FNEvaluateFunctionMatVec_Sqrt_Schur;
  fn->ops->view                      = FNView_Sqrt;
  PetscFunctionReturn(0);
}
