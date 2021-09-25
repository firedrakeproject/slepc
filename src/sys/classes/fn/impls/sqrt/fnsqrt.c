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
  if (x<0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Function not defined in the requested value");
#endif
  *y = PetscSqrtScalar(x);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateDerivative_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  if (x==0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
#if !defined(PETSC_USE_COMPLEX)
  if (x<0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
#endif
  *y = 1.0/(2.0*PetscSqrtScalar(x));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Schur(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = FNSqrtmSchur(fn,n,T,n,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMatVec_Sqrt_Schur(FN fn,Mat A,Vec v)
{
  PetscErrorCode ierr;
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;
  Mat            B;

  PetscFunctionBegin;
  ierr = FN_AllocateWorkMat(fn,A,&B);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = FNSqrtmSchur(fn,n,T,n,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetColumnVector(B,v,0);CHKERRQ(ierr);
  ierr = FN_FreeWorkMat(fn,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_DBP(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = FNSqrtmDenmanBeavers(fn,n,T,n,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_NS(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = FNSqrtmNewtonSchulz(fn,n,Ba,n,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
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
  PetscBLASInt   N,i,it,*piv=NULL,info,lwork=0,query=-1,one=1,zero=0;
  PetscBool      converged=PETSC_FALSE;
  PetscErrorCode ierr;
  unsigned int   ftz;

  PetscFunctionBegin;
  N = n*n;
  tol = PetscSqrtReal((PetscReal)n)*PETSC_MACHINE_EPSILON/2;
  ierr = SlepcSetFlushToZero(&ftz);CHKERRQ(ierr);

  /* query work size */
  PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,A,&ld,piv,&work1,&query,&info));
  ierr = PetscBLASIntCast((PetscInt)PetscRealPart(work1),&lwork);CHKERRQ(ierr);

  ierr = PetscMalloc5(N,&M,N,&M2,N,&G,lwork,&work,n,&piv);CHKERRQ(ierr);
  ierr = PetscArraycpy(M,A,N);CHKERRQ(ierr);

  /* scale M */
  nrm = LAPACKlange_("fro",&n,&n,M,&n,rwork);
  if (nrm>1.0) {
    sqrtnrm = PetscSqrtReal(nrm);
    PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&nrm,&done,&N,&one,M,&N,&info));
    SlepcCheckLapackInfo("lascl",info);
    tol *= nrm;
  }
  ierr = PetscInfo2(fn,"||A||_F = %g, new tol: %g\n",(double)nrm,(double)tol);CHKERRQ(ierr);

  /* X = I */
  ierr = PetscArrayzero(X,N);CHKERRQ(ierr);
  for (i=0;i<n;i++) X[i+i*ld] = 1.0;

  for (it=0;it<MAXIT && !converged;it++) {

    /* G = (5/16)*I + (1/16)*M*(15*I-5*M+M*M) */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,M,&ld,M,&ld,&szero,M2,&ld));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&N,&smfive,M,&one,M2,&one));
    for (i=0;i<n;i++) M2[i+i*ld] += 15.0;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&s1d16,M,&ld,M2,&ld,&szero,G,&ld));
    for (i=0;i<n;i++) G[i+i*ld] += 5.0/16.0;

    /* X = X*G */
    ierr = PetscArraycpy(M2,X,N);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,M2,&ld,G,&ld,&szero,X,&ld));

    /* M = M*inv(G*G) */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,G,&ld,G,&ld,&szero,M2,&ld));
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n,&n,M2,&ld,piv,&info));
    SlepcCheckLapackInfo("getrf",info);
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,M2,&ld,piv,work,&lwork,&info));
    SlepcCheckLapackInfo("getri",info);

    ierr = PetscArraycpy(G,M,N);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,G,&ld,M2,&ld,&szero,M,&ld));

    /* check ||I-M|| */
    ierr = PetscArraycpy(M2,M,N);CHKERRQ(ierr);
    for (i=0;i<n;i++) M2[i+i*ld] -= 1.0;
    Mres = LAPACKlange_("fro",&n,&n,M2,&n,rwork);
    if (PetscIsNanReal(Mres)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"The computed norm is not-a-number");
    if (Mres<=tol) converged = PETSC_TRUE;
    ierr = PetscInfo2(fn,"it: %D res: %g\n",it,(double)Mres);CHKERRQ(ierr);
    ierr = PetscLogFlops(8.0*n*n*n+2.0*n*n+2.0*n*n*n/3.0+4.0*n*n*n/3.0+2.0*n*n*n+2.0*n*n);CHKERRQ(ierr);
  }

  if (Mres>tol) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"SQRTM not converged after %d iterations",MAXIT);

  /* undo scaling */
  if (nrm>1.0) PetscStackCallBLAS("BLASscal",BLASscal_(&N,&sqrtnrm,A,&one));

  ierr = PetscFree5(M,M2,G,work,piv);CHKERRQ(ierr);
  ierr = SlepcResetFlushToZero(&ftz);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;
  cublasStatus_t     cberr;
  cudaError_t        cerr;
  magma_int_t        mierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr); /* For CUDA event timers */
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  magma_init();
  N = n*n;
  tol = PetscSqrtReal((PetscReal)n)*PETSC_MACHINE_EPSILON/2;

  ierr = PetscMalloc1(n,&piv);CHKERRQ(ierr);
  cerr = cudaMalloc((void **)&d_X,sizeof(PetscScalar)*N);CHKERRCUDA(cerr);
  cerr = cudaMalloc((void **)&d_M,sizeof(PetscScalar)*N);CHKERRCUDA(cerr);
  cerr = cudaMalloc((void **)&d_M2,sizeof(PetscScalar)*N);CHKERRCUDA(cerr);
  cerr = cudaMalloc((void **)&d_G,sizeof(PetscScalar)*N);CHKERRCUDA(cerr);

  nb = magma_get_xgetri_nb(n);
  lwork = nb*n;
  cerr = cudaMalloc((void **)&d_work,sizeof(PetscScalar)*lwork);CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);

  /* M = A */
  cerr = cudaMemcpy(d_M,A,sizeof(PetscScalar)*N,cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

  /* scale M */
  cberr = cublasXnrm2(cublasv2handle,N,d_M,one,&nrm);CHKERRCUBLAS(cberr);
  if (nrm>1.0) {
    sqrtnrm = PetscSqrtReal(nrm);
    alpha = 1.0/nrm;
    cberr = cublasXscal(cublasv2handle,N,&alpha,d_M,one);CHKERRCUBLAS(cberr);
    tol *= nrm;
  }
  ierr = PetscInfo2(fn,"||A||_F = %g, new tol: %g\n",(double)nrm,(double)tol);CHKERRQ(ierr);

  /* X = I */
  cerr = cudaMemset(d_X,zero,sizeof(PetscScalar)*N);CHKERRCUDA(cerr);
  ierr = set_diagonal(n,d_X,ld,sone);CHKERRQ(cerr);

  for (it=0;it<MAXIT && !converged;it++) {

    /* G = (5/16)*I + (1/16)*M*(15*I-5*M+M*M) */
    cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_M,ld,d_M,ld,&szero,d_M2,ld);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,N,&smfive,d_M,one,d_M2,one);CHKERRCUBLAS(cberr);
    ierr = shift_diagonal(n,d_M2,ld,s15);CHKERRQ(cerr);
    cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&s1d16,d_M,ld,d_M2,ld,&szero,d_G,ld);CHKERRCUBLAS(cberr);
    ierr = shift_diagonal(n,d_G,ld,5.0/16.0);CHKERRQ(cerr);

    /* X = X*G */
    cerr = cudaMemcpy(d_M2,d_X,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
    cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_M2,ld,d_G,ld,&szero,d_X,ld);CHKERRCUBLAS(cberr);

    /* M = M*inv(G*G) */
    cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_G,ld,d_G,ld,&szero,d_M2,ld);CHKERRCUBLAS(cberr);
    /* magma */
    mierr = magma_xgetrf_gpu(n,n,d_M2,ld,piv,&info);CHKERRMAGMA(mierr);
    if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKgetrf: Illegal value on argument %d",PetscAbsInt(info));
    if (info > 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"LAPACKgetrf: Matrix is singular. U(%d,%d) is zero",info,info);
    mierr = magma_xgetri_gpu(n,d_M2,ld,piv,d_work,lwork,&info);CHKERRMAGMA(mierr);
    if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKgetri: Illegal value on argument %d",PetscAbsInt(info));
    if (info > 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"LAPACKgetri: Matrix is singular. U(%d,%d) is zero",info,info);
    /* magma */
    cerr = cudaMemcpy(d_G,d_M,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
    cberr = cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_G,ld,d_M2,ld,&szero,d_M,ld);CHKERRCUBLAS(cberr);

    /* check ||I-M|| */
    cerr = cudaMemcpy(d_M2,d_M,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
    ierr = shift_diagonal(n,d_M2,ld,-1.0);CHKERRQ(cerr);
    cberr = cublasXnrm2(cublasv2handle,N,d_M2,one,&Mres);CHKERRCUBLAS(cberr);
    if (PetscIsNanReal(Mres)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"The computed norm is not-a-number");
    if (Mres<=tol) converged = PETSC_TRUE;
    ierr = PetscInfo2(fn,"it: %D res: %g\n",it,(double)Mres);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(8.0*n*n*n+2.0*n*n+2.0*n*n*n/3.0+4.0*n*n*n/3.0+2.0*n*n*n+2.0*n*n);CHKERRQ(ierr);
  }

  if (Mres>tol) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"SQRTM not converged after %d iterations", MAXIT);

  if (nrm>1.0) {cberr = cublasXscal(cublasv2handle,N,&sqrtnrm,d_X,one);CHKERRCUBLAS(cberr);}
  cerr = cudaMemcpy(A,d_X,sizeof(PetscScalar)*N,cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

  cerr = cudaFree(d_X);CHKERRCUDA(cerr);
  cerr = cudaFree(d_M);CHKERRCUDA(cerr);
  cerr = cudaFree(d_M2);CHKERRCUDA(cerr);
  cerr = cudaFree(d_G);CHKERRCUDA(cerr);
  cerr = cudaFree(d_work);CHKERRCUDA(cerr);
  ierr = PetscFree(piv);CHKERRQ(ierr);

  magma_finalize();
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MAGMA */
#endif /* PETSC_HAVE_CUDA */

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Sadeghi(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = FNSqrtmSadeghi(fn,n,Ba,n);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
PetscErrorCode FNEvaluateFunctionMat_Sqrt_NS_CUDA(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = FNSqrtmNewtonSchulz_CUDA(fn,n,Ba,n,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MAGMA)
PetscErrorCode FNEvaluateFunctionMat_Sqrt_DBP_CUDAm(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&T);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = FNSqrtmDenmanBeavers_CUDAm(fn,n,T,n,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Sadeghi_CUDAm(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) { ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ierr = FNSqrtmSadeghi_CUDAm(fn,n,Ba,n);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MAGMA */
#endif /* PETSC_HAVE_CUDA */

PetscErrorCode FNView_Sqrt(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
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
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (fn->beta==(PetscScalar)1.0) {
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Square root: sqrt(x)\n");CHKERRQ(ierr);
      } else {
        ierr = SlepcSNPrintfScalar(str,sizeof(str),fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Square root: sqrt(%s*x)\n",str);CHKERRQ(ierr);
      }
    } else {
      ierr = SlepcSNPrintfScalar(str,sizeof(str),fn->beta,PETSC_TRUE);CHKERRQ(ierr);
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Square root: %s*sqrt(x)\n",str);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Square root: %s",str);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
        ierr = SlepcSNPrintfScalar(str,sizeof(str),fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"*sqrt(%s*x)\n",str);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    if (fn->method<nmeth) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing matrix functions with: %s\n",methodname[fn->method]);CHKERRQ(ierr);
    }
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

