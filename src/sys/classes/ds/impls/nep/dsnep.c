/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/dsimpl.h>       /*I "slepcds.h" I*/
#include <slepc/private/rgimpl.h>       /*I "slepcrg.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  PetscInt       nf;                 /* number of functions in f[] */
  FN             f[DS_NUM_EXTRA];    /* functions defining the nonlinear operator */
  PetscInt       max_mid;            /* maximum minimality index */
  PetscInt       nnod;               /* number of nodes for quadrature rules */
  PetscInt       spls;               /* number of sampling columns for quadrature rules */
  PetscInt       Nit;                /* number of refinement iterations */
  PetscReal      rtol;               /* tolerance of Newton refinement */
  RG             rg;                 /* region for contour integral */
  PetscLayout    map;                /* used to distribute work among MPI processes */
  void           *computematrixctx;
  PetscErrorCode (*computematrix)(DS,PetscScalar,PetscBool,DSMatType,void*);
} DS_NEP;

/*
   DSNEPComputeMatrix - Build the matrix associated with a nonlinear operator
   T(lambda) or its derivative T'(lambda), given the parameter lambda, where
   T(lambda) = sum_i E_i*f_i(lambda). The result is written in mat.
*/
static PetscErrorCode DSNEPComputeMatrix(DS ds,PetscScalar lambda,PetscBool deriv,DSMatType mat)
{
  PetscErrorCode ierr;
  DS_NEP         *ctx = (DS_NEP*)ds->data;
  PetscScalar    *T,*E,alpha;
  PetscInt       i,ld,n;
  PetscBLASInt   k,inc=1;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  if (ctx->computematrix) {
    ierr = (*ctx->computematrix)(ds,lambda,deriv,mat,ctx->computematrixctx);CHKERRQ(ierr);
  } else {
    ierr = DSGetDimensions(ds,&n,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSGetLeadingDimension(ds,&ld);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ld*n,&k);CHKERRQ(ierr);
    ierr = DSGetArray(ds,mat,&T);CHKERRQ(ierr);
    ierr = PetscArrayzero(T,k);CHKERRQ(ierr);
    for (i=0;i<ctx->nf;i++) {
      if (deriv) {
        ierr = FNEvaluateDerivative(ctx->f[i],lambda,&alpha);CHKERRQ(ierr);
      } else {
        ierr = FNEvaluateFunction(ctx->f[i],lambda,&alpha);CHKERRQ(ierr);
      }
      E = ds->mat[DSMatExtra[i]];
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&k,&alpha,E,&inc,T,&inc));
    }
    ierr = DSRestoreArray(ds,mat,&T);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSAllocate_NEP(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;
  DS_NEP         *ctx = (DS_NEP*)ds->data;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_X);CHKERRQ(ierr);
  for (i=0;i<ctx->nf;i++) {
    ierr = DSAllocateMat_Private(ds,DSMatExtra[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld*ctx->max_mid,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*ctx->max_mid*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSView_NEP(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  DS_NEP            *ctx = (DS_NEP*)ds->data;
  PetscViewerFormat format;
  PetscInt          i;
  const char        *methodname[] = {
                     "Successive Linear Problems",
                     "Contour Integral"
  };
  const int         nmeth=sizeof(methodname)/sizeof(methodname[0]);

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    if (ds->method<nmeth) {
      ierr = PetscViewerASCIIPrintf(viewer,"solving the problem with: %s\n",methodname[ds->method]);CHKERRQ(ierr);
    }
#if defined(PETSC_USE_COMPLEX)
    if (ds->method==1) {  /* contour integral method */
      ierr = PetscViewerASCIIPrintf(viewer,"number of integration points: %D\n",ctx->nnod);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"maximum minimality index: %D\n",ctx->max_mid);CHKERRQ(ierr);
      if (ctx->spls) { ierr = PetscViewerASCIIPrintf(viewer,"number of sampling columns for quadrature: %D\n",ctx->spls);CHKERRQ(ierr); }
      if (ctx->Nit) { ierr = PetscViewerASCIIPrintf(viewer,"doing iterative refinement (%D its, tolerance %g)\n",ctx->Nit,(double)ctx->rtol);CHKERRQ(ierr); }
      ierr = RGView(ctx->rg,viewer);CHKERRQ(ierr);
    }
#endif
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = PetscViewerASCIIPrintf(viewer,"number of functions: %D\n",ctx->nf);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  for (i=0;i<ctx->nf;i++) {
    ierr = FNView(ctx->f[i],viewer);CHKERRQ(ierr);
    ierr = DSViewMat(ds,viewer,DSMatExtra[i]);CHKERRQ(ierr);
  }
  if (ds->state>DS_STATE_INTERMEDIATE) { ierr = DSViewMat(ds,viewer,DS_MAT_X);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

PetscErrorCode DSVectors_NEP(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscFunctionBegin;
  if (rnorm) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
  switch (mat) {
    case DS_MAT_X:
      break;
    case DS_MAT_Y:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Not implemented yet");
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSort_NEP(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *dummy)
{
  PetscErrorCode ierr;
  DS_NEP         *ctx = (DS_NEP*)ds->data;
  PetscInt       n,l,i,*perm,lds;
  PetscScalar    *A;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  n = ds->n*ctx->max_mid;
  lds = ds->ld*ctx->max_mid;
  l = ds->l;
  A = ds->mat[DS_MAT_A];
  perm = ds->perm;
  for (i=0;i<n;i++) perm[i] = i;
  if (rr) {
    ierr = DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = DSSortEigenvalues_Private(ds,wr,NULL,perm,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=l;i<ds->t;i++) A[i+i*lds] = wr[perm[i]];
  for (i=l;i<ds->t;i++) wr[i] = A[i+i*lds];
  /* n != ds->n */
  ierr = DSPermuteColumns_Private(ds,0,ds->t,ds->n,DS_MAT_X,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSSolve_NEP_SLP(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  PetscScalar    *A,*B,*W,*X,*work,*alpha,*beta;
  PetscScalar    sigma,lambda,mu,re,re2,sone=1.0,szero=0.0;
  PetscBLASInt   info,n,ld,lrwork=0,lwork,one=1,zero=0;
  PetscInt       it,pos,j,maxit=100,result;
  PetscReal      norm,tol,done=1.0;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#else
  PetscReal      *alphai,im,im2;
#endif

  PetscFunctionBegin;
  if (!ds->mat[DS_MAT_A]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  }
  if (!ds->mat[DS_MAT_B]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_B);CHKERRQ(ierr);
  }
  if (!ds->mat[DS_MAT_W]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  }
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscBLASIntCast(2*ds->n+2*ds->n,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(8*ds->n,&lrwork);CHKERRQ(ierr);
#else
  ierr = PetscBLASIntCast(3*ds->n+8*ds->n,&lwork);CHKERRQ(ierr);
#endif
  ierr = DSAllocateWork_Private(ds,lwork,lrwork,0);CHKERRQ(ierr);
  alpha = ds->work;
  beta = ds->work + ds->n;
#if defined(PETSC_USE_COMPLEX)
  work = ds->work + 2*ds->n;
  lwork -= 2*ds->n;
#else
  alphai = ds->work + 2*ds->n;
  work = ds->work + 3*ds->n;
  lwork -= 3*ds->n;
#endif
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  W = ds->mat[DS_MAT_W];
  X = ds->mat[DS_MAT_X];

  sigma = 0.0;
  if (ds->sc->comparison==SlepcCompareTargetMagnitude || ds->sc->comparison==SlepcCompareTargetReal) sigma = *(PetscScalar*)ds->sc->comparisonctx;
  lambda = sigma;
  tol = n*PETSC_MACHINE_EPSILON/PetscSqrtReal(PETSC_SQRT_MACHINE_EPSILON);

  for (it=0;it<maxit;it++) {

    /* evaluate T and T' */
    ierr = DSNEPComputeMatrix(ds,lambda,PETSC_FALSE,DS_MAT_A);CHKERRQ(ierr);
    if (it) {
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,A,&ld,X,&one,&szero,X+ld,&one));
      norm = BLASnrm2_(&n,X+ld,&one);
      if (norm/PetscAbsScalar(lambda)<=tol) break;
    }
    ierr = DSNEPComputeMatrix(ds,lambda,PETSC_TRUE,DS_MAT_B);CHKERRQ(ierr);

    /* compute eigenvalue correction mu and eigenvector u */
#if defined(PETSC_USE_COMPLEX)
    rwork = ds->rwork;
    PetscStackCallBLAS("LAPACKggev",LAPACKggev_("N","V",&n,A,&ld,B,&ld,alpha,beta,NULL,&ld,W,&ld,work,&lwork,rwork,&info));
#else
    PetscStackCallBLAS("LAPACKggev",LAPACKggev_("N","V",&n,A,&ld,B,&ld,alpha,alphai,beta,NULL,&ld,W,&ld,work,&lwork,&info));
#endif
    SlepcCheckLapackInfo("ggev",info);

    /* find smallest eigenvalue */
    j = 0;
    if (beta[j]==0.0) re = (PetscRealPart(alpha[j])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else re = alpha[j]/beta[j];
#if !defined(PETSC_USE_COMPLEX)
    if (beta[j]==0.0) im = (alphai[j]>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else im = alphai[j]/beta[j];
#endif
    pos = 0;
    for (j=1;j<n;j++) {
      if (beta[j]==0.0) re2 = (PetscRealPart(alpha[j])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
      else re2 = alpha[j]/beta[j];
#if !defined(PETSC_USE_COMPLEX)
      if (beta[j]==0.0) im2 = (alphai[j]>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
      else im2 = alphai[j]/beta[j];
      ierr = SlepcCompareSmallestMagnitude(re,im,re2,im2,&result,NULL);CHKERRQ(ierr);
#else
      ierr = SlepcCompareSmallestMagnitude(re,0.0,re2,0.0,&result,NULL);CHKERRQ(ierr);
#endif
      if (result > 0) {
        re = re2;
#if !defined(PETSC_USE_COMPLEX)
        im = im2;
#endif
        pos = j;
      }
    }

#if !defined(PETSC_USE_COMPLEX)
    if (im!=0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"DSNEP found a complex eigenvalue; try rerunning with complex scalars");
#endif
    mu = alpha[pos]/beta[pos];
    ierr = PetscArraycpy(X,W+pos*ld,n);CHKERRQ(ierr);
    norm = BLASnrm2_(&n,X,&one);
    PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&norm,&done,&n,&one,X,&n,&info));
    SlepcCheckLapackInfo("lascl",info);

    /* correct eigenvalue approximation */
    lambda = lambda - mu;
  }

  if (it==maxit) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"DSNEP did not converge");
  ds->t = 1;
  wr[0] = lambda;
  if (wi) wi[0] = 0.0;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
/*
  Newton refinement for eigenpairs computed with contour integral.
  k  - number of eigenpairs to refine
  wr - eigenvalues (eigenvectors are stored in DS_MAT_X)
*/
static PetscErrorCode DSNEPNewtonRefine(DS ds,PetscInt k,PetscScalar *wr)
{
  PetscErrorCode ierr;
  DS_NEP         *ctx = (DS_NEP*)ds->data;
  PetscScalar    *X,*W,*U,*R,sone=1.0,szero=0.0;
  PetscReal      norm;
  PetscInt       i,j,ii,nwu=0,*p,jstart=0,jend=k;
  const PetscInt *range;
  PetscBLASInt   n,*perm,info,ld,one=1,n1;
  PetscMPIInt    len,size,root;
  PetscLayout    map;

  PetscFunctionBegin;
  X = ds->mat[DS_MAT_X];
  W = ds->mat[DS_MAT_W];
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  n1 = n+1;
  p  = ds->perm;
  ierr = PetscArrayzero(p,k);CHKERRQ(ierr);
  ierr = DSAllocateWork_Private(ds,(n+2)*(n+1),0,n+1);CHKERRQ(ierr);
  U    = ds->work+nwu;    nwu += (n+1)*(n+1);
  R    = ds->work+nwu;    nwu += n+1;
  perm = ds->iwork;
  if (ds->pmode==DS_PARALLEL_DISTRIBUTED) {
    ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)ds),PETSC_DECIDE,k,1,&map);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(map,&jstart,&jend);CHKERRQ(ierr);
  }
  for (ii=0;ii<ctx->Nit;ii++) {
    for (j=jstart;j<jend;j++) {
      if (p[j]<2) {
        ierr = DSNEPComputeMatrix(ds,wr[j],PETSC_FALSE,DS_MAT_W);CHKERRQ(ierr);
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,W,&ld,X+ld*j,&one,&szero,R,&one));
        norm = BLASnrm2_(&n,R,&one);
        if (norm/PetscAbsScalar(wr[j]) > ctx->rtol) {
          ierr = PetscInfo2(NULL,"Refining eigenpair %D, residual=%g\n",j,(double)norm/PetscAbsScalar(wr[j]));CHKERRQ(ierr);
          p[j] = 1;
          R[n] = 0.0;
          for (i=0;i<n;i++) {
            ierr = PetscArraycpy(U+i*n1,W+i*ld,n);CHKERRQ(ierr);
            U[n+i*n1] = PetscConj(X[j*ld+i]);
          }
          U[n+n*n1] = 0.0;
          ierr = DSNEPComputeMatrix(ds,wr[j],PETSC_TRUE,DS_MAT_W);CHKERRQ(ierr);
          PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,W,&ld,X+ld*j,&one,&szero,U+n*(n+1),&one));
          /* solve system  */
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n1,&n1,U,&n1,perm,&info));
          SlepcCheckLapackInfo("getrf",info);
          PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N",&n1,&one,U,&n1,perm,R,&n1,&info));
          SlepcCheckLapackInfo("getrs",info);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          wr[j] -= R[n];
          for (i=0;i<n;i++) X[j*ld+i] -= R[i];
          /* normalization */
          norm = BLASnrm2_(&n,X+ld*j,&one);
          for (i=0;i<n;i++) X[ld*j+i] /= norm;
        } else p[j] = 2;
      }
    }
  }
  if (ds->pmode==DS_PARALLEL_DISTRIBUTED) {  /* communicate results */
    ierr = PetscMPIIntCast(k,&len);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_IN_PLACE)
    ierr = MPIU_Allreduce(MPI_IN_PLACE,p,len,MPIU_INT,MPIU_SUM,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
#else
    {
      PetscInt *buffer;
      ierr = PetscMalloc1(len,&buffer);CHKERRQ(ierr);
      ierr = PetscArraycpy(buffer,p,len);CHKERRQ(ierr);
      ierr = MPIU_Allreduce(buffer,p,len,MPIU_INT,MPIU_SUM,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
      ierr = PetscFree(buffer);CHKERRQ(ierr);
    }
#endif
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)ds),&size);CHKERRMPI(ierr);
    ierr = PetscLayoutGetRanges(map,&range);CHKERRQ(ierr);
    for (j=0;j<k;j++) {
      if (p[j]) {  /* j-th eigenpair has been refined */
        for (root=0;root<size;root++) if (range[root+1]>j) break;
        ierr = PetscMPIIntCast(1,&len);CHKERRQ(ierr);
        ierr = MPI_Bcast(wr+j,len,MPIU_SCALAR,root,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
        ierr = PetscMPIIntCast(n,&len);CHKERRQ(ierr);
        ierr = MPI_Bcast(X+ld*j,len,MPIU_SCALAR,root,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
      }
    }
    ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSolve_NEP_Contour(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  DS_NEP         *ctx = (DS_NEP*)ds->data;
  PetscScalar    *alpha,*beta,*A,*B,*X,*W,*work,*Rc,*R,*w,*z,*zn,*S,*U,*V;
  PetscScalar    sone=1.0,szero=0.0,center;
  PetscReal      *rwork,norm,radius,vscale,rgscale,*sigma;
  PetscBLASInt   info,n,*perm,p,pp,ld,lwork,k_,rk_,colA,rowA,one=1;
  PetscInt       mid,lds,nnod=ctx->nnod,k,i,ii,jj,j,s,off,rk,nwu=0,nw,lrwork,*inside,kstart=0,kend=nnod;
  PetscMPIInt    len;
  PetscBool      isellipse;
  PetscRandom    rand;

  PetscFunctionBegin;
  if (!ctx->rg) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ORDER,"The contour solver requires a region passed with DSNEPSetRG()");
  /* Contour parameters */
  ierr = PetscObjectTypeCompare((PetscObject)ctx->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (isellipse) {
    ierr = RGEllipseGetParameters(ctx->rg,&center,&radius,&vscale);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_SUP,"Region must be Ellipse");
  ierr = RGGetScale(ctx->rg,&rgscale);CHKERRQ(ierr);
  if (ds->pmode==DS_PARALLEL_DISTRIBUTED) {
    if (!ctx->map) { ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)ds),PETSC_DECIDE,ctx->nnod,1,&ctx->map);CHKERRQ(ierr); }
    ierr = PetscLayoutGetRange(ctx->map,&kstart,&kend);CHKERRQ(ierr);
  }

  if (!ds->mat[DS_MAT_A]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  } /* size mid*n */
  if (!ds->mat[DS_MAT_B]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_B);CHKERRQ(ierr);
  } /* size mid*n */
  if (!ds->mat[DS_MAT_W]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  } /* size mid*n */
  if (!ds->mat[DS_MAT_U]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_U);CHKERRQ(ierr);
  } /* size mid*n */
  if (!ds->mat[DS_MAT_V]) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_V);CHKERRQ(ierr);
  } /* size n */
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  W = ds->mat[DS_MAT_W];
  U = ds->mat[DS_MAT_U];
  V = ds->mat[DS_MAT_V];
  X = ds->mat[DS_MAT_X];
  mid  = ctx->max_mid;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  p    = n;   /* maximum number of columns for the probing matrix */
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(mid*n,&rowA);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(5*rowA,&lwork);CHKERRQ(ierr);
  nw   = n*(2*p+7*mid)+3*nnod+2*mid*n*p;
  lrwork = mid*n*6+8*n;
  ierr = DSAllocateWork_Private(ds,nw,lrwork,n+1);CHKERRQ(ierr);

  sigma = ds->rwork;
  rwork = ds->rwork+mid*n;
  perm  = ds->iwork;
  z     = ds->work+nwu;    nwu += nnod;         /* quadrature points */
  zn    = ds->work+nwu;    nwu += nnod;         /* normalized quadrature points */
  w     = ds->work+nwu;    nwu += nnod;         /* quadrature weights */
  Rc    = ds->work+nwu;    nwu += n*p;
  R     = ds->work+nwu;    nwu += n*p;
  alpha = ds->work+nwu;    nwu += mid*n;
  beta  = ds->work+nwu;    nwu += mid*n;
  S     = ds->work+nwu;    nwu += 2*mid*n*p;
  work  = ds->work+nwu;    nwu += mid*n*5;

  /* Compute quadrature parameters */
  ierr = RGComputeQuadrature(ctx->rg,RG_QUADRULE_TRAPEZOIDAL,nnod,z,zn,w);CHKERRQ(ierr);

  /* Set random matrix */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)ds),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rand,0x12345678);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rand);CHKERRQ(ierr);
  for (j=0;j<p;j++)
    for (i=0;i<n;i++) { ierr = PetscRandomGetValue(rand,Rc+i+j*n);CHKERRQ(ierr); }
  ierr = PetscArrayzero(S,2*mid*n*p);CHKERRQ(ierr);
  /* Loop of integration points */
  for (k=kstart;k<kend;k++) {
    ierr = PetscInfo1(NULL,"Solving integration point %D\n",k);CHKERRQ(ierr);
    ierr = PetscArraycpy(R,Rc,p*n);CHKERRQ(ierr);
    ierr = DSNEPComputeMatrix(ds,z[k],PETSC_FALSE,DS_MAT_V);CHKERRQ(ierr);

    /* LU factorization */
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n,&n,V,&ld,perm,&info));
    SlepcCheckLapackInfo("getrf",info);
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N",&n,&p,V,&ld,perm,R,&n,&info));
    SlepcCheckLapackInfo("getrs",info);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);

    /* Moments computation */
    for (s=0;s<2*ctx->max_mid;s++) {
      off = s*n*p;
      for (j=0;j<p;j++)
        for (i=0;i<n;i++) S[off+i+j*n] += w[k]*R[j*n+i];
      w[k] *= zn[k];
    }
  }

  if (ds->pmode==DS_PARALLEL_DISTRIBUTED) {  /* compute final S via reduction */
    ierr = PetscMPIIntCast(2*mid*n*p,&len);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_IN_PLACE)
    ierr = MPIU_Allreduce(MPI_IN_PLACE,S,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
#else
    {
      PetscScalar *buffer;
      ierr = PetscMalloc1(len,&buffer);CHKERRQ(ierr);
      ierr = PetscArraycpy(buffer,S,len);CHKERRQ(ierr);
      ierr = MPIU_Allreduce(buffer,S,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
      ierr = PetscFree(buffer);CHKERRQ(ierr);
    }
#endif
  }
  p = ctx->spls?PetscMin(ctx->spls,n):n;
  pp = p;
  do {
    p = pp;
    ierr = PetscBLASIntCast(mid*p,&colA);CHKERRQ(ierr);

    ierr = PetscInfo2(ds,"Computing SVD of size %Dx%D\n",rowA,colA);CHKERRQ(ierr);
    for (jj=0;jj<mid;jj++) {
      for (ii=0;ii<mid;ii++) {
        off = jj*p*rowA+ii*n;
        for (j=0;j<p;j++)
          for (i=0;i<n;i++) A[off+j*rowA+i] = S[((jj+ii)*n+j)*n+i];
      }
    }
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&rowA,&colA,A,&rowA,sigma,U,&rowA,W,&colA,work,&lwork,rwork,&info));
    SlepcCheckLapackInfo("gesvd",info);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);

    rk = colA;
    for (i=1;i<colA;i++) if (sigma[i]/sigma[0]<PETSC_MACHINE_EPSILON*1e4) {rk = i; break;}
    if (rk<colA || p==n) break;
    pp *= 2;
  } while (pp<=n);
  ierr = PetscInfo1(ds,"Solving generalized eigenproblem of size %D\n",rk);CHKERRQ(ierr);
  for (jj=0;jj<mid;jj++) {
    for (ii=0;ii<mid;ii++) {
      off = jj*p*rowA+ii*n;
      for (j=0;j<p;j++)
        for (i=0;i<n;i++) A[off+j*rowA+i] = S[((jj+ii+1)*n+j)*n+i];
    }
  }
  ierr = PetscBLASIntCast(rk,&rk_);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&rowA,&rk_,&colA,&sone,A,&rowA,W,&colA,&szero,B,&rowA));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&rk_,&rk_,&rowA,&sone,U,&rowA,B,&rowA,&szero,A,&rk_));
  ierr = PetscArrayzero(B,n*mid*n*mid);CHKERRQ(ierr);
  for (j=0;j<rk;j++) B[j+j*rk_] = sigma[j];
  PetscStackCallBLAS("LAPACKggev",LAPACKggev_("N","V",&rk_,A,&rk_,B,&rk_,alpha,beta,NULL,&ld,W,&rk_,work,&lwork,rwork,&info));
  for (i=0;i<rk;i++) wr[i] = (center+radius*alpha[i]/beta[i])*rgscale;
  ierr = PetscMalloc1(rk,&inside);CHKERRQ(ierr);
  ierr = RGCheckInside(ctx->rg,rk,wr,wi,inside);CHKERRQ(ierr);
  k=0;
  for (i=0;i<rk;i++)
    if (inside[i]==1) inside[k++] = i;
  /* Discard values outside region */
  lds = ld*mid;
  ierr = PetscArrayzero(A,lds*lds);CHKERRQ(ierr);
  ierr = PetscArrayzero(B,lds*lds);CHKERRQ(ierr);
  for (i=0;i<k;i++) A[i+i*lds] = (center*beta[inside[i]]+radius*alpha[inside[i]])*rgscale;
  for (i=0;i<k;i++) B[i+i*lds] = beta[inside[i]];
  for (i=0;i<k;i++) wr[i] = A[i+i*lds]/B[i+i*lds];
  for (j=0;j<k;j++) for (i=0;i<rk;i++) W[j*rk+i] = sigma[i]*W[inside[j]*rk+i];
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&k_,&rk_,&sone,U,&rowA,W,&rk_,&szero,X,&ld));

  /* Normalize */
  for (j=0;j<k;j++) {
    norm = BLASnrm2_(&n,X+ld*j,&one);
    for (i=0;i<n;i++) X[ld*j+i] /= norm;
  }
  ierr = PetscFree(inside);CHKERRQ(ierr);
  /* Newton refinement */
  ierr = DSNEPNewtonRefine(ds,k,wr);CHKERRQ(ierr);
  ds->t = k;
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode DSSynchronize_NEP(DS ds,PetscScalar eigr[],PetscScalar eigi[])
{
  PetscErrorCode ierr;
  PetscInt       k=0;
  PetscMPIInt    n,rank,size,off=0;

  PetscFunctionBegin;
  if (ds->state>=DS_STATE_CONDENSED) k += ds->n;
  if (eigr) k += 1;
  if (eigi) k += 1;
  ierr = DSAllocateWork_Private(ds,k,0,0);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(k*sizeof(PetscScalar),&size);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ds),&rank);CHKERRMPI(ierr);
  if (!rank) {
    if (ds->state>=DS_STATE_CONDENSED) {
      ierr = MPI_Pack(ds->mat[DS_MAT_X],n,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Pack(eigr,1,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    if (eigi) {
      ierr = MPI_Pack(eigi,1,MPIU_SCALAR,ds->work,size,&off,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
#endif
  }
  ierr = MPI_Bcast(ds->work,size,MPI_BYTE,0,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
  if (rank) {
    if (ds->state>=DS_STATE_CONDENSED) {
      ierr = MPI_Unpack(ds->work,size,&off,ds->mat[DS_MAT_X],n,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
    if (eigr) {
      ierr = MPI_Unpack(ds->work,size,&off,eigr,1,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    if (eigi) {
      ierr = MPI_Unpack(ds->work,size,&off,eigi,1,MPIU_SCALAR,PetscObjectComm((PetscObject)ds));CHKERRMPI(ierr);
    }
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPSetFN_NEP(DS ds,PetscInt n,FN fn[])
{
  PetscErrorCode ierr;
  DS_NEP         *ctx = (DS_NEP*)ds->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (n<=0) SETERRQ1(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Must have one or more functions, you have %D",n);
  if (n>DS_NUM_EXTRA) SETERRQ2(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Too many functions, you specified %D but the limit is %D",n,DS_NUM_EXTRA);
  if (ds->ld) { ierr = PetscInfo(ds,"DSNEPSetFN() called after DSAllocate()\n");CHKERRQ(ierr); }
  for (i=0;i<n;i++) {
    ierr = PetscObjectReference((PetscObject)fn[i]);CHKERRQ(ierr);
  }
  for (i=0;i<ctx->nf;i++) {
    ierr = FNDestroy(&ctx->f[i]);CHKERRQ(ierr);
  }
  for (i=0;i<n;i++) ctx->f[i] = fn[i];
  ctx->nf = n;
  PetscFunctionReturn(0);
}

/*@
   DSNEPSetFN - Sets a number of functions that define the nonlinear
   eigenproblem.

   Collective on ds

   Input Parameters:
+  ds - the direct solver context
.  n  - number of functions
-  fn - array of functions

   Notes:
   The nonlinear eigenproblem is defined in terms of the split nonlinear
   operator T(lambda) = sum_i A_i*f_i(lambda).

   This function must be called before DSAllocate(). Then DSAllocate()
   will allocate an extra matrix A_i per each function, that can be
   filled in the usual way.

   Level: advanced

.seealso: DSNEPGetFN(), DSAllocate()
 @*/
PetscErrorCode DSNEPSetFN(DS ds,PetscInt n,FN fn[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,n,2);
  PetscValidPointer(fn,3);
  for (i=0;i<n;i++) {
    PetscValidHeaderSpecific(fn[i],FN_CLASSID,3);
    PetscCheckSameComm(ds,1,fn[i],3);
  }
  ierr = PetscTryMethod(ds,"DSNEPSetFN_C",(DS,PetscInt,FN[]),(ds,n,fn));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPGetFN_NEP(DS ds,PetscInt k,FN *fn)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  if (k<0 || k>=ctx->nf) SETERRQ1(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %D",ctx->nf-1);
  *fn = ctx->f[k];
  PetscFunctionReturn(0);
}

/*@
   DSNEPGetFN - Gets the functions associated with the nonlinear DS.

   Not collective, though parallel FNs are returned if the DS is parallel

   Input Parameters:
+  ds - the direct solver context
-  k  - the index of the requested function (starting in 0)

   Output Parameter:
.  fn - the function

   Level: advanced

.seealso: DSNEPSetFN()
@*/
PetscErrorCode DSNEPGetFN(DS ds,PetscInt k,FN *fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(fn,3);
  ierr = PetscUseMethod(ds,"DSNEPGetFN_C",(DS,PetscInt,FN*),(ds,k,fn));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPGetNumFN_NEP(DS ds,PetscInt *n)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  *n = ctx->nf;
  PetscFunctionReturn(0);
}

/*@
   DSNEPGetNumFN - Returns the number of functions stored internally by
   the DS.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
.  n - the number of functions passed in DSNEPSetFN()

   Level: advanced

.seealso: DSNEPSetFN()
@*/
PetscErrorCode DSNEPGetNumFN(DS ds,PetscInt *n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidIntPointer(n,2);
  ierr = PetscUseMethod(ds,"DSNEPGetNumFN_C",(DS,PetscInt*),(ds,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPSetMinimality_NEP(DS ds,PetscInt n)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE || n == PETSC_DEFAULT) ctx->max_mid = 4;
  else {
    if (n<1) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"The minimality value must be > 0");
    ctx->max_mid = n;
  }
  PetscFunctionReturn(0);
}

/*@
   DSNEPSetMinimality - Sets the maximum minimality index used internally by
   the DSNEP.

   Logically Collective on ds

   Input Parameters:
+  ds - the direct solver context
-  n  - the maximum minimality index

   Options Database Key:
.  -ds_nep_minimality <n> - sets the maximum minimality index

   Notes:
   The maximum minimality index is used only in the contour integral method,
   and is related to the highest momemts used in the method. The default
   value is 1, an larger value might give better accuracy in some cases, but
   at a higher cost.

   Level: advanced

.seealso: DSNEPGetMinimality()
@*/
PetscErrorCode DSNEPSetMinimality(DS ds,PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,n,2);
  ierr = PetscTryMethod(ds,"DSNEPSetMinimality_C",(DS,PetscInt),(ds,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPGetMinimality_NEP(DS ds,PetscInt *n)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  *n = ctx->max_mid;
  PetscFunctionReturn(0);
}

/*@
   DSNEPGetMinimality - Returns the maximum minimality index used internally by
   the DSNEP.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
.  n - the maximum minimality index passed in DSNEPSetMinimality()

   Level: advanced

.seealso: DSNEPSetMinimality()
@*/
PetscErrorCode DSNEPGetMinimality(DS ds,PetscInt *n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidIntPointer(n,2);
  ierr = PetscUseMethod(ds,"DSNEPGetMinimality_C",(DS,PetscInt*),(ds,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPSetRefine_NEP(DS ds,PetscReal tol,PetscInt its)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  if (tol == PETSC_DEFAULT) ctx->rtol = PETSC_MACHINE_EPSILON/PetscSqrtReal(PETSC_SQRT_MACHINE_EPSILON);
  else {
    if (tol<=0.0) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"The tolerance must be > 0");
    ctx->rtol = tol;
  }
  if (its == PETSC_DECIDE || its == PETSC_DEFAULT) ctx->Nit = 3;
  else {
    if (its<0) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"The number of iterations must be >= 0");
    ctx->Nit = its;
  }
  PetscFunctionReturn(0);
}

/*@
   DSNEPSetRefine - Sets the tolerance and the number of iterations of Newton iterative
   refinement for eigenpairs.

   Logically Collective on ds

   Input Parameters:
+  ds  - the direct solver context
.  tol - the tolerance
-  its - the number of iterations

   Options Database Key:
+  -ds_nep_refine_tol <tol> - sets the tolerance
-  -ds_nep_refine_its <its> - sets the number of Newton iterations

   Notes:
   Iterative refinement of eigenpairs is currently used only in the contour
   integral method.

   Level: advanced

.seealso: DSNEPGetRefine()
@*/
PetscErrorCode DSNEPSetRefine(DS ds,PetscReal tol,PetscInt its)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ds,tol,2);
  PetscValidLogicalCollectiveInt(ds,its,3);
  ierr = PetscTryMethod(ds,"DSNEPSetRefine_C",(DS,PetscReal,PetscInt),(ds,tol,its));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPGetRefine_NEP(DS ds,PetscReal *tol,PetscInt *its)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  if (tol) *tol = ctx->rtol;
  if (its) *its = ctx->Nit;
  PetscFunctionReturn(0);
}

/*@
   DSNEPGetRefine - Returns the tolerance and the number of iterations of Newton iterative
   refinement for eigenpairs.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
+  tol - the tolerance
-  its - the number of iterations

   Level: advanced

.seealso: DSNEPSetRefine()
@*/
PetscErrorCode DSNEPGetRefine(DS ds,PetscReal *tol,PetscInt *its)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  ierr = PetscUseMethod(ds,"DSNEPGetRefine_C",(DS,PetscReal*,PetscInt*),(ds,tol,its));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPSetIntegrationPoints_NEP(DS ds,PetscInt ip)
{
  DS_NEP         *ctx = (DS_NEP*)ds->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) ctx->nnod = 64;
  else {
    if (ip<1) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"The number of integration points must be > 0");
    ctx->nnod = ip;
  }
  ierr = PetscLayoutDestroy(&ctx->map);CHKERRQ(ierr);  /* need to redistribute at next solve */
  PetscFunctionReturn(0);
}

/*@
   DSNEPSetIntegrationPoints - Sets the number of integration points to be
   used in the contour integral method.

   Logically Collective on ds

   Input Parameters:
+  ds - the direct solver context
-  ip - the number of integration points

   Options Database Key:
.  -ds_nep_integration_points <ip> - sets the number of integration points

   Notes:
   This parameter is relevant only in the contour integral method.

   Level: advanced

.seealso: DSNEPGetIntegrationPoints()
@*/
PetscErrorCode DSNEPSetIntegrationPoints(DS ds,PetscInt ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,ip,2);
  ierr = PetscTryMethod(ds,"DSNEPSetIntegrationPoints_C",(DS,PetscInt),(ds,ip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPGetIntegrationPoints_NEP(DS ds,PetscInt *ip)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  *ip = ctx->nnod;
  PetscFunctionReturn(0);
}

/*@
   DSNEPGetIntegrationPoints - Returns the number of integration points used
   in the contour integral method.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
.  ip - the number of integration points

   Level: advanced

.seealso: DSNEPSetIntegrationPoints()
@*/
PetscErrorCode DSNEPGetIntegrationPoints(DS ds,PetscInt *ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidIntPointer(ip,2);
  ierr = PetscUseMethod(ds,"DSNEPGetIntegrationPoints_C",(DS,PetscInt*),(ds,ip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPSetSamplingSize_NEP(DS ds,PetscInt p)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  if (p == PETSC_DECIDE || p == PETSC_DEFAULT) ctx->spls = 0;
  else {
    if (p<20) SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"The sample size can not be smaller than 20");
    ctx->spls = p;
  }
  PetscFunctionReturn(0);
}

/*@
   DSNEPSetSamplingSize - Sets the number of sampling columns to be
   used in the contour integral method.

   Logically Collective on ds

   Input Parameters:
+  ds - the direct solver context
-  p - the number of columns for the sampling matrix

   Options Database Key:
.  -ds_nep_sampling_size <p> - sets the number of sampling columns

   Notes:
   This parameter is relevant only in the contour integral method.

   Level: advanced

.seealso: DSNEPGetSamplingSize()
@*/
PetscErrorCode DSNEPSetSamplingSize(DS ds,PetscInt p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,p,2);
  ierr = PetscTryMethod(ds,"DSNEPSetSamplingSize_C",(DS,PetscInt),(ds,p));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPGetSamplingSize_NEP(DS ds,PetscInt *p)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  *p = ctx->spls;
  PetscFunctionReturn(0);
}

/*@
   DSNEPGetSamplingSize - Returns the number of sampling columns used
   in the contour integral method.

   Not collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameters:
.  p -  the number of columns for the sampling matrix

   Level: advanced

.seealso: DSNEPSetSamplingSize()
@*/
PetscErrorCode DSNEPGetSamplingSize(DS ds,PetscInt *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidIntPointer(p,2);
  ierr = PetscUseMethod(ds,"DSNEPGetSamplingSize_C",(DS,PetscInt*),(ds,p));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPSetComputeMatrixFunction_NEP(DS ds,PetscErrorCode (*fun)(DS,PetscScalar,PetscBool,DSMatType,void*),void *ctx)
{
  DS_NEP *dsctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  dsctx->computematrix    = fun;
  dsctx->computematrixctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DSNEPSetComputeMatrixFunction - Sets a user-provided subroutine to compute
   the matrices T(lambda) or T'(lambda).

   Logically Collective on ds

   Input Parameters:
+  ds  - the direct solver context
.  fun - a pointer to the user function
-  ctx - a context pointer (the last parameter to the user function)

   Calling Sequence of fun:
$   fun(DS ds,PetscScalar lambda,PetscBool deriv,DSMatType mat,void *ctx)

+   ds     - the direct solver object
.   lambda - point where T(lambda) or T'(lambda) must be evaluated
.   deriv  - if true compute T'(lambda), otherwise compute T(lambda)
.   mat    - the DS matrix where the result must be stored
-   ctx    - optional context, as set by DSNEPSetComputeMatrixFunction()

   Note:
   The result is computed as T(lambda) = sum_i E_i*f_i(lambda), and similarly
   for the derivative.

   Level: developer

.seealso: DSNEPGetComputeMatrixFunction()
@*/
PetscErrorCode DSNEPSetComputeMatrixFunction(DS ds,PetscErrorCode (*fun)(DS,PetscScalar,PetscBool,DSMatType,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  ierr = PetscTryMethod(ds,"DSNEPSetComputeMatrixFunction_C",(DS,PetscErrorCode (*)(DS,PetscScalar,PetscBool,DSMatType,void*),void*),(ds,fun,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPGetComputeMatrixFunction_NEP(DS ds,PetscErrorCode (**fun)(DS,PetscScalar,PetscBool,DSMatType,void*),void **ctx)
{
  DS_NEP *dsctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  if (fun) *fun = dsctx->computematrix;
  if (ctx) *ctx = dsctx->computematrixctx;
  PetscFunctionReturn(0);
}

/*@C
   DSNEPGetComputeMatrixFunction - Returns the user-provided callback function
   set in DSNEPSetComputeMatrixFunction().

   Not Collective

   Input Parameter:
.  ds  - the direct solver context

   Output Parameters:
+  fun - the pointer to the user function
-  ctx - the context pointer

   Level: developer

.seealso: DSNEPSetComputeMatrixFunction()
@*/
PetscErrorCode DSNEPGetComputeMatrixFunction(DS ds,PetscErrorCode (**fun)(DS,PetscScalar,PetscBool,DSMatType,void*),void **ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  ierr = PetscUseMethod(ds,"DSNEPGetComputeMatrixFunction_C",(DS,PetscErrorCode (**)(DS,PetscScalar,PetscBool,DSMatType,void*),void**),(ds,fun,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPSetRG_NEP(DS ds,RG rg)
{
  PetscErrorCode ierr;
  DS_NEP         *dsctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)rg);CHKERRQ(ierr);
  ierr = RGDestroy(&dsctx->rg);CHKERRQ(ierr);
  dsctx->rg = rg;
  ierr = PetscLogObjectParent((PetscObject)ds,(PetscObject)dsctx->rg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DSNEPSetRG - Associates a region object to the DSNEP solver.

   Logically Collective on ds

   Input Parameters:
+  ds  - the direct solver context
-  rg  - the region context

   Notes:
   The region is used only in the contour integral method, and
   should enclose the wanted eigenvalues.

   Level: developer

.seealso: DSNEPGetRG()
@*/
PetscErrorCode DSNEPSetRG(DS ds,RG rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (rg) {
    PetscValidHeaderSpecific(rg,RG_CLASSID,2);
    PetscCheckSameComm(ds,1,rg,2);
  }
  ierr = PetscTryMethod(ds,"DSNEPSetRG_C",(DS,RG),(ds,rg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DSNEPGetRG_NEP(DS ds,RG *rg)
{
  PetscErrorCode ierr;
  DS_NEP         *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  if (!ctx->rg) {
    ierr = RGCreate(PetscObjectComm((PetscObject)ds),&ctx->rg);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->rg,(PetscObject)ds,1);CHKERRQ(ierr);
    ierr = RGSetOptionsPrefix(ctx->rg,((PetscObject)ds)->prefix);CHKERRQ(ierr);
    ierr = RGAppendOptionsPrefix(ctx->rg,"ds_nep_");CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ds,(PetscObject)ctx->rg);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)ctx->rg,((PetscObject)ds)->options);CHKERRQ(ierr);
  }
  *rg = ctx->rg;
  PetscFunctionReturn(0);
}

/*@
   DSNEPGetRG - Obtain the region object associated to the DSNEP solver.

   Not Collective

   Input Parameter:
.  ds  - the direct solver context

   Output Parameter:
.  rg  - the region context

   Level: developer

.seealso: DSNEPSetRG()
@*/
PetscErrorCode DSNEPGetRG(DS ds,RG *rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(rg,2);
  ierr = PetscUseMethod(ds,"DSNEPGetRG_C",(DS,RG*),(ds,rg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSSetFromOptions_NEP(PetscOptionItems *PetscOptionsObject,DS ds)
{
  PetscErrorCode ierr;
  PetscInt       k;
  PetscBool      flg;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      r;
  PetscBool      flg1;
  DS_NEP         *ctx = (DS_NEP*)ds->data;
#endif

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"DS NEP Options");CHKERRQ(ierr);

    ierr = PetscOptionsInt("-ds_nep_minimality","Maximum minimality index","DSNEPSetMinimality",4,&k,&flg);CHKERRQ(ierr);
    if (flg) { ierr = DSNEPSetMinimality(ds,k);CHKERRQ(ierr); }

    ierr = PetscOptionsInt("-ds_nep_integration_points","Number of integration points","DSNEPSetIntegrationPoints",64,&k,&flg);CHKERRQ(ierr);
    if (flg) { ierr = DSNEPSetIntegrationPoints(ds,k);CHKERRQ(ierr); }

    ierr = PetscOptionsInt("-ds_nep_sampling_size","Number of sampling columns","DSNEPSetSamplingSize",0,&k,&flg);CHKERRQ(ierr);
    if (flg) { ierr = DSNEPSetSamplingSize(ds,k);CHKERRQ(ierr); }

#if defined(PETSC_USE_COMPLEX)
    r = ctx->rtol;
    ierr = PetscOptionsReal("-ds_nep_refine_tol","Refinement tolerance","DSNEPSetRefine",ctx->rtol,&r,&flg1);CHKERRQ(ierr);
    k = ctx->Nit;
    ierr = PetscOptionsInt("-ds_nep_refine_its","Number of iterative refinement iterations","DSNEPSetRefine",ctx->Nit,&k,&flg);CHKERRQ(ierr);
    if (flg1||flg) { ierr = DSNEPSetRefine(ds,r,k);CHKERRQ(ierr); }

    if (ds->method==1) {
      if (!ctx->rg) { ierr = DSNEPGetRG(ds,&ctx->rg);CHKERRQ(ierr); }
      ierr = RGSetFromOptions(ctx->rg);CHKERRQ(ierr);
    }
#endif

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSDestroy_NEP(DS ds)
{
  PetscErrorCode ierr;
  DS_NEP         *ctx = (DS_NEP*)ds->data;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0;i<ctx->nf;i++) {
    ierr = FNDestroy(&ctx->f[i]);CHKERRQ(ierr);
  }
  ierr = RGDestroy(&ctx->rg);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&ctx->map);CHKERRQ(ierr);
  ierr = PetscFree(ds->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetFN_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetFN_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetNumFN_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetMinimality_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetMinimality_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetRefine_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetRefine_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetIntegrationPoints_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetIntegrationPoints_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetSamplingSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetSamplingSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetRG_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetRG_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetComputeMatrixFunction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetComputeMatrixFunction_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DSMatGetSize_NEP(DS ds,DSMatType t,PetscInt *rows,PetscInt *cols)
{
  DS_NEP *ctx = (DS_NEP*)ds->data;

  PetscFunctionBegin;
  *rows = ds->n;
  *cols = ds->n;
  if (t==DS_MAT_X || t==DS_MAT_Y) *cols *= ctx->max_mid;
  PetscFunctionReturn(0);
}

/*MC
   DSNEP - Dense Nonlinear Eigenvalue Problem.

   Level: beginner

   Notes:
   The problem is expressed as T(lambda)*x = 0, where T(lambda) is a
   parameter-dependent matrix written as T(lambda) = sum_i E_i*f_i(lambda).
   The eigenvalues lambda are the arguments returned by DSSolve()..

   The coefficient matrices E_i are the extra matrices of the DS, and
   the scalar functions f_i are passed via DSNEPSetFN(). Optionally, a
   callback function to fill the E_i matrices can be set with
   DSNEPSetComputeMatrixFunction().

   Used DS matrices:
+  DS_MAT_Ex - coefficient matrices of the split form of T(lambda)
.  DS_MAT_A  - (workspace) T(lambda) evaluated at a given lambda
.  DS_MAT_B  - (workspace) T'(lambda) evaluated at a given lambda
-  DS_MAT_W  - (workspace) eigenvectors of linearization in SLP

   Implemented methods:
+  0 - Successive Linear Problems (SLP), computes just one eigenpair
-  1 - Contour integral, computes all eigenvalues inside a region

.seealso: DSCreate(), DSSetType(), DSType, DSNEPSetFN(), DSNEPSetComputeMatrixFunction()
M*/
SLEPC_EXTERN PetscErrorCode DSCreate_NEP(DS ds)
{
  DS_NEP         *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ds,&ctx);CHKERRQ(ierr);
  ds->data = (void*)ctx;
  ctx->max_mid = 4;
  ctx->nnod    = 64;
  ctx->Nit     = 3;
  ctx->rtol    = PETSC_MACHINE_EPSILON/PetscSqrtReal(PETSC_SQRT_MACHINE_EPSILON);

  ds->ops->allocate       = DSAllocate_NEP;
  ds->ops->setfromoptions = DSSetFromOptions_NEP;
  ds->ops->view           = DSView_NEP;
  ds->ops->vectors        = DSVectors_NEP;
  ds->ops->solve[0]       = DSSolve_NEP_SLP;
#if defined(PETSC_USE_COMPLEX)
  ds->ops->solve[1]       = DSSolve_NEP_Contour;
#endif
  ds->ops->sort           = DSSort_NEP;
  ds->ops->synchronize    = DSSynchronize_NEP;
  ds->ops->destroy        = DSDestroy_NEP;
  ds->ops->matgetsize     = DSMatGetSize_NEP;

  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetFN_C",DSNEPSetFN_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetFN_C",DSNEPGetFN_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetNumFN_C",DSNEPGetNumFN_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetMinimality_C",DSNEPGetMinimality_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetMinimality_C",DSNEPSetMinimality_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetRefine_C",DSNEPGetRefine_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetRefine_C",DSNEPSetRefine_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetIntegrationPoints_C",DSNEPGetIntegrationPoints_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetIntegrationPoints_C",DSNEPSetIntegrationPoints_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetSamplingSize_C",DSNEPGetSamplingSize_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetSamplingSize_C",DSNEPSetSamplingSize_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetRG_C",DSNEPSetRG_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetRG_C",DSNEPGetRG_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPSetComputeMatrixFunction_C",DSNEPSetComputeMatrixFunction_NEP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ds,"DSNEPGetComputeMatrixFunction_C",DSNEPGetComputeMatrixFunction_NEP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

