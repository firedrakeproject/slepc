/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/psimpl.h>      /*I "slepcps.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "PSAllocate_HEP"
PetscErrorCode PSAllocate_HEP(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  ierr = PSAllocateMatReal_Private(ps,PS_MAT_T);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

/*   0       l           k                 n
    -----------------------------------------
    |*       ·           ·                  |
    |  *     ·           ·                  |
    |    *   ·           ·                  |
    |      * ·           ·                  |
    |· · · · o           o                  |
    |          o         o                  |
    |            o       o                  |
    |              o     o                  |
    |                o   o                  |
    |                  o o                  |
    |· · · · o o o o o o o x                |
    |                    x x x              |
    |                      x x x            |
    |                        x x x          |
    |                          x x x        |
    |                            x x x      |
    |                              x x x    |
    |                                x x x  |
    |                                  x x x|
    |                                    x x|
    -----------------------------------------
*/

#undef __FUNCT__  
#define __FUNCT__ "PSSwitchFormat_HEP"
PetscErrorCode PSSwitchFormat_HEP(PS ps,PetscBool tocompact)
{
  PetscReal   *T = ps->rmat[PS_MAT_T];
  PetscScalar *A = ps->mat[PS_MAT_A];
  PetscInt    i,n=ps->n,k=ps->k,ld=ps->ld;

  PetscFunctionBegin;
  if (ps->compact==tocompact) PetscFunctionReturn(0);
  if (tocompact) { /* switch from dense (arrow) to compact storage */
    for (i=0;i<k;i++) {
      T[i] = PetscRealPart(A[i+i*ld]);
      T[i+ld] = PetscRealPart(A[k+i*ld]);
    }
    for (i=k;i<n;i++) {
      T[i] = PetscRealPart(A[i+i*ld]);
      T[i+ld] = PetscRealPart(A[i+1+i*ld]);
    }
  } else { /* switch from compact (arrow) to dense storage */
    for (i=0;i<k;i++) {
      A[i+i*ld] = T[i];
      A[k+i*ld] = T[i+ld];
      A[i+k*ld] = T[i+ld];
    }
    A[k+k*ld] = T[k];
    for (i=k+1;i<n;i++) {
      A[i+i*ld] = T[i];
      A[i-1+i*ld] = T[i-1+ld];
      A[i+(i-1)*ld] = T[i-1+ld];
    } 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_HEP"
PetscErrorCode PSView_HEP(PS ps,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscInt          i,j,r,c;
  PetscReal         value;
  const char        *meth[] = { "LAPACK's _steqr",
                                "LAPACK's _stevr" };

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"solving the problem with: %s\n",meth[ps->method]);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (ps->compact) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",ps->n,ps->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",3*ps->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<ps->n;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,*(ps->rmat[PS_MAT_T]+i));CHKERRQ(ierr);
      }
      for (i=0;i<ps->n-1;i++) {
        r = PetscMax(i+2,ps->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",r,c,*(ps->rmat[PS_MAT_T]+ps->ld+i));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,*(ps->rmat[PS_MAT_T]+ps->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",PSMatName[PS_MAT_T]);CHKERRQ(ierr);
    } else {
      for (i=0;i<ps->n;i++) {
        for (j=0;j<ps->n;j++) {
          if (i==j) value = *(ps->rmat[PS_MAT_T]+i);
          else if ((i<ps->k && j==ps->k) || (i==ps->k && j<ps->k)) value = *(ps->rmat[PS_MAT_T]+ps->ld+PetscMin(i,j));
          else if (i==j+1 && i>ps->k) value = *(ps->rmat[PS_MAT_T]+ps->ld+i-1);
          else if (i+1==j && j>ps->k) value = *(ps->rmat[PS_MAT_T]+ps->ld+j-1);
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",value);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_A);CHKERRQ(ierr); 
  }
  if (ps->state>PS_STATE_INTERMEDIATE) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Q);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_HEP"
PetscErrorCode PSVectors_HEP(PS ps,PSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscScalar    *Q = ps->mat[PS_MAT_Q];
  PetscInt       ld = ps->ld;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ps->state<PS_STATE_CONDENSED) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSSolve() first");
  switch (mat) {
    case PS_MAT_X:
    case PS_MAT_Y:
      if (j) {
        ierr = PetscMemcpy(ps->mat[mat]+(*j)*ld,Q+(*j)*ld,ld*sizeof(PetscScalar));CHKERRQ(ierr);
      } else {
        ierr = PetscMemcpy(ps->mat[mat],Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      if (rnorm) *rnorm = PetscAbsScalar(Q[ps->n-1+(*j)*ld]);
      break;
    case PS_MAT_U:
    case PS_MAT_VT:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
      break;
    default:
      SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_HEP"
PetscErrorCode PSSolve_HEP(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_SYTRD) || defined(SLEPC_MISSING_LAPACK_ORGTR) || defined(SLEPC_MISSING_LAPACK_STEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYTRD/ORGTR/STEQR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   n1,n2,n3,lwork,liwork,info,l,n,m,ld,off,il,iu,*isuppz;
  PetscScalar    *A,*S,*Q,*work,*tau;
  PetscReal      *d,*e,abstol=0.0,vl,vu;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  l  = PetscBLASIntCast(ps->l);
  ld = PetscBLASIntCast(ps->ld);
  n1 = PetscBLASIntCast(ps->k-l+1);  /* size of leading block, excluding locked */
  n2 = PetscBLASIntCast(n-ps->k-1);  /* size of trailing block */
  n3 = n1+n2;
  off = l+l*ld;
  Q  = ps->mat[PS_MAT_Q];
  d  = ps->rmat[PS_MAT_T];
  e  = ps->rmat[PS_MAT_T]+ld;
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<n;i++) Q[i+i*ld] = 1.0;

  if (ps->compact) {

    /* reduce to tridiagonal form */
    if (ps->state<PS_STATE_INTERMEDIATE) {

      ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
      S = ps->mat[PS_MAT_W];
      ierr = PetscMemzero(S,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PSAllocateWork_Private(ps,ld+ld*ld,0,0);CHKERRQ(ierr); 
      tau  = ps->work;
      work = ps->work+ld;
      lwork = ld*ld;

      /* Flip matrix S */
      for (i=l;i<n;i++) S[(n3-1-i+l)+(n3-1-i+l)*ld] = d[i];
      for (i=l;i<ps->k;i++) S[(n3-1-i+l)+(n3-1-ps->k+l)*ld] = e[i];
      for (i=ps->k;i<n-1;i++) S[(n3-1-i+l)+(n3-1-i-1+l)*ld] = e[i];

      /* Reduce (2,2)-block of flipped S to tridiagonal form */
      LAPACKsytrd_("L",&n1,S+n2+n2*ld,&ld,d+l,e+l,tau,work,&lwork,&info);
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);

      /* Flip back diag and subdiag, put them in d and e */
      for (i=0;i<n1-1;i++) {
        d[l+n1-i-1] = PetscRealPart(S[n2+i+(n2+i)*ld]);
        e[l+n1-i-2] = PetscRealPart(S[n2+i+1+(n2+i)*ld]);
      }
      d[l] = PetscRealPart(S[n3-1+(n3-1)*ld]);

      /* Compute the orthogonal matrix used for tridiagonalization */
      LAPACKorgtr_("L",&n1,S+n2+n2*ld,&ld,tau,work,&lwork,&info);
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);

      /* Create full-size Q, flipped back to original order */
      for (i=0;i<n1;i++) 
        for (j=0;j<n1;j++) 
          Q[l+i+(l+j)*ld] = S[n3-i-1+(n3-j-1)*ld];

    }

  } else {

    A  = ps->mat[PS_MAT_A];
    for (i=0;i<l;i++) { d[i] = PetscRealPart(A[i+i*ld]); e[i] = 0.0; }

    if (ps->state<PS_STATE_INTERMEDIATE) {
      /* reduce to tridiagonal form */
      for (j=0;j<n3;j++) {
        ierr = PetscMemcpy(Q+off+j*ld,A+off+j*ld,n3*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      ierr = PSAllocateWork_Private(ps,ld+ld*ld,0,0);CHKERRQ(ierr); 
      tau  = ps->work;
      work = ps->work+ld;
      lwork = ld*ld;
      LAPACKsytrd_("L",&n3,Q+off,&ld,d+l,e+l,tau,work,&lwork,&info);
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);
      LAPACKorgtr_("L",&n3,Q+off,&ld,tau,work,&lwork,&info);
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);
    } else {
      /* copy tridiagonal to d,e */
      for (i=l;i<n;i++) d[i] = PetscRealPart(A[i+i*ld]);
      for (i=l;i<n-1;i++) e[i] = PetscRealPart(A[(i+1)+i*ld]);
    }
  }

  /* Solve the tridiagonal eigenproblem */
  switch (ps->method) {
    case 0:
      ierr = PSAllocateWork_Private(ps,0,2*ld,0);CHKERRQ(ierr); 
      LAPACKsteqr_("V",&n3,d+l,e+l,Q+off,&ld,ps->rwork,&info);
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSTEQR %d",info);
      break;
    case 1:
#if defined(PETSC_USE_COMPLEX)
      ierr = PSAllocateMatReal_Private(ps,PS_MAT_Q);CHKERRQ(ierr);
#endif
      ierr = PSAllocateWork_Private(ps,0,lwork,liwork+2*ld);CHKERRQ(ierr); 
      isuppz = ps->iwork+liwork;
#if defined(PETSC_USE_COMPLEX)
      LAPACKstevr_("V","A",&n3,d+l,e+l,&vl,&vu,&il,&iu,&abstol,&m,wr+l,ps->rmat[PS_MAT_Q]+off,&ld,isuppz,ps->rwork,&lwork,ps->iwork,&liwork,&info);
#else
      LAPACKstevr_("V","A",&n3,d+l,e+l,&vl,&vu,&il,&iu,&abstol,&m,wr+l,Q+off,&ld,isuppz,ps->rwork,&lwork,ps->iwork,&liwork,&info);
#endif
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DSTEVR %d",info);
#if defined(PETSC_USE_COMPLEX)
    for (i=0;i<n;i++) 
      for (j=0;j<n;j++)
        V[i+j*ld] = (ps->rmat[PS_MAT_Q])[i+j*ld];
#endif
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong value of method: %d",ps->method);
  }
  for (i=0;i<n;i++) wr[i] = d[i];
  if (ps->compact) {
    ierr = PetscMemzero(e,(n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  } else {
    ierr = PetscMemzero(A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n;i++) A[i+i*ld] = d[i];
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSSort_HEP"
PetscErrorCode PSSort_HEP(PS ps,PetscScalar *wr,PetscScalar *wi,PetscErrorCode (*comp_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void *comp_ctx)
{
  PetscErrorCode ierr;
  PetscInt       n,l,i,*perm;
  PetscScalar    *A;
  PetscReal      *d;

  PetscFunctionBegin;
  n = ps->n;
  l = ps->l;
  d = ps->rmat[PS_MAT_T];
  ierr = PSAllocateWork_Private(ps,0,0,ps->ld);CHKERRQ(ierr); 
  perm = ps->iwork;
  ierr = PSSortEigenvaluesReal_Private(ps,l,n,d,perm,comp_func,comp_ctx);CHKERRQ(ierr);
  for (i=l;i<n;i++) wr[i] = d[perm[i]];
  ierr = PSPermuteColumns_Private(ps,l,n,PS_MAT_Q,perm);CHKERRQ(ierr);
  if (ps->compact) {
    for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);
  } else {
    A  = ps->mat[PS_MAT_A];
    for (i=l;i<n;i++) A[i+i*ps->ld] = wr[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSCond_HEP"
PetscErrorCode PSCond_HEP(PS ps,PetscReal *cond)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(SLEPC_MISSING_LAPACK_GETRI) || defined(SLEPC_MISSING_LAPACK_LANGE) || defined(SLEPC_MISSING_LAPACK_LANHS)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF/GETRI/LANGE/LANHS - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    *work;
  PetscReal      *rwork;
  PetscBLASInt   *ipiv;
  PetscBLASInt   lwork,info,n,ld;
  PetscReal      hn,hin;
  PetscScalar    *A;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  lwork = 8*ld;
  ierr = PSAllocateWork_Private(ps,lwork,ld,ld);CHKERRQ(ierr); 
  work  = ps->work;
  rwork = ps->rwork;
  ipiv  = ps->iwork;
  ierr = PSSwitchFormat_HEP(ps,PETSC_FALSE);CHKERRQ(ierr);

  /* use workspace matrix W to avoid overwriting A */
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  A = ps->mat[PS_MAT_W];
  ierr = PetscMemcpy(A,ps->mat[PS_MAT_A],sizeof(PetscScalar)*ps->ld*ps->ld);CHKERRQ(ierr);

  /* norm of A */
  hn = LAPACKlange_("I",&n,&n,A,&ld,rwork);

  /* norm of inv(A) */
  LAPACKgetrf_(&n,&n,A,&ld,ipiv,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGETRF %d",info);
  LAPACKgetri_(&n,A,&ld,ipiv,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGETRI %d",info);
  hin = LAPACKlange_("I",&n,&n,A,&ld,rwork);

  *cond = hn*hin;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSTranslateRKS_HEP"
PetscErrorCode PSTranslateRKS_HEP(PS ps,PetscScalar alpha)
{
#if defined(PETSC_MISSING_LAPACK_GEQRF) || defined(SLEPC_MISSING_LAPACK_ORGQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEQRF/ORGQR - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,k=ps->k;
  PetscScalar    *Q,*A,*R,*tau,*work;
  PetscBLASInt   ld,n1,n0,lwork,info;

  PetscFunctionBegin;
  ld = PetscBLASIntCast(ps->ld);
  ierr = PSAllocateWork_Private(ps,ld*ld,0,0);CHKERRQ(ierr);
  tau = ps->work;
  work = ps->work+ld;
  lwork = PetscBLASIntCast(ld*(ld-1));
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  A  = ps->mat[PS_MAT_A];
  Q  = ps->mat[PS_MAT_Q];
  R  = ps->mat[PS_MAT_W];
  /* Copy I+alpha*A */
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(R,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    Q[i+i*ld] = 1.0 + alpha*A[i+i*ld];
    Q[k+i*ld] = alpha*A[k+i*ld];
  }
  /* Compute qr */
  n1 = PetscBLASIntCast(k+1);
  n0 = PetscBLASIntCast(k);
  LAPACKgeqrf_(&n1,&n0,Q,&ld,tau,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  /* Copy R from Q */
  for (j=0;j<k;j++)
    for(i=0;i<=j;i++)
      R[i+j*ld] = Q[i+j*ld];
  /* Compute orthogonal matrix in Q */
  LAPACKorgqr_(&n1,&n1,&n0,Q,&ld,tau,work,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
  /* Compute the updated matrix of projected problem */
  for(j=0;j<k;j++){
    for(i=0;i<k+1;i++)
      A[j*ld+i] = Q[i*ld+j];
  }
  alpha = -1.0/alpha;
  BLAStrsm_("R","U","N","N",&n1,&n0,&alpha,R,&ld,A,&ld);
  for(i=0;i<k;i++)
    A[ld*i+i]-=alpha;
  PetscFunctionReturn(0);
#endif
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_HEP"
PetscErrorCode PSCreate_HEP(PS ps)
{
  PetscFunctionBegin;
  ps->nmeth  = 2;
  ps->ops->allocate      = PSAllocate_HEP;
  ps->ops->view          = PSView_HEP;
  ps->ops->vectors       = PSVectors_HEP;
  ps->ops->solve         = PSSolve_HEP;
  ps->ops->sort          = PSSort_HEP;
  ps->ops->cond          = PSCond_HEP;
  ps->ops->transrks      = PSTranslateRKS_HEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

