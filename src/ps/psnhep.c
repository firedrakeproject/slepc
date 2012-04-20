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
#define __FUNCT__ "PSAllocate_NHEP"
PetscErrorCode PSAllocate_NHEP(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_NHEP"
PetscErrorCode PSView_NHEP(PS ps,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSViewMat_Private(ps,viewer,PS_MAT_A);CHKERRQ(ierr); 
  if (ps->state>PS_STATE_INTERMEDIATE) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Q);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_NHEP_Eigen_Some"
PetscErrorCode PSVectors_NHEP_Eigen_Some(PS ps,PetscInt *k,PetscReal *rnorm,PetscBool left)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   mm,mout,info,ld,n,inc = 1;
  PetscScalar    tmp,done=1.0,zero=0.0;
  PetscReal      norm;
  PetscBool      iscomplex = PETSC_FALSE;
  PetscBLASInt   *select;
  PetscScalar    *A = ps->mat[PS_MAT_A];
  PetscScalar    *Q = ps->mat[PS_MAT_Q];
  PetscScalar    *X = ps->mat[PS_MAT_X];
  PetscScalar    *Y;

  PetscFunctionBegin;
  if (left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for left eigenvectors");
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  if ((*k)<n-1 && A[(*k)+1+(*k)*ld]!=0.0) iscomplex = PETSC_TRUE;
  ierr = PSAllocateWork_Private(ps,0,0,ld);CHKERRQ(ierr); 
  select = ps->iwork;
  for (i=0;i<n;i++) select[i] = (PetscBLASInt)PETSC_FALSE;

  /* Compute k-th eigenvector Y of A */
  Y = X+(*k)*ld;
  mm = iscomplex? 2: 1;
  select[*k] = (PetscBLASInt)PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  if (iscomplex) select[(*k)+1] = (PetscBLASInt)PETSC_TRUE;
  ierr = PSAllocateWork_Private(ps,3*ld,0,0);CHKERRQ(ierr); 
  LAPACKtrevc_("R","S",select,&n,A,&ld,PETSC_NULL,&ld,Y,&ld,&mm,&mout,ps->work,&info);
#else
  ierr = PSAllocateWork_Private(ps,2*ld,ld,0);CHKERRQ(ierr); 
  LAPACKtrevc_("R","S",select,&n,A,&ld,PETSC_NULL,&ld,Y,&ld,&mm,&mout,ps->work,ps->rwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTREVC %d",info);
  if (mout != mm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Inconsistent arguments");

  /* accumulate and normalize eigenvectors */
  if (ps->state>=PS_STATE_CONDENSED) {
    ierr = PetscMemcpy(ps->work,Y,mout*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    BLASgemv_("N",&n,&n,&done,Q,&ld,ps->work,&inc,&zero,Y,&inc);
#if !defined(PETSC_USE_COMPLEX)
    if (iscomplex) BLASgemv_("N",&n,&n,&done,Q,&ld,ps->work+ld,&inc,&zero,Y+ld,&inc);
#endif
    norm = BLASnrm2_(&n,Y,&inc);
#if !defined(PETSC_USE_COMPLEX)
    tmp  = BLASnrm2_(&n,Y+ld,&inc);
    norm = SlepcAbsEigenvalue(norm,tmp);
#endif
    tmp = 1.0 / norm;
    BLASscal_(&n,&tmp,Y,&inc);
#if !defined(PETSC_USE_COMPLEX)
    BLASscal_(&n,&tmp,Y+ld,&inc);
#endif
  }

  /* set output arguments */
  if (iscomplex) (*k)++;
  if (rnorm) {
    if (iscomplex) *rnorm = SlepcAbsEigenvalue(Y[n-1],Y[n-1+ld]);
    else *rnorm = PetscAbsScalar(Y[n-1]);
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_NHEP_Eigen_All"
PetscErrorCode PSVectors_NHEP_Eigen_All(PS ps,PetscBool left)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,mout,info;
  PetscScalar    *X,*Y,*A = ps->mat[PS_MAT_A];
  const char     *side,*back;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  if (left) {
    X = PETSC_NULL;
    Y = ps->mat[PS_MAT_Y];
    side = "L";
  } else {
    X = ps->mat[PS_MAT_X];
    Y = PETSC_NULL;
    side = "R";
  }
  if (ps->state>=PS_STATE_CONDENSED) {
    /* PSSolve() has been called, backtransform with matrix Q */
    back = "B";
    ierr = PetscMemcpy(left?Y:X,ps->mat[PS_MAT_Q],ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  } else back = "A";
#if !defined(PETSC_USE_COMPLEX)
  ierr = PSAllocateWork_Private(ps,3*ld,0,0);CHKERRQ(ierr); 
  LAPACKtrevc_(side,back,PETSC_NULL,&n,A,&ld,Y,&ld,X,&ld,&n,&mout,ps->work,&info);
#else
  ierr = PSAllocateWork_Private(ps,2*ld,ld,0);CHKERRQ(ierr); 
  LAPACKtrevc_(side,back,PETSC_NULL,&n,A,&ld,Y,&ld,X,&ld,&n,&mout,ps->work,ps->rwork,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_NHEP"
PetscErrorCode PSVectors_NHEP(PS ps,PSMatType mat,PetscInt *k,PetscReal *rnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (mat) {
    case PS_MAT_X:
      if (k) {
        ierr = PSVectors_NHEP_Eigen_Some(ps,k,rnorm,PETSC_FALSE);CHKERRQ(ierr);
      } else {
        ierr = PSVectors_NHEP_Eigen_All(ps,PETSC_FALSE);CHKERRQ(ierr);
      }
      break;
    case PS_MAT_Y:
      if (k) {
        ierr = PSVectors_NHEP_Eigen_Some(ps,k,rnorm,PETSC_TRUE);CHKERRQ(ierr);
      } else {
        ierr = PSVectors_NHEP_Eigen_All(ps,PETSC_TRUE);CHKERRQ(ierr);
      }
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
#define __FUNCT__ "PSSolve_NHEP"
PetscErrorCode PSSolve_NHEP(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GEHRD) || defined(SLEPC_MISSING_LAPACK_ORGHR) || defined(PETSC_MISSING_LAPACK_HSEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEHRD/ORGHR/HSEQR - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    *work,*tau;
  PetscInt       i,j;
  PetscBLASInt   ilo,lwork,info,n,ld;
  PetscScalar    *A = ps->mat[PS_MAT_A];
  PetscScalar    *Q = ps->mat[PS_MAT_Q];

  PetscFunctionBegin;
  PetscValidPointer(wi,2);
  n   = PetscBLASIntCast(ps->n);
  ld  = PetscBLASIntCast(ps->ld);
  ilo = PetscBLASIntCast(ps->l+1);
  ierr = PSAllocateWork_Private(ps,ld+ld*ld,0,0);CHKERRQ(ierr); 
  tau  = ps->work;
  work = ps->work+ld;
  lwork = ld*ld;

  /* initialize orthogonal matrix */
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<n;i++) 
    Q[i+i*ld] = 1.0;
  if (n==1) PetscFunctionReturn(0);    

  /* reduce to upper Hessenberg form */
  if (ps->state<PS_STATE_INTERMEDIATE) {
    LAPACKgehrd_(&n,&ilo,&n,A,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",info);
    for (j=0;j<n-1;j++) {
      for (i=j+2;i<n;i++) {
        Q[i+j*ld] = A[i+j*ld];
        A[i+j*ld] = 0.0;
      }
    }
    LAPACKorghr_(&n,&ilo,&n,Q,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGHR %d",info);
  }

  /* compute the (real) Schur form */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKhseqr_("S","V",&n,&ilo,&n,A,&ld,wr,wi,Q,&ld,work,&lwork,&info);
  for (j=0;j<ps->l;j++) {
    if (j==n-1 || A[j+1+j*ld] == 0.0) { 
      /* real eigenvalue */
      wr[j] = A[j+j*ld];
      wi[j] = 0.0;
    } else {
      /* complex eigenvalue */
      wr[j] = A[j+j*ld];
      wr[j+1] = A[j+j*ld];
      wi[j] = PetscSqrtReal(PetscAbsReal(A[j+1+j*ld])) *
              PetscSqrtReal(PetscAbsReal(A[j+(j+1)*ld]));
      wi[j+1] = -wi[j];
      j++;
    }
  }
#else
  LAPACKhseqr_("S","V",&n,&ilo,&n,A,&ld,wr,Q,&ld,work,&lwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",info);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSSort_NHEP"
PetscErrorCode PSSort_NHEP(PS ps,PetscScalar *wr,PetscScalar *wi,PetscErrorCode (*comp_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void *comp_ctx)
{
#if defined(SLEPC_MISSING_LAPACK_TREXC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TREXC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,pos,result;
  PetscBLASInt   ifst,ilst,info,n,ld;
  PetscScalar    *T = ps->mat[PS_MAT_A];
  PetscScalar    *Q = ps->mat[PS_MAT_Q];
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *work;
#endif

  PetscFunctionBegin;
  PetscValidPointer(wi,2);
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PSAllocateWork_Private(ps,ld,0,0);CHKERRQ(ierr); 
  work = ps->work;
#endif
  /* selection sort */
  for (i=ps->l;i<n-1;i++) {
    re = wr[i];
    im = wi[i];
    pos = 0;
    j=i+1; /* j points to the next eigenvalue */
#if !defined(PETSC_USE_COMPLEX)
    if (im != 0) j=i+2;
#endif
    /* find minimum eigenvalue */
    for (;j<n;j++) { 
      ierr = (*comp_func)(re,im,wr[j],wi[j],&result,comp_ctx);CHKERRQ(ierr);
      if (result > 0) {
        re = wr[j];
        im = wi[j];
        pos = j;
      }
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) j++;
#endif
    }
    if (pos) {
      /* interchange blocks */
      ifst = PetscBLASIntCast(pos+1);
      ilst = PetscBLASIntCast(i+1);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKtrexc_("V",&n,T,&ld,Q,&ld,&ifst,&ilst,work,&info);
#else
      LAPACKtrexc_("V",&n,T,&ld,Q,&ld,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
      /* recover original eigenvalues from T matrix */
      for (j=i;j<n;j++) {
        wr[j] = T[j+j*ld];
#if !defined(PETSC_USE_COMPLEX)
        if (j<n-1 && T[j+1+j*ld] != 0.0) {
          /* complex conjugate eigenvalue */
          wi[j] = PetscSqrtReal(PetscAbsReal(T[j+1+j*ld])) *
                  PetscSqrtReal(PetscAbsReal(T[j+(j+1)*ld]));
          wr[j+1] = wr[j];
          wi[j+1] = -wi[j];
          j++;
        } else
#endif
        wi[j] = 0.0;
      }
    }
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0) i++;
#endif
  }
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "PSCond_NHEP"
PetscErrorCode PSCond_NHEP(PS ps,PetscReal *cond)
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

  /* use workspace matrix W to avoid overwriting A */
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  A = ps->mat[PS_MAT_W];
  ierr = PetscMemcpy(A,ps->mat[PS_MAT_A],sizeof(PetscScalar)*ps->ld*ps->ld);CHKERRQ(ierr);

  /* norm of A */
  if (ps->state<PS_STATE_INTERMEDIATE) hn = LAPACKlange_("I",&n,&n,A,&ld,rwork);
  else hn = LAPACKlanhs_("I",&n,A,&ld,rwork);

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
#define __FUNCT__ "PSTranslateHarmonic_NHEP"
PetscErrorCode PSTranslateHarmonic_NHEP(PS ps,PetscScalar tau,PetscReal beta,PetscBool recover,PetscScalar *gin,PetscReal *gamma)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(PETSC_MISSING_LAPACK_GETRS) 
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF/GETRS - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   *ipiv,info,n,ld,one=1,ncol;
  PetscScalar    *A,*B,*Q,*g=gin,*ghat; 
  PetscScalar    done=1.0,dmone=-1.0,dzero=0.0;
  PetscReal      gnorm;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  A  = ps->mat[PS_MAT_A];

  if (!recover) {

    ierr = PSAllocateWork_Private(ps,0,0,ld);CHKERRQ(ierr); 
    ipiv = ps->iwork;
    if (!g) {
      ierr = PSAllocateWork_Private(ps,ld,0,0);CHKERRQ(ierr); 
      g = ps->work;
    }
    /* use workspace matrix W to factor A-tau*eye(n) */
    ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
    B = ps->mat[PS_MAT_W];
    ierr = PetscMemcpy(B,A,sizeof(PetscScalar)*ld*ld);CHKERRQ(ierr);

    /* Vector g initialy stores b = beta*e_n^T */
    ierr = PetscMemzero(g,n*sizeof(PetscScalar));CHKERRQ(ierr);
    g[n-1] = beta;
 
    /* g = (A-tau*eye(n))'\b */
    for (i=0;i<n;i++) 
      B[i+i*ld] -= tau;
    LAPACKgetrf_(&n,&n,B,&ld,ipiv,&info);
    if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
    if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");
    ierr = PetscLogFlops(2.0*n*n*n/3.0);CHKERRQ(ierr);
    LAPACKgetrs_("C",&n,&one,B,&ld,ipiv,g,&ld,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");
    ierr = PetscLogFlops(2.0*n*n-n);CHKERRQ(ierr);

    /* A = A + g*b' */
    for (i=0;i<n;i++) 
      A[i+(n-1)*ld] += g[i]*beta;

  } else { /* recover */

    PetscValidPointer(g,6);
    ierr = PSAllocateWork_Private(ps,ld,0,0);CHKERRQ(ierr); 
    ghat = ps->work;
    Q    = ps->mat[PS_MAT_Q];

    /* g^ = -Q(:,idx)'*g */
    ncol = PetscBLASIntCast(ps->l+ps->k);
    BLASgemv_("C",&n,&ncol,&dmone,Q,&ld,g,&one,&dzero,ghat,&one);

    /* A = A + g^*b' */
    for (i=0;i<ps->l+ps->k;i++)
      for (j=ps->l;j<ps->l+ps->k;j++)
        A[i+j*ld] += ghat[i]*Q[n-1+j*ld]*beta;

    /* g~ = (I-Q(:,idx)*Q(:,idx)')*g = g+Q(:,idx)*g^ */
    BLASgemv_("N",&n,&ncol,&done,Q,&ld,ghat,&one,&done,g,&one);
  }

  /* Compute gamma factor */
  if (gamma) {
    gnorm = 0.0;
    for (i=0;i<n;i++)
      gnorm = gnorm + PetscRealPart(g[i]*PetscConj(g[i]));
    *gamma = PetscSqrtReal(1.0+gnorm);
  }
  PetscFunctionReturn(0);
#endif
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_NHEP"
PetscErrorCode PSCreate_NHEP(PS ps)
{
  PetscFunctionBegin;
  ps->nmeth  = 1;
  ps->ops->allocate      = PSAllocate_NHEP;
  ps->ops->view          = PSView_NHEP;
  ps->ops->vectors       = PSVectors_NHEP;
  ps->ops->solve         = PSSolve_NHEP;
  ps->ops->sort          = PSSort_NHEP;
  ps->ops->cond          = PSCond_NHEP;
  ps->ops->translate     = PSTranslateHarmonic_NHEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

