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

/*
  1) Patterns of A and B
      PS_STATE_RAW:       PS_STATE_INTERM/CONDENSED
       0       n-1              0       n-1
      -------------            -------------
    0 |* * * * * *|          0 |* * * * * *|
      |* * * * * *|            |  * * * * *|
      |* * * * * *|            |    * * * *|
      |* * * * * *|            |    * * * *|
      |* * * * * *|            |        * *|
  n-1 |* * * * * *|        n-1 |          *|
      -------------            -------------

  2) Moreover, P and Q are assumed to be the identity in PS_STATE_INTERMEDIATE.
*/


static PetscErrorCode PSCleanDenseSchur(PetscInt n,PetscInt k,PetscScalar *S,PetscInt ldS,PetscScalar *T,PetscInt ldT,PetscScalar *X,PetscInt ldX,PetscBool doProd);

#undef __FUNCT__  
#define __FUNCT__ "PSAllocate_GNHEP"
PetscErrorCode PSAllocate_GNHEP(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_B);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Z);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_GNHEP"
PetscErrorCode PSView_GNHEP(PS ps,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSViewMat_Private(ps,viewer,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSViewMat_Private(ps,viewer,PS_MAT_B);CHKERRQ(ierr); 
  if (ps->state>PS_STATE_INTERMEDIATE) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Z);CHKERRQ(ierr); 
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Q);CHKERRQ(ierr); 
  }
  if (ps->mat[PS_MAT_X]) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_X);CHKERRQ(ierr); 
  }
  if (ps->mat[PS_MAT_Y]) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Y);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_GNHEP_Eigen_Some"
PetscErrorCode PSVectors_GNHEP_Eigen_Some(PS ps,PetscInt *k,PetscBool left)
{
#if defined(SLEPC_MISSING_LAPACK_TGEVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n,ld,mout,info,*select,mm;
  PetscScalar    *X,*Y,*A = ps->mat[PS_MAT_A],*B = ps->mat[PS_MAT_B],fone=1.0,fzero=0.0;
  PetscBool      iscomplex = PETSC_FALSE;
  const char     *side;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  if (left) {
    X = PETSC_NULL;
    Y = &ps->mat[PS_MAT_Y][ld*(*k)];
    side = "L";
  } else {
    X = &ps->mat[PS_MAT_X][ld*(*k)];
    Y = PETSC_NULL;
    side = "R";
  }
  ierr = PSAllocateWork_Private(ps,0,0,ld);CHKERRQ(ierr); 
  select = ps->iwork;
  for (i=0;i<n;i++) select[i] = 0;
  select[*k] = 1;
  if (ps->state == PS_STATE_INTERMEDIATE) {
    ierr = PSSetIdentity(ps,PS_MAT_Q);CHKERRQ(ierr);
    ierr = PSSetIdentity(ps,PS_MAT_Z);CHKERRQ(ierr);
  }
  ierr = PSCleanDenseSchur(n,0,A,ld,B,ld,ps->mat[PS_MAT_Q],ld,PETSC_TRUE);CHKERRQ(ierr);
  if (ps->state < PS_STATE_CONDENSED) {
    ierr = PSSetState(ps,PS_STATE_CONDENSED);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  mm = 1;
  ierr = PSAllocateWork_Private(ps,2*ld,2*ld,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,"S",select,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&mm,&mout,ps->work,ps->rwork,&info);
#else
  if ((*k)<n-1 && (A[ld*(*k)+(*k)+1] != 0.0 || B[ld*(*k)+(*k)+1] != 0.0)) iscomplex = PETSC_TRUE;
  mm = iscomplex ? 2 : 1;
  ierr = PSAllocateWork_Private(ps,6*ld,0,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,"S",select,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&mm,&mout,ps->work,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  if (select[(*k)] == 0 || mout != mm) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_SUP,"Unsupported the computation of the second vector in a complex pair");
  /* Backtransform: (X/Y) <- (Q/Z) * (X/Y) */
  ierr = PetscMemcpy(ps->work,left?Y:X,mm*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  BLASgemm_("N","N",&n,&mm,&n,&fone,ps->mat[left?PS_MAT_Z:PS_MAT_Q],&ld,ps->work,&ld,&fzero,left?Y:X,&ld);
  /* Update k to the last vector index in the conjugate pair */
  if (iscomplex) (*k)++;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_GNHEP_Eigen_All"
PetscErrorCode PSVectors_GNHEP_Eigen_All(PS ps,PetscBool left)
{
#if defined(SLEPC_MISSING_LAPACK_TGEVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,mout,info;
  PetscScalar    *X,*Y,*A = ps->mat[PS_MAT_A],*B = ps->mat[PS_MAT_B];
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
  ierr = PSCleanDenseSchur(n,0,A,ld,B,ld,ps->mat[PS_MAT_Q],ld,PETSC_TRUE);CHKERRQ(ierr);
  if (ps->state>=PS_STATE_CONDENSED) {
    /* PSSolve() has been called, backtransform with matrix Q */
    back = "B";
    ierr = PetscMemcpy(left?Y:X,ps->mat[left?PS_MAT_Z:PS_MAT_Q],ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    back = "A";
    ierr = PSSetState(ps,PS_STATE_CONDENSED);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = PSAllocateWork_Private(ps,2*ld,2*ld,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,back,PETSC_NULL,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&n,&mout,ps->work,ps->rwork,&info);
#else
  ierr = PSAllocateWork_Private(ps,6*ld,0,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,back,PETSC_NULL,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&n,&mout,ps->work,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_GNHEP"
PetscErrorCode PSVectors_GNHEP(PS ps,PSMatType mat,PetscInt *k,PetscReal *rnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
  switch (mat) {
    case PS_MAT_X:
    case PS_MAT_Y:
      if (k) {
        ierr = PSVectors_GNHEP_Eigen_Some(ps,k,mat == PS_MAT_Y?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr);
      } else {
        ierr = PSVectors_GNHEP_Eigen_All(ps,mat == PS_MAT_Y?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter"); 
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PSNormalize_GNHEP"
PetscErrorCode PSNormalize_GNHEP(PS ps,PSMatType mat,PetscInt col)
{
  PetscErrorCode ierr;
  PetscInt       i,i0,i1;
  PetscBLASInt   ld,n,one = 1;
  PetscScalar    *A = ps->mat[PS_MAT_A],*B = ps->mat[PS_MAT_B],norm,norm0,*x;

  PetscFunctionBegin;
  if(ps->state < PS_STATE_INTERMEDIATE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported state");
  switch (mat) {
    case PS_MAT_X:
    case PS_MAT_Y:
    case PS_MAT_Q:
    case PS_MAT_Z:
      /* Supported matrices */
      break;
    case PS_MAT_U:
    case PS_MAT_VT:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
      break;
    default:
      SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter"); 
  }

  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  ierr = PSGetArray(ps,mat,&x);CHKERRQ(ierr);
  if (col < 0) {
    i0 = 0; i1 = ps->n;
  } else if(col>0 && (A[ps->ld*(col-1)+col] != 0.0 || (B && B[ps->ld*(col-1)+col] != 0.0))) {
    i0 = col-1; i1 = col+1;
  } else {
    i0 = col; i1 = col+1;
  }
  for(i=i0; i<i1; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if(i<n-1 && (A[ps->ld*i+i+1] != 0.0 || (B && B[ps->ld*i+i+1] != 0.0))) {
      norm = BLASnrm2_(&n,&x[ld*i],&one);
      norm0 = BLASnrm2_(&n,&x[ld*(i+1)],&one);
      norm = 1.0/SlepcAbsEigenvalue(norm,norm0);
      BLASscal_(&n,&norm,&x[ld*i],&one);
      BLASscal_(&n,&norm,&x[ld*(i+1)],&one);
      i++;
    } else
#endif
    {
      norm = BLASnrm2_(&n,&x[ld*i],&one);
      norm = 1.0/norm;
      BLASscal_(&n,&norm,&x[ld*i],&one);
     }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GNHEP_Sort"
/*
  Sort the condensed form at the end of any PSSolve_GNHEP_* method. 
*/
static PetscErrorCode PSSolve_GNHEP_Sort(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_TGEXC) || !defined(PETSC_USE_COMPLEX) && (defined(SLEPC_MISSING_LAPACK_LAMCH) || defined(SLEPC_MISSING_LAPACK_LAG2))
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEXC/LAMCH/LAG2 - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,pos,result;
  PetscBLASInt   ifst,ilst,info,n,ld,one=1;
  PetscScalar    *S = ps->mat[PS_MAT_A],*T = ps->mat[PS_MAT_B],*Z = ps->mat[PS_MAT_Z],*Q = ps->mat[PS_MAT_Q];
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt   lwork;
  PetscScalar    *work,a,safmin,scale1,scale2;
#endif

  PetscFunctionBegin;
  if (!ps->comp_fun) PetscFunctionReturn(0);
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
#if !defined(PETSC_USE_COMPLEX)
  lwork = -1;
  LAPACKtgexc_(&one,&one,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,&one,&one,&a,&lwork,&info);
  safmin = LAPACKlamch_("S");
  lwork = a;
  ierr = PSAllocateWork_Private(ps,lwork,0,0);CHKERRQ(ierr); 
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
      ierr = (*ps->comp_fun)(re,im,wr[j],wi[j],&result,ps->comp_ctx);CHKERRQ(ierr);
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
      LAPACKtgexc_(&one,&one,&n,S,&ld,T,&ld,Z,&ld,Q,&ld,&ifst,&ilst,work,&lwork,&info);
#else
      LAPACKtgexc_(&one,&one,&n,S,&ld,T,&ld,Z,&ld,Q,&ld,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xTGEXC %i",info);
      /* recover original eigenvalues from T and S matrices */
      for (j=i;j<n;j++) {
#if !defined(PETSC_USE_COMPLEX)
        if (j<n-1 && S[j*ld+j+1] != 0.0) {
          /* complex conjugate eigenvalue */
          LAPACKlag2_(S+j*ld+j,&ld,T+j*ld+j,&ld,&safmin,&scale1,&scale2,&re,&a,&im);
          wr[j] = re / scale1;
          wi[j] = im / scale1;
          wr[j+1] = a / scale2;
          wi[j+1] = -wi[j];
          j++;
        } else
#endif
        {
          if (T[j*ld+j] == 0.0) wr[j] = (PetscRealPart(S[j*ld+j])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
          else wr[j] = S[j*ld+j] / T[j*ld+j];
          wi[j] = 0.0;
        }
      }
    }
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0.0) i++;
#endif
  }
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__
#define __FUNCT__ "PSCleanDenseSchur"
/* Write zeros from the column k to n in the lower triangular part of the
   matrices S and T, and inside 2-by-2 diagonal blocks of T in order to
   make (S,T) a valid Schur decompositon.
*/
static PetscErrorCode PSCleanDenseSchur(PetscInt n,PetscInt k,PetscScalar *S,PetscInt ldS,PetscScalar *T,PetscInt ldT,PetscScalar *X,PetscInt ldX,PetscBool doProd)
{
#if defined(SLEPC_MISSING_LAPACK_LASV2)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LASV2 - Lapack routine is unavailable.");
#else
  PetscInt        i, j;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar     s;
#else
  PetscErrorCode  ierr;
#endif

  PetscFunctionBegin;
  PetscValidScalarPointer(S,3);
  if (T) { PetscValidScalarPointer(T,5); }

  if (!doProd && X) {
    for (i=0; i<n; i++) for (j=0; j<n; j++) X[ldX*i+j] = 0.0;
    for (i=0; i<n; i++) X[ldX*i+i] = 1.0;
  }

#if defined(PETSC_USE_COMPLEX)
  for (i=k; i<n; i++) {
    /* Some functions need the diagonal elements in T be real */
    if (T && PetscImaginaryPart(T[ldT*i+i]) != 0.0) {
      s = PetscConj(T[ldT*i+i])/PetscAbsScalar(T[ldT*i+i]);
      for(j=0; j<=i; j++)
        T[ldT*i+j]*= s,
        S[ldS*i+j]*= s;
      T[ldT*i+i] = PetscRealPart(T[ldT*i+i]);
      if (X) for(j=0; j<n; j++) X[ldX*i+j]*= s;
    }
    if ((j=i+1) < n) {
      S[ldS*i+j] = 0.0;
      if (T) T[ldT*i+j] = 0.0;
    }
  }
#else
  for (i=k; i<n; i++) {
    if (S[ldS*i+i+1] != 0.0) {
      if ((j=i+2) < n) S[ldS*(i+1)+j] = 0.0;
      if (T) {
        /* T[ldT*(i+1)+i] = 0.0; */
        {
          /* Check if T(i+1,i) is negligible */
          if (PetscAbs(T[ldT*(i+1)+i])+PetscAbs(T[ldT*i+i+1]) > (PetscAbs(T[ldT*i+i])+PetscAbs(T[ldT*(i+1)+i+1]))*PETSC_MACHINE_EPSILON) {
            PetscBLASInt    ldS_,ldT_,n_i,n_i_1,one=1,n_,i_1,i_;
            PetscScalar     b11,b22,sr,cr,sl,cl;
            ldS_ = PetscBLASIntCast(ldS);
            ldT_ = PetscBLASIntCast(ldT);
            n_i = PetscBLASIntCast(n-i);
            n_i_1 = n_i - 1;
            i_1 = PetscBLASIntCast(i+1);
            i_ = PetscBLASIntCast(i);
            n_ = PetscBLASIntCast(n);
            ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
            LAPACKlasv2_(&T[ldT*i+i],&T[ldT*i+i+1],&T[ldT*(i+1)+i+1],&b22,&b11,&sr,&cr,&sl,&cl);
            ierr = PetscFPTrapPop();CHKERRQ(ierr);
            if (b11 < 0.0) { cr=-cr; sr=-sr; b11=-b11; b22=-b22; }
            BLASrot_(&n_i,&S[ldS*i+i],&ldS_,&S[ldS*i+i+1],&ldS_,&cl,&sl);
            BLASrot_(&i_1,&S[ldS*i],&one,&S[ldS*(i+1)],&one,&cr,&sr);
            if (n_i_1>0) BLASrot_(&n_i_1,&T[ldT*(i+2)+i],&ldT_,&T[ldT*(i+2)+i],&ldT_,&cl,&sl);
            BLASrot_(&i_,&T[ldT*i],&one,&T[ldT*(i+1)],&one,&cr,&sr);
            if (X) BLASrot_(&n_,&X[ldX*i],&one,&X[ldX*(i+1)],&one,&cr,&sr);
            T[ldT*i+i] = b11; T[ldT*i+i+1] = T[ldT*(i+1)+i] = 0.0; T[ldT*(i+1)+i+1] = b22;
          } else {
            T[ldT*(i+1)+i] = T[ldT*i+i+1] = 0.0;
          }
        }
        if ((j=i+1) < n) T[ldT*i+j] = 0.0;
        if ((j=i+2) < n) T[ldT*(i+1)+j] = 0.0;
      }
      i++;
    } else {
      if ((j=i+1) < n) {
        S[ldS*i+j] = 0.0;
        if (T) T[ldT*i+j] = 0.0;
      }
    }
  }
#endif

  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GNHEP"
PetscErrorCode PSSolve_GNHEP(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GGES)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GGES - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    *work,*beta,a;
  PetscInt       i;
  PetscBLASInt   lwork,info,n,ld,iaux;
  PetscScalar    *A = ps->mat[PS_MAT_A],*B = ps->mat[PS_MAT_B],*Z = ps->mat[PS_MAT_Z],*Q = ps->mat[PS_MAT_Q];

  PetscFunctionBegin;
  PetscValidPointer(wi,3);
  n   = PetscBLASIntCast(ps->n);
  ld  = PetscBLASIntCast(ps->ld);
  if (ps->state==PS_STATE_INTERMEDIATE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for the intermediate state");
  lwork = -1;
#if !defined(PETSC_USE_COMPLEX)
  LAPACKgges_("V","V","N",PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&ld,PETSC_NULL,&ld,&a,&lwork,PETSC_NULL,&info);
  lwork = (PetscBLASInt)a;
  ierr = PSAllocateWork_Private(ps,lwork+ld,0,0);CHKERRQ(ierr); 
  beta = ps->work;
  work = beta+ps->n;
  lwork = PetscBLASIntCast(ps->lwork-ps->n);
  LAPACKgges_("V","V","N",PETSC_NULL,&n,A,&ld,B,&ld,&iaux,wr,wi,beta,Z,&ld,Q,&ld,work,&lwork,PETSC_NULL,&info);
#else
  LAPACKgges_("V","V","N",PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&ld,PETSC_NULL,&ld,&a,&lwork,PETSC_NULL,PETSC_NULL,&info);
  lwork = (PetscBLASInt)PetscRealPart(a);
  ierr = PSAllocateWork_Private(ps,lwork+ld,8*ld,0);CHKERRQ(ierr); 
  beta = ps->work;
  work = beta+ps->n;
  lwork = PetscBLASIntCast(ps->lwork-ps->n);
  LAPACKgges_("V","V","N",PETSC_NULL,&n,A,&ld,B,&ld,&iaux,wr,beta,Z,&ld,Q,&ld,work,&lwork,ps->rwork,PETSC_NULL,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xGGES %i",info);
  for (i=0;i<n;i++) {
    if (beta[i]==0.0) wr[i] = (PetscRealPart(wr[i])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wr[i] /= beta[i];
#if !defined(PETSC_USE_COMPLEX)
    if (beta[i]==0.0) wi[i] = (wi[i]>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wi[i] /= beta[i];
#endif
  }
  ierr = PSSolve_GNHEP_Sort(ps,wr,wi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_GNHEP"
PetscErrorCode PSCreate_GNHEP(PS ps)
{
  PetscFunctionBegin;
  ps->ops->allocate      = PSAllocate_GNHEP;
  ps->ops->view          = PSView_GNHEP;
  ps->ops->vectors       = PSVectors_GNHEP;
  ps->ops->solve[0]      = PSSolve_GNHEP;
  ps->ops->normalize     = PSNormalize_GNHEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

