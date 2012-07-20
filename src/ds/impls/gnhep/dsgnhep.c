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

#include <slepc-private/dsimpl.h>      /*I "slepcds.h" I*/
#include <slepcblaslapack.h>

/*
  1) Patterns of A and B
      DS_STATE_RAW:       DS_STATE_INTERM/CONDENSED
       0       n-1              0       n-1
      -------------            -------------
    0 |* * * * * *|          0 |* * * * * *|
      |* * * * * *|            |  * * * * *|
      |* * * * * *|            |    * * * *|
      |* * * * * *|            |    * * * *|
      |* * * * * *|            |        * *|
  n-1 |* * * * * *|        n-1 |          *|
      -------------            -------------

  2) Moreover, P and Q are assumed to be the identity in DS_STATE_INTERMEDIATE.
*/


static PetscErrorCode CleanDenseSchur(PetscInt n,PetscInt k,PetscScalar *S,PetscInt ldS,PetscScalar *T,PetscInt ldT,PetscScalar *X,PetscInt ldX,PetscScalar *Y,PetscInt ldY,PetscBool doProd);

#undef __FUNCT__  
#define __FUNCT__ "DSAllocate_GNHEP"
PetscErrorCode DSAllocate_GNHEP(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr); 
  ierr = DSAllocateMat_Private(ds,DS_MAT_B);CHKERRQ(ierr); 
  ierr = DSAllocateMat_Private(ds,DS_MAT_Z);CHKERRQ(ierr); 
  ierr = DSAllocateMat_Private(ds,DS_MAT_Q);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSView_GNHEP"
PetscErrorCode DSView_GNHEP(DS ds,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSViewMat_Private(ds,viewer,DS_MAT_A);CHKERRQ(ierr); 
  ierr = DSViewMat_Private(ds,viewer,DS_MAT_B);CHKERRQ(ierr); 
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ierr = DSViewMat_Private(ds,viewer,DS_MAT_Z);CHKERRQ(ierr); 
    ierr = DSViewMat_Private(ds,viewer,DS_MAT_Q);CHKERRQ(ierr); 
  }
  if (ds->mat[DS_MAT_X]) {
    ierr = DSViewMat_Private(ds,viewer,DS_MAT_X);CHKERRQ(ierr); 
  }
  if (ds->mat[DS_MAT_Y]) {
    ierr = DSViewMat_Private(ds,viewer,DS_MAT_Y);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSVectors_GNHEP_Eigen_Some"
PetscErrorCode DSVectors_GNHEP_Eigen_Some(DS ds,PetscInt *k,PetscBool left)
{
#if defined(SLEPC_MISSING_LAPACK_TGEVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n,ld,mout,info,*select,mm;
  PetscScalar    *X,*Y,*A = ds->mat[DS_MAT_A],*B = ds->mat[DS_MAT_B],fone=1.0,fzero=0.0;
  PetscBool      iscomplex = PETSC_FALSE;
  const char     *side;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ds->n);
  ld = PetscBLASIntCast(ds->ld);
  if (left) {
    X = PETSC_NULL;
    Y = &ds->mat[DS_MAT_Y][ld*(*k)];
    side = "L";
  } else {
    X = &ds->mat[DS_MAT_X][ld*(*k)];
    Y = PETSC_NULL;
    side = "R";
  }
  ierr = DSAllocateWork_Private(ds,0,0,ld);CHKERRQ(ierr); 
  select = ds->iwork;
  for (i=0;i<n;i++) select[i] = 0;
  select[*k] = 1;
  if (ds->state == DS_STATE_INTERMEDIATE) {
    ierr = DSSetIdentity(ds,DS_MAT_Q);CHKERRQ(ierr);
    ierr = DSSetIdentity(ds,DS_MAT_Z);CHKERRQ(ierr);
  }
  ierr = CleanDenseSchur(n,0,A,ld,B,ld,ds->mat[DS_MAT_Q],ld,ds->mat[DS_MAT_Z],ld,PETSC_TRUE);CHKERRQ(ierr);
  if (ds->state < DS_STATE_CONDENSED) {
    ierr = DSSetState(ds,DS_STATE_CONDENSED);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  mm = 1;
  ierr = DSAllocateWork_Private(ds,2*ld,2*ld,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,"S",select,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&mm,&mout,ds->work,ds->rwork,&info);
#else
  if ((*k)<n-1 && (A[ld*(*k)+(*k)+1] != 0.0 || B[ld*(*k)+(*k)+1] != 0.0)) iscomplex = PETSC_TRUE;
  mm = iscomplex ? 2 : 1;
  ierr = DSAllocateWork_Private(ds,6*ld,0,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,"S",select,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&mm,&mout,ds->work,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  if (select[(*k)] == 0 || mout != mm) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_SUP,"Unsupported the computation of the second vector in a complex pair");
  /* Backtransform: (X/Y) <- (Q/Z) * (X/Y) */
  ierr = PetscMemcpy(ds->work,left?Y:X,mm*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  BLASgemm_("N","N",&n,&mm,&n,&fone,ds->mat[left?DS_MAT_Z:DS_MAT_Q],&ld,ds->work,&ld,&fzero,left?Y:X,&ld);
  /* Update k to the last vector index in the conjugate pair */
  if (iscomplex) (*k)++;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "DSVectors_GNHEP_Eigen_All"
PetscErrorCode DSVectors_GNHEP_Eigen_All(DS ds,PetscBool left)
{
#if defined(SLEPC_MISSING_LAPACK_TGEVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,mout,info;
  PetscScalar    *X,*Y,*A = ds->mat[DS_MAT_A],*B = ds->mat[DS_MAT_B];
  const char     *side,*back;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ds->n);
  ld = PetscBLASIntCast(ds->ld);
  if (left) {
    X = PETSC_NULL;
    Y = ds->mat[DS_MAT_Y];
    side = "L";
  } else {
    X = ds->mat[DS_MAT_X];
    Y = PETSC_NULL;
    side = "R";
  }
  ierr = CleanDenseSchur(n,0,A,ld,B,ld,ds->mat[DS_MAT_Q],ld,ds->mat[DS_MAT_Z],ld,PETSC_TRUE);CHKERRQ(ierr);
  if (ds->state>=DS_STATE_CONDENSED) {
    /* DSSolve() has been called, backtransform with matrix Q */
    back = "B";
    ierr = PetscMemcpy(left?Y:X,ds->mat[left?DS_MAT_Z:DS_MAT_Q],ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    back = "A";
    ierr = DSSetState(ds,DS_STATE_CONDENSED);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = DSAllocateWork_Private(ds,2*ld,2*ld,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,back,PETSC_NULL,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&n,&mout,ds->work,ds->rwork,&info);
#else
  ierr = DSAllocateWork_Private(ds,6*ld,0,0);CHKERRQ(ierr); 
  LAPACKtgevc_(side,back,PETSC_NULL,&n,A,&ld,B,&ld,Y,&ld,X,&ld,&n,&mout,ds->work,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "DSVectors_GNHEP"
PetscErrorCode DSVectors_GNHEP(DS ds,DSMatType mat,PetscInt *k,PetscReal *rnorm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rnorm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
  switch (mat) {
    case DS_MAT_X:
    case DS_MAT_Y:
      if (k) {
        ierr = DSVectors_GNHEP_Eigen_Some(ds,k,mat == DS_MAT_Y?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr);
      } else {
        ierr = DSVectors_GNHEP_Eigen_All(ds,mat == DS_MAT_Y?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter"); 
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DSNormalize_GNHEP"
PetscErrorCode DSNormalize_GNHEP(DS ds,DSMatType mat,PetscInt col)
{
  PetscErrorCode ierr;
  PetscInt       i,i0,i1;
  PetscBLASInt   ld,n,one = 1;
  PetscScalar    *A = ds->mat[DS_MAT_A],*B = ds->mat[DS_MAT_B],norm,*x;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    norm0;
#endif

  PetscFunctionBegin;
  if(ds->state < DS_STATE_INTERMEDIATE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported state");
  switch (mat) {
    case DS_MAT_X:
    case DS_MAT_Y:
    case DS_MAT_Q:
    case DS_MAT_Z:
      /* Supported matrices */
      break;
    case DS_MAT_U:
    case DS_MAT_VT:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
      break;
    default:
      SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter"); 
  }

  n  = PetscBLASIntCast(ds->n);
  ld = PetscBLASIntCast(ds->ld);
  ierr = DSGetArray(ds,mat,&x);CHKERRQ(ierr);
  if (col < 0) {
    i0 = 0; i1 = ds->n;
  } else if(col>0 && (A[ds->ld*(col-1)+col] != 0.0 || (B && B[ds->ld*(col-1)+col] != 0.0))) {
    i0 = col-1; i1 = col+1;
  } else {
    i0 = col; i1 = col+1;
  }
  for(i=i0; i<i1; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if(i<n-1 && (A[ds->ld*i+i+1] != 0.0 || (B && B[ds->ld*i+i+1] != 0.0))) {
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
#define __FUNCT__ "DSSort_GNHEP"
PetscErrorCode DSSort_GNHEP(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_TGEXC) || !defined(PETSC_USE_COMPLEX) && (defined(SLEPC_MISSING_LAPACK_LAMCH) || defined(SLEPC_MISSING_LAPACK_LAG2))
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TGEXC/LAMCH/LAG2 - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,pos,result;
  PetscBLASInt   ifst,ilst,info,n,ld,one=1;
  PetscScalar    *S = ds->mat[DS_MAT_A],*T = ds->mat[DS_MAT_B],*Z = ds->mat[DS_MAT_Z],*Q = ds->mat[DS_MAT_Q];
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt   lwork;
  PetscScalar    *work,a,safmin,scale1,scale2;
#endif

  PetscFunctionBegin;
  if (!ds->comp_fun) PetscFunctionReturn(0);
  n  = PetscBLASIntCast(ds->n);
  ld = PetscBLASIntCast(ds->ld);
#if !defined(PETSC_USE_COMPLEX)
  lwork = -1;
  LAPACKtgexc_(&one,&one,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,&one,&one,&a,&lwork,&info);
  safmin = LAPACKlamch_("S");
  lwork = a;
  ierr = DSAllocateWork_Private(ds,lwork,0,0);CHKERRQ(ierr); 
  work = ds->work;
#endif
  /* selection sort */
  for (i=ds->l;i<n-1;i++) {
    re = wr[i];
    im = wi[i];
    pos = 0;
    j=i+1; /* j points to the next eigenvalue */
#if !defined(PETSC_USE_COMPLEX)
    if (im != 0) j=i+2;
#endif
    /* find minimum eigenvalue */
    for (;j<n;j++) { 
      ierr = (*ds->comp_fun)(re,im,wr[j],wi[j],&result,ds->comp_ctx);CHKERRQ(ierr);
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
      if (info) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_LIB,"Error in Lapack xTGEXC %i",info);
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
#define __FUNCT__ "CleanDenseSchur"
/*
   Write zeros from the column k to n in the lower triangular part of the
   matrices S and T, and inside 2-by-2 diagonal blocks of T in order to
   make (S,T) a valid Schur decompositon.
*/
static PetscErrorCode CleanDenseSchur(PetscInt n,PetscInt k,PetscScalar *S,PetscInt ldS,PetscScalar *T,PetscInt ldT,PetscScalar *X,PetscInt ldX,PetscScalar *Y,PetscInt ldY,PetscBool doProd)
{
#if defined(SLEPC_MISSING_LAPACK_LASV2)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LASV2 - Lapack routine is unavailable.");
#else
  PetscInt       i,j;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    s;
#else
  PetscBLASInt   ldS_,ldT_,n_i,n_i_2,one=1,n_,i_2,i_;
  PetscScalar    b11,b22,sr,cr,sl,cl;
#endif

  PetscFunctionBegin;
  if (!doProd && X) {
    for (i=0;i<n;i++) for (j=0;j<n;j++) X[ldX*i+j] = 0.0;
    for (i=0;i<n;i++) X[ldX*i+i] = 1.0;
  }
  if (!doProd && Y) {
    for (i=0;i<n;i++) for (j=0;j<n;j++) Y[ldY*i+j] = 0.0;
    for (i=0;i<n;i++) Y[ldX*i+i] = 1.0;
  }

#if defined(PETSC_USE_COMPLEX)
  for (i=k; i<n; i++) {
    /* Some functions need the diagonal elements in T be real */
    if (T && PetscImaginaryPart(T[ldT*i+i]) != 0.0) {
      s = PetscConj(T[ldT*i+i])/PetscAbsScalar(T[ldT*i+i]);
      for(j=0;j<=i;j++) {
        T[ldT*i+j] *= s;
        S[ldS*i+j] *= s;
      }
      T[ldT*i+i] = PetscRealPart(T[ldT*i+i]);
      if (X) for(j=0;j<n;j++) X[ldX*i+j] *= s;
    }
    j = i+1;
    if (j<n) {
      S[ldS*i+j] = 0.0;
      if (T) T[ldT*i+j] = 0.0;
    }
  }
#else
  ldS_ = PetscBLASIntCast(ldS);
  ldT_ = PetscBLASIntCast(ldT);
  n_   = PetscBLASIntCast(n);
  for (i=k;i<n-1;i++) {
    if (S[ldS*i+i+1] != 0.0) {
      /* Check if T(i+1,i) and T(i,i+1) are zero */
      if (T[ldT*(i+1)+i] != 0.0 || T[ldT*i+i+1] != 0.0) {
        /* Check if T(i+1,i) and T(i,i+1) are negligible */
        if (PetscAbs(T[ldT*(i+1)+i])+PetscAbs(T[ldT*i+i+1]) < (PetscAbs(T[ldT*i+i])+PetscAbs(T[ldT*(i+1)+i+1]))*PETSC_MACHINE_EPSILON) {
          T[ldT*i+i+1] = 0.0;
          T[ldT*(i+1)+i] = 0.0;

        } else {
          /* If one of T(i+1,i) or T(i,i+1) is negligible, we make zero the other element */
          if (PetscAbs(T[ldT*i+i+1]) < (PetscAbs(T[ldT*i+i])+PetscAbs(T[ldT*(i+1)+i+1])+PetscAbs(T[ldT*(i+1)+i]))*PETSC_MACHINE_EPSILON) {
            LAPACKlasv2_(&T[ldT*i+i],&T[ldT*(i+1)+i],&T[ldT*(i+1)+i+1],&b22,&b11,&sl,&cl,&sr,&cr);
          } else if (PetscAbs(T[ldT*(i+1)+i]) < (PetscAbs(T[ldT*i+i])+PetscAbs(T[ldT*(i+1)+i+1])+PetscAbs(T[ldT*i+i+1]))*PETSC_MACHINE_EPSILON) {
            LAPACKlasv2_(&T[ldT*i+i],&T[ldT*i+i+1],&T[ldT*(i+1)+i+1],&b22,&b11,&sr,&cr,&sl,&cl);
          } else {
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported format. Call DSSolve before this function.");
          }
          n_i = PetscBLASIntCast(n-i);
          n_i_2 = n_i - 2;
          i_2 = PetscBLASIntCast(i+2);
          i_ = PetscBLASIntCast(i);
          if (b11 < 0.0) { cr=-cr; sr=-sr; b11=-b11; b22=-b22; }
          BLASrot_(&n_i,&S[ldS*i+i],&ldS_,&S[ldS*i+i+1],&ldS_,&cl,&sl);
          BLASrot_(&i_2,&S[ldS*i],&one,&S[ldS*(i+1)],&one,&cr,&sr);
          BLASrot_(&n_i_2,&T[ldT*(i+2)+i],&ldT_,&T[ldT*(i+2)+i+1],&ldT_,&cl,&sl);
          BLASrot_(&i_,&T[ldT*i],&one,&T[ldT*(i+1)],&one,&cr,&sr);
          if (X) BLASrot_(&n_,&X[ldX*i],&one,&X[ldX*(i+1)],&one,&cr,&sr);
          if (Y) BLASrot_(&n_,&Y[ldY*i],&one,&X[ldY*(i+1)],&one,&cl,&sl);
          T[ldT*i+i] = b11;
          T[ldT*i+i+1] = 0.0;
          T[ldT*(i+1)+i] = 0.0;
          T[ldT*(i+1)+i+1] = b22;
        }
      }
    i++;
    }
  }
#endif
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "DSSolve_GNHEP"
PetscErrorCode DSSolve_GNHEP(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GGES)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GGES - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    *work,*beta,a;
  PetscInt       i;
  PetscBLASInt   lwork,info,n,ld,iaux;
  PetscScalar    *A = ds->mat[DS_MAT_A],*B = ds->mat[DS_MAT_B],*Z = ds->mat[DS_MAT_Z],*Q = ds->mat[DS_MAT_Q];

  PetscFunctionBegin;
  PetscValidPointer(wi,3);
  n   = PetscBLASIntCast(ds->n);
  ld  = PetscBLASIntCast(ds->ld);
  if (ds->state==DS_STATE_INTERMEDIATE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for the intermediate state");
  lwork = -1;
#if !defined(PETSC_USE_COMPLEX)
  LAPACKgges_("V","V","N",PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&ld,PETSC_NULL,&ld,&a,&lwork,PETSC_NULL,&info);
  lwork = (PetscBLASInt)a;
  ierr = DSAllocateWork_Private(ds,lwork+ld,0,0);CHKERRQ(ierr); 
  beta = ds->work;
  work = beta+ds->n;
  lwork = PetscBLASIntCast(ds->lwork-ds->n);
  LAPACKgges_("V","V","N",PETSC_NULL,&n,A,&ld,B,&ld,&iaux,wr,wi,beta,Z,&ld,Q,&ld,work,&lwork,PETSC_NULL,&info);
#else
  LAPACKgges_("V","V","N",PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,&ld,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&ld,PETSC_NULL,&ld,&a,&lwork,PETSC_NULL,PETSC_NULL,&info);
  lwork = (PetscBLASInt)PetscRealPart(a);
  ierr = DSAllocateWork_Private(ds,lwork+ld,8*ld,0);CHKERRQ(ierr); 
  beta = ds->work;
  work = beta+ds->n;
  lwork = PetscBLASIntCast(ds->lwork-ds->n);
  LAPACKgges_("V","V","N",PETSC_NULL,&n,A,&ld,B,&ld,&iaux,wr,beta,Z,&ld,Q,&ld,work,&lwork,ds->rwork,PETSC_NULL,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_LIB,"Error in Lapack xGGES %i",info);
  for (i=0;i<n;i++) {
    if (beta[i]==0.0) wr[i] = (PetscRealPart(wr[i])>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wr[i] /= beta[i];
#if !defined(PETSC_USE_COMPLEX)
    if (beta[i]==0.0) wi[i] = (wi[i]>0.0)? PETSC_MAX_REAL: PETSC_MIN_REAL;
    else wi[i] /= beta[i];
#endif
  }
  PetscFunctionReturn(0);
#endif
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "DSCreate_GNHEP"
PetscErrorCode DSCreate_GNHEP(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate      = DSAllocate_GNHEP;
  ds->ops->view          = DSView_GNHEP;
  ds->ops->vectors       = DSVectors_GNHEP;
  ds->ops->solve[0]      = DSSolve_GNHEP;
  ds->ops->sort          = DSSort_GNHEP;
  ds->ops->normalize     = DSNormalize_GNHEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END

