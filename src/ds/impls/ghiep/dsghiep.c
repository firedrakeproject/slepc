/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#undef __FUNCT__  
#define __FUNCT__ "DSAllocate_GHIEP"
PetscErrorCode DSAllocate_GHIEP(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr); 
  ierr = DSAllocateMat_Private(ds,DS_MAT_B);CHKERRQ(ierr); 
  ierr = DSAllocateMat_Private(ds,DS_MAT_Q);CHKERRQ(ierr); 
  ierr = DSAllocateMatReal_Private(ds,DS_MAT_T);CHKERRQ(ierr); 
  ierr = DSAllocateMatReal_Private(ds,DS_MAT_D);CHKERRQ(ierr);
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc(ld*sizeof(PetscInt),&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ds,ld*sizeof(PetscInt));CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSwitchFormat_GHIEP"
PetscErrorCode DSSwitchFormat_GHIEP(DS ds,PetscBool tocompact)
{
  PetscErrorCode ierr;
  PetscReal      *T,*S;
  PetscScalar    *A,*B;
  PetscInt       i,n,ld;

  PetscFunctionBegin;
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  T = ds->rmat[DS_MAT_T];
  S = ds->rmat[DS_MAT_D];
  n = ds->n;
  ld = ds->ld;
  if (tocompact) { /* switch from dense (arrow) to compact storage */
    ierr = PetscMemzero(T,3*ld*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(S,ld*sizeof(PetscReal));CHKERRQ(ierr);
    for (i=0;i<n-1;i++) {
      T[i] = PetscRealPart(A[i+i*ld]);
      T[ld+i] = PetscRealPart(A[i+1+i*ld]);
      S[i] = PetscRealPart(B[i+i*ld]);
    }
    T[n-1] = PetscRealPart(A[n-1+(n-1)*ld]);
    S[n-1] = PetscRealPart(B[n-1+(n-1)*ld]);
    for (i=ds->l;i< ds->k;i++) T[2*ld+i] = PetscRealPart(A[ds->k+i*ld]); 
  }else { /* switch from compact (arrow) to dense storage */
    ierr = PetscMemzero(A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(B,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n-1;i++) {
      A[i+i*ld] = T[i];
      A[i+1+i*ld] = T[ld+i];
      A[i+(i+1)*ld] = T[ld+i];
      B[i+i*ld] = S[i];
    }
    A[n-1+(n-1)*ld] = T[n-1];
    B[n-1+(n-1)*ld] = S[n-1];
    for (i=ds->l;i<ds->k;i++) {
      A[ds->k+i*ld] = T[2*ld+i];
      A[i+ds->k*ld] = T[2*ld+i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSView_GHIEP"
PetscErrorCode DSView_GHIEP(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscInt          i,j;
  PetscReal         value;
  const char *methodname[] = {
                     "HR method",
                     "QR + Inverse Iteration",
                     "QR",
                     "DQDS + Inverse Iteration "
  };

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"solving the problem with: %s\n",methodname[ds->method]);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (ds->compact) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",ds->n,ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",3*ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<ds->n;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,*(ds->rmat[DS_MAT_T]+i));CHKERRQ(ierr);
      }
      for (i=0;i<ds->n-1;i++) {
        if (*(ds->rmat[DS_MAT_T]+ds->ld+i) !=0 && i!=ds->k-1) {
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+2,i+1,*(ds->rmat[DS_MAT_T]+ds->ld+i));CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+2,*(ds->rmat[DS_MAT_T]+ds->ld+i));CHKERRQ(ierr);
        }
      }
      for (i = ds->l;i<ds->k;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",ds->k+1,i+1,*(ds->rmat[DS_MAT_T]+2*ds->ld+i));CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,ds->k+1,*(ds->rmat[DS_MAT_T]+2*ds->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_A]);CHKERRQ(ierr);
      
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",ds->n,ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"omega = zeros(%D,3);\n",3*ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"omega = [\n");CHKERRQ(ierr);
      for (i=0;i<ds->n;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,*(ds->rmat[DS_MAT_D]+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(omega);\n",DSMatName[DS_MAT_B]);CHKERRQ(ierr);

    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"T\n");CHKERRQ(ierr);
      for (i=0;i<ds->n;i++) {
        for (j=0;j<ds->n;j++) {
          if (i==j) value = *(ds->rmat[DS_MAT_T]+i);
          else if (i==j+1 || j==i+1) value = *(ds->rmat[DS_MAT_T]+ds->ld+PetscMin(i,j));
          else if ((i<ds->k && j==ds->k) || (i==ds->k && j<ds->k)) value = *(ds->rmat[DS_MAT_T]+2*ds->ld+PetscMin(i,j));
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",value);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"omega\n");CHKERRQ(ierr);
      for (i=0;i<ds->n;i++) {
        for (j=0;j<ds->n;j++) {
          if (i==j) value = *(ds->rmat[DS_MAT_D]+i);
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",value);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    ierr = DSViewMat_Private(ds,viewer,DS_MAT_A);CHKERRQ(ierr);
    ierr = DSViewMat_Private(ds,viewer,DS_MAT_B);CHKERRQ(ierr);
  }
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ierr = DSViewMat_Private(ds,viewer,DS_MAT_Q);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSVectors_GHIEP_Eigen_Some"
static PetscErrorCode DSVectors_GHIEP_Eigen_Some(DS ds,PetscInt *idx,PetscReal *rnorm)
{
  PetscErrorCode ierr;
  PetscReal      b[4],M[4],d1,d2,s1,s2,e;
  PetscReal      scal1,scal2,wr1,wr2,wi,ep,norm;
  PetscScalar    *Q,*X,Y[4],alpha,zeroS = 0.0;
  PetscInt       k;
  PetscBLASInt   two = 2,n_,ld,one=1; 
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt   four=4;
#endif
  
  PetscFunctionBegin;
  X = ds->mat[DS_MAT_X];
  Q = ds->mat[DS_MAT_Q];
  k = *idx;
  n_ = PetscBLASIntCast(ds->n);
  ld = PetscBLASIntCast(ds->ld);
  if (k < ds->n-1) {
   e = (ds->compact)?*(ds->rmat[DS_MAT_T]+ld+k):PetscRealPart(*(ds->mat[DS_MAT_A]+(k+1)+ld*k));
  } else e = 0.0;
  if (e == 0.0) {/* Real */
     if (ds->state>=DS_STATE_CONDENSED) {
       ierr = PetscMemcpy(X+k*ld,Q+k*ld,ld*sizeof(PetscScalar));CHKERRQ(ierr);
     } else {
       ierr = PetscMemzero(X+k*ds->ld,ds->ld*sizeof(PetscScalar));
       X[k+k*ds->ld] = 1.0;
     }
     if (rnorm) {
       *rnorm = PetscAbsScalar(X[ds->n-1+k*ld]);
     }
  } else { /* 2x2 block */
    if (ds->compact) {
      s1 = *(ds->rmat[DS_MAT_D]+k);
      d1 = *(ds->rmat[DS_MAT_T]+k);
      s2 = *(ds->rmat[DS_MAT_D]+k+1);
      d2 = *(ds->rmat[DS_MAT_T]+k+1);
    } else {
      s1 = PetscRealPart(*(ds->mat[DS_MAT_B]+k*ld+k));
      d1 = PetscRealPart(*(ds->mat[DS_MAT_A]+k+k*ld));
      s2 = PetscRealPart(*(ds->mat[DS_MAT_B]+(k+1)*ld+k+1));
      d2 = PetscRealPart(*(ds->mat[DS_MAT_A]+k+1+(k+1)*ld));
    }
    M[0] = d1; M[1] = e; M[2] = e; M[3]= d2;
    b[0] = s1; b[1] = 0.0; b[2] = 0.0; b[3] = s2;
    ep = LAPACKlamch_("S");
    /* Compute eigenvalues of the block */
    LAPACKlag2_(M, &two, b, &two, &ep, &scal1, &scal2, &wr1, &wr2, &wi);
    if (wi==0.0) { /* Real eigenvalues */
      SETERRQ(PETSC_COMM_SELF,1,"Real block in DSVectors_GHIEP");
    } else { /* Complex eigenvalues */
      if (scal1<ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Nearly infinite eigenvalue");
      wr1 /= scal1; wi /= scal1;
#if !defined(PETSC_USE_COMPLEX)
      if ( SlepcAbs(s1*d1-wr1,wi)<SlepcAbs(s2*d2-wr1,wi)) { 
        Y[0] = wr1-s2*d2; Y[1] = s2*e; Y[2] = wi; Y[3] = 0.0;
      } else { 
        Y[0] = s1*e; Y[1] = wr1-s1*d1; Y[2] = 0.0; Y[3] = wi;
      }
      norm = BLASnrm2_(&four,Y,&one);
      norm = 1/norm;
      if (ds->state >= DS_STATE_CONDENSED) {
        alpha = norm;
        BLASgemm_("N","N",&n_,&two,&two,&alpha,ds->mat[DS_MAT_Q]+k*ld,&ld,Y,&two,&zeroS,X+k*ld,&ld);
        if (rnorm) *rnorm = SlepcAbsEigenvalue(X[ds->n-1+k*ld],X[ds->n-1+(k+1)*ld]);
      } else {
        ierr = PetscMemzero(X+k*ld,2*ld*sizeof(PetscScalar));CHKERRQ(ierr);
        X[k*ld+k] = Y[0]*norm; X[k*ld+k+1] = Y[1]*norm;
        X[(k+1)*ld+k] = Y[2]*norm; X[(k+1)*ld+k+1] = Y[3]*norm;
      }
#else
      if ( SlepcAbs(s1*d1-wr1,wi)<SlepcAbs(s2*d2-wr1,wi)) { 
        Y[0] = wr1-s2*d2+PETSC_i*wi; Y[1] = s2*e;
      } else { 
        Y[0] = s1*e; Y[1] = wr1-s1*d1+PETSC_i*wi;
      }
      norm = BLASnrm2_(&two,Y,&one);
      norm = 1/norm;
      if (ds->state >= DS_STATE_CONDENSED) {
        alpha = norm;
        BLASgemv_("N",&n_,&two,&alpha,ds->mat[DS_MAT_Q]+k*ld,&ld,Y,&one,&zeroS,X+k*ld,&one);
        if (rnorm) *rnorm = PetscAbsScalar(X[ds->n-1+k*ld]);
      } else {
        ierr = PetscMemzero(X+k*ld,2*ld*sizeof(PetscScalar));CHKERRQ(ierr);
        X[k*ld+k] = Y[0]*norm; X[k*ld+k+1] = Y[1]*norm;
      }
      X[(k+1)*ld+k] = PetscConj(X[k*ld+k]); X[(k+1)*ld+k+1] = PetscConj(X[k*ld+k+1]);
#endif
      (*idx)++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSVectors_GHIEP"
PetscErrorCode DSVectors_GHIEP(DS ds,DSMatType mat,PetscInt *k,PetscReal *rnorm)
{
  PetscInt       i;
  PetscReal      e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_X:
      if (k) {
        ierr = DSVectors_GHIEP_Eigen_Some(ds,k,rnorm);CHKERRQ(ierr);
      } else {
        for (i=0; i<ds->n; i++) {
          e = (ds->compact)?*(ds->rmat[DS_MAT_T]+ds->ld+i):PetscRealPart(*(ds->mat[DS_MAT_A]+(i+1)+ds->ld*i));
          if (e == 0.0) {/* real */
            if (ds->state >= DS_STATE_CONDENSED) {
              ierr = PetscMemcpy(ds->mat[mat]+i*ds->ld,ds->mat[DS_MAT_Q]+i*ds->ld,ds->ld*sizeof(PetscScalar));CHKERRQ(ierr);
            } else {
              ierr = PetscMemzero(ds->mat[mat]+i*ds->ld,ds->ld*sizeof(PetscScalar));CHKERRQ(ierr);
              *(ds->mat[mat]+i+i*ds->ld) = 1.0;
            }
          } else {
            ierr = DSVectors_GHIEP_Eigen_Some(ds,&i,rnorm);CHKERRQ(ierr);
          }
        }
      }
      break;
    case DS_MAT_Y:
    case DS_MAT_U:
    case DS_MAT_VT:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
      break;
    default:
      SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSGHIEPComplexEigs"
/*
  Extract the eigenvalues contained in the block-diagonal of the indefinite problem.
  Only the index range n0..n1 is processed.
*/
PetscErrorCode DSGHIEPComplexEigs(DS ds, PetscInt n0, PetscInt n1, PetscScalar *wr, PetscScalar *wi)
{
  PetscInt     k,ld;
  PetscBLASInt two=2;
  PetscScalar  *A,*B;
  PetscReal    *D,*T;
  PetscReal    b[4],M[4],d1,d2,s1,s2,e;
  PetscReal    scal1,scal2,ep,wr1,wr2,wi1;

  PetscFunctionBegin;
  ld = ds->ld;
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  D = ds->rmat[DS_MAT_D];
  T = ds->rmat[DS_MAT_T];
  for (k=n0;k<n1;k++) {
    if (k < n1-1) {
      e = (ds->compact)?T[ld+k]:PetscRealPart(A[(k+1)+ld*k]);
    }else e = 0.0;
    if (e==0.0) { 
      /* real eigenvalue */
      wr[k] = (ds->compact)?T[k]/D[k]:A[k+k*ld]/B[k+k*ld];
      wi[k] = 0.0 ;
    } else {
      /* diagonal block */
      if (ds->compact) {
        s1 = D[k];
        d1 = T[k];
        s2 = D[k+1];
        d2 = T[k+1];
      } else {
        s1 = PetscRealPart(B[k*ld+k]);
        d1 = PetscRealPart(A[k+k*ld]);
        s2 = PetscRealPart(B[(k+1)*ld+k+1]);
        d2 = PetscRealPart(A[k+1+(k+1)*ld]);
      }
      M[0] = d1; M[1] = e; M[2] = e; M[3]= d2;
      b[0] = s1; b[1] = 0.0; b[2] = 0.0; b[3] = s2;
      ep = LAPACKlamch_("S");
      /* Compute eigenvalues of the block */
      LAPACKlag2_(M, &two, b, &two, &ep, &scal1, &scal2, &wr1, &wr2, &wi1);
      if (scal1<ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Nearly infinite eigenvalue");
      wr[k] = wr1/scal1;
      if (wi1==0.0) { /* Real eigenvalues */
        if (scal2<ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Nearly infinite eigenvalue");
        wr[k+1] = wr2/scal2;
        wi[k] = 0.0;
        wi[k+1] = 0.0;
      } else { /* Complex eigenvalues */
#if !defined(PETSC_USE_COMPLEX)
        wr[k+1] = wr[k];
        wi[k] = wi1/scal1;
        wi[k+1] = -wi[k];
#else
        wr[k] += PETSC_i*wi1/scal1;
        wr[k+1] = PetscConj(wr[k]);
        wi[k] = 0.0;
        wi[k+1] = 0.0;
#endif
      }
      k++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSort_GHIEP"
PetscErrorCode DSSort_GHIEP(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;
  PetscInt       n,i,*perm;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
  n = ds->n;
  d = ds->rmat[DS_MAT_T];
  e = d + ds->ld;
  s = ds->rmat[DS_MAT_D];
  ierr = DSAllocateWork_Private(ds,ds->ld,ds->ld,0);CHKERRQ(ierr); 
  perm = ds->perm;
  if (!rr) {
    rr = wr;
    ri = wi;
  }
  ierr = DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_TRUE);CHKERRQ(ierr);
  if (!ds->compact) {ierr = DSSwitchFormat_GHIEP(ds,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = PetscMemcpy(ds->work,wr,n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=ds->l;i<n;i++) {
    wr[i] = *(ds->work + perm[i]);
  }
  ierr = PetscMemcpy(ds->work,wi,n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=ds->l;i<n;i++) {
    wi[i] = *(ds->work + perm[i]);
  }
  ierr = PetscMemcpy(ds->rwork,s,n*sizeof(PetscReal));CHKERRQ(ierr);
  for (i=ds->l;i<n;i++) {
    s[i] = *(ds->rwork+perm[i]);
  } 
  ierr = PetscMemcpy(ds->rwork,d,n*sizeof(PetscReal));CHKERRQ(ierr);
  for (i=ds->l;i<n;i++) {
    d[i] = *(ds->rwork  + perm[i]);
  }
  ierr = PetscMemcpy(ds->rwork,e,(n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(e+ds->l,(n-1-ds->l)*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=ds->l;i<n-1;i++) {
    if (perm[i]<n-1) e[i] = *(ds->rwork + perm[i]);
  }
  if (!ds->compact) { ierr = DSSwitchFormat_GHIEP(ds,PETSC_FALSE);CHKERRQ(ierr);}
  ierr = DSPermuteColumns_Private(ds,ds->l,n,DS_MAT_Q,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "HRGen"
/*
  Generates a hyperbolic rotation
    if x1*x1 - x2*x2 != 0 
      r = sqrt( |x1*x1 - x2*x2| )
      c = x1/r  s = x2/r
     
      | c -s||x1|   |d*r|
      |-s  c||x2| = | 0 | 
      where d = 1 for type==1 and -1 for type==2
  Returns the condition number of the reduction
*/
static PetscErrorCode HRGen(PetscReal x1,PetscReal x2,PetscInt *type,PetscReal *c,PetscReal *s,PetscReal *r,PetscReal *cond)
{
  PetscReal t,n2,xa,xb;
  PetscInt  type_;

  PetscFunctionBegin;
  if (x2==0) {
    *r = PetscAbsReal(x1);
    *c = (x1>=0)?1.0:-1.0;
    *s = 0.0;
    if (type) *type = 1;
    PetscFunctionReturn(0);
  }
  if (PetscAbsReal(x1) == PetscAbsReal(x2)) {
    /* hyperbolic rotation doesn't exist */
    *c = 0;
    *s = 0;
    *r = 0;
    if (type) *type = 0;
    *cond = PETSC_MAX_REAL;
    PetscFunctionReturn(0);
  }
  
  if (PetscAbsReal(x1)>PetscAbsReal(x2)) {
    xa = x1; xb = x2; type_ = 1;
  } else {
    xa = x2; xb = x1; type_ = 2;
  } 
  t = xb/xa;
  n2 = PetscAbsReal(1 - t*t);
  *r = PetscSqrtReal(n2)*PetscAbsReal(xa);
  *c = x1/(*r);
  *s = x2/(*r);
  if (type_ == 2) *r *= -1;
  if (type) *type = type_;
  if (cond) *cond = (PetscAbsReal(*c) + PetscAbsReal(*s))/PetscAbsReal(PetscAbsReal(*c) - PetscAbsReal(*s));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "HRApply"
/*
                                |c  s|
  Applies an hyperbolic rotator |s  c|
           |c  s|
    [x1 x2]|s  c| 
*/
PetscErrorCode HRApply(PetscInt n, PetscScalar *x1,PetscInt inc1, PetscScalar *x2, PetscInt inc2,PetscReal c, PetscReal s)
{
  PetscInt    i;
  PetscReal   t;
  PetscScalar tmp;
  
  PetscFunctionBegin;
  if (PetscAbsReal(c)>PetscAbsReal(s)) { /* Type I */
    t = s/c;
    for (i=0;i<n;i++) {
      x1[i*inc1] = c*x1[i*inc1] + s*x2[i*inc2];
      x2[i*inc2] = t*x1[i*inc1] + x2[i*inc2]/c;
    }
  } else { /* Type II */
    t = c/s;
    for (i=0;i<n;i++) {
      tmp = x1[i*inc1];
      x1[i*inc1] = c*x1[i*inc1] + s*x2[i*inc2];
      x2[i*inc2] = t*x1[i*inc1] + tmp/s;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TridiagDiag_HHR"
/*
  Reduction to tridiagonal-diagonal form (see F. Tisseur, SIMAX 26(1), 2004).

  Input:
    A symmetric (only lower triangular part is refered)
    s vector +1 and -1 (signature matrix)
  Output:
    d,e
    s
    Q s-orthogonal matrix whith Q^T*A*Q = T (symmetric tridiagonal matrix)
*/
static PetscErrorCode TridiagDiag_HHR(PetscInt n,PetscScalar *A,PetscInt lda,PetscReal *s,PetscScalar* Q,PetscInt ldq,PetscBool flip,PetscReal *d,PetscReal *e,PetscInt *perm_,PetscScalar *w,PetscInt lw)
{
#if defined(PETSC_MISSING_LAPACK_LARFG) || defined(PETSC_MISSING_LAPACK_LARF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LARFG/LARF - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,k,*ii,*jj,i0,ik,tmp,type,*perm,nwall,nwu;
  PetscReal      *ss,cond=1.0,cs,sn,r;
  PetscScalar    *work,tau,t,*AA;
  PetscBLASInt   n0,n1,ni,inc=1,m,n_,lda_,ldq_;
  PetscBool      breakdown = PETSC_TRUE;
  
  PetscFunctionBegin;
  if (n<3) {
    if (n==1)Q[0]=1;
    if (n==2) {Q[0] = Q[1+ldq] = 1; Q[1] = Q[ldq] = 0;}
    PetscFunctionReturn(0);
  }
  lda_ = PetscBLASIntCast(lda);
  n_   = PetscBLASIntCast(n);
  ldq_ = PetscBLASIntCast(ldq);
  nwall = n*n+n;
  nwu = 0;
  if (!w || lw < nwall) {
    ierr = PetscMalloc(nwall*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }else work = w;
  ierr = PetscMalloc(n*sizeof(PetscReal),&ss);CHKERRQ(ierr); 
  ierr = PetscMalloc(n*sizeof(PetscInt),&perm);CHKERRQ(ierr);
  AA = work;
  for (i=0;i<n;i++) {
    ierr = PetscMemcpy(AA+i*n,A+i*lda,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  nwu += n*n;
  k=0;
  while (breakdown && k<n) {
    breakdown = PETSC_FALSE;
    /* Classify (and flip) A and s according to sign */
    if (flip) {
      for (i=0;i<n;i++) {
        perm[i] = n-1-perm_[i];
        if (perm[i]==0) i0 = i;
        if (perm[i]==k) ik = i;
      }
    } else {
      for (i=0;i<n;i++) {
        perm[i] = perm_[i];
        if (perm[i]==0) i0 = i;
        if (perm[i]==k) ik = i;
      }
    }
    perm[ik] = 0;
    perm[i0] = k;
    i=1;
    while (i<n-1 && s[perm[i-1]]==s[perm[0]]) {
      if (s[perm[i]]!=s[perm[0]]) {
        j=i+1;
        while (j<n-1 && s[perm[j]]!=s[perm[0]])j++;
        tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
      }
      i++;
    }
    for (i=0;i<n;i++) {
      ss[i] = s[perm[i]];
    }
    if (flip) { ii = &j; jj = &i;} else { ii = &i; jj = &j;}
    for (i=0;i<n;i++)
      for (j=0;j<n;j++)
        A[i+j*lda] = AA[perm[*ii]+perm[*jj]*n];
    /* Initialize Q */
    for (i=0;i<n;i++) {
      ierr = PetscMemzero(Q+i*ldq,n*sizeof(PetscScalar));
      Q[perm[i]+i*ldq] = 1.0;
    }
    for (ni=1;ni<n && ss[ni]==ss[0]; ni++);
    n0 = ni-1; n1 = PetscBLASIntCast(n)-ni;
    for (j=0;j<n-2;j++) {
      m = PetscBLASIntCast(n-j-1);
      /* Forming and applying reflectors */
      if ( n0 > 1 ) {
        LAPACKlarfg_(&n0, A+ni-n0+j*lda, A+ni-n0+j*lda+1,&inc,&tau);
        /* Apply reflector */
        if ( PetscAbsScalar(tau) != 0.0 ) {
          t=*( A+ni-n0+j*lda);  *(A+ni-n0+j*lda)=1.0;
          LAPACKlarf_("R",&m,&n0,A+ni-n0+j*lda,&inc,&tau,A+j+1+(j+1)*lda,&lda_,work+nwu);
          LAPACKlarf_("L",&n0,&m,A+ni-n0+j*lda,&inc,&tau,A+j+1+(j+1)*lda,&lda_,work+nwu);
          /* Update Q */
          LAPACKlarf_("R",&n_,&n0,A+ni-n0+j*lda,&inc,&tau,Q+(j+1)*ldq,&ldq_,work+nwu);
          *(A+ni-n0+j*lda) = t;
          for (i=1;i<n0;i++) {
            *(A+ni-n0+j*lda+i) = 0.0;  *(A+j+(ni-n0+i)*lda) = 0.0;
          }
          *(A+j+(ni-n0)*lda) = *(A+ni-n0+j*lda);
        }
      }
      if ( n1 > 1 ) {
        LAPACKlarfg_(&n1, A+n-n1+j*lda, A+n-n1+j*lda+1,&inc,&tau);
        /* Apply reflector */
        if ( PetscAbsScalar(tau) != 0.0 ) {
          t=*( A+n-n1+j*lda);  *(A+n-n1+j*lda)=1.0;
          LAPACKlarf_("R",&m,&n1,A+n-n1+j*lda,&inc,&tau,A+j+1+(n-n1)*lda,&lda_,work+nwu);
          LAPACKlarf_("L",&n1,&m,A+n-n1+j*lda,&inc,&tau,A+n-n1+(j+1)*lda,&lda_,work+nwu);
          /* Update Q */
          LAPACKlarf_("R",&n_,&n1,A+n-n1+j*lda,&inc,&tau,Q+(n-n1)*ldq,&ldq_,work+nwu);
          *(A+n-n1+j*lda) = t;
          for (i=1;i<n1;i++) {
            *(A+n-n1+i+j*lda) = 0.0;  *(A+j+(n-n1+i)*lda) = 0.0;
          }
          *(A+j+(n-n1)*lda) = *(A+n-n1+j*lda);
        }
      }
      /* Hyperbolic rotation */
      if ( n0 > 0 && n1 > 0) {
        ierr = HRGen(PetscRealPart(A[ni-n0+j*lda]),PetscRealPart(A[n-n1+j*lda]),&type,&cs,&sn,&r,&cond);CHKERRQ(ierr);
        /* Check condition number */
        if (cond > 1.0/(10*PETSC_SQRT_MACHINE_EPSILON)) {
          breakdown = PETSC_TRUE;
          k++;
          if (k==n || flip)
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Breakdown in construction of hyperbolic transformation");
          break;  
        }
        A[ni-n0+j*lda] = r; A[n-n1+j*lda] = 0.0;
        A[j+(ni-n0)*lda] = r; A[j+(n-n1)*lda] = 0.0;
        /* Apply to A */
        ierr = HRApply(m, A+j+1+(ni-n0)*lda,1, A+j+1+(n-n1)*lda,1, cs, -sn);CHKERRQ(ierr);
        ierr = HRApply(m, A+ni-n0+(j+1)*lda,lda, A+n-n1+(j+1)*lda,lda, cs, -sn);CHKERRQ(ierr);
        
        /* Update Q */
        ierr = HRApply(n, Q+(ni-n0)*ldq,1, Q+(n-n1)*ldq,1, cs, -sn);CHKERRQ(ierr);
        if (type==2) {
          ss[ni-n0] = -ss[ni-n0]; ss[n-n1] = -ss[n-n1];
          n0++;ni++;n1--;
        }
      }
      if (n0>0) n0--;else n1--;
    }
  }

/* flip matrices */
    if (flip) {
      for (i=0;i<n-1;i++) {
        d[i] = PetscRealPart(A[n-i-1+(n-i-1)*lda]);
        e[i] = PetscRealPart(A[n-i-1+(n-i-2)*lda]);
        s[i] = ss[n-i-1];
      }
      s[n-1] = ss[0];
      d[n-1] = PetscRealPart(A[0]);
      for (i=0;i<n;i++) {
        ierr=PetscMemcpy(work+i*n,Q+i*ldq,n*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      for (i=0;i<n;i++)
        for (j=0;j<n;j++)
          Q[i+j*ldq] = work[i+(n-j-1)*n];
    } else {
      for (i=0;i<n-1;i++) {
        d[i] = PetscRealPart(A[i+i*lda]);
        e[i] = PetscRealPart(A[i+1+i*lda]);
        s[i] = ss[i];
      }
      s[n-1] = ss[n-1];
      d[n-1] = PetscRealPart(A[n-1 + (n-1)*lda]);
    }

  ierr = PetscFree(ss);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "IndefOrthog"
/*
  compute x = x - y*ss^{-1}*y^T*s*x where ss=y^T*s*y
  s diagonal (signature matrix)
*/
static PetscErrorCode IndefOrthog(PetscReal *s, PetscScalar *y, PetscReal ss, PetscScalar *x, PetscScalar *h,PetscInt n)
{
  PetscInt    i;
  PetscScalar h_,r;

  PetscFunctionBegin;
  if (y) {
    h_ = 0.0; /* h_=(y^Tdiag(s)*y)^{-1}*y^T*diag(s)*x*/
    for (i=0;i<n;i++) { h_+=y[i]*s[i]*x[i];}
    h_ /= ss;
    for (i=0;i<n;i++) {x[i] -= h_*y[i];} /* x = x-h_*y */
    /* repeat */
    r = 0.0;
    for (i=0;i<n;i++) { r+=y[i]*s[i]*x[i];}
    r /= ss;
    for (i=0;i<n;i++) {x[i] -= r*y[i];}
    h_ += r;
  }else h_ = 0.0;
  if (h) *h = h_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IndefNorm"
/* 
   normalization with a indefinite norm
*/
static PetscErrorCode IndefNorm(PetscReal *s,PetscScalar *x, PetscReal *norm,PetscInt n)
{
  PetscInt  i;
  PetscReal norm_;

  PetscFunctionBegin;
  /* s-normalization */
  norm_ = 0.0;
  for (i=0;i<n;i++) {norm_ += PetscRealPart(x[i]*s[i]*x[i]);}
  if (norm_<0) {norm_ = -PetscSqrtReal(-norm_);}
  else {norm_ = PetscSqrtReal(norm_);}
  for (i=0;i<n;i++)x[i] /= norm_;
  if (norm) *norm = norm_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSEigenVectorsPseudoOrthog"
static PetscErrorCode DSEigenVectorsPseudoOrthog(DS ds, DSMatType mat, PetscScalar *wr, PetscScalar *wi,PetscBool accum)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,off;
  PetscBLASInt   ld,n1,one=1;
  PetscScalar    PQ[4],xx,yx,xy,yy,*y,h,oneS=1.0,zeroS=0.0,*X,*W,*B;
  PetscReal      *ss,*s,*d,*e,d1,d2,toldeg=PETSC_SQRT_MACHINE_EPSILON*100;

  PetscFunctionBegin;
  ld = PetscBLASIntCast(ds->ld);
  n1 = PetscBLASIntCast(ds->n - ds->l);
  ierr = DSAllocateWork_Private(ds,ld*ld+2*ld,ld,2*ld);CHKERRQ(ierr);
  s = ds->rmat[DS_MAT_D];
  d = ds->rmat[DS_MAT_T];
  e = d + ld;
  off = ds->l+ds->l*ld;
  if (!ds->compact) {
    B = ds->mat[DS_MAT_B];
    for (i=ds->l;i<ds->n;i++) {
      s[i] = PetscRealPart(B[i+i*ld]);
    }
  }

  /* compute real s-orthonormal base */
  X = ds->mat[mat];
  ss = ds->rwork;
  y = ds->work;

#if defined(PETSC_USE_COMPLEX)
  /* with complex scalars we need to operate as in real scalar */
  for (i=ds->l;i<ds->n;i++) {
    if (PetscImaginaryPart(wr[i])!=0.0) {
      for (j=ds->l;j<ds->n;j++) {
        X[j+(i+1)*ld] = PetscImaginaryPart(X[j+i*ld]);
        X[j+i*ld] = PetscRealPart(X[j+i*ld]);
      }
      i++;
    }
  }
#endif

  for (i=ds->l;i<ds->n;i++) {
#if defined(PETSC_USE_COMPLEX)
    if (PetscImaginaryPart(wr[i])==0.0) { /* real */
#else
    if (wi[i]==0.0) { /* real */
#endif
      for (j=ds->l;j<i;j++) {
         /* s-orthogonalization with close eigenvalues */
        if (wi[j]==0.0) {
          if ( PetscAbsScalar(wr[j]-wr[i])<toldeg) {
            ierr = IndefOrthog(s+ds->l, X+j*ld+ds->l, ss[j],X+i*ld+ds->l, PETSC_NULL,n1);CHKERRQ(ierr);
          }
        }else j++;
      }
      ierr = IndefNorm(s+ds->l,X+i*ld+ds->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0.0)?-1:1;
      d[i] = PetscRealPart(wr[i]*ss[i]); e[i] = 0.0;
    } else {
      for (j=ds->l;j<i;j++) {
        /* s-orthogonalization of Xi and Xi+1*/
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(wr[j])!=0.0) {
#else
        if (wi[j]!=0.0) {
#endif
          if (PetscAbsScalar(wr[j]-wr[i])<toldeg && PetscAbsScalar(PetscAbsScalar(wi[j])-PetscAbsScalar(wi[i]))<toldeg) {
            for (k=ds->l;k<ds->n;k++) y[k] = s[k]*X[k+i*ld];
            xx = BLASdot_(&n1,X+ds->l+j*ld,&one,y+ds->l,&one);
            yx = BLASdot_(&n1,X+ds->l+(j+1)*ld,&one,y+ds->l,&one);
            for (k=ds->l;k<ds->n;k++) y[k] = s[k]*X[k+(i+1)*ld];
            xy = BLASdot_(&n1,X+ds->l+j*ld,&one,y+ds->l,&one);
            yy = BLASdot_(&n1,X+ds->l+(j+1)*ld,&one,y+ds->l,&one);
            PQ[0] = ss[j]*xx; PQ[1] = ss[j+1]*yx; PQ[2] = ss[j]*xy; PQ[3] = ss[j+1]*yy;
            for (k=ds->l;k<ds->n;k++) {
              X[k+i*ld] -= PQ[0]*X[k+j*ld]+PQ[1]*X[k+(j+1)*ld];
              X[k+(i+1)*ld] -= PQ[2]*X[k+j*ld]+PQ[3]*X[k+(j+1)*ld];
            }
            /* Repeat */
            for (k=ds->l;k<ds->n;k++) y[k] = s[k]*X[k+i*ld];
            xx = BLASdot_(&n1,X+ds->l+j*ld,&one,y+ds->l,&one);
            yx = BLASdot_(&n1,X+ds->l+(j+1)*ld,&one,y+ds->l,&one);
            for (k=ds->l;k<ds->n;k++) y[k] = s[k]*X[k+(i+1)*ld];
            xy = BLASdot_(&n1,X+ds->l+j*ld,&one,y+ds->l,&one);
            yy = BLASdot_(&n1,X+ds->l+(j+1)*ld,&one,y+ds->l,&one);
            PQ[0] = ss[j]*xx; PQ[1] = ss[j+1]*yx; PQ[2] = ss[j]*xy; PQ[3] = ss[j+1]*yy;
            for (k=ds->l;k<ds->n;k++) {
              X[k+i*ld] -= PQ[0]*X[k+j*ld]+PQ[1]*X[k+(j+1)*ld];
              X[k+(i+1)*ld] -= PQ[2]*X[k+j*ld]+PQ[3]*X[k+(j+1)*ld];
            }
          }
          j++;
        }
      }
      ierr = IndefNorm(s+ds->l,X+i*ld+ds->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0)?-1:1;
      ierr = IndefOrthog(s+ds->l, X+i*ld+ds->l, ss[i],X+(i+1)*ld+ds->l, &h,n1);CHKERRQ(ierr);
      ierr = IndefNorm(s+ds->l,X+(i+1)*ld+ds->l,&d2,n1);CHKERRQ(ierr);
      ss[i+1] = (d2<0)?-1:1;
      d[i] = PetscRealPart((wr[i]-wi[i]*h/d1)*ss[i]);
      d[i+1] = PetscRealPart((wr[i]+wi[i]*h/d1)*ss[i+1]);
      e[i] = PetscRealPart(wi[i]*d2/d1*ss[i]); e[i+1] = 0.0;
      i++;
    }
  }
  for (i=ds->l;i<ds->n;i++) s[i] = ss[i];
  /* accumulate previous Q */
  if (accum && mat!=DS_MAT_Q) {
    ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
    W = ds->mat[DS_MAT_W];
    ierr = DSCopyMatrix_Private(ds,DS_MAT_W,DS_MAT_Q);CHKERRQ(ierr);
    BLASgemm_("N","N",&n1,&n1,&n1,&oneS,W+off,&ld,ds->mat[DS_MAT_X]+off,&ld,&zeroS,ds->mat[DS_MAT_Q]+off,&ld);
  }
  if (!ds->compact) {ierr = DSSwitchFormat_GHIEP(ds,PETSC_FALSE);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSGHIEPPseudoOrthogInverseIteration"
/*
  Get eigenvectors with inverse iteration.
  The system matrix is in Hessenberg form.
*/
PetscErrorCode DSGHIEPPseudoOrthogInverseIteration(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(PETSC_MISSING_LAPACK_HSEIN)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HSEIN - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,off;
  PetscBLASInt   *select,*infoC,ld,n1,mout,info;
  PetscScalar    *A,*B,*H,*X;
  PetscReal      *s,*d,*e;

  PetscFunctionBegin;
  ld = PetscBLASIntCast(ds->ld);
  n1 = PetscBLASIntCast(ds->n - ds->l);
  ierr = DSAllocateWork_Private(ds,ld*ld+2*ld,ld,2*ld);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  H = ds->mat[DS_MAT_W];
  s = ds->rmat[DS_MAT_D];
  d = ds->rmat[DS_MAT_T];
  e = d + ld;
  select = ds->iwork;
  infoC = ds->iwork + ld;
  off = ds->l+ds->l*ld;
  if (ds->compact) {
    H[off] = d[ds->l]*s[ds->l];
    H[off+ld] = e[ds->l]*s[ds->l];
    for (i=ds->l+1;i<ds->n-1;i++) {
      H[i+(i-1)*ld] = e[i-1]*s[i];
      H[i+i*ld] = d[i]*s[i];
      H[i+(i+1)*ld] = e[i]*s[i];
    }
    H[ds->n-1+(ds->n-2)*ld] = e[ds->n-2]*s[ds->n-1];
    H[ds->n-1+(ds->n-1)*ld] = d[ds->n-1]*s[ds->n-1];
  } else {
    s[ds->l] = PetscRealPart(B[off]);
    H[off] = A[off]*s[ds->l];
    H[off+ld] = A[off+ld]*s[ds->l];
    for (i=ds->l+1;i<ds->n-1;i++) {
      s[i] = PetscRealPart(B[i+i*ld]);
      H[i+(i-1)*ld] = A[i+(i-1)*ld]*s[i];
      H[i+i*ld]     = A[i+i*ld]*s[i];
      H[i+(i+1)*ld] = A[i+(i+1)*ld]*s[i];
    }
    s[ds->n-1] = PetscRealPart(B[ds->n-1+(ds->n-1)*ld]);
    H[ds->n-1+(ds->n-2)*ld] = A[ds->n-1+(ds->n-2)*ld]*s[ds->n-1];
    H[ds->n-1+(ds->n-1)*ld] = A[ds->n-1+(ds->n-1)*ld]*s[ds->n-1];
  }
  ierr = DSAllocateMat_Private(ds,DS_MAT_X);CHKERRQ(ierr);
  X = ds->mat[DS_MAT_X];
  for (i=0;i<n1;i++)select[i]=1;
#if !defined(PETSC_USE_COMPLEX)
  LAPACKhsein_("R","N","N",select,&n1,H+off,&ld,wr+ds->l,wi+ds->l,PETSC_NULL,&ld,X+off,&ld,&n1,&mout,ds->work,PETSC_NULL,infoC,&info);
#else
  LAPACKhsein_("R","N","N",select,&n1,H+off,&ld,wr+ds->l,PETSC_NULL,&ld,X+off,&ld,&n1,&mout,ds->work,ds->rwork,PETSC_NULL,infoC,&info);
#endif
  if (info<0)SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in hsein routine %d",-i);
  if (info>0) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Convergence error in hsein routine %d",i);
  }

  ierr = DSEigenVectorsPseudoOrthog(ds, DS_MAT_X, wr, wi,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "DSIntermediate_GHIEP"
/*
   Reduce to tridiagonal-diagonal pair by means of TridiagDiag_HHR.
*/
PetscErrorCode DSIntermediate_GHIEP(DS ds)
{
  PetscErrorCode ierr;
  PetscInt       i,ld,off;
  PetscScalar    *A,*B,*Q;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
  ld = ds->ld;
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  Q = ds->mat[DS_MAT_Q];
  d = ds->rmat[DS_MAT_T];
  e = ds->rmat[DS_MAT_T]+ld;
  s = ds->rmat[DS_MAT_D];
  off = ds->l+ds->l*ld;
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = DSAllocateWork_Private(ds,ld*ld,0,0);CHKERRQ(ierr);

  for (i=0;i<ds->n;i++) Q[i+i*ld]=1.0;
  for (i=0;i<ds->n-ds->l;i++) *(ds->perm+i)=i;
  if (ds->compact) {
    if (ds->state < DS_STATE_INTERMEDIATE) {
      ierr = DSSwitchFormat_GHIEP(ds,PETSC_FALSE);CHKERRQ(ierr);
      ierr = TridiagDiag_HHR(ds->k-ds->l+1,A+off,ld,s+ds->l,Q+off,ld,PETSC_TRUE,d+ds->l,e+ds->l,ds->perm,ds->work,ld*ld);CHKERRQ(ierr);
      ds->k = ds->l;
      ierr = PetscMemzero(d+2*ld+ds->l,(ds->n-ds->l)*sizeof(PetscReal));CHKERRQ(ierr);
    }
  } else {
    if (ds->state < DS_STATE_INTERMEDIATE) {
      for (i=0;i<ds->n;i++)
        s[i] = PetscRealPart(B[i+i*ld]);
      ierr = TridiagDiag_HHR(ds->n-ds->l,A+off,ld,s+ds->l,Q+off,ld,PETSC_FALSE,d+ds->l,e+ds->l,ds->perm,ds->work,ld*ld);CHKERRQ(ierr);
      ierr = PetscMemzero(d+2*ld,(ds->n)*sizeof(PetscReal));CHKERRQ(ierr);
      ds->k = ds->l;
      ierr = DSSwitchFormat_GHIEP(ds,PETSC_FALSE);CHKERRQ(ierr);
    } else { ierr = DSSwitchFormat_GHIEP(ds,PETSC_TRUE);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSGHIEPRealBlocks"
/*
   Undo 2x2 blocks that have real eigenvalues.
*/
PetscErrorCode DSGHIEPRealBlocks(DS ds)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      e,d1,d2,s1,s2,ss1,ss2,t,dd,ss;
  PetscReal      maxy,ep,scal1,scal2,snorm;
  PetscReal      *T,*D,b[4],M[4],wr1,wr2,wi;
  PetscScalar    *A,*B,Y[4],oneS = 1.0,zeroS = 0.0;
  PetscBLASInt   m,two=2,ld;
  PetscBool      isreal;

  PetscFunctionBegin;
  ld = PetscBLASIntCast(ds->ld);
  m = PetscBLASIntCast(ds->n-ds->l);
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  T = ds->rmat[DS_MAT_T];
  D = ds->rmat[DS_MAT_D];
  ierr = DSAllocateWork_Private(ds,2*m,0,0);CHKERRQ(ierr);
  for (i=ds->l;i<ds->n-1;i++) {
    e = (ds->compact)?T[ld+i]:PetscRealPart(A[(i+1)+ld*i]);
    if (e != 0.0) { /* 2x2 block */
      if (ds->compact) {
        s1 = D[i];
        d1 = T[i];
        s2 = D[i+1];
        d2 = T[i+1];
      } else {
        s1 = PetscRealPart(B[i*ld+i]);
        d1 = PetscRealPart(A[i*ld+i]);
        s2 = PetscRealPart(B[(i+1)*ld+i+1]);
        d2 = PetscRealPart(A[(i+1)*ld+i+1]);
      }
      isreal = PETSC_FALSE;
      if (s1==s2) { /* apply a Jacobi rotation to compute the eigendecomposition */
        dd = d1-d2;
        if (2*PetscAbsReal(e) <= dd) {
          t = 2*e/dd;
          t = t/(1 + PetscSqrtReal(1+t*t));
        } else {
          t = dd/(2*e);
          ss = (t>=0)?1.0:-1.0;
          t = ss/(PetscAbsReal(t)+PetscSqrtReal(1+t*t));
        }
        Y[0] = 1/PetscSqrtReal(1 + t*t); Y[3] = Y[0]; /* c */
        Y[1] = Y[0]*t; Y[2] = -Y[1]; /* s */
        wr1 = d1+t*e;
        wr2 = d2-t*e;
        ss1 = s1; ss2 = s2;
        isreal = PETSC_TRUE;
      } else {
        ss1 = 1.0; ss2 = 1.0,
        M[0] = d1; M[1] = e; M[2] = e; M[3]= d2;
        b[0] = s1; b[1] = 0.0; b[2] = 0.0; b[3] = s2;
        ep = LAPACKlamch_("S");
        /* Compute eigenvalues of the block */
        LAPACKlag2_(M, &two, b, &two,&ep , &scal1, &scal2, &wr1, &wr2, &wi);
        if (wi==0.0) { /* Real eigenvalues */
          isreal = PETSC_TRUE;
          if (scal1<ep||scal2<ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Nearly infinite eigenvalue");
          wr1 /= scal1; wr2 /= scal2;
          if ( PetscAbsReal(s1*d1-wr1)<PetscAbsReal(s2*d2-wr1)) { Y[0] = wr1-s2*d2; Y[1] =s2*e;}
          else{ Y[0] = s1*e; Y[1] = wr1-s1*d1; }
          /* normalize with a signature*/
          maxy = PetscMax(PetscAbsScalar(Y[0]),PetscAbsScalar(Y[1]));
          scal1 = PetscRealPart(Y[0])/maxy; scal2 = PetscRealPart(Y[1])/maxy;
          snorm = scal1*scal1*s1 + scal2*scal2*s2;
          if (snorm<0) {ss1 = -1.0; snorm = -snorm;}
          snorm = maxy*PetscSqrtReal(snorm); Y[0] = Y[0]/snorm; Y[1] = Y[1]/snorm;
          if ( PetscAbsReal(s1*d1-wr2)<PetscAbsReal(s2*d2-wr2)) { Y[2] = wr2-s2*d2; Y[3] =s2*e;}
          else{ Y[2] = s1*e; Y[3] = wr2-s1*d1; }
          maxy = PetscMax(PetscAbsScalar(Y[2]),PetscAbsScalar(Y[3]));
          scal1 = PetscRealPart(Y[2])/maxy; scal2 = PetscRealPart(Y[3])/maxy;
          snorm = scal1*scal1*s1 + scal2*scal2*s2;
          if (snorm<0) {ss2 = -1.0; snorm = -snorm;}
          snorm = maxy*PetscSqrtReal(snorm);Y[2] = Y[2]/snorm; Y[3] = Y[3]/snorm;
        }
        wr1 *= ss1; wr2 *= ss2;
      }
      if (isreal) {
        if (ds->compact) {
          D[i] = ss1;;
          T[i] = wr1;
          D[i+1] = ss2;
          T[i+1] = wr2;
          T[ld+i] = 0.0;
        }else {
          B[i*ld+i] = ss1;
          A[i*ld+i] = wr1;
          B[(i+1)*ld+i+1] = ss2;
          A[(i+1)*ld+i+1] = wr2;
          A[(i+1)+ld*i] = 0.0;
          A[i+ld*(i+1)] = 0.0;
        }
        BLASgemm_("N","N",&m,&two,&two,&oneS,ds->mat[DS_MAT_Q]+ds->l+i*ld,&ld,Y,&two,&zeroS,ds->work,&m);
        ierr = PetscMemcpy(ds->mat[DS_MAT_Q]+ds->l+i*ld,ds->work,m*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscMemcpy(ds->mat[DS_MAT_Q]+ds->l+(i+1)*ld,ds->work+m,m*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      i++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSolve_GHIEP_QR_II"
PetscErrorCode DSSolve_GHIEP_QR_II(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(PETSC_MISSING_LAPACK_HSEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HSEQR - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,off;
  PetscBLASInt   n1,ld,one,info,lwork;
  PetscScalar    *H,*A,*B,*Q,*work;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
  one = 1;
  n1 = PetscBLASIntCast(ds->n - ds->l);
  ld = PetscBLASIntCast(ds->ld);
  off = ds->l + ds->l*ld;
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  Q = ds->mat[DS_MAT_Q];
  d = ds->rmat[DS_MAT_T];
  e = ds->rmat[DS_MAT_T] + ld;
  s = ds->rmat[DS_MAT_D];
  ierr = DSAllocateWork_Private(ds,ld*ld,2*ld,ld*2);CHKERRQ(ierr); 
  work = ds->work;
  lwork = ld*ld;

  /* Quick return if possible */
  if (n1 == 1) {
    *(Q+off) = 1;
    if (ds->compact) {
      wr[ds->l] = d[ds->l]/s[ds->l];
      wi[ds->l] = 0.0;
    } else {
      d[ds->l] = PetscRealPart(A[off]);
      s[ds->l] = PetscRealPart(B[off]);
      wr[ds->l] = d[ds->l]/s[ds->l];
      wi[ds->l] = 0.0;  
    }
    PetscFunctionReturn(0);
  }
  /* Reduce to pseudotriadiagonal form */
  ierr = DSIntermediate_GHIEP( ds);CHKERRQ(ierr);

  /* Compute Eigenvalues (QR)*/
  ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  H = ds->mat[DS_MAT_W];
  if (ds->compact) {
    H[off] = d[ds->l]*s[ds->l];
    H[off+ld] = e[ds->l]*s[ds->l];
    for (i=ds->l+1;i<ds->n-1;i++) {
      H[i+(i-1)*ld] = e[i-1]*s[i];
      H[i+i*ld]     = d[i]*s[i];
      H[i+(i+1)*ld] = e[i]*s[i];
    }
    H[ds->n-1+(ds->n-2)*ld] = e[ds->n-2]*s[ds->n-1];
    H[ds->n-1+(ds->n-1)*ld] = d[ds->n-1]*s[ds->n-1];
  } else {
    s[ds->l] = PetscRealPart(B[off]);
    H[off] = A[off]*s[ds->l];
    H[off+ld] = A[off+ld]*s[ds->l];
    for (i=ds->l+1;i<ds->n-1;i++) {
      s[i] = PetscRealPart(B[i+i*ld]);
      H[i+(i-1)*ld] = A[i+(i-1)*ld]*s[i];
      H[i+i*ld]     = A[i+i*ld]*s[i];
      H[i+(i+1)*ld] = A[i+(i+1)*ld]*s[i];
    }
    s[ds->n-1] = PetscRealPart(B[ds->n-1+(ds->n-1)*ld]);
    H[ds->n-1+(ds->n-2)*ld] = A[ds->n-1+(ds->n-2)*ld]*s[ds->n-1];
    H[ds->n-1+(ds->n-1)*ld] = A[ds->n-1+(ds->n-1)*ld]*s[ds->n-1];
  }

#if !defined(PETSC_USE_COMPLEX)
  LAPACKhseqr_("E","N",&n1,&one,&n1,H+off,&ld,wr+ds->l,wi+ds->l,PETSC_NULL,&ld,work,&lwork,&info);
#else
  LAPACKhseqr_("E","N",&n1,&one,&n1,H+off,&ld,wr,PETSC_NULL,&ld,work,&lwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",&info);

  /* Compute Eigenvectors with Inverse Iteration */
  ierr = DSGHIEPPseudoOrthogInverseIteration(ds,wr,wi);CHKERRQ(ierr);

  /* Recover eigenvalues from diagonal */
  ierr = DSGHIEPComplexEigs(ds, 0, ds->l, wr, wi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "DSSolve_GHIEP_QR"
PetscErrorCode DSSolve_GHIEP_QR(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GEHRD) || defined(SLEPC_MISSING_LAPACK_ORGHR) || defined(PETSC_MISSING_LAPACK_HSEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEHRD/ORGHR/HSEQR - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,off;
  PetscBLASInt   lwork,info,n1,one=1,mout,ld;
  PetscScalar    *A,*B,*H,*Q,*work,*tau;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
  n1 = PetscBLASIntCast(ds->n - ds->l);
  ld = PetscBLASIntCast(ds->ld);
  off = ds->l + ds->l*ld;
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  Q = ds->mat[DS_MAT_Q];
  d = ds->rmat[DS_MAT_T];
  e = ds->rmat[DS_MAT_T] + ld;
  s = ds->rmat[DS_MAT_D];
  ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  H = ds->mat[DS_MAT_W];
  ierr = DSAllocateWork_Private(ds,ld+ld*ld,ld,0);CHKERRQ(ierr); 
  tau  = ds->work;
  work = ds->work+ld;
  lwork = ld*ld;

   /* initialize orthogonal matrix */
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i< ds->n;i++) 
    Q[i+i*ld] = 1.0;
  /* quick return */
  if (n1 == 1) {
    if (ds->compact) {
      wr[ds->l] = d[ds->l]/s[ds->l];
      wi[ds->l] = 0.0;
    } else {
      d[ds->l] = PetscRealPart(A[off]);
      s[ds->l] = PetscRealPart(B[off]);
      wr[ds->l] = d[ds->l]/s[ds->l];
      wi[ds->l] = 0.0;  
    }
    PetscFunctionReturn(0);
  }

  /* form standard problem in H */
  if (ds->compact) {
    ierr = PetscMemzero(H,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=ds->l; i < ds->n-1; i++) {
      H[i+i*ld] = d[i]/s[i];
      H[(i+1)+i*ld] = e[i]/s[i+1];
      H[i+(i+1)*ld] = e[i]/s[i];
    } 
    H[ds->n-1 + (ds->n-1)*ld] = d[ds->n-1]/s[ds->n-1];

    for (i=ds->l; i < ds->k; i++) {
      H[ds->k+i*ld] = *(ds->rmat[DS_MAT_T]+2*ld+i)/s[ds->k];
      H[i+ds->k*ld] = *(ds->rmat[DS_MAT_T]+2*ld+i)/s[i];
    }
  } else {
    for (j=ds->l; j<ds->n; j++) {
      for (i=ds->l; i<ds->n; i++) {
        H[i+j*ld] = A[i+j*ld]/B[i+i*ld];
      }
    }
  }
  /* reduce to upper Hessenberg form */
  if (ds->state<DS_STATE_INTERMEDIATE) {
    LAPACKgehrd_(&n1,&one,&n1,H+off,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",&info);
    for (j=ds->l;j<ds->n-1;j++) {
      for (i=j+2;i<ds->n;i++) {
        Q[i+j*ld] = H[i+j*ld];
        H[i+j*ld] = 0.0;
      }
    }
    LAPACKorghr_(&n1,&one,&n1,Q+off,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGHR %d",&info);
  }

  /* Compute the real Schur form */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKhseqr_("S","V",&n1,&one,&n1,H+off,&ld,wr+ds->l,wi+ds->l,Q+off,&ld,work,&lwork,&info);
#else
  LAPACKhseqr_("S","V",&n1,&one,&n1,H+off,&ld,wr,Q+off,&ld,work,&lwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",&info);
  
  /* Compute eigenvectors */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKtrevc_("R","B",PETSC_NULL,&n1,H+off,&ld,PETSC_NULL,&ld,Q+off,&ld,&n1,&mout,ds->work,&info);
#else
  LAPACKtrevc_("R","B",PETSC_NULL,&n1,H+off,&ld,PETSC_NULL,&ld,Q+off,&ld,&n1,&mout,work,ds->rwork,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",&info);

  /* Compute real s-orthonormal basis */
  ierr = DSEigenVectorsPseudoOrthog(ds, DS_MAT_Q, wr, wi,PETSC_FALSE);CHKERRQ(ierr);

  /* Undo from diagonal the blocks whith real eigenvalues*/
  ierr = DSGHIEPRealBlocks(ds);CHKERRQ(ierr);

  /* Recover eigenvalues from diagonal */
  ierr = DSGHIEPComplexEigs(ds, 0, ds->l, wr, wi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "DSNormalize_GHIEP"
PetscErrorCode DSNormalize_GHIEP(DS ds,DSMatType mat,PetscInt col)
{
  PetscErrorCode ierr;
  PetscInt       i,i0,i1;
  PetscBLASInt   ld,n,one = 1;
  PetscScalar    *A = ds->mat[DS_MAT_A],norm,*x;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    norm0;
#endif

  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_X:
    case DS_MAT_Y:
    case DS_MAT_Q:
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
  } else if (col>0 && A[ds->ld*(col-1)+col] != 0.0) {
    i0 = col-1; i1 = col+1;
  } else {
    i0 = col; i1 = col+1;
  }
  for (i=i0; i<i1; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (i<n-1 && A[ds->ld*i+i+1] != 0.0) {
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

extern PetscErrorCode DSSolve_GHIEP_HZ(DS,PetscScalar*,PetscScalar*);
extern PetscErrorCode DSSolve_GHIEP_DQDS_II(DS,PetscScalar*,PetscScalar*);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "DSCreate_GHIEP"
PetscErrorCode DSCreate_GHIEP(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate      = DSAllocate_GHIEP;
  ds->ops->view          = DSView_GHIEP;
  ds->ops->vectors       = DSVectors_GHIEP;
  ds->ops->solve[0]      = DSSolve_GHIEP_HZ;
  ds->ops->solve[1]      = DSSolve_GHIEP_QR_II;
  ds->ops->solve[2]      = DSSolve_GHIEP_QR;
  ds->ops->solve[3]      = DSSolve_GHIEP_DQDS_II;
  ds->ops->sort          = DSSort_GHIEP;
  ds->ops->normalize     = DSNormalize_GHIEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END
