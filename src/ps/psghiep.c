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

PetscInt dbPS=0;

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
  if(y){
    h_ = 0.0; /* h_=(y^Tdiag(s)*y)^{-1}*y^T*diag(s)*x*/
    for(i=0;i<n;i++){ h_+=y[i]*s[i]*x[i];}
    h_ /= ss;
    for(i=0;i<n;i++){x[i] -= h_*y[i];} /* x = x-h_*y */
    /* repeat */
    r = 0.0;
    for(i=0;i<n;i++){ r+=y[i]*s[i]*x[i];}
    r /= ss;
    for(i=0;i<n;i++){x[i] -= r*y[i];}
    h_ += r;
  }else h_ = 0.0;
  if(h) *h = h_;
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
  for(i=0;i<n;i++){norm_ += PetscRealPart(x[i]*s[i]*x[i]);}
  if(norm_<0){norm_ = -PetscSqrtReal(-norm_);}
  else {norm_ = PetscSqrtReal(norm_);}
  for(i=0;i<n;i++)x[i] /= norm_;
  if(norm) *norm = norm_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSAllocate_GHIEP"
PetscErrorCode PSAllocate_GHIEP(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_A);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_B);CHKERRQ(ierr); 
  ierr = PSAllocateMat_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  ierr = PSAllocateMatReal_Private(ps,PS_MAT_T);CHKERRQ(ierr); 
  ierr = PSAllocateMatReal_Private(ps,PS_MAT_D);CHKERRQ(ierr);
  ierr = PetscFree(ps->perm);CHKERRQ(ierr);
  ierr = PetscMalloc(ld*sizeof(PetscInt),&ps->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ps,ld*sizeof(PetscInt));CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSwitchFormat_GHIEP"
PetscErrorCode PSSwitchFormat_GHIEP(PS ps,PetscBool tocompact)
{
  PetscErrorCode ierr;
  PetscReal      *T,*S;
  PetscScalar    *A,*B;
  PetscInt       i,n,ld;

  PetscFunctionBegin;
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  T = ps->rmat[PS_MAT_T];
  S = ps->rmat[PS_MAT_D];
  n = ps->n;
  ld = ps->ld;
  if (tocompact) { /* switch from dense (arrow) to compact storage */
    ierr = PetscMemzero(T,3*ld*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(S,ld*sizeof(PetscReal));CHKERRQ(ierr);
    for(i=0;i<n-1;i++){
      T[i] = PetscRealPart(A[i+i*ld]);
      T[ld+i] = PetscRealPart(A[i+1+i*ld]);
      S[i] = PetscRealPart(B[i+i*ld]);
    }
    T[n-1] = PetscRealPart(A[n-1+(n-1)*ld]);
    S[n-1] = PetscRealPart(B[n-1+(n-1)*ld]);
    for(i=ps->l;i< ps->k;i++) T[2*ld+i] = PetscRealPart(A[ps->k+i*ld]); 
  }else { /* switch from compact (arrow) to dense storage */
    ierr = PetscMemzero(A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(B,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for(i=0;i<n-1;i++){
      A[i+i*ld] = T[i];
      A[i+1+i*ld] = T[ld+i];
      A[i+(i+1)*ld] = T[ld+i];
      B[i+i*ld] = S[i];
    }
    A[n-1+(n-1)*ld] = T[n-1];
    B[n-1+(n-1)*ld] = S[n-1];
    for(i=ps->l;i<ps->k;i++){
      A[ps->k+i*ld] = T[2*ld+i];
      A[i+ps->k*ld] = T[2*ld+i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_GHIEP"
PetscErrorCode PSView_GHIEP(PS ps,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscInt          i,j;
  PetscReal         value;
  const char *methodname[] = {
                     "HR method",
                     "QR + Inverse Iteration",
                     "QR"
  };

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"solving the problem with: %s\n",methodname[ps->method]);CHKERRQ(ierr);
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
        if (*(ps->rmat[PS_MAT_T]+ps->ld+i) !=0 && i!=ps->k-1){
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+2,i+1,*(ps->rmat[PS_MAT_T]+ps->ld+i));CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+2,*(ps->rmat[PS_MAT_T]+ps->ld+i));CHKERRQ(ierr);
        }
      }
      for (i = ps->l;i<ps->k;i++){
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",ps->k+1,i+1,*(ps->rmat[PS_MAT_T]+2*ps->ld+i));CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,ps->k+1,*(ps->rmat[PS_MAT_T]+2*ps->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",PSMatName[PS_MAT_A]);CHKERRQ(ierr);
      
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",ps->n,ps->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"omega = zeros(%D,3);\n",3*ps->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"omega = [\n");CHKERRQ(ierr);
      for (i=0;i<ps->n;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,*(ps->rmat[PS_MAT_D]+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(omega);\n",PSMatName[PS_MAT_B]);CHKERRQ(ierr);

    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"T\n");CHKERRQ(ierr);
      for (i=0;i<ps->n;i++) {
        for (j=0;j<ps->n;j++) {
          if (i==j) value = *(ps->rmat[PS_MAT_T]+i);
          else if (i==j+1 || j==i+1) value = *(ps->rmat[PS_MAT_T]+ps->ld+PetscMin(i,j));
          else if ((i<ps->k && j==ps->k) || (i==ps->k && j<ps->k)) value = *(ps->rmat[PS_MAT_T]+2*ps->ld+PetscMin(i,j));
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",value);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"omega\n");CHKERRQ(ierr);
      for (i=0;i<ps->n;i++) {
        for (j=0;j<ps->n;j++) {
          if (i==j) value = *(ps->rmat[PS_MAT_D]+i);
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
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_B);CHKERRQ(ierr);
  }
  if (ps->state>PS_STATE_INTERMEDIATE) {
    ierr = PSViewMat_Private(ps,viewer,PS_MAT_Q);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_GHIEP_Eigen_Some"
static PetscErrorCode PSVectors_GHIEP_Eigen_Some(PS ps,PetscInt *idx,PetscReal *rnorm,PetscBool left)
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
  if (left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for left eigenvectors");
  else X = ps->mat[PS_MAT_X];
  Q = ps->mat[PS_MAT_Q];
  k = *idx;
  n_ = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  if(k < ps->n-1){
   e = (ps->compact)?*(ps->rmat[PS_MAT_T]+ld+k):PetscRealPart(*(ps->mat[PS_MAT_A]+(k+1)+ld*k));
  }else e = 0.0;
  if(e == 0.0){/* Real */
     if(ps->state >= PS_STATE_CONDENSED){
       ierr = PetscMemcpy(X+k*ld,Q+k*ld,ld*sizeof(PetscScalar));CHKERRQ(ierr);
     }else{
       ierr = PetscMemzero(X+k*ps->ld,ps->ld*sizeof(PetscScalar));
       X[k+k*ps->ld] = 1.0;
     }
     if (rnorm) {
       *rnorm = PetscAbsScalar(X[ps->n-1+k*ld]);
     }
  }else{ /* 2x2 block */
    if(ps->compact){
      s1 = *(ps->rmat[PS_MAT_D]+k);
      d1 = *(ps->rmat[PS_MAT_T]+k);
      s2 = *(ps->rmat[PS_MAT_D]+k+1);
      d2 = *(ps->rmat[PS_MAT_T]+k+1);
    }else{
      s1 = PetscRealPart(*(ps->mat[PS_MAT_B]+k*ld+k));
      d1 = PetscRealPart(*(ps->mat[PS_MAT_A]+k+k*ld));
      s2 = PetscRealPart(*(ps->mat[PS_MAT_B]+(k+1)*ld+k+1));
      d2 = PetscRealPart(*(ps->mat[PS_MAT_A]+k+1+(k+1)*ld));
    }
    M[0] = d1; M[1] = e; M[2] = e; M[3]= d2;
    b[0] = s1; b[1] = 0.0; b[2] = 0.0; b[3] = s2;
    ep = LAPACKlamch_("S");
    /* Compute eigenvalues of the block */
    LAPACKlag2_(M, &two, b, &two, &ep, &scal1, &scal2, &wr1, &wr2, &wi);
    if(wi==0.0){ /* Real eigenvalues */
      SETERRQ(PETSC_COMM_SELF,1,"Real block in PSVectors_GHIEP");
    }else{ /* Complex eigenvalues */
      if(scal1<ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Nearly infinite eigenvalue");
      wr1 /= scal1; wi /= scal1;
#if !defined(PETSC_USE_COMPLEX)
      if( SlepcAbs(s1*d1-wr1,wi)<SlepcAbs(s2*d2-wr1,wi)){ 
        Y[0] = wr1-s2*d2; Y[1] = s2*e; Y[2] = wi; Y[3] = 0.0;
      }else{ 
        Y[0] = s1*e; Y[1] = wr1-s1*d1; Y[2] = 0.0; Y[3] = wi;
      }
      norm = BLASnrm2_(&four,Y,&one);
      norm = 1/norm;
      if(ps->state >= PS_STATE_CONDENSED){
        alpha = norm;
        BLASgemm_("N","N",&n_,&two,&two,&alpha,ps->mat[PS_MAT_Q]+k*ld,&ld,Y,&two,&zeroS,X+k*ld,&ld);
        if (rnorm) *rnorm = SlepcAbsEigenvalue(X[ps->n-1+k*ld],X[ps->n-1+(k+1)*ld]);
      }else{
        ierr = PetscMemzero(X+k*ld,2*ld*sizeof(PetscScalar));CHKERRQ(ierr);
        X[k*ld+k] = Y[0]*norm; X[k*ld+k+1] = Y[1]*norm;
        X[(k+1)*ld+k] = Y[2]*norm; X[(k+1)*ld+k+1] = Y[3]*norm;
      }
#else
      if( SlepcAbs(s1*d1-wr1,wi)<SlepcAbs(s2*d2-wr1,wi)){ 
        Y[0] = wr1-s2*d2+PETSC_i*wi; Y[1] = s2*e;
      }else{ 
        Y[0] = s1*e; Y[1] = wr1-s1*d1+PETSC_i*wi;
      }
      norm = BLASnrm2_(&two,Y,&one);
      norm = 1/norm;
      if(ps->state >= PS_STATE_CONDENSED){
        alpha = norm;
        BLASgemv_("N",&n_,&two,&alpha,ps->mat[PS_MAT_Q]+k*ld,&ld,Y,&one,&zeroS,X+k*ld,&one);
        if (rnorm) *rnorm = PetscAbsScalar(X[ps->n-1+k*ld]);
      }else{
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
#define __FUNCT__ "PSVectors_GHIEP"
PetscErrorCode PSVectors_GHIEP(PS ps,PSMatType mat,PetscInt *k,PetscReal *rnorm)
{
  PetscInt       i;
  PetscBool      left;
  PetscReal      e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (mat) {
    case PS_MAT_Y:
      left = PETSC_TRUE;
    case PS_MAT_X:
      left = PETSC_FALSE;
      if (k){
        ierr = PSVectors_GHIEP_Eigen_Some(ps,k,rnorm,left);CHKERRQ(ierr);
      }else{
        for(i=0; i<ps->n; i++){
          e = (ps->compact)?*(ps->rmat[PS_MAT_T]+ps->ld+i):PetscRealPart(*(ps->mat[PS_MAT_A]+(i+1)+ps->ld*i));
          if(e == 0.0){/* real */
            if(ps->state >= PS_STATE_CONDENSED){
              ierr = PetscMemcpy(ps->mat[mat]+i*ps->ld,ps->mat[PS_MAT_Q]+i*ps->ld,ps->ld*sizeof(PetscScalar));CHKERRQ(ierr);
            }else{
              ierr = PetscMemzero(ps->mat[mat]+i*ps->ld,ps->ld*sizeof(PetscScalar));
              *(ps->mat[mat]+i+i*ps->ld) = 1.0;
            }
          }else{
            ierr = PSVectors_GHIEP_Eigen_Some(ps,&i,rnorm,left);CHKERRQ(ierr);
          }
        }
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
#define __FUNCT__ "PSGHIEPComplexEigs"
/*
  Extract the eigenvalues contained in the block-diagonal of the indefinite problem.
  Only the index range n0..n1 is processed.
*/
static PetscErrorCode PSGHIEPComplexEigs(PS ps, PetscInt n0, PetscInt n1, PetscScalar *wr, PetscScalar *wi)
{
  PetscInt     k,ld;
  PetscBLASInt two=2;
  PetscScalar  *A,*B;
  PetscReal    *D,*T;
  PetscReal    b[4],M[4],d1,d2,s1,s2,e;
  PetscReal    scal1,scal2,ep,wr1,wr2,wi1;

  PetscFunctionBegin;
  ld = ps->ld;
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  D = ps->rmat[PS_MAT_D];
  T = ps->rmat[PS_MAT_T];
  for (k=n0;k<n1;k++) {
    if(k < n1-1){
      e = (ps->compact)?T[ld+k]:PetscRealPart(A[(k+1)+ld*k]);
    }else e = 0.0;
    if (e==0.0) { 
      /* real eigenvalue */
      wr[k] = (ps->compact)?T[k]/D[k]:A[k+k*ld]/B[k+k*ld];
      wi[k] = 0.0 ;
    } else {
      /* diagonal block */
      if(ps->compact){
        s1 = D[k];
        d1 = T[k];
        s2 = D[k+1];
        d2 = T[k+1];
      }else{
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
      if(scal1<ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Nearly infinite eigenvalue");
      wr[k] = wr1/scal1;
      if(wi1==0.0){ /* Real eigenvalues */
        if(scal2<ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Nearly infinite eigenvalue");
        wr[k+1] = wr2/scal2;
        wi[k] = 0.0;
        wi[k+1] = 0.0;
      }else{ /* Complex eigenvalues */
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
#define __FUNCT__ "PSSortEigenvalues_Private"
static PetscErrorCode PSSortEigenvalues_Private(PS ps,PetscScalar *wr,PetscScalar *wi,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,result,tmp1,tmp2,d=1;

  PetscFunctionBegin;
  for (i=0;i<ps->n;i++) perm[i] = i;
  /* insertion sort */
  i=ps->l+1;
#if !defined(PETSC_USE_COMPLEX)
  if(wi[perm[i-1]]!=0.0) i++; /* initial value is complex */
#else
  if(PetscImaginaryPart(wr[perm[i-1]])!=0.0) i++;
#endif
  for (;i<ps->n;i+=d) {
    re = wr[perm[i]];
    im = wi[perm[i]];
    tmp1 = perm[i];
#if !defined(PETSC_USE_COMPLEX)
    if(im!=0.0) {d = 2; tmp2 = perm[i+1];}else d = 1;
#else
    if(PetscImaginaryPart(re)!=0.0) {d = 2; tmp2 = perm[i+1];}else d = 1;
#endif
    j = i-1;
    ierr = (*ps->comp_fun)(re,im,wr[perm[j]],wi[perm[j]],&result,ps->comp_ctx);CHKERRQ(ierr);
    while (result<0 && j>=ps->l) {
      perm[j+d]=perm[j]; j--;
#if !defined(PETSC_USE_COMPLEX)
      if(wi[perm[j+1]]!=0)
#else
      if(PetscImaginaryPart(wr[perm[j+1]])!=0)
#endif
        {perm[j+d]=perm[j]; j--;}

     if (j>=ps->l) {
       ierr = (*ps->comp_fun)(re,im,wr[perm[j]],wi[perm[j]],&result,ps->comp_ctx);CHKERRQ(ierr);
     }
    }
    perm[j+1] = tmp1;
    if(d==2) perm[j+2] = tmp2;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PSSolve_GHIEP_Sort"
/*
  Sort the eigendecomposition at the end of any PSSolve_GHIEP_* method. 
*/
static PetscErrorCode PSSolve_GHIEP_Sort(PS ps,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  PetscInt       n,i,*perm;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
  n = ps->n;
  d = ps->rmat[PS_MAT_T];
  e = d + ps->ld;
  s = ps->rmat[PS_MAT_D];
  ierr = PSAllocateWork_Private(ps,ps->ld,ps->ld,0);CHKERRQ(ierr); 
  perm = ps->perm;
  ierr = PSSortEigenvalues_Private(ps,wr,wi,perm);CHKERRQ(ierr);
  if(!ps->compact){ierr = PSSwitchFormat_GHIEP(ps,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = PetscMemcpy(ps->work,wr,n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=ps->l;i<n;i++) {
    wr[i] = *(ps->work + perm[i]);
  }
  ierr = PetscMemcpy(ps->work,wi,n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=ps->l;i<n;i++) {
    wi[i] = *(ps->work + perm[i]);
  }
  ierr = PetscMemcpy(ps->rwork,s,n*sizeof(PetscReal));CHKERRQ(ierr);
  for (i=ps->l;i<n;i++) {
    s[i] = *(ps->rwork+perm[i]);
  } 
  ierr = PetscMemcpy(ps->rwork,d,n*sizeof(PetscReal));CHKERRQ(ierr);
  for (i=ps->l;i<n;i++) {
    d[i] = *(ps->rwork  + perm[i]);
  }
  ierr = PetscMemcpy(ps->rwork,e,(n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemzero(e+ps->l,(n-1-ps->l)*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=ps->l;i<n-1;i++) {
    if(perm[i]<n-1) e[i] = *(ps->rwork + perm[i]);
  }
  if(!ps->compact){ ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);}
  ierr = PSPermuteColumns_Private(ps,ps->l,n,PS_MAT_Q,perm);CHKERRQ(ierr);
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
  if(x2==0) {
    *r = PetscAbsReal(x1);
    *c = (x1>=0)?1.0:-1.0;
    *s = 0.0;
    if(type) *type = 1;
    PetscFunctionReturn(0);
  }
  if(PetscAbsReal(x1) == PetscAbsReal(x2)){
    /* hyperbolic rotation doesn't exist */
    *c = 0;
    *s = 0;
    *r = 0;
    if(type) *type = 0;
    *cond = PETSC_MAX_REAL;
    PetscFunctionReturn(0);
  }
  
  if(PetscAbsReal(x1)>PetscAbsReal(x2)){
    xa = x1; xb = x2; type_ = 1;
  } else {
    xa = x2; xb = x1; type_ = 2;
  } 
  t = xb/xa;
  n2 = PetscAbsReal(1 - t*t);
  *r = PetscSqrtReal(n2)*PetscAbsReal(xa);
  *c = x1/(*r);
  *s = x2/(*r);
  if(type_ == 2) *r *= -1;
  if(type) *type = type_;
  if(cond) *cond = (PetscAbsReal(*c) + PetscAbsReal(*s))/PetscAbsReal(PetscAbsReal(*c) - PetscAbsReal(*s));
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
static PetscErrorCode HRApply(PetscInt n, PetscScalar *x1,PetscInt inc1, PetscScalar *x2, PetscInt inc2,PetscReal c, PetscReal s)
{
  PetscInt    i;
  PetscReal   t;
  PetscScalar tmp;
  
  PetscFunctionBegin;
  if(PetscAbsReal(c)>PetscAbsReal(s)){ /* Type I */
    t = s/c;
    for(i=0;i<n;i++){
      x1[i*inc1] = c*x1[i*inc1] + s*x2[i*inc2];
      x2[i*inc2] = t*x1[i*inc1] + x2[i*inc2]/c;
    }
  }else{ /* Type II */
    t = c/s;
    for(i=0;i<n;i++){
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
  Input:
    A symmetric (only lower triangular part is refered)
    s vector +1 and -1 (signature matrix)
  Output:
    d,e
    s
    Q s-orthogonal matrix whith Q^T*A*Q = T (symmetric tridiagonal matrix)
*/
static PetscErrorCode TridiagDiag_HHR(PetscInt n,PetscScalar *A,PetscInt lda,PetscReal *s,PetscScalar* Q,PetscInt ldq,PetscBool flip,PetscReal *d,PetscReal *e,PetscScalar *w,PetscInt lw,PetscInt *perm)
{
#if defined(PETSC_MISSING_LAPACK_LARFG) || defined(PETSC_MISSING_LAPACK_LARF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LARFG/LARF - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,*ii,*jj,tmp,type;
  PetscReal      *ss,cond,cs,sn,r,maxCond=1.0;
  PetscScalar    *work,tau,t;
  PetscBLASInt   n0,n1,ni,inc=1,m,n_,lda_,ldq_;

  PetscFunctionBegin;
  if(n<3){
    if(n==1)Q[0]=1;
    if(n==2){Q[0] = Q[1+ldq] = 1; Q[1] = Q[ldq] = 0;}
    PetscFunctionReturn(0);
  }
  lda_ = PetscBLASIntCast(lda);
  n_   = PetscBLASIntCast(n);
  ldq_ = PetscBLASIntCast(ldq);
  if (!w || lw < n*n) {
    ierr = PetscMalloc(n*n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }else work = w;

  /* Classify (and flip) A and s according to sign */
  ierr = PetscMalloc(n*sizeof(PetscReal),&ss);CHKERRQ(ierr); 
  if(flip) for(i=0;i<n;i++) {perm[i] = n-1-perm[i];}
  i=1;
  while(i<n-1 && s[perm[i-1]]==s[perm[0]]){
    if(s[perm[i]]!=s[perm[0]]){
      j=i+1;
      while(j<n-1 && s[perm[j]]!=s[perm[0]])j++;
      tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    i++;
  }
  for(i=0;i<n;i++){
    ierr = PetscMemcpy(work+i*n,A+i*lda,n*sizeof(PetscScalar));CHKERRQ(ierr);
    ss[i] = s[perm[i]];
  }
  if (flip){ ii = &j; jj = &i;} else { ii = &i; jj = &j;}
  for(i=0;i<n;i++)
    for(j=0;j<n;j++)
      A[i+j*lda] = work[perm[*ii]+perm[*jj]*n];
  /* Initialize Q */
  for(i=0;i<n;i++){
    ierr = PetscMemzero(Q+i*ldq,n*sizeof(PetscScalar));
    Q[perm[i]+i*ldq] = 1.0;
  }
  for(ni=1;ni<n && ss[ni]==ss[0]; ni++);
  n0 = ni-1; n1 = PetscBLASIntCast(n)-ni;
  for(j=0;j<n-2;j++){
    m = PetscBLASIntCast(n-j-1);
    /* Forming and applying reflectors */
    if( n0 > 1 ){
      LAPACKlarfg_(&n0, A+ni-n0+j*lda, A+ni-n0+j*lda+1,&inc,&tau);
      /* Apply reflector */
      if( PetscAbsScalar(tau) != 0.0 ){
        t=*( A+ni-n0+j*lda);  *(A+ni-n0+j*lda)=1.0;
        LAPACKlarf_("R",&m,&n0,A+ni-n0+j*lda,&inc,&tau,A+j+1+(j+1)*lda,&lda_,work);
        LAPACKlarf_("L",&n0,&m,A+ni-n0+j*lda,&inc,&tau,A+j+1+(j+1)*lda,&lda_,work);
        /* Update Q */
        LAPACKlarf_("R",&n_,&n0,A+ni-n0+j*lda,&inc,&tau,Q+(j+1)*ldq,&ldq_,work);
        *(A+ni-n0+j*lda) = t;
        for(i=1;i<n0;i++) {
          *(A+ni-n0+j*lda+i) = 0.0;  *(A+j+(ni-n0+i)*lda) = 0.0;
        }
        *(A+j+(ni-n0)*lda) = *(A+ni-n0+j*lda);
      }
    }
    if( n1 > 1 ){
      LAPACKlarfg_(&n1, A+n-n1+j*lda, A+n-n1+j*lda+1,&inc,&tau);
      /* Apply reflector */
      if( PetscAbsScalar(tau) != 0.0 ){
        t=*( A+n-n1+j*lda);  *(A+n-n1+j*lda)=1.0;
        LAPACKlarf_("R",&m,&n1,A+n-n1+j*lda,&inc,&tau,A+j+1+(n-n1)*lda,&lda_,work);
        LAPACKlarf_("L",&n1,&m,A+n-n1+j*lda,&inc,&tau,A+n-n1+(j+1)*lda,&lda_,work);
        /* Update Q */
        LAPACKlarf_("R",&n_,&n1,A+n-n1+j*lda,&inc,&tau,Q+(n-n1)*ldq,&ldq_,work);
        *(A+n-n1+j*lda) = t;
        for(i=1;i<n1;i++) {
          *(A+n-n1+i+j*lda) = 0.0;  *(A+j+(n-n1+i)*lda) = 0.0;
        }
        *(A+j+(n-n1)*lda) = *(A+n-n1+j*lda);
      }
    }
    /* Hyperbolic rotation */
    if( n0 > 0 && n1 > 0){
      ierr = HRGen(PetscRealPart(A[ni-n0+j*lda]),PetscRealPart(A[n-n1+j*lda]),&type,&cs,&sn,&r,&cond);CHKERRQ(ierr);
      if(cond>maxCond) maxCond = cond;
      A[ni-n0+j*lda] = r; A[n-n1+j*lda] = 0.0;
      A[j+(ni-n0)*lda] = r; A[j+(n-n1)*lda] = 0.0;
      /* Apply to A */
      ierr = HRApply(m, A+j+1+(ni-n0)*lda,1, A+j+1+(n-n1)*lda,1, cs, -sn);CHKERRQ(ierr);
      ierr = HRApply(m, A+ni-n0+(j+1)*lda,lda, A+n-n1+(j+1)*lda,lda, cs, -sn);CHKERRQ(ierr);
      
      /* Update Q */
      ierr = HRApply(n, Q+(ni-n0)*ldq,1, Q+(n-n1)*ldq,1, cs, -sn);CHKERRQ(ierr);
      if(type==2){
        ss[ni-n0] = -ss[ni-n0]; ss[n-n1] = -ss[n-n1];
        n0++;ni++;n1--;
      }
    }
    if(n0>0) n0--;else n1--;
  }
  /* flip matrices */
  if(flip){
    for(i=0;i<n-1;i++){
      d[i] = PetscRealPart(A[n-i-1+(n-i-1)*lda]);
      e[i] = PetscRealPart(A[n-i-1+(n-i-2)*lda]);
      s[i] = ss[n-i-1];
    }
    s[n-1] = ss[0];
    d[n-1] = PetscRealPart(A[0]);
    for(i=0;i<n;i++){
      ierr=PetscMemcpy(work+i*n,Q+i*ldq,n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    for(i=0;i<n;i++)
      for(j=0;j<n;j++)
        Q[i+j*ldq] = work[i+(n-j-1)*n];
  }else{
    for(i=0;i<n-1;i++){
      d[i] = PetscRealPart(A[i+i*lda]);
      e[i] = PetscRealPart(A[i+1+i*lda]);
      s[i] = ss[i];
    }
    s[n-1] = ss[n-1];
    d[n-1] = PetscRealPart(A[n-1 + (n-1)*lda]);
  }
  ierr = PetscFree(ss);CHKERRQ(ierr);
/* ///////////////////////////////// */
if(dbPS>=1){
PetscPrintf(PETSC_COMM_WORLD," maxCond in triadDiag: %g\n",maxCond); }
/* //////////////////////////////// */
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PSEigenVectorsPseudoOrthog"
static PetscErrorCode PSEigenVectorsPseudoOrthog(PS ps, PSMatType mat, PetscScalar *wr, PetscScalar *wi,PetscBool accum)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,off;
  PetscBLASInt   ld,n1,one=1;
  PetscScalar    PQ[4],xx,yx,xy,yy,*y,h,oneS=1.0,zeroS=0.0,*X,*W,*B;
  PetscReal      *ss,*s,*d,*e,d1,d2,toldeg=1e-5;/* ////////////// */

  PetscFunctionBegin;
  ld = PetscBLASIntCast(ps->ld);
  n1 = PetscBLASIntCast(ps->n - ps->l);
  ierr = PSAllocateWork_Private(ps,ld*ld+2*ld,ld,2*ld);CHKERRQ(ierr);
  s = ps->rmat[PS_MAT_D];
  d = ps->rmat[PS_MAT_T];
  e = d + ld;
  off = ps->l+ps->l*ld;
  if(!ps->compact){
    B = ps->mat[PS_MAT_B];
    for(i=ps->l;i<ps->n;i++){
      s[i] = PetscRealPart(B[i+i*ld]);
    }
  }

  /* compute real s-orthonormal base */
  X = ps->mat[mat];
  ss = ps->rwork;
  y = ps->work;
  //ierr = PSSwitchFormat_GHIEP(ps,(ps->compact)?PETSC_FALSE:PETSC_TRUE);CHKERRQ(ierr); /* only when final A is calculated from vectors */

#if defined(PETSC_USE_COMPLEX)
  /* with complex scalars we need to operate as in real scalar */
  for(i=ps->l;i<ps->n;i++){
    if(PetscImaginaryPart(wr[i])!=0.0){
      for(j=ps->l;j<ps->n;j++){
        X[j+(i+1)*ld] = PetscImaginaryPart(X[j+i*ld]);
        X[j+i*ld] = PetscRealPart(X[j+i*ld]);
      }
      i++;
    }
  }
#endif

  for(i=ps->l;i<ps->n;i++){
#if defined(PETSC_USE_COMPLEX)
    if(PetscImaginaryPart(wr[i])==0.0) { /* real */
#else
    if(wi[i]==0.0) { /* real */
#endif
      for(j=ps->l;j<i;j++){
         /* s-orthogonalization with close eigenvalues */
        if(wi[j]==0.0){
          if( PetscAbsScalar(wr[j]-wr[i])<toldeg){
            ierr = IndefOrthog(s+ps->l, X+j*ld+ps->l, ss[j],X+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
          }
        }else j++;
      }
      ierr = IndefNorm(s+ps->l,X+i*ld+ps->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0.0)?-1:1;
      d[i] = PetscRealPart(wr[i]*ss[i]); e[i] = 0.0;
    }else{
      for(j=ps->l;j<i;j++){
        /* s-orthogonalization of Xi and Xi+1*/
#if defined(PETSC_USE_COMPLEX)
        if(PetscImaginaryPart(wr[j])!=0.0) {
#else
        if(wi[j]!=0.0) {
#endif
          if(PetscAbsScalar(wr[j]-wr[i])<toldeg && PetscAbsScalar(PetscAbsScalar(wi[j])-PetscAbsScalar(wi[i]))<toldeg){
            for(k=ps->l;k<ps->n;k++) y[k] = s[k]*X[k+i*ld];
            xx = BLASdot_(&n1,X+ps->l+j*ld,&one,y+ps->l,&one);
            yx = BLASdot_(&n1,X+ps->l+(j+1)*ld,&one,y+ps->l,&one);
            for(k=ps->l;k<ps->n;k++) y[k] = s[k]*X[k+(i+1)*ld];
            xy = BLASdot_(&n1,X+ps->l+j*ld,&one,y+ps->l,&one);
            yy = BLASdot_(&n1,X+ps->l+(j+1)*ld,&one,y+ps->l,&one);
            PQ[0] = ss[j]*xx; PQ[1] = ss[j+1]*yx; PQ[2] = ss[j]*xy; PQ[3] = ss[j+1]*yy;
            for(k=ps->l;k<ps->n;k++){
              X[k+i*ld] -= PQ[0]*X[k+j*ld]+PQ[1]*X[k+(j+1)*ld];
              X[k+(i+1)*ld] -= PQ[2]*X[k+j*ld]+PQ[3]*X[k+(j+1)*ld];
            }
            /* Repeat */
            for(k=ps->l;k<ps->n;k++) y[k] = s[k]*X[k+i*ld];
            xx = BLASdot_(&n1,X+ps->l+j*ld,&one,y+ps->l,&one);
            yx = BLASdot_(&n1,X+ps->l+(j+1)*ld,&one,y+ps->l,&one);
            for(k=ps->l;k<ps->n;k++) y[k] = s[k]*X[k+(i+1)*ld];
            xy = BLASdot_(&n1,X+ps->l+j*ld,&one,y+ps->l,&one);
            yy = BLASdot_(&n1,X+ps->l+(j+1)*ld,&one,y+ps->l,&one);
            PQ[0] = ss[j]*xx; PQ[1] = ss[j+1]*yx; PQ[2] = ss[j]*xy; PQ[3] = ss[j+1]*yy;
            for(k=ps->l;k<ps->n;k++){
              X[k+i*ld] -= PQ[0]*X[k+j*ld]+PQ[1]*X[k+(j+1)*ld];
              X[k+(i+1)*ld] -= PQ[2]*X[k+j*ld]+PQ[3]*X[k+(j+1)*ld];
            }
          }
          j++;
        }
      }
      ierr = IndefNorm(s+ps->l,X+i*ld+ps->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0)?-1:1;
      ierr = IndefOrthog(s+ps->l, X+i*ld+ps->l, ss[i],X+(i+1)*ld+ps->l, &h,n1);CHKERRQ(ierr);
      ierr = IndefNorm(s+ps->l,X+(i+1)*ld+ps->l,&d2,n1);CHKERRQ(ierr);
      ss[i+1] = (d2<0)?-1:1;
/*
      BLASgemv_("N",&n1,&n1,&oneS,ps->mat[PS_MAT_A]+off,&ld,X+i*ld+ps->l,&one,&zeroS,y,&one);
      d[i] = BLASdot_(&n1,X+i*ld+ps->l,&one,y,&one);
      BLASgemv_("N",&n1,&n1,&oneS,ps->mat[PS_MAT_A]+off,&ld,X+(i+1)*ld+ps->l,&one,&zeroS,y,&one);
      d[i+1] = BLASdot_(&n1,X+(i+1)*ld+ps->l,&one,y,&one);
      e[i] = BLASdot_(&n1,X+(i)*ld+ps->l,&one,y,&one); e[i+1] = 0.0;
*/
      d[i] = PetscRealPart((wr[i]-wi[i]*h/d1)*ss[i]);
      d[i+1] = PetscRealPart((wr[i]+wi[i]*h/d1)*ss[i+1]);
      e[i] = PetscRealPart(wi[i]*d2/d1*ss[i]); e[i+1] = 0.0;
      i++;
    }
  }
  for(i=ps->l;i<ps->n;i++) s[i] = ss[i];
  /* accumulate previous Q */
  if(accum && mat!=PS_MAT_Q){
    ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
    W = ps->mat[PS_MAT_W];
    ierr = PSCopyMatrix_Private(ps,PS_MAT_W,PS_MAT_Q);CHKERRQ(ierr);
    BLASgemm_("N","N",&n1,&n1,&n1,&oneS,W+off,&ld,ps->mat[PS_MAT_X]+off,&ld,&zeroS,ps->mat[PS_MAT_Q]+off,&ld);
  }
  if(!ps->compact){ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PSGHIEPPseudoOrthogInverseIteration"
/*
  Get eigenvectors with inverse iteration.
  The system matrix is in Hessenberg form.
*/
static PetscErrorCode PSGHIEPPseudoOrthogInverseIteration(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(PETSC_MISSING_LAPACK_HSEIN)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HSEIN - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,off;
  PetscBLASInt   *select,*infoC,ld,n1,mout,info;
  PetscScalar    *A,*B,*H,*X;
  PetscReal      *s,*d,*e;

  PetscFunctionBegin;
  ld = PetscBLASIntCast(ps->ld);
  n1 = PetscBLASIntCast(ps->n - ps->l);
  ierr = PSAllocateWork_Private(ps,ld*ld+2*ld,ld,2*ld);CHKERRQ(ierr);
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  H = ps->mat[PS_MAT_W];
  s = ps->rmat[PS_MAT_D];
  d = ps->rmat[PS_MAT_T];
  e = d + ld;
  select = ps->iwork;
  infoC = ps->iwork + ld;
  off = ps->l+ps->l*ld;
  if(ps->compact){
    H[off] = d[ps->l]*s[ps->l];
    H[off+ld] = e[ps->l]*s[ps->l];
    for(i=ps->l+1;i<ps->n-1;i++){
      H[i+(i-1)*ld] = e[i-1]*s[i];
      H[i+i*ld] = d[i]*s[i];
      H[i+(i+1)*ld] = e[i]*s[i];
    }
    H[ps->n-1+(ps->n-2)*ld] = e[ps->n-2]*s[ps->n-1];
    H[ps->n-1+(ps->n-1)*ld] = d[ps->n-1]*s[ps->n-1];
  }else{
    s[ps->l] = PetscRealPart(B[off]);
    H[off] = A[off]*s[ps->l];
    H[off+ld] = A[off+ld]*s[ps->l];
    for(i=ps->l+1;i<ps->n-1;i++){
      s[i] = PetscRealPart(B[i+i*ld]);
      H[i+(i-1)*ld] = A[i+(i-1)*ld]*s[i];
      H[i+i*ld]     = A[i+i*ld]*s[i];
      H[i+(i+1)*ld] = A[i+(i+1)*ld]*s[i];
    }
    s[ps->n-1] = PetscRealPart(B[ps->n-1+(ps->n-1)*ld]);
    H[ps->n-1+(ps->n-2)*ld] = A[ps->n-1+(ps->n-2)*ld]*s[ps->n-1];
    H[ps->n-1+(ps->n-1)*ld] = A[ps->n-1+(ps->n-1)*ld]*s[ps->n-1];
  }
  ierr = PSAllocateMat_Private(ps,PS_MAT_X);CHKERRQ(ierr);
  X = ps->mat[PS_MAT_X];
  for(i=0;i<n1;i++)select[i]=1;
#if !defined(PETSC_USE_COMPLEX)
  LAPACKhsein_("R","N","N",select,&n1,H+off,&ld,wr+ps->l,wi+ps->l,PETSC_NULL,&ld,X+off,&ld,&n1,&mout,ps->work,PETSC_NULL,infoC,&info);
#else
  LAPACKhsein_("R","N","N",select,&n1,H+off,&ld,wr+ps->l,PETSC_NULL,&ld,X+off,&ld,&n1,&mout,ps->work,ps->rwork,PETSC_NULL,infoC,&info);
#endif
  if(info<0)SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in hsein routine %d",-i);
  if(info>0)SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Convergence error in hsein routine %d",i);

  ierr = PSEigenVectorsPseudoOrthog(ps, PS_MAT_X, wr, wi,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSIntermediate_GHIEP"
/*
   Reduce to tridiagonal-diagonal pair by means of TridiagDiag_HHR.
*/
static PetscErrorCode PSIntermediate_GHIEP(PS ps)
{
  PetscErrorCode ierr;
  PetscInt       i,ld,off;
  PetscScalar    *A,*B,*Q;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
  ld = ps->ld;
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  Q = ps->mat[PS_MAT_Q];
  d = ps->rmat[PS_MAT_T];
  e = ps->rmat[PS_MAT_T]+ld;
  s = ps->rmat[PS_MAT_D];
  off = ps->l+ps->l*ld;
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PSAllocateWork_Private(ps,ld*ld,0,0);CHKERRQ(ierr);

  for (i=0;i<ps->n;i++) Q[i+i*ld]=1.0;
  for (i=0;i<ps->n-ps->l;i++) *(ps->perm+i)=i;
  if(ps->compact){
    if(ps->state < PS_STATE_INTERMEDIATE){
      ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);
      ierr = TridiagDiag_HHR(ps->k-ps->l+1,A+off,ld,s+ps->l,Q+off,ld,PETSC_TRUE,d+ps->l,e+ps->l,ps->work,ld*ld,ps->perm);CHKERRQ(ierr);
      ps->k = ps->l;
      ierr = PetscMemzero(d+2*ld+ps->l,(ps->n-ps->l)*sizeof(PetscReal));CHKERRQ(ierr);
    }
  }else{
    if(ps->state < PS_STATE_INTERMEDIATE){
      for(i=0;i<ps->n;i++)
        s[i] = PetscRealPart(B[i+i*ld]);
      ierr = TridiagDiag_HHR(ps->n-ps->l,A+off,ld,s+ps->l,Q+off,ld,PETSC_FALSE,d+ps->l,e+ps->l,ps->work,ld*ld,ps->perm);CHKERRQ(ierr);
      ierr = PetscMemzero(d+2*ld,(ps->n)*sizeof(PetscReal));CHKERRQ(ierr);
      ps->k = ps->l;
      ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);
    }else{ ierr = PSSwitchFormat_GHIEP(ps,PETSC_TRUE);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PSGHIEPRealBlocks"
/*
   Undo 2x2 blocks that have real eigenvalues.
*/
PetscErrorCode PSGHIEPRealBlocks(PS ps)
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
  ld = PetscBLASIntCast(ps->ld);
  m = PetscBLASIntCast(ps->n-ps->l);
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  T = ps->rmat[PS_MAT_T];
  D = ps->rmat[PS_MAT_D];
  ierr = PSAllocateWork_Private(ps,2*m,0,0);CHKERRQ(ierr);
  for(i=ps->l;i<ps->n-1;i++){
    e = (ps->compact)?T[ld+i]:PetscRealPart(A[(i+1)+ld*i]);
    if(e != 0.0){ /* 2x2 block */
      if(ps->compact){
        s1 = D[i];
        d1 = T[i];
        s2 = D[i+1];
        d2 = T[i+1];
      }else{
        s1 = PetscRealPart(B[i*ld+i]);
        d1 = PetscRealPart(A[i*ld+i]);
        s2 = PetscRealPart(B[(i+1)*ld+i+1]);
        d2 = PetscRealPart(A[(i+1)*ld+i+1]);
      }
      isreal = PETSC_FALSE;
      if(s1==s2){ /* apply a Jacobi rotation to compute the eigendecomposition */
        dd = d1-d2;
        if(2*PetscAbsReal(e) <= dd){
          t = 2*e/dd;
          t = t/(1 + PetscSqrtReal(1+t*t));
        }else{
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
      }else{
        ss1 = 1.0; ss2 = 1.0,
        M[0] = d1; M[1] = e; M[2] = e; M[3]= d2;
        b[0] = s1; b[1] = 0.0; b[2] = 0.0; b[3] = s2;
        ep = LAPACKlamch_("S");
        /* Compute eigenvalues of the block */
        LAPACKlag2_(M, &two, b, &two,&ep , &scal1, &scal2, &wr1, &wr2, &wi);
        if(wi==0.0){ /* Real eigenvalues */
          isreal = PETSC_TRUE;
          if(scal1<ep||scal2<ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Nearly infinite eigenvalue");
          wr1 /= scal1; wr2 /= scal2;
          if( PetscAbsReal(s1*d1-wr1)<PetscAbsReal(s2*d2-wr1)){ Y[0] = wr1-s2*d2; Y[1] =s2*e;}
          else{ Y[0] = s1*e; Y[1] = wr1-s1*d1; }
          /* normalize with a signature*/
          maxy = PetscMax(PetscAbsScalar(Y[0]),PetscAbsScalar(Y[1]));
          scal1 = PetscRealPart(Y[0])/maxy; scal2 = PetscRealPart(Y[1])/maxy;
          snorm = scal1*scal1*s1 + scal2*scal2*s2;
          if(snorm<0){ss1 = -1.0; snorm = -snorm;}
          snorm = maxy*PetscSqrtReal(snorm); Y[0] = Y[0]/snorm; Y[1] = Y[1]/snorm;
          if( PetscAbsReal(s1*d1-wr2)<PetscAbsReal(s2*d2-wr2)){ Y[2] = wr2-s2*d2; Y[3] =s2*e;}
          else{ Y[2] = s1*e; Y[3] = wr2-s1*d1; }
          maxy = PetscMax(PetscAbsScalar(Y[2]),PetscAbsScalar(Y[3]));
          scal1 = PetscRealPart(Y[2])/maxy; scal2 = PetscRealPart(Y[3])/maxy;
          snorm = scal1*scal1*s1 + scal2*scal2*s2;
          if(snorm<0){ss2 = -1.0; snorm = -snorm;}
          snorm = maxy*PetscSqrtReal(snorm);Y[2] = Y[2]/snorm; Y[3] = Y[3]/snorm;
        }
        wr1 *= ss1; wr2 *= ss2;
      }
      if(isreal){
        if(ps->compact) {
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
        BLASgemm_("N","N",&m,&two,&two,&oneS,ps->mat[PS_MAT_Q]+ps->l+i*ld,&ld,Y,&two,&zeroS,ps->work,&m);
        ierr = PetscMemcpy(ps->mat[PS_MAT_Q]+ps->l+i*ld,ps->work,m*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscMemcpy(ps->mat[PS_MAT_Q]+ps->l+(i+1)*ld,ps->work+m,m*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      i++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GHIEP_QR_II"
PetscErrorCode PSSolve_GHIEP_QR_II(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(PETSC_MISSING_LAPACK_HSEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HSEQR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,off;
  PetscBLASInt   n1,ld,one,info,lwork;
  PetscScalar    *H,*A,*B,*Q,*work;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
/* ////////////////////// */
ierr = PetscOptionsGetInt(PETSC_NULL,"-dbPS",&dbPS,PETSC_NULL);CHKERRQ(ierr);
/* ////////////////////// */
  one = 1;
  n1 = PetscBLASIntCast(ps->n - ps->l);
  ld = PetscBLASIntCast(ps->ld);
  off = ps->l + ps->l*ld;
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  Q = ps->mat[PS_MAT_Q];
  d = ps->rmat[PS_MAT_T];
  e = ps->rmat[PS_MAT_T] + ld;
  s = ps->rmat[PS_MAT_D];
  ierr = PSAllocateWork_Private(ps,ld*ld,2*ld,ld*2);CHKERRQ(ierr); 
  work = ps->work;
  lwork = ld*ld;

  /* Quick return if possible */
  if (n1 == 1) {
    *(Q+off) = 1;
    if(ps->compact){
      wr[ps->l] = d[ps->l]/s[ps->l];
      wi[ps->l] = 0.0;
    }else{
      d[ps->l] = PetscRealPart(A[off]);
      s[ps->l] = PetscRealPart(B[off]);
      wr[ps->l] = d[ps->l]/s[ps->l];
      wi[ps->l] = 0.0;  
    }
    PetscFunctionReturn(0);
  }
  /* Reduce to pseudotriadiagonal form */
  ierr = PSIntermediate_GHIEP( ps);CHKERRQ(ierr);

/* //////////////////////////// */
PetscViewer viewer;
PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);
if(dbPS>1){
printf("tras tridiagonalizar\n");
PSView(ps,viewer);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
/* ///////////////////// */

  /* Compute Eigenvalues (QR)*/
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  H = ps->mat[PS_MAT_W];
  if(ps->compact){
    H[off] = d[ps->l]*s[ps->l];
    H[off+ld] = e[ps->l]*s[ps->l];
    for(i=ps->l+1;i<ps->n-1;i++){
      H[i+(i-1)*ld] = e[i-1]*s[i];
      H[i+i*ld]     = d[i]*s[i];
      H[i+(i+1)*ld] = e[i]*s[i];
    }
    H[ps->n-1+(ps->n-2)*ld] = e[ps->n-2]*s[ps->n-1];
    H[ps->n-1+(ps->n-1)*ld] = d[ps->n-1]*s[ps->n-1];
  }else{
    s[ps->l] = PetscRealPart(B[off]);
    H[off] = A[off]*s[ps->l];
    H[off+ld] = A[off+ld]*s[ps->l];
    for(i=ps->l+1;i<ps->n-1;i++){
      s[i] = PetscRealPart(B[i+i*ld]);
      H[i+(i-1)*ld] = A[i+(i-1)*ld]*s[i];
      H[i+i*ld]     = A[i+i*ld]*s[i];
      H[i+(i+1)*ld] = A[i+(i+1)*ld]*s[i];
    }
    s[ps->n-1] = PetscRealPart(B[ps->n-1+(ps->n-1)*ld]);
    H[ps->n-1+(ps->n-2)*ld] = A[ps->n-1+(ps->n-2)*ld]*s[ps->n-1];
    H[ps->n-1+(ps->n-1)*ld] = A[ps->n-1+(ps->n-1)*ld]*s[ps->n-1];
  }

#if !defined(PETSC_USE_COMPLEX)
  LAPACKhseqr_("E","N",&n1,&one,&n1,H+off,&ld,wr+ps->l,wi+ps->l,PETSC_NULL,&ld,work,&lwork,&info);
#else
  LAPACKhseqr_("E","N",&n1,&one,&n1,H+off,&ld,wr,PETSC_NULL,&ld,work,&lwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",&info);

  /* Compute Eigenvectors with Inverse Iteration */
  ierr = PSGHIEPPseudoOrthogInverseIteration(ps,wr,wi);CHKERRQ(ierr);
/* ////////////////////// */
if(dbPS>1){
printf("PseudoOrthog\n");
ierr = PSView(ps,viewer);CHKERRQ(ierr);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
/* ///////////////////// */

  ierr = PSSolve_GHIEP_Sort(ps,wr,wi);CHKERRQ(ierr);

/* ////////////////////// */
if(dbPS>1){
printf("SORT\n");
ierr = PSView(ps,viewer);CHKERRQ(ierr);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
/* ////////////////////// */
    PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GHIEP_QR"
PetscErrorCode PSSolve_GHIEP_QR(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GEHRD) || defined(SLEPC_MISSING_LAPACK_ORGHR) || defined(PETSC_MISSING_LAPACK_HSEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEHRD/ORGHR/HSEQR - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,off;
  PetscBLASInt   lwork,info,n1,one=1,mout,ld;
  PetscScalar    *A,*B,*H,*Q,*work,*tau;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
  n1 = PetscBLASIntCast(ps->n - ps->l);
  ld = PetscBLASIntCast(ps->ld);
  off = ps->l + ps->l*ld;
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  Q = ps->mat[PS_MAT_Q];
  d = ps->rmat[PS_MAT_T];
  e = ps->rmat[PS_MAT_T] + ld;
  s = ps->rmat[PS_MAT_D];
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  H = ps->mat[PS_MAT_W];
  ierr = PSAllocateWork_Private(ps,ld+ld*ld,ld,0);CHKERRQ(ierr); 
  tau  = ps->work;
  work = ps->work+ld;
  lwork = ld*ld;

/* //////////////////////// */
PetscViewer viewer;
ierr = PetscOptionsGetInt(PETSC_NULL,"-dbPS",&dbPS,PETSC_NULL);CHKERRQ(ierr);
ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
/* /////////////////////// */

   /* initialize orthogonal matrix */
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i< ps->n;i++) 
    Q[i+i*ld] = 1.0;
  /* quick return */
  if (n1 == 1) {
    if(ps->compact){
      wr[ps->l] = d[ps->l]/s[ps->l];
      wi[ps->l] = 0.0;
    }else{
      d[ps->l] = PetscRealPart(A[off]);
      s[ps->l] = PetscRealPart(B[off]);
      wr[ps->l] = d[ps->l]/s[ps->l];
      wi[ps->l] = 0.0;  
    }
    PetscFunctionReturn(0);
  }

  /* form standard problem in H */
  if (ps->compact) {
    ierr = PetscMemzero(H,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for(i=ps->l; i < ps->n-1; i++){
      H[i+i*ld] = d[i]/s[i];
      H[(i+1)+i*ld] = e[i]/s[i+1];
      H[i+(i+1)*ld] = e[i]/s[i];
    } 
    H[ps->n-1 + (ps->n-1)*ld] = d[ps->n-1]/s[ps->n-1];

    for(i=ps->l; i < ps->k; i++){
      H[ps->k+i*ld] = *(ps->rmat[PS_MAT_T]+2*ld+i)/s[ps->k];
      H[i+ps->k*ld] = *(ps->rmat[PS_MAT_T]+2*ld+i)/s[i];
    }
  }else{
    for(j=ps->l; j<ps->n; j++){
      for(i=ps->l; i<ps->n; i++){
        H[i+j*ld] = A[i+j*ld]/B[i+i*ld];
      }
    }
  }
  /* reduce to upper Hessenberg form */
  if (ps->state<PS_STATE_INTERMEDIATE) {
    LAPACKgehrd_(&n1,&one,&n1,H+off,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",&info);
    for (j=ps->l;j<ps->n-1;j++) {
      for (i=j+2;i<ps->n;i++) {
        Q[i+j*ld] = H[i+j*ld];
        H[i+j*ld] = 0.0;
      }
    }
    LAPACKorghr_(&n1,&one,&n1,Q+off,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGHR %d",&info);
  }

  /* compute the real Schur form */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKhseqr_("S","V",&n1,&one,&n1,H+off,&ld,wr+ps->l,wi+ps->l,Q+off,&ld,work,&lwork,&info);
#else
  LAPACKhseqr_("S","V",&n1,&one,&n1,H+off,&ld,wr,Q+off,&ld,work,&lwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",&info);
  
  /* compute eigenvectors */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKtrevc_("R","B",PETSC_NULL,&n1,H+off,&ld,PETSC_NULL,&ld,Q+off,&ld,&n1,&mout,ps->work,&info);
#else
  LAPACKtrevc_("R","B",PETSC_NULL,&n1,H+off,&ld,PETSC_NULL,&ld,Q+off,&ld,&n1,&mout,work,ps->rwork,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",&info);

  /* compute real s-orthonormal basis */
  ierr = PSEigenVectorsPseudoOrthog(ps, PS_MAT_Q, wr, wi,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PSGHIEPRealBlocks(ps);CHKERRQ(ierr);
  ierr = PSSolve_GHIEP_Sort(ps,wr,wi);CHKERRQ(ierr);
/* ////////////////////// */
if(dbPS>1){
printf("pseudoOrthog\n");
ierr = PSView(ps,viewer);CHKERRQ(ierr);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
/* ////////////////////// */
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "UnifiedRotation"
/*
   Sets up a 2-by-2 matrix to eliminate y in the vector [x y]'.
   Transformation is rotator if sygn = 1 and hyperbolic if sygn = -1.
*/
static PetscErrorCode UnifiedRotation(PetscReal x,PetscReal y,PetscReal sygn,PetscReal *rot,PetscReal *rcond,PetscBool *swap)
{
  PetscReal nrm,c,s;

  PetscFunctionBegin;
  *swap = PETSC_FALSE;
  if (y == 0) {
    rot[0] = 1.0; rot[1] = 0.0; rot[2] = 0.0; rot[3] = 1.0;
    *rcond = 1.0;
  } else {
    nrm = PetscMax(PetscAbs(x),PetscAbs(y));
    c = x/nrm; s = y/nrm;
    if (sygn == 1.0) {  /* set up a rotator */
      nrm = PetscSqrtReal(c*c+s*s);     
      c = c/nrm; s = s/nrm;
      /* rot = [c s; -s c]; */
      rot[0] = c; rot[1] = -s; rot[2] = s; rot[3] = c;
      *rcond = 1.0;
    } else if (sygn == -1) {  /* set up a hyperbolic transformation */
      nrm = c*c-s*s;
      if (nrm > 0) nrm = PetscSqrtReal(nrm);
      else if (nrm < 0) {
        nrm = PetscSqrtReal(-nrm);
        *swap = PETSC_TRUE;
      } else {  /* breakdown */
        SETERRQ(PETSC_COMM_SELF,1,"Breakdown in construction of hyperbolic transformation.");
        rot[0] = 1.0; rot[1] = 0.0; rot[2] = 0.0; rot[3] = 1.0;
        *rcond = 0.0;
        PetscFunctionReturn(0);
      }
      c = c/nrm; s = s/nrm;
      /* rot = [c -s; -s c]; */
      rot[0] = c; rot[1] = -s; rot[2] = -s; rot[3] = c;
      *rcond = PetscAbs(PetscAbs(s)-PetscAbs(c))/(PetscAbs(s)+PetscAbs(c));
    } else SETERRQ(PETSC_COMM_SELF,1,"Value of sygn sent to transetup must be 1 or -1.");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "HZStep"
static PetscErrorCode HZStep(PetscBLASInt ntop,PetscBLASInt nn,PetscReal tr,PetscReal dt,PetscReal *aa,PetscReal *bb,PetscReal *dd,PetscScalar *uu,PetscInt n,PetscInt ld,PetscBool *flag)
{
  PetscErrorCode ierr;
  PetscBLASInt   one=1;
  PetscInt       k,jj;
  PetscBLASInt   n_;
  PetscReal      bulge10,bulge20,bulge30,bulge31,bulge41,bulge42;
  PetscReal      sygn,rcond,worstcond,rot[4],buf[2];
  PetscScalar    rtmp;
  PetscBool      swap;

  PetscFunctionBegin;
  worstcond = 1.0;
  n_ = PetscBLASIntCast(n);

  /* Build initial bulge that sets step in motion */
  bulge10 = dd[ntop+1]*(aa[ntop]*(aa[ntop] - dd[ntop]*tr) + dt*dd[ntop]*dd[ntop]) + dd[ntop]*bb[ntop]*bb[ntop]; 
  bulge20 = bb[ntop]*(dd[ntop+1]*aa[ntop] + dd[ntop]*aa[ntop+1] - dd[ntop]*dd[ntop+1]*tr);
  bulge30 = bb[ntop]*bb[ntop+1]*dd[ntop];
  bulge31 = 0.0;
  bulge41 = 0.0;
  bulge42 = 0.0;

  /* Chase the bulge */
  for (jj=ntop;jj<nn-1;jj++) {
  
    /* Check for trivial bulge */
    if (jj>ntop && PetscMax(PetscMax(PetscAbs(bulge10),PetscAbs(bulge20)),PetscAbs(bulge30))<PETSC_MACHINE_EPSILON*(PetscAbs(aa[jj]) + PetscAbs(aa[jj+1]))) {
      bb[jj-1] = 0.0;  /* deflate and move on */
  
    } else { /* carry out the step */

      /* Annihilate tip entry bulge30 */
      if (bulge30 != 0.0) { 
      
        /* Make an interchange if necessary to ensure that the 
           first transformation is othogonal, not hyperbolic.  */
        if (dd[jj+1] != dd[jj+2]) { /* make an interchange */
          if (dd[jj] != dd[jj+1]) {  /* interchange 1st and 2nd */
            buf[0] = bulge20; bulge20 = bulge10; bulge10 = buf[0];
            buf[0] = aa[jj]; aa[jj] = aa[jj+1]; aa[jj+1] = buf[0];
            buf[0] = bb[jj+1]; bb[jj+1] = bulge31; bulge31 = buf[0];
            buf[0] = dd[jj]; dd[jj] = dd[jj+1]; dd[jj+1] = buf[0];
            for (k=0;k<n;k++) {
              rtmp = uu[k+jj*ld]; uu[k+jj*ld] = uu[k+(jj+1)*ld]; uu[k+(jj+1)*ld] = rtmp;
            }
          } else {  /* interchange 1st and 3rd */
            buf[0] = bulge30; bulge30 = bulge10; bulge10 = buf[0];
            buf[0] = aa[jj]; aa[jj] = aa[jj+2]; aa[jj+2] = buf[0];
            buf[0] = bb[jj]; bb[jj] = bb[jj+1]; bb[jj+1] = buf[0];
            buf[0] = dd[jj]; dd[jj] = dd[jj+2]; dd[jj+2] = buf[0];
            if (jj + 2 < nn-1) {
              bulge41 = bb[jj+2];
              bb[jj+2] = 0;
            }
            for (k=0;k<n;k++) {
              rtmp = uu[k+jj*ld]; uu[k+jj*ld] = uu[k+(jj+2)*ld]; uu[k+(jj+2)*ld] = rtmp;
            }
          }
        }
    
        /* Set up transforming matrix rot. */
        ierr = UnifiedRotation(bulge20,bulge30,1,rot,&rcond,&swap);CHKERRQ(ierr);

        /* Apply transforming matrix rot to T. */
        bulge20 = rot[0]*bulge20 + rot[2]*bulge30;
        buf[0] = rot[0]*bb[jj] + rot[2]*bulge31;
        buf[1] = rot[1]*bb[jj] + rot[3]*bulge31;
        bb[jj] = buf[0];
        bulge31 = buf[1];
        buf[0] = rot[0]*rot[0]*aa[jj+1] + 2.0*rot[0]*rot[2]*bb[jj+1] + rot[2]*rot[2]*aa[jj+2];
        buf[1] = rot[1]*rot[1]*aa[jj+1] + 2.0*rot[3]*rot[1]*bb[jj+1] + rot[3]*rot[3]*aa[jj+2];
        bb[jj+1] = rot[1]*rot[0]*aa[jj+1] + rot[3]*rot[2]*aa[jj+2] + (rot[3]*rot[0] + rot[1]*rot[2])*bb[jj+1];
        aa[jj+1] = buf[0];
        aa[jj+2] = buf[1];
        if (jj + 2 < nn-1) {
          bulge42 = bb[jj+2]*rot[2];
          bb[jj+2] = bb[jj+2]*rot[3];
        }

        /* Accumulate transforming matrix */
        BLASrot_(&n_,uu+(jj+1)*ld,&one,uu+(jj+2)*ld,&one,&rot[0],&rot[2]);
      }

      /* Annihilate inner entry bulge20 */
      if (bulge20 != 0.0) {

        /* Begin by setting up transforming matrix rot */
        sygn = dd[jj]*dd[jj+1];
        ierr = UnifiedRotation(bulge10,bulge20,sygn,rot,&rcond,&swap);CHKERRQ(ierr);
        if (rcond<PETSC_MACHINE_EPSILON) {
          SETERRQ1(PETSC_COMM_SELF,0,"Transforming matrix is numerically singular rcond=%g.",rcond);
          *flag = PETSC_TRUE;
          PetscFunctionReturn(0);
        }
        if (rcond < worstcond) worstcond = rcond;

        /* Apply transforming matrix rot to T */
        if (jj > ntop) bb[jj-1] = rot[0]*bulge10 + rot[2]*bulge20;
        buf[0] = rot[0]*rot[0]*aa[jj] + 2*rot[0]*rot[2]*bb[jj] + rot[2]*rot[2]*aa[jj+1];
        buf[1] = rot[1]*rot[1]*aa[jj] + 2*rot[3]*rot[1]*bb[jj] + rot[3]*rot[3]*aa[jj+1];
        bb[jj] = rot[1]*rot[0]*aa[jj] + rot[3]*rot[2]*aa[jj+1] + (rot[3]*rot[0] + rot[1]*rot[2])*bb[jj];
        aa[jj] = buf[0];
        aa[jj+1] = buf[1];
        if (jj + 1 < nn-1) {
          /* buf = [ bulge31 bb(jj+1) ] * rot' */
          buf[0] = rot[0]*bulge31 + rot[2]*bb[jj+1];
          buf[1] = rot[1]*bulge31 + rot[3]*bb[jj+1];
          bulge31 = buf[0];
          bb[jj+1] = buf[1];
        }
        if (jj + 2 < nn-1) {
          /* buf = [bulge41 bulge42] * rot' */
          buf[0] = rot[0]*bulge41 + rot[2]*bulge42;
          buf[1] = rot[1]*bulge41 + rot[3]*bulge42;
          bulge41 = buf[0];
          bulge42 = buf[1];
        }

        /* Apply transforming matrix rot to D */
        if (swap == 1) {
          buf[0] = dd[jj]; dd[jj] = dd[jj+1]; dd[jj+1] = buf[0];
        }

        /* Accumulate transforming matrix, uu(jj:jj+1,:) = rot*uu(jj:jj+1,:) */
        if (sygn==1) {
          BLASrot_(&n_,uu+jj*ld,&one,uu+(jj+1)*ld,&one,&rot[0],&rot[2]);
        } else {
          ierr = HRApply(n,uu+jj*ld,1,uu+(jj+1)*ld,1,rot[0],rot[1]);CHKERRQ(ierr);
        }
      }
    }

    /* Adjust bulge for next step */
    bulge10 = bb[jj];
    bulge20 = bulge31;
    bulge30 = bulge41;
    bulge31 = bulge42; 
    bulge41 = 0.0;
    bulge42 = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "HZIteration"
static PetscErrorCode HZIteration(PetscBLASInt nn,PetscBLASInt cgd,PetscReal *aa,PetscReal *bb,PetscReal *dd,PetscScalar *uu,PetscBLASInt ld)
{
  PetscErrorCode ierr;
  PetscBLASInt   j2,one=1;
  PetscInt       its,nits,nstop,jj,ntop,nbot,ntry;
  PetscReal      htr,det,dis,dif,tn,kt,c,s,tr,dt;
  PetscBool      flag=PETSC_FALSE;

  PetscFunctionBegin;
  its = 0;
  nbot = nn-1;
  nits = 0;
  nstop = 40*(nn - cgd);

  while (nbot >= cgd && nits < nstop) {

    /* Check for zeros on the subdiagonal */
    jj = nbot - 1;
    while (jj>=cgd && PetscAbs(bb[jj])>PETSC_MACHINE_EPSILON*(PetscAbs(aa[jj])+PetscAbs(aa[jj+1]))) jj = jj-1;
    if (jj>=cgd) bb[jj]=0;
    ntop = jj + 1;  /* starting point for step */
    if (ntop == nbot) {  /* isolate single eigenvalue */
      nbot = ntop - 1;
      its = 0;
    } else if (ntop+1 == nbot) {  /* isolate pair of eigenvalues */
      htr = 0.5*(aa[ntop]*dd[ntop] + aa[nbot]*dd[nbot]);
      det = dd[ntop]*dd[nbot]*(aa[ntop]*aa[nbot]-bb[ntop]*bb[ntop]);
      dis = htr*htr - det;
      if (dis > 0) {  /* distinct real eigenvalues */
        if (dd[ntop] == dd[nbot]) {  /* separate the eigenvalues by a Jacobi rotator */
          dif = aa[ntop]-aa[nbot];
          if (2.0*PetscAbs(bb[ntop])<=dif) {
            tn = 2*bb[ntop]/dif;
            tn = tn/(1.0 + PetscSqrtScalar(1.0+tn*tn));
          } else {
            kt = dif/(2.0*bb[ntop]);
            tn = PetscSign(kt)/(PetscAbs(kt)+PetscSqrtScalar(1.0+kt*kt));
          }
          c = 1.0/PetscSqrtScalar(1.0 + tn*tn);
          s = c*tn;
          aa[ntop] = aa[ntop] + tn*bb[ntop];
          aa[nbot] = aa[nbot] - tn*bb[ntop];
          bb[ntop] = 0;
          j2 = nn-cgd;
          BLASrot_(&j2,uu+ntop*ld+cgd,&one,uu+nbot*ld+cgd,&one,&c,&s);
        } else {
          dis = PetscSqrtScalar(dis);
          if (htr < 0) dis = -dis;
        }
      }
      nbot = ntop - 1;
    } else {  /* Do an HZ iteration */
      its = its + 1;
      nits = nits + 1;
      tr = aa[nbot-1]*dd[nbot-1] + aa[nbot]*dd[nbot];
      dt = dd[nbot-1]*dd[nbot]*(aa[nbot-1]*aa[nbot]-bb[nbot-1]*bb[nbot-1]);
      for (ntry=1;ntry<=6;ntry++) {
        ierr = HZStep(ntop,nbot+1,tr,dt,aa,bb,dd,uu,nn,ld,&flag);CHKERRQ(ierr);
        if (!flag) break;
        else if (ntry == 6) 
          SETERRQ(PETSC_COMM_SELF,1,"Unable to complete hz step on six tries");
        else {
          tr = 0.9*tr; dt = 0.81*dt;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GHIEP_HZ"
PetscErrorCode PSSolve_GHIEP_HZ(PS ps,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  PetscInt       off;
  PetscBLASInt   n1,ld;
  PetscScalar    *A,*B,*Q;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
////////////////////
ierr = PetscOptionsGetInt(PETSC_NULL,"-dbPS",&dbPS,PETSC_NULL);CHKERRQ(ierr);
PetscViewer viewer;
PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);
if(dbPS>1){
printf("En PSSolve_HZ");
PSView(ps,viewer);
}
/////////////////////
  n1  = ps->n - ps->l;
  ld = PetscBLASIntCast(ps->ld);
  off = ps->l + ps->l*ld;
  A  = ps->mat[PS_MAT_A];
  B  = ps->mat[PS_MAT_B];
  Q = ps->mat[PS_MAT_Q];
  d = ps->rmat[PS_MAT_T];
  e = ps->rmat[PS_MAT_T] + ld;
  s  = ps->rmat[PS_MAT_D];
  /* Quick return */
  if (n1 == 1) {
    *(Q+off) = 1;
    if(ps->compact){
      wr[ps->l] = d[ps->l]/s[ps->l]; wi[ps->l] = 0.0;
    }else{
      d[ps->l] = PetscRealPart(A[off]); s[ps->l] = PetscRealPart(B[off]);
      wr[ps->l] = d[ps->l]/s[ps->l]; wi[ps->l] = 0.0;  
    }
    PetscFunctionReturn(0);
  }
  /* Reduce to pseudotriadiagonal form */
  ierr = PSIntermediate_GHIEP(ps);CHKERRQ(ierr);
/////////////////////
if(dbPS>1){
printf("tridiagonalización \n");
PSView(ps,viewer);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
////////////////////

  ierr = HZIteration(ps->n,ps->l,d,e,s,Q,ld);CHKERRQ(ierr);
  if(!ps->compact){
    ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);
  }
/////////////////////
if(dbPS>1){
printf("Tras HZ \n");
PSView(ps,viewer);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
////////////////////

   ierr = PSGHIEPRealBlocks(ps);CHKERRQ(ierr);
/////////////////////
if(dbPS>1){
printf("Blocks \n");
PSView(ps,viewer);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
////////////////////
  /* Recover eigenvalues from diagonal */
  ierr = PSGHIEPComplexEigs(ps, 0, ps->n, wr, wi);CHKERRQ(ierr);/* /////////////// */
  ierr = PSSolve_GHIEP_Sort(ps,wr,wi);CHKERRQ(ierr);
/////////////////////
if(dbPS>1){
printf("Sort \n");
PSView(ps,viewer);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
////////////////////
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ScanJ"
/*
  INPUT:
    a ---- diagonal of J
    b ---- subdiagonal of J;
    the superdiagonal of J is all 1's

  OUTPUT:
    For an eigenvalue lambda of J we have:
      gl=<real(lambda)<=gr
      -sigma=<imag(lambda)<=sigma
*/
static PetscErrorCode ScanJ(PetscInt n,PetscReal *a,PetscReal *b,PetscReal *gl,PetscReal *gr,PetscReal *sigma)
{
  PetscInt  i;
  PetscReal b0,b1,rad;

  PetscFunctionBegin;
  /* For original matrix C, C_bal=T+S; T-symmetric and S=skew-symmetric
   C_bal is the balanced form of C */ 
  /* Bounds on the imaginary part of C (Gersgorin bound for S)*/
  *sigma = 0.0;
  b0 = 0.0;
  for(i=0;i<n-1;i++){
    if(b[i]<0.0) b1 = PetscSqrtReal(-b[i]);
    else b1 = 0.0;
    *sigma = PetscMax(*sigma,b1+b0);
    b0 = b1;
  }
  *sigma = PetscMax(*sigma,b0);
  /* Gersgorin bounds for T (=Gersgorin bounds on the real part for C) */
  rad = (b[0]>0.0)?PetscSqrtReal(b[0]):0.0; /* rad = b1+b0, b0 = 0 */
  *gr = a[0]+rad;
  *gl = a[0]-rad;
  b0 = rad;
  for(i=1;i<n-1;i++){
    b1 = (b[i]>0.0)?PetscSqrtReal(b[i]):0.0;
    rad = b0+b1;
    *gr = PetscMax(*gr,a[i]+rad);
    *gl = PetscMin(*gl,a[i]-rad);
    b0 = b1;
  }
  rad = b0;
  *gr = PetscMax(*gr,a[n-1]+rad);
  *gl = PetscMin(*gl,a[n-1]-rad);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Prologue"
/* 
  INPUT: 
    a  - vector with the diagonal elements
    b  - vector with the subdiagonal elements
    gl - Gersgorin left bound (real axis)
    gr - Gersgorin right bound (real axis)
  OUTPUT:
    eigvalue - multiple eigenvalue (if there is an eigenvalue)
    m        - its multiplicity    (m=0 if there isn't a multiple eigenvalue)
    X        - matrix of generalized eigenvectors  
    shift    
*/
//static PetscErrorCode Prologue(PetscInt n,PetscReal *a,PetscReal *b,PetscReal gl,PetscReal gr,PetscInt *m,PetscScalar *X,PetscReal *shift,PetscReal *w,PetscReal nw)
static PetscErrorCode Prologue(PetscInt n,PetscReal *a,PetscReal *b,PetscReal gl,PetscReal gr,PetscInt *m,PetscReal *shift,PetscReal *w,PetscReal nw)
{

////////////////// eliminar parámetro X
  PetscErrorCode ierr;
  PetscReal      mu,tol,*a1,*work,*y,*yp,*x,*xp;
  PetscInt       i,k,nwall=0;

  PetscFunctionBegin;
  *m = 0;
  mu = 0.0;
  for(i=0;i<n;i++) mu += a[i];
  mu /= n;
  tol=n*PETSC_MACHINE_EPSILON*(gr-gl);
  nwall = 5*n+4;
  if(w && nw>=nwall) {
    work = w;
    nwall = nw;
  }else {
    ierr = PetscMalloc(nwall*sizeof(PetscReal),&work);CHKERRQ(ierr);
    ierr = PetscMemzero(work,nwall*sizeof(PetscReal));CHKERRQ(ierr);
  }
  a1 = work; /* size n */
  y = work+n; /* size n+1 */
  yp = y+n+1; /* size n+1. yp is the derivative of y (p for "prime") */
  x = yp+n+1; /* size n+1 */
  xp = x+n+1; /* size n+1 */
  for(i=0;i<n;i++) a1[i] = mu-a[i];
  x[0] = 1;
  xp[0] = 0;
  x[1] = a1[0];
  xp[1] = 1;
  for(i=1;i<n;i++){
    x[i+1]=a1[i]*x[i]-b[i-1]*x[i-1];
    xp[i+1]=a1[i]*xp[i]+x[i]-b[i-1]*xp[i-1];
  }
  *shift = mu;
  if( PetscAbsReal(x[n])<tol){   
    /* mu is an eigenvalue */
    *m = *m+1;
    if( PetscAbsReal(xp[n])<tol ){
      /* mu is a multiple eigenvalue; Is it the one-point spectrum case? */
      //ierr = PetscMemcpy(x,X,n*sizeof(PetscReal));CHKERRQ(ierr); /* matrix of eigenvectors */
      k = 0;
      while(PetscAbsReal(xp[n])<tol && k<n-1){
           ierr = PetscMemcpy(x,y,(n+1)*sizeof(PetscReal));CHKERRQ(ierr);
           ierr = PetscMemcpy(xp,yp,(n+1)*sizeof(PetscReal));CHKERRQ(ierr);
           //x(1:beg)=zeros(1,beg);
           x[k] = 0.0;
           k++;
           x[k] = 1.0;
           xp[k] = 0.0;
           x[k+1] = a1[k] + y[k];
           xp[k+1] = 1+yp[k];
           for(i=k+1;i<n;i++){
             x[i+1] = a1[i]*x[i]-b[i-1]*x[i-1]+y[i];
             xp[i+1]=a1[i]*xp[i]+x[i]-b[i-1]*xp[i-1]+yp[i];
           }
           *m = *m+1;
           //ierr = PetscMemcpy(x,X+k*n,n*sizeof(PetscReal));CHKERRQ(ierr);
      }
    }     
  }
  if(work != w){
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
/*
  When mu is not an eigenvalue or it it an eigenvalue but it is not the one-point spectrum case, we will always have shift=mu

  Need to check for overflow!

  After calling Prologue, eigenComplexdqds and eigen3dqds will test if m==n in which case we have the one-point spectrum case; 
  If m!=0, the only output to be used is the shift returned.
*/
  PetscFunctionReturn(0);
}  


#undef __FUNCT__
#define __FUNCT__ "LUfac"
static PetscErrorCode LUfac(PetscInt n,PetscReal *a,PetscReal *b,PetscReal shift,PetscReal tol,PetscReal norm,PetscReal *L,PetscReal *U,PetscInt *fail,PetscReal *w,PetscInt nw){
  PetscErrorCode ierr;
  PetscInt       nwall,i;
  PetscReal      *work,*a1;

  PetscFunctionBegin;
  nwall = n;
  if(w && nw>=nwall){
    work = w;
    nwall = nw;
  }else{
    ierr = PetscMalloc(nwall*sizeof(PetscReal),&work);CHKERRQ(ierr);
  }
  a1 = work;
  for(i=0;i<n;i++) a1[i] = a[i]-shift;
  *fail = 0;
  for(i=0;i<n-1;i++){
    U[i] = a1[i];
    L[i] = b[i]/U[i];
    a1[i+1] = a1[i+1]-L[i];
  }
  U[n-1] = a1[n-1];

  /* Check if there are NaN values */
  for(i=0;i<n-1 && *fail==0;i++){
    if(PetscIsInfOrNanReal(L[i])) *fail=1;
    if(PetscIsInfOrNanReal(U[i])) *fail=1;
  }
  if(*fail==0 && PetscIsInfOrNanReal(U[n-1])) *fail=1;

  for(i=0;i<n-1 && *fail==0;i++){
    if( PetscAbsReal(L[i])>tol*norm) *fail = 1;  /* This demands IEEE arithmetic */
    if( PetscAbsReal(U[i])>tol*norm) *fail = 1;
  }
  if( *fail==0 && PetscAbsReal(U[n-1])>tol*norm) *fail = 1;
  
  if(work != w){
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "realDQDS"
static PetscErrorCode realDQDS(PetscInt n,PetscReal *L,PetscReal *U,PetscReal shift,PetscReal tol,PetscReal norm,PetscReal *L1,PetscReal *U1,PetscInt *fail)
{
  PetscReal d;
  PetscInt  i;

  PetscFunctionBegin;
  *fail = 0;
  d = U[0]-shift;
  for(i=0;i<n-1;i++){
    U1[i] = d+L[i];
    L1[i] = L[i]*(U[i+1]/U1[i]);
    d = d*(U[i+1]/U1[i])-shift;
  }
  U1[n-1]=d;

  /* The following demands IEEE arithmetic */
  for(i=0;i<n-1 && *fail==0;i++){
    if(PetscIsInfOrNanReal(L1[i])) *fail=1;
    if(PetscIsInfOrNanReal(U1[i])) *fail=1;
  }
  if(*fail==0 && PetscIsInfOrNanReal(U1[n-1])) *fail=1;
  for(i=0;i<n-1 && *fail==0;i++){
    if(PetscAbsReal(L1[i])>tol*norm) *fail=1;
    if(PetscAbsReal(U1[i])>tol*norm) *fail=1;
  }
  if(*fail==0 && PetscAbsReal(U1[n-1])>tol*norm) *fail=1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "tridqdsZhuang3"
static PetscErrorCode tridqdsZhuang3(PetscInt n,PetscReal *e,PetscReal *q,PetscReal sum,PetscReal prod,PetscReal tol,PetscReal norm,PetscReal tolDef,PetscInt *fail)
{
  PetscReal xl,yl,xr,yr,zr,t;
  PetscInt  i;

  PetscFunctionBegin;
  *fail = 0;
  xr = 1.0;
  yr = e[0];
  zr = 0.0;
  /* Step 1 */
  /* the efect of Z1 */
  xr = xr*q[0]+yr;
  /* the inverse of L1 */
  xl = (q[0]+e[0])*(q[0]+e[0])+q[1]*e[0]-sum*(q[0]+e[0])+prod;
  yl = -(q[2]*e[1]*q[1]*e[0])/xl;
  xl = -(q[1]*e[0]*(q[0]+e[0]+q[1]+e[1]-sum))/xl;
  /* the efect of L1 */
  q[0] = xr-xl;
  xr = yr-xl;
  yr = zr-yl-xl*e[1];
  /*the inverse of Y1 */
  xr = xr/q[0];
  yr = yr/q[0];
  /*the effect of Y1 inverse */
  e[0] = xl+yr+xr*q[1];
  xl = yl+zr+yr*q[2];      /* zr=0  when n=3 */
  /*the effect of Y1 */
  xr = 1.0-xr;
  yr = e[1]-yr;

  /* STEP n-1 */

  if (PetscAbsReal(e[n-3])>tolDef*PetscAbsReal(xl) || PetscAbsReal(e[n-3])>tolDef*PetscAbsReal(q[n-3])){
    /* the efect of Zn-1 */
    xr = xr*q[n-2]+yr;
    /* the inverse of Ln-1 */
    xl = -xl/e[n-3];
    /* the efect of Ln-1 */
    q[n-2] = xr-xl;
    xr = yr-xl;
    /*the inverse of Yn-1 */
    xr = xr/q[n-2];
    /*the effect of the inverse of Yn-1 */
    e[n-2] = xl+xr*q[n-1];
    /*the effects of Yn-1 */
    xr = 1.0-xr;
    /* STEP n */
    /*the effect of Zn */
    xr = xr*q[n-1];
    /*the inverse of Ln=I */
    /*the effect of Ln */
    q[n-1] = xr;
    /* the inverse of  Yn-1=I */

  } else { /* Free deflation */
    e[n-2] = (e[n-3]+(xr*q[n-2]+yr)+q[n-1])*0.5;       /* Sum=trace/2 */
    q[n-2] = (e[n-3]+q[n-2]*xr)*q[n-1]-xl;             /* det */
    t = ((e[n-3]+(xr*q[n-2]+yr)-q[n-1])*0.5);
    q[n-1] = t*t+(xl+q[n-1]*yr);
    *fail = 2;
  }

  /* The following demands IEEE arithmetic */
  for(i=0;i<n-1 && *fail==0;i++){
    if(PetscIsInfOrNanReal(e[i])) *fail=1;
    if(PetscIsInfOrNanReal(q[i])) *fail=1;
  }
  if(*fail==0 && PetscIsInfOrNanReal(q[n-1])) *fail=1;
  for(i=0;i<n-1 && *fail==0;i++){
    if(PetscAbsReal(e[i])>tol*norm) *fail=1;
    if(PetscAbsReal(q[i])>tol*norm) *fail=1;
  }
  if(*fail==0 && PetscAbsReal(q[n-1])>tol*norm) *fail=1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "tridqdsZhuang"
static PetscErrorCode tridqdsZhuang(PetscInt n,PetscReal *e,PetscReal *q,PetscReal sum,PetscReal prod,PetscReal tol,PetscReal norm,PetscReal tolDef,PetscReal *e1,PetscReal *q1,PetscInt *fail){
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      xl,yl,xr,yr,zr,t;

  PetscFunctionBegin;
  for(i=0;i<n-1;i++){
    e1[i] = e[i];
    q1[i] = q[i];
  }
  q1[n-1] = q[n-1];
  *fail = 0;
  if(n>3){   /* For n>3 */
    *fail = 0;
    xr = 1;
    yr = e1[0];
    zr = 0;
    /* step 1 */
    /* the efect of Z1 */ 
    xr = xr*q1[0]+yr;
    /* the inverse of L1 */
    xl = (q1[0]+e1[0])*(q1[0]+e1[0])+q1[1]*e1[0]-sum*(q1[0]+e1[0])+prod;
    yl = -(q1[2]*e1[1]*q1[1]*e1[0])/xl;
    xl = -(q1[1]*e1[0]*(q1[0]+e1[0]+q1[1]+e1[1]-sum))/xl;
    /* the efect of L1 */
    q1[0] = xr-xl;
    xr = yr-xl;
    yr = zr-yl-xl*e1[1];
    zr = -yl*e1[2];
    /* the inverse of Y1 */
    xr = xr/q1[0];
    yr = yr/q1[0];
    zr = zr/q1[0];
    /* the effect of Y1 inverse */
    e1[0] = xl+yr+xr*q1[1];
    xl = yl+zr+yr*q1[2];
    yl = zr*q1[3];
    /* the effect of Y1 */
    xr = 1-xr;
    yr = e1[1]-yr;
    zr = -zr; 
    /* step i=2,...,n-3 */
    for (i=1;i<n-3;i++) {
      /* the efect of Zi */
      xr = xr*q1[i]+yr;
      /* the inverse of Li */
      xl = -xl/e1[i-1];
      yl = -yl/e1[i-1];
      /* the efect of Li */
      q1[i] = xr-xl;
      xr = yr-xl;
      yr = zr-yl-xl*e1[i+1];
      zr = -yl*e1[i+2];
      /* the inverse of Yi */
      xr = xr/q1[i];
      yr = yr/q1[i];
      zr = zr/q1[i];
      /* the effect of the inverse of Yi */
      e1[i] = xl+yr+xr*q1[i+1];
      xl = yl+zr+yr*q1[i+2];
      yl = zr*q1[i+3];
      /* the effects of Yi */
      xr = 1.0-xr;
      yr = e1[i+1]-yr;
      zr = -zr;
    }

    /* STEP n-2            zr is no longer needed */

    /* the efect of Zn-2 */
    xr = xr*q1[n-3]+yr;
    /* the inverse of Ln-2 */
    xl = -xl/e1[n-4];
    yl = -yl/e1[n-4];
    /* the efect of Ln-2 */
    q1[n-3] = xr-xl;
    xr = yr-xl;
    yr = zr-yl-xl*e1[n-2];
    /* the inverse of Yn-2 */
    xr = xr/q1[n-3];
    yr = yr/q1[n-3];
    /* the effect of the inverse of Yn-2 */
    e1[n-3] = xl+yr+xr*q1[n-2];
    xl = yl+yr*q1[n-1];
    /* the effect of Yn-2 */
    xr = 1.0-xr;
    yr = e1[n-2]-yr;
   
    /* STEP n-1           yl and yr are no longer needed */
    /* Testing for EARLY DEFLATION */

    if (PetscAbsReal(e1[n-3])>tolDef*PetscAbsReal(xl) || PetscAbsReal(e1[n-3])>tolDef*PetscAbsReal(q1[n-3])) {
      /* the efect of Zn-1 */
      xr = xr*q1[n-2]+yr;
      /* the inverse of Ln-1 */
      xl = -xl/e1[n-3];
      /* the efect of Ln-1 */
      q1[n-2] = xr-xl;
      xr = yr-xl;
      /*the inverse of Yn-1 */
      xr = xr/q1[n-2];
      /*the effect of the inverse of Yn-1 */
      e1[n-2] = xl+xr*q1[n-1];
      /*the effects of Yn-1 */
      xr = 1.0-xr;
   
      /* STEP n;     xl no longer needed */
      /* the effect of Zn */
      xr = xr*q1[n-1];
      /* the inverse of Ln = I */
      /* the effect of Ln */
      q1[n-1] = xr;
      /* the inverse of  Yn-1=I */

    } else {  /* FREE DEFLATION */
      e1[n-2] = (e1[n-3]+xr*q1[n-2]+yr+q1[n-1])*0.5;     /* sum=trace/2 */
      q1[n-2] = (e1[n-3]+q1[n-2]*xr)*q1[n-1]-xl;         /* det */
      t = (e1[n-3]+xr*q1[n-2]+yr-q1[n-1])*0.5;
      q1[n-1] = t*t+xl+q1[n-1]*yr;
      *fail = 2;
    }

    for(i=0;i<n-1 && *fail==0;i++){
      if(PetscIsInfOrNanReal(e1[i])) *fail=1;
      if(PetscIsInfOrNanReal(q1[i])) *fail=1;
    }
    if(*fail==0 && PetscIsInfOrNanReal(q1[n-1])) *fail=1;
    for(i=0;i<n-1 && *fail==0;i++){
      if(PetscAbsReal(e1[i])>tol*norm) *fail = 1;  /* This demands IEEE arithmetic */
      if(PetscAbsReal(q1[i])>tol*norm) *fail = 1;
    }
    if( *fail==0 && PetscAbsReal(q1[n-1])>tol*norm) *fail = 1;
  
  } else {  /* The case n=3 */
    ierr = tridqdsZhuang3(n,e1,q1,sum,prod,tol,norm,tolDef,fail);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "PSGHIEP_Eigen3DQDS"
static PetscErrorCode PSGHIEP_Eigen3DQDS(PetscInt n,PetscReal *a,PetscReal *b,PetscReal *c, PetscScalar *wr, PetscScalar *wi,PetscReal *w,PetscInt nw)
{
  PetscInt       totalIt=0;       /* Total Number of Iterations  */
  PetscInt       totalFail=0;     /* Total number of failures */
  PetscInt       nFail=0;         /* Number of failures per transformation */
  PetscReal      tolZero=1.0/16;  /* Tolerance for zero shifts */
  PetscInt       maxIt=10*n;      /* Maximum number of iterations */
  PetscInt       maxFail=10*n;    /* Maximum number of failures allowed per each transformation */
  PetscReal      tolDef=PETSC_MACHINE_EPSILON;      /* Tolerance for deflation eps, 10*eps, 100*eps */                      
  PetscReal      tolGrowth=100000; /*1/PetscSqrtReal(PETSC_MACHINE_EPSILON);*/
  PetscErrorCode ierr;        
  PetscInt       i,k,nwu,nwall,begin,ind,flag,dim,m;
  PetscReal      norm,gr,gl,sigma,delta,meanEig,*work,*U,*L,*U1,*L1,*split;              
  PetscReal      acShift,initialShift,shift,sum,det,disc,prod,x1,x2,maxEntry;
  PetscInt       realSteps,complexSteps,earlyDef,lastSplit,splitCount;
  PetscBool      test1,test2;

//eigValFail=zeros(n,1); %In case of failure

  PetscFunctionBegin;
  /* First normalization */
  dim = n;
  maxEntry = 0.0;
  for (i=0;i<dim-1;i++) {
    if (PetscAbsReal(a[i])>maxEntry) maxEntry = PetscAbsReal(a[i]);
    if (PetscAbsReal(b[i])>maxEntry) maxEntry = PetscAbsReal(b[i]);
    if (PetscAbsReal(c[i])>maxEntry) maxEntry = PetscAbsReal(c[i]);
  }
  if (PetscAbsReal(a[dim-1])>maxEntry) maxEntry = PetscAbsReal(a[dim-1]);
  for (i=0;i<dim-1;i++) {
    a[i] /= maxEntry;
    b[i] /= maxEntry;
    c[i] /= maxEntry;
  }
  a[dim-1] /= maxEntry;
/*
maxEntry=max(abs([A B C]));
maxEntry=1;
A=A./maxEntry;
B=B./maxEntry;
C=C./maxEntry;
*/
  /* Test if the matrix is unreduced */
  for(i=0;i<n-1;i++){
    if(PetscAbsReal(b[i])==0 || PetscAbsReal(c[i])==0){
      SETERRQ(PETSC_COMM_SELF,1,"Initial tridiagonal matrix is not unreduced");
    }
  }
  nwall = 9*n+4;
  if(w && nw>=nwall){
    work = w;
    nwall = nw;
  }else{
    ierr = PetscMalloc(nwall*sizeof(PetscReal),&work);CHKERRQ(ierr);   /////////////// falta liberar
  }
  U = work;
  L = work+n;
  U1 = work+2*n;
  L1 = work+3*n;
  nwu = 4*n;
  ierr = PetscMemzero(wi,n*sizeof(PetscScalar));
  /* Normalization - the J form of C */
  for(i=0;i<n-1;i++) b[i] *= c[i]; /* subdiagonal of the J form */

/*
  %Test if the matrix is symmetrizable 
signB=(B>0);
if sum(signB)==n-1
    disp('Warning: Matrix is symetrizable.')
end
*/
  /* Scan matrix J  ---- Finding a box of inclusion for the eigenvalues */
  norm = 0.0;
  for(i=0;i<n-1;i++){
    norm = PetscMax(norm,PetscMax(PetscAbsReal(a[i]),PetscAbsReal(b[i])));
  }
  norm = PetscMax(norm,PetscMax(1,PetscAbsReal(a[n-1])));
  ierr = ScanJ(n,a,b,&gl,&gr,&sigma);CHKERRQ(ierr);
  delta = (gr-gl)/n; /* How much to add to the shift, in case of failure (element growth) */
  meanEig = 0.0;
  for(i=0;i<n;i++) meanEig += a[i];
  meanEig /= n; /* shift = initial shift = mean of eigenvalues */
  //ierr = Prologue(n,a,b,gl,gr,&m,X,&shift,work+nwu,nwall-nwu);CHKERRQ(ierr);
  ierr = Prologue(n,a,b,gl,gr,&m,&shift,work+nwu,nwall-nwu);CHKERRQ(ierr);
  if (m==n) { /* Multiple eigenvalue, we have the one-point spectrum case */
    for(i=0;i<dim;i++) {
      wr[i] = shift;  /////////////?????????????????
      wi[i] = 0.0;
    }
    PetscFunctionReturn(0);
  }
  /* Initial LU Factorization */
  if( delta==0 ) shift=0;/* The case when all eigenvalues are pure imaginary */
  ierr = LUfac(n,a,b,shift,tolGrowth,norm,L,U,&flag,work+nwu,nwall-nwu);CHKERRQ(ierr); /* flag=1 failure; flag=0 successful transformation*/
  while(flag==1 && nFail<maxFail){
    shift=shift+delta;  
    if(shift>gr || shift<gl){ /* Successive failures */
      shift=meanEig;
      delta=-delta;
    }
    nFail=nFail+1;
    ierr = LUfac(n,a,b,shift,tolGrowth,norm,L,U,&flag,work+nwu,nwall-nwu);CHKERRQ(ierr); /* flag=1 failure; flag=0 successful transformation*/
  }
  if(nFail==maxFail){
    SETERRQ(PETSC_COMM_SELF,1,"Maximun number of failures reached in Initial LU factorization");
  }
  /* Successful Initial transformation */
  totalFail = totalFail+nFail;
  nFail = 0;
  acShift = 0;
  initialShift = shift;
  shift = 0;

  realSteps = 1;  ////// eliminar variable
  complexSteps = 0;  ///// eliminar variable
  earlyDef = 0;  /// eliminar??
  begin = 0;
  lastSplit = 0;
  split = work+nwu;
  nwu += n;
  split[lastSplit] = begin;
  splitCount = 0;
  while (begin!=-1){   ///////////////?????????
    while(n-begin>2 && totalIt<maxIt){
      /* Check for deflation before performing a transformation */
      test1 = ((PetscAbsReal(L[n-2])<tolDef*PetscAbsReal(U[n-2])) && (PetscAbsReal(L[n-2])<tolDef*PetscAbsReal(U[n-1]+acShift)) && (PetscAbsReal(L[n-2]*U[n])<tolDef*PetscAbsReal(acShift+U[n-1])) && (PetscAbsReal(L[n-2])*(PetscAbsReal(U[n-2])+1)<tolDef*PetscAbsReal(acShift+U[n-1])))? PETSC_TRUE: PETSC_FALSE;
      if(flag==2){  /* Early 2x2 deflation */
        earlyDef=earlyDef+1;
        test2 = PETSC_TRUE;  
      }else{ 
        if(n-begin>4){
          test2 = ((PetscAbsReal(L[n-3])<tolDef*PetscAbsReal(U[n-3])) && (PetscAbsReal(L[n-3]*(U[n-4]+L[n-4]))< tolDef*PetscAbsReal(U[n-4]*(U[n-3]+L[n-3])+L[n-4]*L[n-3])))? PETSC_TRUE: PETSC_FALSE;
        }else{ /* n-begin+1=3 */
          test2 = (PetscAbsReal(L[begin])<tolDef*PetscAbsReal(U[begin]))? PETSC_TRUE: PETSC_FALSE;
        }
      }
      while(test2 || test1){
        /* 2x2 deflation */
        if(test2){
          if(flag==2){ /* Early deflation */
            sum = L[n-2];
            det = U[n-2];
            disc = U[n-1];
            flag = 0;
          }else{
            sum = (L[n-2]+(U[n-2]+U[n-1]))/2;  
            disc = (L[n-2]*(L[n-2]+2*(U[n-2]+U[n-1]))+(U[n-2]-U[n-1])*(U[n-2]-U[n-1]))/4;
            det = U[n-2]*U[n-1];
          }
          if(disc<=0){
#if !defined(PETSC_USE_COMPLEX)
            wr[--n] = sum+acShift; wi[n] = PetscSqrtReal(-disc);
            wr[--n] = sum+acShift; wi[n] = -PetscSqrtReal(-disc);
#else
            wr[--n] = sum-PETSC_i*PetscSqrtReal(-disc)+acShift; wi[n] = 0.0;
            wr[--n] = sum+PETSC_i*PetscSqrtReal(-disc)+acShift; wi[n] = 0.0;
#endif
          }else{  
            if(sum==0){
              x1 = PetscSqrtReal(disc);
              x2 = -x1;
            }else{ 
              x1 = ((sum>=0.0)?1.0:-1.0)*(PetscAbsReal(sum)+PetscSqrtReal(disc));
              x2 = det/x1;
            }
            wr[--n] = x1+acShift;
            wr[--n] = x2+acShift;
          }
        }else{ /* test1 -- 1x1 deflation */
          x1 = U[n-1]+acShift;
          wr[--n] = x1;
        }
        
        if(n<=begin+2){
          break;
        }else{
          test1 = ((PetscAbsReal(L[n-2])<tolDef*PetscAbsReal(U[n-2])) && (PetscAbsReal(L[n-2])<tolDef*PetscAbsReal(U[n-1]+acShift)) && (PetscAbsReal(L[n-2]*U[n-1])<tolDef*PetscAbsReal(acShift+U[n-1])) && (PetscAbsReal(L[n-2])*(PetscAbsReal(U[n-2])+1)< tolDef*PetscAbsReal(acShift+U[n-1])))? PETSC_TRUE: PETSC_FALSE;
          if(n-begin>4){
            test2 = ((PetscAbsReal(L[n-3])<tolDef*PetscAbsReal(U[n-3])) && (PetscAbsReal(L[n-3]*(U[n-4]+L[n-4]))< tolDef*PetscAbsReal(U[n-4]*(U[n-3]+L[n-3])+L[n-4]*L[n-3])))? PETSC_TRUE: PETSC_FALSE;
          }else{ /* n-begin+1=3 */
            test2 = (PetscAbsReal(L[begin])<tolDef*PetscAbsReal(U[begin]))? PETSC_TRUE: PETSC_FALSE;
          }
        }
      } /* end "WHILE deflations" */
      /* After deflation */
      if(n>begin+3){
        ind = begin;
        for(k=n-4;k>=begin+1;k--){
          if( (PetscAbsReal(L[k])<tolDef*PetscAbsReal(U[k]))&&(PetscAbsReal(L[k]*U[k+1]*(U[k+2]+L[k+2])*(U[k-1]+L[k-1]))<tolDef*PetscAbsReal((U[k-1]*(U[k]+L[k])+L[k-1]*L[k])*(U[k+1]*(U[k+2]+L[k+2])+L[k+1]*L[k+2]))) ){
             ind=k;
             break;
          }
        }
        if( ind>begin || PetscAbsReal(L[begin]) <tolDef*PetscAbsReal(U[begin]) ){
          lastSplit = lastSplit+1;
          split[lastSplit] = begin;
          L[ind] = acShift; /* Use of L[ind] to save acShift */
          begin = ind+1;
          splitCount = splitCount+1;
        }
      }
    
      if(n>begin+2){
        disc = (L[n-2]*(L[n-2]+2*(U[n-2]+U[n-1]))+(U[n-2]-U[n-1])*(U[n-2]-U[n-1]))/4;
        if( (PetscAbsReal(L[n-2])>tolZero) && (PetscAbsReal(L[n-3])>tolZero)){ /* L's are big */
          shift = 0;
          sum = 0; /* Needed in case of failure */
          prod = 0;
          ierr = realDQDS(n-begin,L+begin,U+begin,0,tolGrowth,norm,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
          realSteps++;
          if(flag){  /* Failure */
            ierr = tridqdsZhuang(n-begin,L+begin,U+begin,0.0,0.0,tolGrowth,norm,tolDef,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
            shift = 0.0;
            while (flag==1 && nFail<maxFail) {  /* Successive failures */
              shift = shift+delta;
              if (shift>gr-acShift || shift<gl-acShift) {
                shift = meanEig-acShift;
                delta = -delta;
              }
              nFail++;
              ierr = realDQDS(n-begin,L+begin,U+begin,0,tolGrowth,norm,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
            }
          }
        }else{ /* L's are small */
          if (disc<0){  /* disc <0   Complex case; Francis shift; 3dqds */
            sum = U[n-2]+L[n-2]+U[n-1];
            prod = U[n-2]*U[n-1];
            ierr = tridqdsZhuang(n-begin,L+begin,U+begin,sum,prod,tolGrowth,norm,tolDef,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
            complexSteps++;
            shift = 0.0; /* Restoring transformation */
            while (flag==1 && nFail<maxFail) { /* In case of failure */
              shift = shift+U[n-1];  /* first time shift=0 */
              ierr = realDQDS(n-begin,L+begin,U+begin,shift,tolGrowth,norm,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
              nFail++;
            }
          } else  { /* disc >0  Real case; real Wilkinson shift; dqds */
            sum = (L[n-2]+U[n-2]+U[n-1])/2;
            if (sum==0){
              x1 = PetscSqrtReal(disc);
              x2 = -x1;
            } else {
              x1 = ((sum>=0)?1.0:-1.0)*(PetscAbsReal(sum)+PetscSqrtReal(disc));
              x2 = U[n-2]*U[n-1]/x1;
            }
            /* Take the eigenvalue closest to UL(n,n) */
            if (PetscAbsReal(x1-U[n-1])<PetscAbsReal(x2-U[n-1])) {
              shift = x1;
            } else {
              shift = x2;
            }
            ierr = realDQDS(n-begin,L+begin,U+begin,shift,tolGrowth,norm,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
            realSteps++;
            /* In case of failure */
            while (flag==1 && nFail<maxFail) {
              sum = 2*shift;
              prod = shift*shift;
              ierr = tridqdsZhuang(n-1-begin,L+begin,U+begin,sum,prod,tolGrowth,norm,tolDef,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
              /* In case of successive failures */
              if (shift==0) {
                shift = PetscMin(PetscAbsReal(L[n-2]),PetscAbsReal(L[n-3]))*delta;
              } else {
                shift=shift+delta;
              }
              if (shift>gr-acShift || shift<gl-acShift) {
                shift = meanEig-acShift;
                delta = -delta;
              }
              if (flag==0) { /* We changed from real dqds to 3dqds */
                shift=0;
              }
              nFail++;
            }
          }
        } /* end "if tolZero" */
        if (nFail==maxFail) {
          SETERRQ(PETSC_COMM_SELF,1,"Maximun number of failures reached. No convergence in DQDS");
        }
        /* Successful Transformation; flag==0 */
        totalIt++;
        acShift = shift+acShift;
        for(i=begin;i<n-1;i++){
          L[i] = L1[i];
          U[i] = U1[i];          
        }
        U[n-1] = U1[n-1];          
        totalFail = totalFail+nFail;
        nFail = 0;
      }  /* end "if n>begin+1" */ 
    }  /* end WHILE 1 */
    if (totalIt>=maxIt) { 
      SETERRQ(PETSC_COMM_SELF,1,"Maximun number of iterations reached. No convergence in DQDS");
    }
    /* END: n=2 or n=1  % n=begin+1 or n=begin */
    if (n==begin+2) {
      sum = (L[n-2]+U[n-2]+U[n-1])/2;  
      disc = (L[n-2]*(L[n-2]+2*(U[n-2]+U[n-1]))+(U[n-2]-U[n-1])*(U[n-2]-U[n-1]))/4;
        if (disc<=0)  {  /* Complex case */
        /* Deflation 2 */
#if !defined(PETSC_USE_COMPLEX)
        wr[--n] = sum+acShift; wi[n] = PetscSqrtReal(-disc);
        wr[--n] = sum+acShift; wi[n] = -PetscSqrtReal(-disc);
#else
        x1 = sum+PETSC_i*PetscSqrtReal(-disc);
        x2 = sum-PETSC_i*PetscSqrtReal(-disc);
        wr[--n] = x2+acShift; wi[n] = 0.0;
        wr[--n] = x1+acShift; wi[n] = 0.0;
#endif
      }else  { /* Real case */
        if (sum==0) {
          x1 = PetscSqrtReal(disc);
          x2 = -x1;
        } else {
          x1 = ((sum>=0)?1.0:-1.0)*(PetscAbsReal(sum)+PetscSqrtReal(disc));
          x2 = U[n-2]*U[n-1]/x1;
        }
        /* Deflation 2 */
        wr[--n] = x2+acShift;       
        wr[--n] = x1+acShift;       
      }
    }else { /* n=1   n=begin */
      /* deflation 1 */
      x1 = U[n-1]+acShift;
      wr[--n] = x1;
    }
    switch (n) {
      case 0:
        begin = -1;  ///////????????'
        break;
      case 1:
        acShift = L[begin-1];
        begin = split[lastSplit];
        lastSplit--;
        break;
      default : /* n>=2 */
        acShift = L[begin-1];
        begin = split[lastSplit];
        lastSplit--;
    }
  }/* While begin~=0 */
  for(i=0;i<dim;i++){
    wr[i] = (wr[i]+initialShift)*maxEntry;
#if !defined(PETSC_USE_COMPLEX)
    wi[i] *= maxEntry;
#endif
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GHIEP_DQDS_II"
PetscErrorCode PSSolve_GHIEP_DQDS_II(PS ps,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  PetscInt       i,off,ld,nwall,nwu;
  PetscScalar    *A,*B,*Q;
  PetscReal      *d,*e,*s,*a,*b,*c;

  PetscFunctionBegin;
/* ////////////////////// */
ierr = PetscOptionsGetInt(PETSC_NULL,"-dbPS",&dbPS,PETSC_NULL);CHKERRQ(ierr);
/* ////////////////////// */
  ld = ps->ld;
  off = ps->l + ps->l*ld;
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  Q = ps->mat[PS_MAT_Q];
  d = ps->rmat[PS_MAT_T];
  e = ps->rmat[PS_MAT_T] + ld;
  c = ps->rmat[PS_MAT_T] + 2*ld;
  s = ps->rmat[PS_MAT_D];
  /* Quick return if possible */
  if (ps->n-ps->l == 1) {
    *(Q+off) = 1;
    if(ps->compact){
      wr[ps->l] = d[ps->l]/s[ps->l];
      wi[ps->l] = 0.0;
    }else{
      d[ps->l] = PetscRealPart(A[off]);
      s[ps->l] = PetscRealPart(B[off]);
      wr[ps->l] = d[ps->l]/s[ps->l];
      wi[ps->l] = 0.0;  
    }
    PetscFunctionReturn(0);
  }
  nwall = 12*ld+4;
  ierr = PSAllocateWork_Private(ps,0,nwall,0);CHKERRQ(ierr); 
  /* Reduce to pseudotriadiagonal form */
  ierr = PSIntermediate_GHIEP( ps);CHKERRQ(ierr);
  
/* //////////////////////////// */
PetscViewer viewer;
PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);
if(dbPS>1){
printf("tras tridiagonalizar\n");
PSView(ps,viewer);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
/* ///////////////////// */

  /* Compute Eigenvalues (DQDS)*/
  /* Form pseudosymmetric tridiagonal */
  a = ps->rwork;
  b = a+ld;
  c = b+ld;
  nwu = 3*ld;
  if(ps->compact){
    for(i=ps->l;i<ps->n-1;i++){
      a[i] = d[i]*s[i];
      b[i] = e[i]*s[i+1];
      c[i] = e[i]*s[i];
    }
    a[ps->n-1] = d[ps->n-1]*s[ps->n-1];
  }else{
    for(i=ps->l;i<ps->n-1;i++){
      a[i] = PetscRealPart(A[i+i*ld]*B[i+i*ld]);
      b[i] = PetscRealPart(A[i+1+i*ld]*s[i+1]);
      c[i] = PetscRealPart(A[i+(i+1)*ld]*s[i]);
    }
    a[ps->n-1] = PetscRealPart(A[ps->n-1+(ps->n-1)*ld]*B[ps->n-1+(ps->n-1)*ld]);
  }
  ierr = PSGHIEP_Eigen3DQDS(ps->n-ps->l,a+ps->l,b+ps->l,c+ps->l,wr+ps->l,wi+ps->l,ps->rwork+nwu,nwall-nwu);
/* ///////////////// */
PetscPrintf(PETSC_COMM_WORLD,"vp=[\n");
for(i=0;i<ps->n;i++)
PetscPrintf(PETSC_COMM_WORLD,"%.14g\n",wr[i]);
PetscPrintf(PETSC_COMM_WORLD,"];\n");
/* /////////////////// */
  /* Compute Eigenvectors with Inverse Iteration */
  ierr = PSGHIEPPseudoOrthogInverseIteration(ps,wr,wi);CHKERRQ(ierr);
/* ////////////////////// */
if(dbPS>1){
printf("PseudoOrthog\n");
ierr = PSView(ps,viewer);CHKERRQ(ierr);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
/* ///////////////////// */

  ierr = PSSolve_GHIEP_Sort(ps,wr,wi);CHKERRQ(ierr);

/* ////////////////////// */
if(dbPS>1){
printf("SORT\n");
ierr = PSView(ps,viewer);CHKERRQ(ierr);
PSViewMat_Private(ps,viewer,PS_MAT_Q);
}
/* ////////////////////// */
    PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PSNormalize_GHIEP"
PetscErrorCode PSNormalize_GHIEP(PS ps,PSMatType mat,PetscInt col)
{
  PetscErrorCode ierr;
  PetscInt       i,i0,i1;
  PetscBLASInt   ld,n,one = 1;
  PetscScalar    *A = ps->mat[PS_MAT_A],norm,*x;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    norm0;
#endif

  PetscFunctionBegin;
  if(ps->state < PS_STATE_INTERMEDIATE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported state");
  switch (mat) {
    case PS_MAT_X:
    case PS_MAT_Y:
    case PS_MAT_Q:
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
  } else if(col>0 && A[ps->ld*(col-1)+col] != 0.0) {
    i0 = col-1; i1 = col+1;
  } else {
    i0 = col; i1 = col+1;
  }
  for(i=i0; i<i1; i++) {
#if !defined(PETSC_USE_COMPLEX)
    if(i<n-1 && A[ps->ld*i+i+1] != 0.0) {
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_GHIEP"
PetscErrorCode PSCreate_GHIEP(PS ps)
{
  PetscFunctionBegin;
  ps->ops->allocate      = PSAllocate_GHIEP;
  ps->ops->view          = PSView_GHIEP;
  ps->ops->vectors       = PSVectors_GHIEP;
  ps->ops->solve[0]      = PSSolve_GHIEP_HZ;
  ps->ops->solve[1]      = PSSolve_GHIEP_DQDS_II;
  ps->ops->solve[2]      = PSSolve_GHIEP_QR;
  ps->ops->solve[3]      = PSSolve_GHIEP_QR_II;
  ps->ops->normalize     = PSNormalize_GHIEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END
