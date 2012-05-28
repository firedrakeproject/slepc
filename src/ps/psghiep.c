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

/*extern PetscErrorCode ArrowTridiag(PetscBLASInt *n,PetscReal *d,PetscReal *e,PetscScalar *Q,PetscBLASInt *ldq);*/
/*
  compute X = X - Y*ss^{-1}*Y^T*s*X where ss=Y^T*s*Y
  s diagonal (signature matrix)
*/
#undef __FUNCT__  
#define __FUNCT__ "IndefOrthog"
static PetscErrorCode IndefOrthog(PetscReal *s, PetscScalar *Y, PetscReal ss, PetscScalar *X, PetscScalar *h,PetscInt n)
{
  PetscInt i;
  PetscScalar h_,r;
  PetscFunctionBegin;
  if(Y){
    h_ = 0.0; /* h_=(Y^Tdiag(s)*Y)^{-1}*Y^T*diag(s)*X*/
    for(i=0;i<n;i++){ h_+=Y[i]*s[i]*X[i];}
    h_ /= ss;
    for(i=0;i<n;i++){X[i] -= h_*Y[i];} /* X = X-h_*Y */
    /* repeat */
    r = 0.0;
    for(i=0;i<n;i++){ r+=Y[i]*s[i]*X[i];}
    r /= ss;
    for(i=0;i<n;i++){X[i] -= r*Y[i];}
    h_ += r;
  }else h_ = 0.0;
  if(h) *h = h_;
  PetscFunctionReturn(0);
}
/* normalization with a indefinite norm */
#undef __FUNCT__
#define __FUNCT__ "IndefNorm"
static PetscErrorCode IndefNorm(PetscReal *s,PetscScalar *X, PetscReal *norm,PetscInt n)
{
  PetscInt	i;
  PetscReal   	norm_;
  /* s-normalization */
  norm_ = 0.0;
  for(i=0;i<n;i++){norm_ += PetscRealPart(X[i]*s[i]*X[i]);}
  if(norm_<0){norm_ = -PetscSqrtReal(-norm_);}
  else {norm_ = PetscSqrtReal(norm_);}
  for(i=0;i<n;i++)X[i] /= norm_;
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
                     "QR method",
                     "EA + Inverse Iteration"
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
        if (*(ps->rmat[PS_MAT_T]+ps->ld+i) !=0){
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
static PetscErrorCode PSVectors_GHIEP_Eigen_Some(PS ps,PetscInt *k,PetscReal *rnorm,PetscBool left)
{

  /* to complete  */
/*
  PetscScalar    *Q = ps->mat[PS_MAT_Q];
  PetscReal	 s1,s2,d1,d2,b;
  PetscInt       ld = ps->ld,k_;
  PetscErrorCode ierr;
  PSMatType	 mat;
 */ 
  PetscFunctionBegin;
#if 0
  if (left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for left eigenvectors");
  else mat = PS_MAT_Q;
  k_ = *k;
  if(k_ < ps->n-1){
   b = (ps->compact)?*(ps->rmat[PS_MAT_T]+ld+k_):PetscRealPart(*(ps->mat[PS_MAT_A]+(k_+1)+ld*k_));
  }else b = 0.0;
  if(b == 0.0){/* real */
    ierr = PetscMemcpy(ps->mat[mat]+(k_)*ld,Q+(k_)*ld,ld*sizeof(PetscScalar));CHKERRQ(ierr);
  }else{ /* complex block */
    if(ps->compact){
      s1 = *(ps->rmat[PS_MAT_T]+2*ld+k_);
      d1 = *(ps->rmat[PS_MAT_T]+k_);
      s2 = *(ps->rmat[PS_MAT_T]+2*ld+k_+1);
      d2 = *(ps->rmat[PS_MAT_T]+k_+1);
    }else{
      s1 = PetscRealPart(*(ps->mat[PS_MAT_B]+2*ld+k_));
      d1 = PetscRealPart(*(ps->mat[PS_MAT_A]+k_));
      s2 = PetscRealPart(*(ps->mat[PS_MAT_B]+2*ld+k_+1));
      d2 = PetscRealPart(*(ps->mat[PS_MAT_A]+k_+1));
    }
  }
  if (rnorm) *rnorm = PetscAbsScalar(Q[ps->n-1+(k_)*ld]);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSVectors_GHIEP"
PetscErrorCode PSVectors_GHIEP(PS ps,PSMatType mat,PetscInt *k,PetscReal *rnorm)
{
  PetscInt       i;
  PetscBool	 left;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ps->state<PS_STATE_CONDENSED) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSSolve() first");
  switch (mat) {
    case PS_MAT_Y:
      left = PETSC_TRUE;
    case PS_MAT_X:
      if (k){
        ierr = PSVectors_GHIEP_Eigen_Some(ps,k,rnorm,left);CHKERRQ(ierr);
      }else{
        for(i=0; i<ps->n; i++){
          ierr = PSVectors_GHIEP_Eigen_Some(ps,&i,rnorm,left);CHKERRQ(ierr);
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
static PetscErrorCode PSGHIEPComplexEigs(PS ps, PetscInt n0, PetscInt n1, PetscScalar *wr, PetscScalar *wi){
  PetscInt	j,ld;
  PetscScalar	*A,*B;
  PetscReal	*d,*e,*s,d1,d2,e1,e2,disc;

  PetscFunctionBegin;
  ld = ps->ld;
  if (ps->compact){
    d = ps->rmat[PS_MAT_T];
    e = d + ld;
    s = ps->rmat[PS_MAT_D];
    for (j=n0;j<n1;j++) {
      if (j==n1-1 || e[j] == 0.0) { 
        /* real eigenvalue */
        wr[j] = d[j]/s[j];
        wi[j] = 0.0 ;
      } else {
      /* diagonal block */
        d1 = d[j]/s[j]; d2 = d[j+1]/s[j+1]; e1 = e[j]/s[j+1]; e2 = e[j]/s[j];
        wr[j] = (d1+d2)/2;  wr[j+1] = wr[j];
        disc = (d1-d2)*(d1-d2) - (e1-e2)*(e1-e2);
        if (disc<0){ /* complex eigenvalues */
          wi[j] = PetscSqrtReal(-disc)/2; wi[j+1] = -wi[j];
        }else{ /* real eigenvalues */
          disc = PetscSqrtReal(disc)/2;
          wr[j] = wr[j]+disc; wr[j+1]=wr[j+1]-disc; wi[j] = 0.0; wi[j+1] = 0.0;
        }
        j++;
      }
    }
  }else{
    A = ps->mat[PS_MAT_A];
    B = ps->mat[PS_MAT_B];
    for (j=n0;j<n1;j++) {
      if (j==n1-1 || A[(j+1)+j*ld] == 0.0) { 
        /* real eigenvalue */
        wr[j] = A[j+j*ld]/B[j+j*ld];
        wi[j] = 0.0 ;
      } else {
      /* diagonal block */
        d1 = PetscRealPart(A[j+j*ld]/B[j+j*ld]);
        d2 = PetscRealPart(A[(j+1)+(j+1)*ld]/B[(j+1)+(j+1)*ld]);
        e1 = PetscRealPart(A[j+(j+1)*ld]/B[j+j*ld]);
        e2 = PetscRealPart(A[(j+1)+j*ld]/B[(j+1)+(j+1)*ld]);
        wr[j] = (d1+d2)/2;  wr[j+1] = wr[j];
        disc = (d1-d2)*(d1-d2) - (e1-e2)*(e1-e2);
        if (disc<0.0){ /* complex eigenvalues */
          wi[j] = PetscSqrtReal(-disc)/2; wi[j+1] = -wi[j];
        }else{ /* real eigenvalues */
          disc = PetscSqrtReal(disc)/2;
          wr[j] = wr[j]+disc; wr[j+1]=wr[j+1]-disc; wi[j] = 0.0; wi[j+1] = 0.0;
        }
        j++;
      }
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
  PetscValidScalarPointer(wi,3);

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
  ierr = PetscMemcpy(ps->rwork,e,n*sizeof(PetscReal));CHKERRQ(ierr);
  for (i=ps->l;i<n-1;i++) {
    e[i] = *(ps->rwork + perm[i]);
  }
  if(!ps->compact){ ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);}
  ierr = PSPermuteColumns_Private(ps,ps->l,n,PS_MAT_Q,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
  Generates an hyperbolic rotation
    if x1*x1 - x2*x2 != 0 
      r = sqrt( |x1*x1 - x2*x2| )
      c = x1/r  s = x2/r
     
      | c -s||x1|   |d*r|
      |-s  c||x2| = | 0 | 
      where d = 1 for type==1 and -1 for type==2
  Returns the condition number of the reduction
*/
#undef __FUNCT__
#define __FUNCT__ "HRGen"
static PetscErrorCode HRGen(PetscReal x1,PetscReal x2,PetscInt *type,PetscReal *c,PetscReal *s,PetscReal *r,PetscReal *cond)
{
  PetscReal t,n2,xa,xb;
  PetscInt  type_;
  PetscFunctionBegin;
  if(x2==0) {
    *r = PetscAbsReal(x1); *c = (x1>=0)?1.0:-1.0; *s = 0.0; *type = 1;
    PetscFunctionReturn(0);
  }
  if(PetscAbsReal(x1) == PetscAbsReal(x2)){
  /* Doesn't exist hyperbolic rotation */
    *c = 0; *s = 0; *r = 0; *type = 0; *cond = PETSC_MAX_REAL;
    PetscFunctionReturn(0);
  }
  
  if(PetscAbsReal(x1)>PetscAbsReal(x2)){
    xa = x1; xb = x2; type_ =1;
  }else{ xa = x2; xb = x1; type_ =2;} 
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
/*

				|c  s|
  Applies an hyperbolic rotator	|s  c|
           |c  s|
    [X1 X2]|s  c| 
*/
#undef __FUNCT__
#define __FUNCT__ "HRApply"
static PetscErrorCode HRApply(PetscInt n, PetscScalar *X1,PetscInt inc1, PetscScalar *X2, PetscInt inc2,PetscReal c, PetscReal s)
{
  PetscInt	i;
  PetscReal	t;
  PetscScalar  	tmp;
  
  PetscFunctionBegin;
  if(PetscAbsReal(c)>PetscAbsReal(s)){ /* Type I */
    t = s/c;
    for(i=0;i<n;i++){
      *(X1+i*inc1) = c*(*(X1+i*inc1)) + s*(*(X2+i*inc2));
      *(X2+i*inc2) = t*(*(X1+i*inc1)) + (*(X2+i*inc2))/c;
    }
  }else{ /* Type II */
    t = c/s;
    for(i=0;i<n;i++){
      tmp = *(X1+i*inc1);
      *(X1+i*inc1) = c*(*(X1+i*inc1)) + s*(*(X2+i*inc2));
      *(X2+i*inc2) = t*(*(X1+i*inc1)) + tmp/s;
    }
  }
PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "OmegaArrowPerm"
//static PetscErrorCode ArrowOmegaPerm(PetscInt n,PetscReal *d,PetscReal *e,PetscReal *r,PetscReal *Omega,PetscScalar *Q,PetscInt ldq,PetscBool id, PetscInt *n1_, PetscInt *n2_)
static PetscErrorCode OmegaArrowPerm(PetscInt n,PetscReal *d,PetscReal *e,PetscReal *Omega, PetscInt *k,PetscInt *perm)
{
  PetscErrorCode  ierr;
  PetscInt	  i,j,n1,n2,k1,k2,;
  PetscReal       *rw;
  PetscFuntionBegin;
  ierr = PetscMalloc(n*sizeof(PetscInt),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&rw);CHKERRQ(ierr);
  n1 = 0; n2 = 0;
  for(i=0;i<n-1;i++){
    if(e[i] != 0.0) { 
      if(Omega[i] > 0.0) n1++; else n2++;
    }else i++;
  }
  if(i < n ){if(Omega[i] > 0.0) n1++; else n2++;}
  
  if( n2 > n1){ d = -1; j = n1; n1 = n2; n2 = j;} 
  k1 = 0; k2 = n1; j = n1 + n2;
  for(i=0;i<n-1;i++){
    if(e[i] != 0.0) { 
      if(d*Omega[i] > 0.0) perm[k1++] = i; else perm[k2++] = i;
    }else{
      perm[j++] = i++;
      perm[j++] = i;
    }
  }
  if(k) *k = ;
  PetscFunctionReturn(0);
}

/*
  Reduce an arrowhead symmetric-diagonal pair to tridiagonal-diagonal
  Omega: signature matrix
*/
#undef __FUNCT__  
#define __FUNCT__ "ArrowTridiagDiag"
//static PetscErrorCode ArrowTridiagDiag(PetscInt k,PetscInt n,PetscReal *d,PetscReal *e,PetscReal *Omega,PetscScalar *Q,PetscInt ldq,PetscBool id)
static PetscErrorCode ArrowTridiagDiag(PetscInt n,PetscReal *d,PetscReal *e,PetscReal *r,PetscReal *Omega,PetscScalar *Q,PetscInt ldq)
{
  PetscFunctionBegin;
  PetscBLASInt 	  j2,ld,one,n_;
  PetscInt   	  i,j,type,n1,n2,perm;
  PetscReal    	  c,s,p,off,e1,e0,d1,d0,temp;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (n<=2) PetscFunctionReturn(0);
  ld = PetscBLASIntCast(ldq);
  one = 1;
  
  for (j=0;j<n-2;j++) {
    /* Eliminate entry e(j) by a rotation in the planes (j,j+1) */
    type = (Omega[j]*Omega[j+1]>0.0)?1:-1;
    if(PetscAbsReal(e[j+1]) < PetscAbsReal(e[j])) type = 2;
    e0 = e[j]; e1 = e[j+1];
    d0 = d[j]; d1 = d[j+1];
    temp = e[j+1];
    LAPACKlartg_(&temp,&e[j],&c,&s,&e[j+1]);
    s = -s;
    /* Apply rotation to diagonal elements */
    temp   = d[j+1];
    e[j]   = c*s*(temp-d[j]);
    d[j+1] = s*s*d[j] + c*c*temp;
    d[j]   = c*c*d[j] + s*s*temp;

    /* Apply rotation to Q */
    j2 = j+2;
    BLASrot_(&j2,Q+j*ld,&one,Q+(j+1)*ld,&one,&c,&s);
    /* Chase newly introduced off-diagonal entry to the top left corner */
    for (i=j-1;i>=0;i--) {
      e[i] = c*e[i];
      temp = e[i+1];
      LAPACKlartg_(&temp,&off,&c,&s,&e[i+1]);
      s = -s;
      temp = (d[i]-d[i+1])*s - 2.0*c*e[i];
      p = s*temp;
      d[i+1] += p;
      d[i] -= p;
      e[i] = -e[i] - c*temp;
      j2 = j+2;
      BLASrot_(&j2,Q+i*ld,&one,Q+(i+1)*ld,&one,&c,&s);
    }
  }
  PetscFunctionReturn(0);
}
#endif

/*
  Input:
    A symmetric (only lower triangular part is refered)
    s vector +1 and -1 (signature matrix)
  Output:
    d,e
    s
    Q s-orthogonal matrix whith Q^T*A*Q = T (symmetric tridiagonal matrix)
*/
#undef __FUNCT__
#define __FUNCT__ "TridiagDiag_HHR"
PetscErrorCode TridiagDiag_HHR(PetscInt n,PetscScalar *A,PetscInt lda,PetscReal *s,PetscScalar* Q,PetscInt ldq,PetscBool flip,PetscReal *d,PetscReal *e,PetscScalar *w,PetscInt lw,PetscInt *perm)
{
#if defined(PETSC_MISSING_LAPACK_LARFG) || defined(PETSC_MISSING_LAPACK_LARF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LARFG/LARF - Lapack routines are unavailable.");
#else
  PetscErrorCode  ierr;
  PetscInt	  i,j,*ii,*jj,tmp,type;
  PetscReal 	  *ss,cond,cs,sn,t,r;
  PetscScalar	  *work,tau;
  PetscBLASInt	  n0,n1,ni,inc=1,m,n_,lda_,ldq_;

  PetscFunctionBegin;
  if(n<3){
    if(n==1)Q[0]=1;
    if(n==2){Q[0] = Q[1+ldq] = 1; Q[1] = Q[ldq] = 0;}
    PetscFunctionReturn(0);
  }
  lda_ = PetscBLASIntCast(lda);
  n_ = PetscBLASIntCast(n);
  ldq_ = PetscBLASIntCast(ldq);
  if ( !w || lw < n*n ){
    ierr = PetscMalloc(n*n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }else work = w;

  /* Sort (and flip) A and s */
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
          *(A+ni-n0+j*lda+i) = 0.0;  *(A+j+(ni-n0+i)*lda) = 0.0;/**/
        }
        *(A+j+(ni-n0)*lda) = *(A+ni-n0+j*lda);/**/
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
          *(A+n-n1+i+j*lda) = 0.0;  *(A+j+(n-n1+i)*lda) = 0.0;/**/
        }
        *(A+j+(n-n1)*lda) = *(A+n-n1+j*lda);/**/
      }
    }
    /* Hyperbolic rotation */
    if( n0 > 0 && n1 > 0){
      ierr = HRGen(PetscRealPart(A[ni-n0+j*lda]),PetscRealPart(A[n-n1+j*lda]),&type,&cs,&sn,&r,&cond);CHKERRQ(ierr);
      if(r>1e6){ierr = PetscPrintf(PETSC_COMM_WORLD,"Condition number of hyperbolic rotation %g\n",cond);CHKERRQ(ierr);}
      A[ni-n0+j*lda] = r; A[n-n1+j*lda] = 0.0;
      A[j+(ni-n0)*lda] = r; A[j+(n-n1)*lda] = 0.0; /**/
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
#endif
PetscFunctionReturn(0);
}


/*
  The system matrix is in Hessenberg form
*/
#undef __FUNCT__
#define __FUNCT__ "PSGHIEPPseudoOrthogInverseIteration"
static PetscErrorCode PSGHIEPPseudoOrthogInverseIteration(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(PETSC_MISSING_LAPACK_HSEIN)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HSEIN - Lapack routine is unavailable.");
#else
#if defined(PETSC_USE_COMPLEX)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"  PSGHIEP implemented only for real scalars ");
#else
  PetscErrorCode  ierr;
  PetscInt	  i,j,off;
  PetscBLASInt	  *select,*infoC,ld,n1,mout,info;
  PetscScalar	  *H,*X;
  PetscReal	  *s,*ss,*d,*e,h,d1,d2,toldeg=1e-5;/* ////////////// */

  PetscFunctionBegin;
  ld = PetscBLASIntCast(ps->ld);
  n1  = PetscBLASIntCast(ps->n - ps->l);
  ierr = PSAllocateWork_Private(ps,ld*ld,ld,2*ld);CHKERRQ(ierr);
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  H = ps->mat[PS_MAT_W];
  s = ps->rmat[PS_MAT_D];
  d = ps->rmat[PS_MAT_T];
  e = d + ld;
  select = ps->iwork;
  infoC = ps->iwork + ld;
  ierr = PetscMemzero(H,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  off = ps->l+ps->l*ld;
  if(ps->compact){
     H[off] = d[ps->l]*s[ps->l]; H[off+ld] = e[ps->l]*s[ps->l];
    for(i=ps->l+1;i<ps->n-1;i++){
      H[i+(i-1)*ld] = e[i-1]*s[i]; H[i+i*ld] = d[i]*s[i]; H[i+(i+1)*ld] = e[i]*s[i];
    }
    H[ps->n-1+(ps->n-2)*ld] = e[ps->n-2]*s[ps->n-1]; H[ps->n-1+(ps->n-1)*ld] = d[ps->n-1]*s[ps->n-1];
  }else{
    s[ps->l] = *(ps->mat[PS_MAT_B]+off);
    H[off] = *(ps->mat[PS_MAT_A]+off)*s[ps->l]; H[off+ld] = *(ps->mat[PS_MAT_A]+off+ld)*s[ps->l];
    for(i=ps->l+1;i<ps->n-1;i++){
      s[i] = *(ps->mat[PS_MAT_B]+i+i*ld);
      H[i+(i-1)*ld] =  *(ps->mat[PS_MAT_A]+i+(i-1)*ld)*s[i]; H[i+i*ld] =  *(ps->mat[PS_MAT_A]+i+i*ld)*s[i]; H[i+(i+1)*ld] =  *(ps->mat[PS_MAT_A]+i+(i+1)*ld)*= s[i];
    }
    s[ps->n-1] = *(ps->mat[PS_MAT_B]+ps->n-1+(ps->n-1)*ld);
    H[ps->n-1+(ps->n-2)*ld] =  *(ps->mat[PS_MAT_A]+ps->n-1+(ps->n-2)*ld)*s[ps->n-1]; H[ps->n-1+(ps->n-1)*ld] =  *(ps->mat[PS_MAT_A]+ps->n-1+(ps->n-1)*ld)*s[ps->n-1];
  }
  ierr = PSAllocateMat_Private(ps,PS_MAT_X);CHKERRQ(ierr);
  X = ps->mat[PS_MAT_X];
  for(i=0;i<n1;i++)select[i]=1;
  LAPACKhsein_("R","N","N",select,&n1,H+off,&ld,wr+ps->l,wi+ps->l,PETSC_NULL,&ld,X+off,&ld,&n1,&mout,ps->work,PETSC_NULL,infoC,&info);
  /* compute real s-orthonormal base */
  ss = ps->rwork;
  for(i=ps->l;i<ps->n;i++){
    if(wi[i]==0.0){/* real */
      for(j=i-1;j>=ps->l;j--){
         /* s-orthogonalization with close eigenvalues */
        if(wi[j]==0.0 && PetscAbsScalar(wr[j]-wr[i])<toldeg){
          ierr = IndefOrthog(s+ps->l, X+j*ld+ps->l, ss[j],X+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
        }
      }
      ierr = IndefNorm(s+ps->l,X+i*ld+ps->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0.0)?-1:1;
      d[i] = PetscRealPart(wr[i]*ss[i]); e[i] = 0.0;
    }else{
      for(j=i-1;j>=ps->l;j--){
        /* s-orthogonalization of Xi and Xi+1*/
        if(PetscAbsScalar(wr[j]-wr[i])<toldeg && PetscAbsScalar(PetscAbsScalar(wi[j])-PetscAbsScalar(wi[i]))<toldeg){
          ierr = IndefOrthog(s+ps->l, X+j*ld+ps->l, ss[j],X+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
          ierr = IndefOrthog(s+ps->l, X+j*ld+ps->l, ss[j],X+(i+1)*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
        }
      }
      ierr = IndefNorm(s+ps->l,X+i*ld+ps->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0)?-1:1;
      ierr = IndefOrthog(s+ps->l, X+i*ld+ps->l, ss[i],X+(i+1)*ld+ps->l, &h,n1);CHKERRQ(ierr);
      ierr = IndefNorm(s+ps->l,X+(i+1)*ld+ps->l,&d2,n1);CHKERRQ(ierr);
      ss[i+1] = (d2<0)?-1:1;
      d[i] = PetscRealPart((wr[i]-wi[i]*h/d1)*ss[i]);
      d[i+1] = PetscRealPart((wr[i]+wi[i]*h/d1)*ss[i+1]);
      e[i] = PetscRealPart(wi[i]*d2/d1*ss[i]); e[i+1] = 0.0;
      i++;
    }
  }
  for(i=ps->l;i<ps->n;i++) s[i] = ss[i];
#endif
#endif
  PetscFunctionReturn(0);
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
    ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TridiagDiag_HHR(ps->k-ps->l+1,A+off,ld,s+ps->l,Q+off,ld,PETSC_TRUE,d+ps->l,e+ps->l,ps->work,ld*ld,ps->perm);CHKERRQ(ierr);
    ps->k = ps->l;
    ierr = PetscMemzero(d+2*ld+ps->l,(ps->n-ps->l)*sizeof(PetscReal));CHKERRQ(ierr);
  }else{
    for(i=0;i<ps->n;i++)
      s[i] = PetscRealPart(B[i+i*ld]);
    ierr = TridiagDiag_HHR(ps->n-ps->l,A+off,ld,s+ps->l,Q+off,ld,PETSC_FALSE,d+ps->l,e+ps->l,ps->work,ld*ld,ps->perm);CHKERRQ(ierr);
    ierr = PetscMemzero(d+2*ld,(ps->n)*sizeof(PetscReal));CHKERRQ(ierr);
    ps->k = ps->l;
    ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GHIEP_EA_II"
PetscErrorCode PSSolve_GHIEP_EA_II(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(PETSC_MISSING_LAPACK_HSEIN)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"HSEIN - Lapack routine is unavailable.");
#else
#if defined(PETSC_USE_COMPLEX)
  PetscFunctionBegin;
  SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_SUP," In PSSolve, EA + II method not implemented for complex indefinite problems");
#else

  PetscErrorCode ierr;
  PetscInt       i,j,off;
  PetscBLASInt   lwork,n1,one,ld,*select,*infoC;
  PetscScalar    *A,*B,*W,*Q,zero,oneS,tmp;
  PetscReal      *d,*e,*s,*ss,t,*dw;

  PetscFunctionBegin;
  n1  = PetscBLASIntCast(ps->n - ps->l);
  one = PetscBLASIntCast(1);
  oneS = 1.0;
  zero = 0.0;
  ld = PetscBLASIntCast(ps->ld);
  off = ps->l + ps->l*ld;
  A  = ps->mat[PS_MAT_A];
  B  = ps->mat[PS_MAT_B];
  Q = ps->mat[PS_MAT_Q];
  d = ps->rmat[PS_MAT_T];
  e = ps->rmat[PS_MAT_T] + ld;
  s  = ps->rmat[PS_MAT_D];
  ierr = PSAllocateWork_Private(ps,ld+ld*ld,2*ld,ld*2);CHKERRQ(ierr); 
  lwork = ld*ld;
  select = ps->iwork;
  infoC = ps->iwork + ld;
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
  ierr = PSIntermediate_GHIEP( ps);CHKERRQ(ierr);

  /* Compute Eigenvalues (EA)*/
  /* routine dca require normalized form */
  ierr = PSSwitchFormat_GHIEP(ps,!ps->compact);CHKERRQ(ierr);
  ss = ps->rwork;
  dw = ps->rwork+ld;
  ss[ps->l] = s[ps->l];
  dw[ps->l] = d[ps->l];
  t = 1.0/e[ps->l]; t *= t;
  for(i=ps->l+1;i<ps->n-1;i++){
    dw[i] = d[i]*t; 
    ss[i] = s[i]*t;
    t = 1/(e[i]*t*e[i]);
  }
  dw[ps->n-1] = d[ps->n-1]*t; ss[ps->n-1] = s[ps->n-1]*t;

  SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_SUP,"Aberth fortran module unavailable");
#if 0
  __eigensolve_MOD_eigen(&n1,dw+ps->l,ss+ps->l,wr+ps->l,wi+ps->l);
#endif
  for(i=0;i<n1;i++)if(PetscAbsScalar(*(wi+ps->l+i))<PETSC_MACHINE_EPSILON)*(wi+ps->l+i)=0.0; /* ///// */
  /* Sort for having consecutive conjugate pairs */
  for(i=ps->l;i<ps->n;i++){
    if(PetscAbsScalar(wi[i])<PETSC_MACHINE_EPSILON) wi[i]=0.0;
    else{ /* complex eigenvalue */
      j=i+1;
      while(j<ps->n && (PetscAbsScalar((wr[i]-wr[j])/wr[i])>1e+4*PETSC_MACHINE_EPSILON || PetscAbsScalar((wi[i]+wi[j])/wi[i])>1e+4*PETSC_MACHINE_EPSILON))j++;
      if(j==ps->n){
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"IN EA COMPLEX WITHOUT CONJUGATE PAIR %d",1);
      }
      tmp = wi[i+1]; wi[i+1] = wi[j]; wi[j] = tmp;
      tmp = wr[i+1]; wr[i+1] = wr[j]; wr[j] = tmp;
      i++;
    }
  }
  /* Compute Eigenvectors with Inverse Iteration */
  ierr = PSGHIEPPseudoOrthogInverseIteration(ps,wr,wi);CHKERRQ(ierr);
  
/* accumulate previous Q */
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  W = ps->mat[PS_MAT_W];
  ierr = PSCopyMatrix_Private(ps,PS_MAT_W,PS_MAT_Q);CHKERRQ(ierr);

  BLASgemm_("N","N",&n1,&n1,&n1,&oneS,W+off,&ld,ps->mat[PS_MAT_X]+off,&ld,&zero,Q+off,&ld);

  /* The result is stored in both places (compact and regular) */
  if (!ps->compact) {
    ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);
  }
   /* Recover eigenvalues from diagonal */
  ierr = PSGHIEPComplexEigs(ps, 0, ps->n, wr, wi);CHKERRQ(ierr);/* /////////////// */
  ierr = PSSolve_GHIEP_Sort(ps,wr,wi);CHKERRQ(ierr);
#endif
#endif
    PetscFunctionReturn(0);
}


/*
Parameters:
ps (In/Out): On input the ps object contains (T,S) symmetric pencil with S  indefinite diagonal (signature matrix)
	On output ps contains Q and (D,SS), equivalent symmetric pencil whit D block diagonal and SS diagonal, 
	verifying: Q^T*T*Q = D and Q^T*S*Q = SS
wr,wi (Out): eigenvalues of equivalent pencils

(Modified only rows and columns ps->l to ps->n in T and S)
*/
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
  PetscBLASInt   lwork,info,n1,one,mout,ld;
  PetscScalar    *A,*B,*Q,*work,*tau;
  PetscReal      *d,*e,*s;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar	 h;
  PetscReal      *ss,toldeg=1e-5,d1,d2;
#endif

  PetscFunctionBegin;
  n1  = PetscBLASIntCast(ps->n - ps->l);
  one = PetscBLASIntCast(1);
  ld = PetscBLASIntCast(ps->ld);
  off = ps->l + ps->l*ld;
  A  = ps->mat[PS_MAT_A];
  B  = ps->mat[PS_MAT_B];
  Q = ps->mat[PS_MAT_Q];
  d  = ps->rmat[PS_MAT_T];
  e  = ps->rmat[PS_MAT_T] + ld;
  s  = ps->rmat[PS_MAT_D];
  ierr = PSAllocateWork_Private(ps,ld+ld*ld,ld,0);CHKERRQ(ierr); 
  tau  = ps->work;
  work = ps->work+ld;
  lwork = ld*ld;

   /* initialize orthogonal matrix */
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i< ps->n;i++) 
    Q[i+i*ld] = 1.0;
  if (n1 == 1) {
    if(ps->compact){
      wr[ps->l] = d[ps->l]/s[ps->l]; wi[ps->l] = 0.0;
    }else{
      d[ps->l] = PetscRealPart(A[off]); s[ps->l] = PetscRealPart(B[off]);
      wr[ps->l] = d[ps->l]/s[ps->l]; wi[ps->l] = 0.0;  
    }
    PetscFunctionReturn(0);
  }

  /* form standard problem in A */
  if (ps->compact) {
    ierr = PetscMemzero(A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    for(i=ps->l; i < ps->n-1; i++){
      A[i+i*ld] = d[i]/s[i];
      A[(i+1)+i*ld] = e[i]/s[i+1];
      A[i+(i+1)*ld] = e[i]/s[i];
    } 
    A[ps->n-1 + (ps->n-1)*ld] = d[ps->n-1]/s[ps->n-1];

    for(i=ps->l; i < ps->k; i++){
      A[ps->k+i*ld] = *(ps->rmat[PS_MAT_T]+2*ld+i)/s[ps->k];
      A[i + ps->k*ld] = *(ps->rmat[PS_MAT_T]+2*ld+i)/s[i];
    }
  }else{
    for(j=ps->l; j<ps->n; j++){
      for(i=ps->l; i<ps->n; i++){
        A[i+j*ld] /= B[i+i*ld];
      }
    }
  }
  
  /* reduce to upper Hessenberg form */
  if (ps->state<PS_STATE_INTERMEDIATE) {
  LAPACKgehrd_(&n1,&one,&n1,A+off,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEHRD %d",&info);
    for (j=ps->l;j<ps->n-1;j++) {
      for (i=j+2;i<ps->n;i++) {
        Q[i+j*ld] = A[i+j*ld];
        A[i+j*ld] = 0.0;
      }
    }
    LAPACKorghr_(&n1,&one,&n1,Q+off,&ld,tau,work,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGHR %d",&info);
  }

  /* compute the real Schur form */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKhseqr_("S","V",&n1,&one,&n1,A+off,&ld,wr+ps->l,wi+ps->l,Q+off,&ld,work,&lwork,&info);
#else
  LAPACKhseqr_("S","V",&n1,&one,&n1,A+off,&ld,wr,Q+off,&ld,work,&lwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",&info);
  
  /* compute eigenvectors */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKtrevc_("R","B",PETSC_NULL,&n1,A+off,&ld,PETSC_NULL,&ld,Q+off,&ld,&n1,&mout,ps->work,&info);
#else
  LAPACKtrevc_("R","B",PETSC_NULL,&n1,A+off,&ld,PETSC_NULL,&ld,Q+off,&ld,&n1,&mout,work,ps->rwork,&info);
#endif
  if (info) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_LIB,"Error in Lapack xTREVC %i",&info);
  /* compute real s-orthonormal base */
#if !defined(PETSC_USE_COMPLEX)
  ss = ps->rwork;
  for(i=ps->l;i<ps->n;i++){
    if(wi[i]==0.0){/* real */
      for(j=i-1;j>=ps->l;j--){
         /* s-orthogonalization with close eigenvalues */
        if(wi[j]==0 && PetscAbsScalar(wr[j]-wr[i])<toldeg){
          ierr = IndefOrthog(s+ps->l, Q+j*ld+ps->l, ss[j],Q+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
        }
      }
      ierr = IndefNorm(s+ps->l,Q+i*ld+ps->l,&h,n1);CHKERRQ(ierr);
      ss[i] = (h<0)?-1:1;
      d[i] = PetscRealPart(wr[i]*ss[i]); e[i] = 0.0;
    }else{
      for(j=i-1;j>=ps->l;j--){
        /* s-orthogonalization of Qi and Qi+1*/
        if(PetscAbsScalar(wr[j]-wr[i])<toldeg && PetscAbsReal(PetscAbsScalar(wi[j])-PetscAbsScalar(wi[i]))<toldeg){
          ierr =  IndefOrthog(s+ps->l, Q+j*ld+ps->l, ss[j],Q+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
          ierr = IndefOrthog(s+ps->l, Q+j*ld+ps->l, ss[j],Q+(i+1)*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
        }
      }
      ierr = IndefNorm(s+ps->l,Q+i*ld+ps->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0)?-1:1;
      ierr = IndefOrthog(s+ps->l, Q+i*ld+ps->l, ss[i],Q+(i+1)*ld+ps->l, &h,n1);CHKERRQ(ierr);
      ierr = IndefNorm(s+ps->l,Q+(i+1)*ld+ps->l,&d2,n1);CHKERRQ(ierr);
      ss[i+1] = (d2<0)?-1:1;
      d[i] = PetscRealPart((wr[i]-wi[i]*h/d1)*ss[i]);
      d[i+1] = PetscRealPart((wr[i]+wi[i]*h/d1)*ss[i+1]);
      e[i] = PetscRealPart(wi[i]*d2/d1*ss[i]); e[i+1] = 0.0;
      i++;
    }
  }
  for(i=ps->l;i<ps->n;i++) s[i] = ss[i];
  ps->k = ps->l;
  /* The result is stored in both places (compact and regular) */
  if (!ps->compact) {
    ierr = PetscMemzero(A+ps->l*ld,n1*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PSSwitchFormat_GHIEP(ps,PETSC_FALSE);CHKERRQ(ierr);
  }
  /* Recover eigenvalues from diagonal */
  ierr = PSGHIEPComplexEigs(ps, 0, ps->n, wr, wi);CHKERRQ(ierr);
  ierr = PSSolve_GHIEP_Sort(ps,wr,wi);CHKERRQ(ierr);
#else
  SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP," In PSSolve, QR method not implemented for complex indefinite problems",info);
#endif
  PetscFunctionReturn(0);
#endif
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
  ps->ops->solve[0]      = PSSolve_GHIEP_QR;
  ps->ops->solve[1]      = PSSolve_GHIEP_EA_II;
  PetscFunctionReturn(0);
}
EXTERN_C_END
