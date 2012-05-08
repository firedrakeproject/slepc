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
  compute X = X - Y*ss^{-1}*Y^T*s*X where ss=Y^T*s*Y
  s diagonal (signature matrix)
*/
#undef __FUNCT__  
#define __FUNCT__ "PSOrthog_private"
static PetscErrorCode PSOrthog_private(PetscReal *s, PetscScalar *Y, PetscReal ss, PetscScalar *X, PetscScalar *h,PetscInt n)
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
#define __FUNCT__ "PSNormIndef_private"
static PetscErrorCode PSNormIndef_private(PetscReal *s,PetscScalar *X, PetscReal *norm,PetscInt n)
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSwitchFormat_GHIEP"
PetscErrorCode PSSwitchFormat_GHIEP(PS ps,PetscBool tocompact)
{
  PetscReal	*T;
  PetscScalar	*A,*B;
  PetscInt	i,k,ld;

  PetscFunctionBegin;
  A = ps->mat[PS_MAT_A];
  B = ps->mat[PS_MAT_B];
  T = ps->rmat[PS_MAT_T];
  k = ps->k;
  ld = ps->ld;
  if(tocompact){ /* switch from dense (arrow) to compact */
    for(i=0; i < k; i++){
      T[i] = PetscRealPart(A[i*(1+ld)]);
      T[ld +i] = PetscRealPart(A[k+i*ld]);
      T[2*ld +i] = PetscRealPart(B[i*(1+ld)]);
    }
    for(i=k; i < ps->n; i++){
      T[i] = PetscRealPart(A[i*(1+ld)]);
      T[ld +i] = PetscRealPart(A[i*(ld+1)+1]);
      T[2*ld +i] = PetscRealPart(B[i*(1+ld)]);
    }
  }else{ /* switch from compact (arrow) to dense */
    for(i=0; i < k; i++){
      A[i*(1+ld)] = T[i];
      A[k+i*ld] = T[ld+i];
      A[i+k*ld] = T[ld+i];
      B[i*(1+ld)] = T[2*ld+i];
    }
    A[k*(ld+1)] = T[k];
    B[k*(ld+1)] = T[2*ld+k];
    for(i=k+1; i < ps->n; i++){
      A[i*(1+ld)] = T[i];
      A[i*ld + i-1] = T[i-1+ld];
      A[(i-1)*ld + i] = T[i-1+ld];
      B[i*(1+ld)] = T[2*ld+i];
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
  PetscInt          i,j,r,c;
  PetscReal         value;
  const char *methodname[] = {
                     "QR method",
                     "QR + Inverse Iteration"
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
        r = PetscMax(i+2,ps->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",r,c,*(ps->rmat[PS_MAT_T]+ps->ld+i));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,*(ps->rmat[PS_MAT_T]+ps->ld+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",PSMatName[PS_MAT_T]);CHKERRQ(ierr);
      
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",ps->n,ps->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"omega = zeros(%D,3);\n",3*ps->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"omega = [\n");CHKERRQ(ierr);
      for (i=0;i<ps->n;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,*(ps->rmat[PS_MAT_T]+2*(ps->ld)+i));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(omega);\n",PSMatName[PS_MAT_B]);CHKERRQ(ierr);

    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"T\n");CHKERRQ(ierr);
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
      ierr = PetscViewerASCIIPrintf(viewer,"omega\n");CHKERRQ(ierr);
      for (i=0;i<ps->n;i++) {
        for (j=0;j<ps->n;j++) {
          if (i==j) value = *(ps->rmat[PS_MAT_T]+2*(ps->ld)+i);
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
PetscErrorCode PSVectors_GHIEP_Eigen_Some(PS ps,PetscInt *k,PetscReal *rnorm,PetscBool left)
{

  /* to complete  */
  PetscScalar    *Q = ps->mat[PS_MAT_Q];
  PetscReal	 s1,s2,d1,d2,b;
  PetscInt       ld = ps->ld,k_;
  PetscErrorCode ierr;
  PSMatType	 mat;
  
  PetscFunctionBegin;
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
#define __FUNCT__ "PSComplexEigs_private"
static PetscErrorCode PSComplexEigs_private(PS ps, PetscInt n0, PetscInt n1, PetscScalar *wr, PetscScalar *wi){
  PetscInt	j,ld;
  PetscScalar	*A,*B;
  PetscReal	*d,*e,*s,d1,d2,e1,e2,disc;

  PetscFunctionBegin;
  ld = ps->ld;
  if (ps->compact){
    d = ps->rmat[PS_MAT_T];
    e = d + ld;
    s = e + ld;
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
        d1 = PetscRealPart(A[j+j*ld])/PetscRealPart(B[j+j*ld]);
        d2 = PetscRealPart(A[(j+1)+(j+1)*ld])/PetscRealPart(B[(j+1)+(j+1)*ld]);
        e1 = PetscRealPart(A[j+(j+1)*ld])/PetscRealPart(B[j+j*ld]);
        e2 = PetscRealPart(A[(j+1)+j*ld])/PetscRealPart(B[(j+1)+(j+1)*ld]);
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

/*
  Form an hyperbolic rotator
    if x1*x1 - x2*x2 != 0 
      r = sqrt( |x1*x1 - x2*x2| )
      c = x1/r  s = x2/r
     
      | c -s||x1|   |d*r|
      |-s  c||x2| = | 0 | 
      where d = 1 for type==1 and -1 for type==2
*/
#undef __FUNCT__
#define __FUNCT__ "PSlarth_private"
static PetscErrorCode PSlarth_private(PetscReal x1,PetscReal x2,PetscInt *type,PetscReal *c,PetscReal *s,PetscReal *r)
{
  PetscReal t,n2,xa,xb;
  PetscInt  type_;
  PetscFunctionBegin;
  if(x2==0) {
    *c = 1.0; *s = 0.0; *r = PetscAbsReal(x1); *type = 1;
    PetscFunctionReturn(0);
  }
  if(PetscAbsReal(x1) == PetscAbsReal(x2)){
  /* Not exists hyperbolic rotator */
    *c = 0; *s = 0; *r = 0; *type = 0;
    PetscFunctionReturn(0);
  }
  
  if(PetscAbsReal(x1)>PetscAbsReal(x2)){
    xa = x1; xb = x2; type_ =1;
  }else{ xa = x2; xb = x1; type_ =2;
  } 
  t = xb/xa;
  n2 = PetscAbsReal(1 - t*t);
  *r = PetscSqrtReal(n2)*PetscAbsReal(xa);
  *c = x1/(*r);
  *s = x2/(*r);
  if(type_ == 2) *r *= -1;
  if(type) *type = type_;
  PetscFunctionReturn(0);
}

/* 
				| c  -s|
  Apply an hyperbolic rotator	|-s   c|
           |c  s|
    [X1 X2]|s  c| 
*/
#undef __FUNCT__
#define __FUNCT__ "PSRoth_private"
static PetscErrorCode PSRoth_private(PetscInt n, PetscScalar *X1, PetscScalar *X2, PetscReal c, PetscReal s)
{
PetscFunctionBegin;

PetscFunctionReturn(0);
}
/*
  Reduce an arrowhead symmetric-diagonal pair to tridiagonal-diagonal
  Omega: signature matrix
*/
#undef __FUNCT__  
#define __FUNCT__ "ArrowTridiagDiag"
static PetscErrorCode ArrowTridiagDiag(PetscInt k,PetscInt n,PetscReal *d,PetscReal *e,PetscReal *Omega,PetscScalar *Q,PetscInt ldq)
{
  PetscFunctionBegin;
  PetscBLASInt 	  j2,ld=ldq,one=1;
  PetscInt   	  i,j,type;
  PetscReal    	  c,s,p,off,e1,e0,d1,d0,temp;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (n<=2) PetscFunctionReturn(0);
  
  for (j=0;j<n-2;j++) {
    /* Eliminate entry e(j) by a rotation in the planes (j,j+1) */
    type = (Omega[j]*Omega[j+1]>0.0)?1:-1;
    if(PetscAbsReal(e[j+1]) < PetscAbsReal(e[j])) type = 2;
    e0 = e[j]; e1 = e[j+1];
    d0 = d[j]; d1 = d[j+1];
    temp = e[j+1];
    if( type > 0){ /* unitary rotator */
      LAPACKlartg_(&temp,&e[j],&c,&s,&e[j+1]);
    }else{/* hyperbolic rotator */
      ierr = PSlarth_private(temp,e[j],PETSC_NULL,&c,&s,&e[j+1]);CHKERRQ(ierr);
    }
    s = -s;
    /* Apply rotation to diagonal elements */
    temp   = d[j+1];
    e[j]   = c*s*(temp-d[j]);
    d[j+1] = s*s*d[j] + c*c*temp;
    d[j]   = c*c*d[j] + s*s*temp;

    /* Apply rotation to Q */
    j2 = j+2;
    if(type == 0) {
      BLASrot_(&j2,Q+j*ld,&one,Q+(j+1)*ld,&one,&c,&s);
    }else{
      ierr = PSRoth_private(j2, Q+j*ld,Q+(j+1)*ld,c,s);CHKERRQ(ierr);
    }
    /* Chase newly introduced off-diagonal entry to the top left corner */
    for (i=j-1;i>=0;i--) {
      off  = -type*s*e[i];
      e[i] = c*e[i];
      temp = e[i+1];
      type = (Omega[i]*Omega[i+1]>0.0)?1:-1;
      if(type > 0){
        LAPACKlartg_(&temp,&off,&c,&s,&e[i+1]);
      }else{
        ierr = PSlarth_private(e[i+1],off,PETSC_NULL,&c,&s,&e[i+1]);CHKERRQ(ierr);
      }
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


#undef __FUNCT__  
#define __FUNCT__ "PSSolve_GHIEP_QR_II"
PetscErrorCode PSSolve_GHIEP_QR_II(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_GEHRD) || defined(SLEPC_MISSING_LAPACK_ORGHR) || defined(PETSC_MISSING_LAPACK_HSEQR) || defined(PETSC_MISSING_LAPACK_HSEIN)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEHRD/ORGHR/HSEQR/HSEIN - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,off;
  PetscBLASInt   lwork,info,n1,one,ld,*select,*infoC,mout;
  PetscScalar    *A,*B,*W,*Q,*work,*tau,zero,oneS,h;
  PetscReal      *d,*e,*s,*ss,toldeg=1e-5,d1,d2;

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
  d  = ps->rmat[PS_MAT_T];
  e  = ps->rmat[PS_MAT_T] + ld;
  s  = ps->rmat[PS_MAT_T] + 2*ld;
  ierr = PSAllocateWork_Private(ps,ld+ld*ld,ld,ld*2);CHKERRQ(ierr); 
  tau  = ps->work;
  work = ps->work+ld;
  lwork = ld*ld;
  select = ps->iwork;
  infoC = ps->iwork + ld;
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
    for(i=ps->l; i < ps->k; i++){
      A[i+i*ld] = d[i]/s[i];
      A[ps->k+i*ld] = e[i]/s[ps->k];
      A[i + ps->k*ld] = e[i]/s[i];
    }
    A[ps->k + ps->k*ld] = d[ps->k]/s[ps->k];
    for(i=ps->k+1; i < ps->n; i++){
      A[i+i*ld] = d[i]/s[i];
      A[(i-1)+i*ld] = e[i-1]/s[i-1];
      A[i+(i-1)*ld] = e[i-1]/s[i];
    } 
  }else{
    for(j=ps->l; j<ps->n; j++){
      for(i=ps->l; i<ps->n; i++){
        A[i+j*ld] /= B[i+i*ld];
      }
    }
  }
  
  /* reduce to upper Hessemberg form */
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
  
  /* Compute Eigenvalues (QR)*/
  ierr = PSAllocateMat_Private(ps,PS_MAT_W);CHKERRQ(ierr);
  W = ps->mat[PS_MAT_W];
  ierr = PetscMemcpy(W,A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  
#if !defined(PETSC_USE_COMPLEX)
  LAPACKhseqr_("E","N",&n1,&one,&n1,W+off,&ld,wr+ps->l,wi+ps->l,PETSC_NULL,&ld,work,&lwork,&info);
#else
  LAPACKhseqr_("E","N",&n1,&one,&n1,W+off,&ld,wr+ps->l,PETSC_NULL,&ld,work,&lwork,&info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xHSEQR %d",&info);

  /* Compute Eigenvectors with Inverse Iteration */
#if !defined(PETSC_USE_COMPLEX)  
  for(i=0;i<n1;i++)select[i]=1;
  LAPACKhsein_("R","N","N",select,&n1,A+off,&ld,wr+ps->l,wi+ps->l,PETSC_NULL,&ld,W+off,&ld,&n1,&mout,work,PETSC_NULL,infoC,&info);
#else
  SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP," In PSSolve, QR + II method not implemented for complex indefinite problems",info);
#endif
  /* accumulate previous Q */ 
  if (ps->state<PS_STATE_INTERMEDIATE) {  
    BLASgemm_("N","N",&n1,&n1,&n1,&oneS,Q+off,&ld,W+off,&ld,&zero,A+off,&ld);
    ierr = PSCopyMatrix_Private(ps,PS_MAT_Q,PS_MAT_A);CHKERRQ(ierr); 
  }else {ierr = PSCopyMatrix_Private(ps,PS_MAT_Q,PS_MAT_W);CHKERRQ(ierr);}
  /* compute real s-orthonormal base */
  ss = ps->rwork;
  for(i=ps->l;i<ps->n;i++){
    if(wi[i]==0.0){/* real */
      for(j=i-1;j>=ps->l;j--){
         /* s-orthogonalization with close eigenvalues */
        if(wi[j]==0.0 && PetscAbsScalar(wr[j]-wr[i])<toldeg){
          ierr = PSOrthog_private(s+ps->l, Q+j*ld+ps->l, ss[j],Q+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
        }
      }
      ierr = PSNormIndef_private(s+ps->l,Q+i*ld+ps->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0.0)?-1:1;
      d[i] = PetscRealPart(wr[i])*ss[i]; e[i] = 0.0;
    }else{
      for(j=i-1;j>=ps->l;j--){
        /* s-orthogonalization of Qi and Qi+1*/
        if(PetscAbsScalar(wr[j]-wr[i])<toldeg && PetscAbsScalar(PetscAbsScalar(wi[j])-PetscAbsScalar(wi[i]))<toldeg){
          ierr = PSOrthog_private(s+ps->l, Q+j*ld+ps->l, ss[j],Q+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
          ierr = PSOrthog_private(s+ps->l, Q+j*ld+ps->l, ss[j],Q+(i+1)*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
        }
      }
      ierr = PSNormIndef_private(s+ps->l,Q+i*ld+ps->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0)?-1:1;
      ierr = PSOrthog_private(s+ps->l, Q+i*ld+ps->l, ss[i],Q+(i+1)*ld+ps->l, &h,n1);CHKERRQ(ierr);
      ierr = PSNormIndef_private(s+ps->l,Q+(i+1)*ld+ps->l,&d2,n1);CHKERRQ(ierr);
      ss[i+1] = (d2<0)?-1:1;
      d[i] = (PetscRealPart(wr[i])-PetscRealPart(wi[i])*h/d1)*ss[i];
      d[i+1] = (PetscRealPart(wr[i])+PetscRealPart(wi[i])*h/d1)*ss[i+1];
      e[i] = PetscRealPart(wi[i])*d2/d1*ss[i]; e[i+1] = 0.0;
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
  ierr = PSComplexEigs_private(ps, 0, ps->n, wr, wi);CHKERRQ(ierr);
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
  PetscScalar	 h;
  PetscReal      *d,*e,*s,*ss,toldeg=1e-5,d1,d2;

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
  s  = ps->rmat[PS_MAT_T] + 2*ld;
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
    for(i=ps->l; i < ps->k; i++){
      A[i+i*ld] = d[i]/s[i];
      A[ps->k+i*ld] = e[i]/s[ps->k];
      A[i + ps->k*ld] = e[i]/s[i];
    }
    A[ps->k + ps->k*ld] = d[ps->k]/s[ps->k];
    for(i=ps->k+1; i < ps->n; i++){
      A[i+i*ld] = d[i]/s[i];
      A[(i-1)+i*ld] = e[i-1]/s[i-1];
      A[i+(i-1)*ld] = e[i-1]/s[i];
    } 
  }else{
    for(j=ps->l; j<ps->n; j++){
      for(i=ps->l; i<ps->n; i++){
        A[i+j*ld] /= B[i+i*ld];
      }
    }
  }
  
  /* reduce to upper Hessemberg form */
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
          ierr = PSOrthog_private(s+ps->l, Q+j*ld+ps->l, ss[j],Q+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
        }
      }
      ierr = PSNormIndef_private(s+ps->l,Q+i*ld+ps->l,&h,n1);CHKERRQ(ierr);
      ss[i] = (h<0)?-1:1;
      d[i] = PetscRealPart(wr[i])*ss[i]; e[i] = 0.0;
    }else{
      for(j=i-1;j>=ps->l;j--){
        /* s-orthogonalization of Qi and Qi+1*/
        if(PetscAbsScalar(wr[j]-wr[i])<toldeg && PetscAbsReal(PetscAbsScalar(wi[j])-PetscAbsScalar(wi[i]))<toldeg){
          ierr =  PSOrthog_private(s+ps->l, Q+j*ld+ps->l, ss[j],Q+i*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
          ierr = PSOrthog_private(s+ps->l, Q+j*ld+ps->l, ss[j],Q+(i+1)*ld+ps->l, PETSC_NULL,n1);CHKERRQ(ierr);
        }
      }
      ierr = PSNormIndef_private(s+ps->l,Q+i*ld+ps->l,&d1,n1);CHKERRQ(ierr);
      ss[i] = (d1<0)?-1:1;
      ierr = PSOrthog_private(s+ps->l, Q+i*ld+ps->l, ss[i],Q+(i+1)*ld+ps->l, &h,n1);CHKERRQ(ierr);
      ierr = PSNormIndef_private(s+ps->l,Q+(i+1)*ld+ps->l,&d2,n1);CHKERRQ(ierr);
      ss[i+1] = (d2<0)?-1:1;
      d[i] = (PetscRealPart(wr[i])-PetscRealPart(wi[i])*h/d1)*ss[i];
      d[i+1] = (PetscRealPart(wr[i])+PetscRealPart(wi[i])*h/d1)*ss[i+1];
      e[i] = PetscRealPart(wi[i])*d2/d1*ss[i]; e[i+1] = 0.0;
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
  ierr = PSComplexEigs_private(ps, 0, ps->n, wr, wi);CHKERRQ(ierr);
#else
  SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_SUP," In PSSolve, QR method not implemented for complex indefinite problems",info);
#endif
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PSSortEigenvalues_Private"
static PetscErrorCode PSSortEigenvalues_Private(PS ps,PetscScalar *wr,PetscScalar *wi,PetscInt *perm,PetscErrorCode (*comp_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void *comp_ctx)
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
    ierr = (*comp_func)(re,im,wr[perm[j]],wi[perm[j]],&result,comp_ctx);CHKERRQ(ierr);
    while (result<0 && j>=ps->l) {
      perm[j+d]=perm[j]; j--;
#if !defined(PETSC_USE_COMPLEX)
      if(wi[perm[j+1]]!=0)
#else
      if(PetscImaginaryPart(wr[perm[j+1]])!=0)
#endif
        {perm[j+d]=perm[j]; j--;}

     if (j>=ps->l) {
        ierr = (*comp_func)(re,im,wr[perm[j]],wi[perm[j]],&result,comp_ctx);CHKERRQ(ierr);
      }
    }
    perm[j+1] = tmp1;
    if(d==2) perm[j+2] = tmp2;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PSSort_GHIEP"
PetscErrorCode PSSort_GHIEP(PS ps,PetscScalar *wr,PetscScalar *wi,PetscErrorCode (*comp_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void *comp_ctx)
{
  PetscErrorCode ierr;
  PetscInt       n,i,*perm;
  PetscReal      *d,*e,*s;

  PetscFunctionBegin;
  n = ps->n;
  d = ps->rmat[PS_MAT_T];
  e = d + ps->ld;
  s = d + 2*ps->ld;
  ierr = PSAllocateWork_Private(ps,ps->ld,ps->ld,ps->ld);CHKERRQ(ierr); 
  perm = ps->perm;
  ierr = PSSortEigenvalues_Private(ps,wr,wi,perm,comp_func,comp_ctx);CHKERRQ(ierr);
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
  ps->ops->solve[1]      = PSSolve_GHIEP_QR_II;
  ps->ops->sort          = PSSort_GHIEP;
  PetscFunctionReturn(0);
}
EXTERN_C_END
