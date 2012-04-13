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
#define __FUNCT__ "PSAllocate_ArrowTrid"
PetscErrorCode PSAllocate_ArrowTrid(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PSAllocateMat_Private(ps,PS_MAT_X);CHKERRQ(ierr); 
  ierr = PSAllocateMatReal_Private(ps,PS_MAT_T);CHKERRQ(ierr); 
  ierr = PSAllocateMatReal_Private(ps,PS_MAT_Q);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView_ArrowTrid"
PetscErrorCode PSView_ArrowTrid(PS ps,PetscViewer viewer)
{
  PetscInt          i,j,r,c;
  PetscReal         value;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
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
      ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",r,c,*(ps->rmat[PS_MAT_T]+ps->n+i));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,*(ps->rmat[PS_MAT_T]+ps->n+i));CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",PSMatName[PS_MAT_T]);CHKERRQ(ierr);
  } else {
    for (i=0;i<ps->n;i++) {
      for (j=0;j<ps->n;j++) {
        if (i==j) value = *(ps->rmat[PS_MAT_T]+i);
        else if ((i<ps->k && j==ps->k) || (i==ps->k && j<ps->k)) value = *(ps->rmat[PS_MAT_T]+ps->n+PetscMin(i,j));
        else if (i==j+1 && i>ps->k) value = *(ps->rmat[PS_MAT_T]+ps->n+i-1);
        else if (i+1==j && j>ps->k) value = *(ps->rmat[PS_MAT_T]+ps->n+j-1);
        else value = 0.0;
        ierr = PetscViewerASCIIPrintf(viewer," %7.5e ",value);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PSViewMat_Private(ps,viewer,PS_MAT_X);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSolve_ArrowTrid"
PetscErrorCode PSSolve_ArrowTrid(PS ps,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_SYTRD) || defined(SLEPC_MISSING_LAPACK_ORGTR) || defined(SLEPC_MISSING_LAPACK_STEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYTRD/ORGTR/STEQR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBLASInt   n1,n2,lwork,info,n,ld;
  PetscReal      *S,*Q,*d,*e;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  Q  = ps->rmat[PS_MAT_Q];
  /* quick return */
  if (n == 1) {
    Q[0] = 1.0;
    PetscFunctionReturn(0);    
  }
  ld = PetscBLASIntCast(ps->ld);
  n1 = PetscBLASIntCast(ps->k+1);    /* size of leading block, including residuals */
  n2 = PetscBLASIntCast(n-ps->k-1);  /* size of trailing block */
  d  = ps->rmat[PS_MAT_T];
  e  = ps->rmat[PS_MAT_T]+ld;

  /* reduce to tridiagonal form */
  if (ps->state<PS_STATE_INTERMEDIATE) {

    if (!ps->rmat[PS_MAT_W]) { ierr = PSAllocateMatReal_Private(ps,PS_MAT_W);CHKERRQ(ierr); }
    S = ps->rmat[PS_MAT_W];
    ierr = PetscMemzero(S,ld*ld*sizeof(PetscReal));CHKERRQ(ierr);

    /* Flip matrix S */
    for (i=0;i<n;i++) 
      S[(n-1-i)+(n-1-i)*ld] = d[i];
    for (i=0;i<ps->k;i++)
      S[(n-1-i)+(n-1-ps->k)*ld] = e[i];
    for (i=ps->k;i<n-1;i++)
      S[(n-1-i)+(n-1-i-1)*ld] = e[i];

    /* Reduce (2,2)-block of flipped S to tridiagonal form */
    lwork = PetscBLASIntCast(ps->ld*ps->ld-ps->ld);
    LAPACKsytrd_("L",&n1,S+n2+n2*ld,&ld,d,e,Q,Q+ld,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);

    /* Flip back diag and subdiag, put them in d and e */
    for (i=0;i<n-1;i++) {
      d[n-i-1] = S[i+i*ld];
      e[n-i-2] = S[i+1+i*ld];
    }
    d[0] = S[n-1+(n-1)*ld];

    /* Compute the orthogonal matrix used for tridiagonalization */
    LAPACKorgtr_("L",&n1,S+n2+n2*ld,&ld,Q,Q+ld,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);

    /* Create full-size Q, flipped back to original order */
    for (i=0;i<n;i++) 
      for (j=0;j<n;j++) 
        Q[i+j*ld] = 0.0;
    for (i=n1;i<n;i++) 
      Q[i+i*ld] = 1.0;
    for (i=0;i<n1;i++) 
      for (j=0;j<n1;j++) 
        Q[i+j*ld] = S[n-i-1+(n-j-1)*ld];

  } else {
    /* initialize orthogonal matrix */
    ierr = PetscMemzero(Q,ld*ld*sizeof(PetscReal));CHKERRQ(ierr);
    for (i=0;i<n;i++) 
      Q[i+i*ld] = 1.0;
  }

  /* Solve the tridiagonal eigenproblem */
  if (ps->state<PS_STATE_CONDENSED) {
    ierr = PSAllocateWork_Private(ps,0,2*ld,0);CHKERRQ(ierr); 
    LAPACKsteqr_("V",&n,d,e,Q,&ld,ps->rwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSTEQR %d",info);
    for (i=0;i<n;i++) wr[i] = d[i];
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PSSort_ArrowTrid"
PetscErrorCode PSSort_ArrowTrid(PS ps,PetscScalar *wr,PetscScalar *wi,PetscErrorCode (*comp_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void *comp_ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,p,*perm;
  PetscBLASInt   n,ld;
  PetscReal      *Q,*X,*d,rtmp;

  PetscFunctionBegin;
  n  = PetscBLASIntCast(ps->n);
  ld = PetscBLASIntCast(ps->ld);
  d  = ps->rmat[PS_MAT_T];
  Q  = ps->rmat[PS_MAT_Q];
  X  = ps->mat[PS_MAT_X];
  ierr = PSAllocateWork_Private(ps,0,0,ld);CHKERRQ(ierr); 
  perm = ps->iwork;
  ierr = PSSortEigenvaluesReal_Private(ps,n,d,perm,comp_func,comp_ctx);CHKERRQ(ierr);
  for (i=0;i<n;i++)
    wr[i] = d[perm[i]];
  for (i=0;i<n;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap eigenvectors i and j */
      for (k=0;k<n;k++) {
        rtmp = Q[k+p*ld]; Q[k+p*ld] = Q[k+i*ld]; Q[k+i*ld] = rtmp;
      }
    }
  }
  for (i=0;i<n;i++) 
    for (j=0;j<n;j++) 
      X[i+j*ld] = Q[i+j*ld];
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PSCreate_ArrowTrid"
PetscErrorCode PSCreate_ArrowTrid(PS ps)
{
  PetscFunctionBegin;
  ps->ops->allocate      = PSAllocate_ArrowTrid;
  ps->ops->view          = PSView_ArrowTrid;
  //ps->ops->computevector = PSComputeVector_ArrowTrid;
  ps->ops->solve         = PSSolve_ArrowTrid;
  ps->ops->sort          = PSSort_ArrowTrid;
  PetscFunctionReturn(0);
}
EXTERN_C_END

