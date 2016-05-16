/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/dsimpl.h>
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "DSAllocate_HEP"
PetscErrorCode DSAllocate_HEP(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DSAllocateMat_Private(ds,DS_MAT_A);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_Q);CHKERRQ(ierr);
  ierr = DSAllocateMatReal_Private(ds,DS_MAT_T);CHKERRQ(ierr);
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld,&ds->perm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ds,ld*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*   0       l           k                 n-1
    -----------------------------------------
    |*       .           .                  |
    |  *     .           .                  |
    |    *   .           .                  |
    |      * .           .                  |
    |. . . . o           o                  |
    |          o         o                  |
    |            o       o                  |
    |              o     o                  |
    |                o   o                  |
    |                  o o                  |
    |. . . . o o o o o o o x                |
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
#define __FUNCT__ "DSSwitchFormat_HEP"
static PetscErrorCode DSSwitchFormat_HEP(DS ds,PetscBool tocompact)
{
  PetscErrorCode ierr;
  PetscReal      *T = ds->rmat[DS_MAT_T];
  PetscScalar    *A = ds->mat[DS_MAT_A];
  PetscInt       i,n=ds->n,k=ds->k,ld=ds->ld;

  PetscFunctionBegin;
  if (tocompact) { /* switch from dense (arrow) to compact storage */
    ierr = PetscMemzero(T,3*ld*sizeof(PetscReal));CHKERRQ(ierr);
    for (i=0;i<k;i++) {
      T[i] = PetscRealPart(A[i+i*ld]);
      T[i+ld] = PetscRealPart(A[k+i*ld]);
    }
    for (i=k;i<n-1;i++) {
      T[i] = PetscRealPart(A[i+i*ld]);
      T[i+ld] = PetscRealPart(A[i+1+i*ld]);
    }
    T[n-1] = PetscRealPart(A[n-1+(n-1)*ld]);
    if (ds->extrarow) T[n-1+ld] = PetscRealPart(A[n+(n-1)*ld]);
  } else { /* switch from compact (arrow) to dense storage */
    ierr = PetscMemzero(A,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
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
    if (ds->extrarow) A[n+(n-1)*ld] = T[n-1+ld];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSView_HEP"
PetscErrorCode DSView_HEP(DS ds,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscInt          i,j,r,c,rows;
  PetscReal         value;
  const char        *methodname[] = {
                     "Implicit QR method (_steqr)",
                     "Relatively Robust Representations (_stevr)",
                     "Divide and Conquer method (_stedc)",
                     "Block Divide and Conquer method (dsbtdc)"
  };
  const int         nmeth=sizeof(methodname)/sizeof(methodname[0]);

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    if (ds->bs>1) {
      ierr = PetscViewerASCIIPrintf(viewer,"block size: %D\n",ds->bs);CHKERRQ(ierr);
    }
    if (ds->method>=nmeth) {
      ierr = PetscViewerASCIIPrintf(viewer,"solving the problem with: INVALID METHOD\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"solving the problem with: %s\n",methodname[ds->method]);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  if (ds->compact) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    rows = ds->extrarow? ds->n+1: ds->n;
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",rows,ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",3*ds->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
      for (i=0;i<ds->n;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,i+1,*(ds->rmat[DS_MAT_T]+i));CHKERRQ(ierr);
      }
      for (i=0;i<rows-1;i++) {
        r = PetscMax(i+2,ds->k+1);
        c = i+1;
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",r,c,*(ds->rmat[DS_MAT_T]+ds->ld+i));CHKERRQ(ierr);
        if (i<ds->n-1 && ds->k<ds->n) { /* do not print vertical arrow when k=n */
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",c,r,*(ds->rmat[DS_MAT_T]+ds->ld+i));CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer,"];\n%s = spconvert(zzz);\n",DSMatName[DS_MAT_T]);CHKERRQ(ierr);
    } else {
      for (i=0;i<rows;i++) {
        for (j=0;j<ds->n;j++) {
          if (i==j) value = *(ds->rmat[DS_MAT_T]+i);
          else if ((i<ds->k && j==ds->k) || (i==ds->k && j<ds->k)) value = *(ds->rmat[DS_MAT_T]+ds->ld+PetscMin(i,j));
          else if (i==j+1 && i>ds->k) value = *(ds->rmat[DS_MAT_T]+ds->ld+i-1);
          else if (i+1==j && j>ds->k) value = *(ds->rmat[DS_MAT_T]+ds->ld+j-1);
          else value = 0.0;
          ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",value);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    ierr = DSViewMat(ds,viewer,DS_MAT_A);CHKERRQ(ierr);
  }
  if (ds->state>DS_STATE_INTERMEDIATE) {
    ierr = DSViewMat(ds,viewer,DS_MAT_Q);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSVectors_HEP"
PetscErrorCode DSVectors_HEP(DS ds,DSMatType mat,PetscInt *j,PetscReal *rnorm)
{
  PetscScalar    *Q = ds->mat[DS_MAT_Q];
  PetscInt       ld = ds->ld,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_X:
    case DS_MAT_Y:
      if (j) {
        if (ds->state>=DS_STATE_CONDENSED) {
          ierr = PetscMemcpy(ds->mat[mat]+(*j)*ld,Q+(*j)*ld,ld*sizeof(PetscScalar));CHKERRQ(ierr);
        } else {
          ierr = PetscMemzero(ds->mat[mat]+(*j)*ld,ld*sizeof(PetscScalar));CHKERRQ(ierr);
          *(ds->mat[mat]+(*j)+(*j)*ld) = 1.0;
        }
      } else {
        if (ds->state>=DS_STATE_CONDENSED) {
          ierr = PetscMemcpy(ds->mat[mat],Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
        } else {
          ierr = PetscMemzero(ds->mat[mat],ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
          for (i=0;i<ds->n;i++) *(ds->mat[mat]+i+i*ld) = 1.0;
        }
      }
      if (rnorm && j) *rnorm = PetscAbsScalar(Q[ds->n-1+(*j)*ld]);
      break;
    case DS_MAT_U:
    case DS_MAT_VT:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSNormalize_HEP"
PetscErrorCode DSNormalize_HEP(DS ds,DSMatType mat,PetscInt col)
{
  PetscFunctionBegin;
  switch (mat) {
    case DS_MAT_X:
    case DS_MAT_Y:
    case DS_MAT_Q:
      /* All the matrices resulting from DSVectors and DSSolve are already normalized */
      break;
    case DS_MAT_U:
    case DS_MAT_VT:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented yet");
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_OUTOFRANGE,"Invalid mat parameter");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrowTridiag"
/*
  ARROWTRIDIAG reduces a symmetric arrowhead matrix of the form

                [ d 0 0 0 e ]
                [ 0 d 0 0 e ]
            A = [ 0 0 d 0 e ]
                [ 0 0 0 d e ]
                [ e e e e d ]

  to tridiagonal form

                [ d e 0 0 0 ]
                [ e d e 0 0 ]
   T = Q'*A*Q = [ 0 e d e 0 ],
                [ 0 0 e d e ]
                [ 0 0 0 e d ]

  where Q is an orthogonal matrix. Rutishauser's algorithm is used to
  perform the reduction, which requires O(n**2) flops. The accumulation
  of the orthogonal factor Q, however, requires O(n**3) flops.

  Arguments
  =========

  N       (input) INTEGER
          The order of the matrix A.  N >= 0.

  D       (input/output) DOUBLE PRECISION array, dimension (N)
          On entry, the diagonal entries of the matrix A to be
          reduced.
          On exit, the diagonal entries of the reduced matrix T.

  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
          On entry, the off-diagonal entries of the matrix A to be
          reduced.
          On exit, the subdiagonal entries of the reduced matrix T.

  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ, N)
          On exit, the orthogonal matrix Q.

  LDQ     (input) INTEGER
          The leading dimension of the array Q.

  Note
  ====
  Based on Fortran code contributed by Daniel Kressner
*/
static PetscErrorCode ArrowTridiag(PetscBLASInt n,PetscReal *d,PetscReal *e,PetscScalar *Q,PetscBLASInt ld)
{
#if defined(SLEPC_MISSING_LAPACK_LARTG)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"LARTG - Lapack routine is unavailable");
#else
  PetscBLASInt i,j,j2,one=1;
  PetscReal    c,s,p,off,temp;

  PetscFunctionBegin;
  if (n<=2) PetscFunctionReturn(0);

  for (j=0;j<n-2;j++) {

    /* Eliminate entry e(j) by a rotation in the planes (j,j+1) */
    temp = e[j+1];
    PetscStackCallBLAS("LAPACKlartg",LAPACKlartg_(&temp,&e[j],&c,&s,&e[j+1]));
    s = -s;

    /* Apply rotation to diagonal elements */
    temp   = d[j+1];
    e[j]   = c*s*(temp-d[j]);
    d[j+1] = s*s*d[j] + c*c*temp;
    d[j]   = c*c*d[j] + s*s*temp;

    /* Apply rotation to Q */
    j2 = j+2;
    PetscStackCallBLAS("BLASrot",BLASrot_(&j2,Q+j*ld,&one,Q+(j+1)*ld,&one,&c,&s));

    /* Chase newly introduced off-diagonal entry to the top left corner */
    for (i=j-1;i>=0;i--) {
      off  = -s*e[i];
      e[i] = c*e[i];
      temp = e[i+1];
      PetscStackCallBLAS("LAPACKlartg",LAPACKlartg_(&temp,&off,&c,&s,&e[i+1]));
      s = -s;
      temp = (d[i]-d[i+1])*s - 2.0*c*e[i];
      p = s*temp;
      d[i+1] += p;
      d[i] -= p;
      e[i] = -e[i] - c*temp;
      j2 = j+2;
      PetscStackCallBLAS("BLASrot",BLASrot_(&j2,Q+i*ld,&one,Q+(i+1)*ld,&one,&c,&s));
    }
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSIntermediate_HEP"
/*
   Reduce to tridiagonal form by means of ArrowTridiag.
*/
static PetscErrorCode DSIntermediate_HEP(DS ds)
{
#if defined(SLEPC_MISSING_LAPACK_SYTRD) || defined(SLEPC_MISSING_LAPACK_ORGTR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYTRD/ORGTR - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n1,n2,n3,lwork,info,l,n,ld,off;
  PetscScalar    *A,*Q,*work,*tau;
  PetscReal      *d,*e;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->l,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->k-l+1,&n1);CHKERRQ(ierr); /* size of leading block, excl. locked */
  ierr = PetscBLASIntCast(n-ds->k-1,&n2);CHKERRQ(ierr); /* size of trailing block */
  n3 = n1+n2;
  off = l+l*ld;
  A  = ds->mat[DS_MAT_A];
  Q  = ds->mat[DS_MAT_Q];
  d  = ds->rmat[DS_MAT_T];
  e  = ds->rmat[DS_MAT_T]+ld;
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<n;i++) Q[i+i*ld] = 1.0;

  if (ds->compact) {

    if (ds->state<DS_STATE_INTERMEDIATE) ArrowTridiag(n1,d+l,e+l,Q+off,ld);

  } else {

    for (i=0;i<l;i++) { d[i] = PetscRealPart(A[i+i*ld]); e[i] = 0.0; }

    if (ds->state<DS_STATE_INTERMEDIATE) {
      ierr = DSCopyMatrix_Private(ds,DS_MAT_Q,DS_MAT_A);CHKERRQ(ierr);
      ierr = DSAllocateWork_Private(ds,ld+ld*ld,0,0);CHKERRQ(ierr);
      tau  = ds->work;
      work = ds->work+ld;
      lwork = ld*ld;
      PetscStackCallBLAS("LAPACKsytrd",LAPACKsytrd_("L",&n3,Q+off,&ld,d+l,e+l,tau,work,&lwork,&info));
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);
      PetscStackCallBLAS("LAPACKorgtr",LAPACKorgtr_("L",&n3,Q+off,&ld,tau,work,&lwork,&info));
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);
    } else {
      /* copy tridiagonal to d,e */
      for (i=l;i<n;i++) d[i] = PetscRealPart(A[i+i*ld]);
      for (i=l;i<n-1;i++) e[i] = PetscRealPart(A[(i+1)+i*ld]);
    }
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSSort_HEP"
PetscErrorCode DSSort_HEP(DS ds,PetscScalar *wr,PetscScalar *wi,PetscScalar *rr,PetscScalar *ri,PetscInt *k)
{
  PetscErrorCode ierr;
  PetscInt       n,l,i,*perm,ld=ds->ld;
  PetscScalar    *A;
  PetscReal      *d;

  PetscFunctionBegin;
  if (!ds->sc) PetscFunctionReturn(0);
  n = ds->n;
  l = ds->l;
  A = ds->mat[DS_MAT_A];
  d = ds->rmat[DS_MAT_T];
  perm = ds->perm;
  if (!rr) {
    ierr = DSSortEigenvaluesReal_Private(ds,d,perm);CHKERRQ(ierr);
  } else {
    ierr = DSSortEigenvalues_Private(ds,rr,ri,perm,PETSC_FALSE);CHKERRQ(ierr);
  }
  for (i=l;i<n;i++) wr[i] = d[perm[i]];
  ierr = DSPermuteColumns_Private(ds,l,n,DS_MAT_Q,perm);CHKERRQ(ierr);
  for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);
  if (!ds->compact) {
    for (i=l;i<n;i++) A[i+i*ld] = wr[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSUpdateExtraRow_HEP"
PetscErrorCode DSUpdateExtraRow_HEP(DS ds)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n,ld,incx=1;
  PetscScalar    *A,*Q,*x,*y,one=1.0,zero=0.0;
  PetscReal      *e,beta;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  A  = ds->mat[DS_MAT_A];
  Q  = ds->mat[DS_MAT_Q];
  e  = ds->rmat[DS_MAT_T]+ld;

  if (ds->compact) {
    beta = e[n-1];   /* in compact, we assume all entries are zero except the last one */
    for (i=0;i<n;i++) e[i] = PetscRealPart(beta*Q[n-1+i*ld]);
    ds->k = n;
  } else {
    ierr = DSAllocateWork_Private(ds,2*ld,0,0);CHKERRQ(ierr);
    x = ds->work;
    y = ds->work+ld;
    for (i=0;i<n;i++) x[i] = PetscConj(A[n+i*ld]);
    PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&one,Q,&ld,x,&incx,&zero,y,&incx));
    for (i=0;i<n;i++) A[n+i*ld] = PetscConj(y[i]);
    ds->k = n;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSolve_HEP_QR"
PetscErrorCode DSSolve_HEP_QR(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(PETSC_MISSING_LAPACK_STEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"STEQR - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n1,n2,n3,info,l,n,ld,off;
  PetscScalar    *Q,*A;
  PetscReal      *d,*e;

  PetscFunctionBegin;
  if (ds->bs>1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This method is not prepared for bs>1");
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->l,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->k-l+1,&n1);CHKERRQ(ierr); /* size of leading block, excl. locked */
  ierr = PetscBLASIntCast(n-ds->k-1,&n2);CHKERRQ(ierr); /* size of trailing block */
  n3 = n1+n2;
  off = l+l*ld;
  Q  = ds->mat[DS_MAT_Q];
  A  = ds->mat[DS_MAT_A];
  d  = ds->rmat[DS_MAT_T];
  e  = ds->rmat[DS_MAT_T]+ld;

  /* Reduce to tridiagonal form */
  ierr = DSIntermediate_HEP(ds);CHKERRQ(ierr);

  /* Solve the tridiagonal eigenproblem */
  for (i=0;i<l;i++) wr[i] = d[i];

  ierr = DSAllocateWork_Private(ds,0,2*ld,0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKsteqr",LAPACKsteqr_("V",&n3,d+l,e+l,Q+off,&ld,ds->rwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSTEQR %d",info);
  for (i=l;i<n;i++) wr[i] = d[i];

  /* Create diagonal matrix as a result */
  if (ds->compact) {
    ierr = PetscMemzero(e,(n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  } else {
    for (i=l;i<n;i++) {
      ierr = PetscMemzero(A+l+i*ld,(n-l)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    for (i=l;i<n;i++) A[i+i*ld] = d[i];
  }

  /* Set zero wi */
  if (wi) for (i=l;i<n;i++) wi[i] = 0.0;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSSolve_HEP_MRRR"
PetscErrorCode DSSolve_HEP_MRRR(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_STEVR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"STEVR - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n1,n2,n3,lwork,liwork,info,l,n,m,ld,off,il,iu,*isuppz;
  PetscScalar    *A,*Q,*W=NULL,one=1.0,zero=0.0;
  PetscReal      *d,*e,abstol=0.0,vl,vu;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       j;
  PetscReal      *ritz;
#endif

  PetscFunctionBegin;
  if (ds->bs>1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This method is not prepared for bs>1");
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->l,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->k-l+1,&n1);CHKERRQ(ierr); /* size of leading block, excl. locked */
  ierr = PetscBLASIntCast(n-ds->k-1,&n2);CHKERRQ(ierr); /* size of trailing block */
  n3 = n1+n2;
  off = l+l*ld;
  A  = ds->mat[DS_MAT_A];
  Q  = ds->mat[DS_MAT_Q];
  d  = ds->rmat[DS_MAT_T];
  e  = ds->rmat[DS_MAT_T]+ld;

  /* Reduce to tridiagonal form */
  ierr = DSIntermediate_HEP(ds);CHKERRQ(ierr);

  /* Solve the tridiagonal eigenproblem */
  for (i=0;i<l;i++) wr[i] = d[i];

  if (ds->state<DS_STATE_INTERMEDIATE) {  /* Q contains useful info */
    ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
    ierr = DSCopyMatrix_Private(ds,DS_MAT_W,DS_MAT_Q);CHKERRQ(ierr);
    W = ds->mat[DS_MAT_W];
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = DSAllocateMatReal_Private(ds,DS_MAT_Q);CHKERRQ(ierr);
#endif
  lwork = 20*ld;
  liwork = 10*ld;
  ierr = DSAllocateWork_Private(ds,0,lwork+ld,liwork+2*ld);CHKERRQ(ierr);
  isuppz = ds->iwork+liwork;
#if defined(PETSC_USE_COMPLEX)
  ritz = ds->rwork+lwork;
  PetscStackCallBLAS("LAPACKstevr",LAPACKstevr_("V","A",&n3,d+l,e+l,&vl,&vu,&il,&iu,&abstol,&m,ritz+l,ds->rmat[DS_MAT_Q]+off,&ld,isuppz,ds->rwork,&lwork,ds->iwork,&liwork,&info));
  for (i=l;i<n;i++) wr[i] = ritz[i];
#else
  PetscStackCallBLAS("LAPACKstevr",LAPACKstevr_("V","A",&n3,d+l,e+l,&vl,&vu,&il,&iu,&abstol,&m,wr+l,Q+off,&ld,isuppz,ds->rwork,&lwork,ds->iwork,&liwork,&info));
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack DSTEVR %d",info);
#if defined(PETSC_USE_COMPLEX)
  for (i=l;i<n;i++)
    for (j=l;j<n;j++)
      Q[i+j*ld] = (ds->rmat[DS_MAT_Q])[i+j*ld];
#endif
  if (ds->state<DS_STATE_INTERMEDIATE) {  /* accumulate previous Q */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n3,&n3,&n3,&one,W+off,&ld,Q+off,&ld,&zero,A+off,&ld));
    ierr = DSCopyMatrix_Private(ds,DS_MAT_Q,DS_MAT_A);CHKERRQ(ierr);
  }
  for (i=l;i<n;i++) d[i] = PetscRealPart(wr[i]);

  /* Create diagonal matrix as a result */
  if (ds->compact) {
    ierr = PetscMemzero(e,(n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  } else {
    for (i=l;i<n;i++) {
      ierr = PetscMemzero(A+l+i*ld,(n-l)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    for (i=l;i<n;i++) A[i+i*ld] = d[i];
  }

  /* Set zero wi */
  if (wi) for (i=l;i<n;i++) wi[i] = 0.0;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSSolve_HEP_DC"
PetscErrorCode DSSolve_HEP_DC(DS ds,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_STEDC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"STEDC - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   n1,info,l,ld,off,lrwork,liwork;
  PetscScalar    *Q,*A;
  PetscReal      *d,*e;
#if defined(PETSC_USE_COMPLEX)
  PetscBLASInt   lwork;
  PetscInt       j;
#endif

  PetscFunctionBegin;
  if (ds->bs>1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This method is not prepared for bs>1");
  ierr = PetscBLASIntCast(ds->l,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->n-ds->l,&n1);CHKERRQ(ierr);
  off = l+l*ld;
  Q  = ds->mat[DS_MAT_Q];
  A  = ds->mat[DS_MAT_A];
  d  = ds->rmat[DS_MAT_T];
  e  = ds->rmat[DS_MAT_T]+ld;

  /* Reduce to tridiagonal form */
  ierr = DSIntermediate_HEP(ds);CHKERRQ(ierr);

  /* Solve the tridiagonal eigenproblem */
  for (i=0;i<l;i++) wr[i] = d[i];

  lrwork = 5*n1*n1+3*n1+1;
  liwork = 5*n1*n1+6*n1+6;
#if !defined(PETSC_USE_COMPLEX)
  ierr = DSAllocateWork_Private(ds,0,lrwork,liwork);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKstedc",LAPACKstedc_("V",&n1,d+l,e+l,Q+off,&ld,ds->rwork,&lrwork,ds->iwork,&liwork,&info));
#else
  lwork = ld*ld;
  ierr = DSAllocateWork_Private(ds,lwork,lrwork,liwork);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKstedc",LAPACKstedc_("V",&n1,d+l,e+l,Q+off,&ld,ds->work,&lwork,ds->rwork,&lrwork,ds->iwork,&liwork,&info));
  /* Fixing Lapack bug*/
  for (j=ds->l;j<ds->n;j++)
    for (i=0;i<ds->l;i++) Q[i+j*ld] = 0.0;
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSTEDC %d",info);
  for (i=l;i<ds->n;i++) wr[i] = d[i];

  /* Create diagonal matrix as a result */
  if (ds->compact) {
    ierr = PetscMemzero(e,(ds->n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  } else {
    for (i=l;i<ds->n;i++) {
      ierr = PetscMemzero(A+l+i*ld,(ds->n-l)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    for (i=l;i<ds->n;i++) A[i+i*ld] = d[i];
  }

  /* Set zero wi */
  if (wi) for (i=l;i<ds->n;i++) wi[i] = 0.0;
  PetscFunctionReturn(0);
#endif
}

#if !defined(PETSC_USE_COMPLEX)
#undef __FUNCT__
#define __FUNCT__ "DSSolve_HEP_BDC"
PetscErrorCode DSSolve_HEP_BDC(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  PetscBLASInt   i,j,k,m,n,info,nblks,bs,ld,lde,lrwork,liwork,*ksizes,*iwork,mingapi;
  PetscScalar    *Q,*A;
  PetscReal      *D,*E,*d,*e,tol=PETSC_MACHINE_EPSILON/2,tau1=1e-16,tau2=1e-18,*rwork,mingap;

  PetscFunctionBegin;
  if (ds->l>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This method is not prepared for l>1");
  if (ds->compact) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for compact storage");
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->bs,&bs);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  nblks = n/bs;
  Q  = ds->mat[DS_MAT_Q];
  A  = ds->mat[DS_MAT_A];
  d  = ds->rmat[DS_MAT_T];
  e  = ds->rmat[DS_MAT_T]+ld;
  lrwork = 4*n*n+60*n+1;
  liwork = 5*n+5*nblks-1;
  lde = 2*bs+1;
  ierr = DSAllocateWork_Private(ds,bs*n+lde*lde*(nblks-1),lrwork,nblks+liwork);CHKERRQ(ierr);
  D      = ds->work;
  E      = ds->work+bs*n;
  rwork  = ds->rwork;
  ksizes = ds->iwork;
  iwork  = ds->iwork+nblks;
  ierr = PetscMemzero(iwork,liwork*sizeof(PetscBLASInt));CHKERRQ(ierr);

  /* Copy matrix to block tridiagonal format */
  j=0;
  for (i=0;i<nblks;i++) {
    ksizes[i]=bs;
    for (k=0;k<bs;k++)
      for (m=0;m<bs;m++)
        D[k+m*bs+i*bs*bs] = PetscRealPart(A[j+k+(j+m)*n]);
    j = j + bs;
  }
  j=0;
  for (i=0;i<nblks-1;i++) {
    for (k=0;k<bs;k++)
      for (m=0;m<bs;m++)
        E[k+m*lde+i*lde*lde] = PetscRealPart(A[j+bs+k+(j+m)*n]);
    j = j + bs;
  }

  /* Solve the block tridiagonal eigenproblem */
  BDC_dsbtdc_("D","A",n,nblks,ksizes,D,bs,bs,E,lde,lde,tol,tau1,tau2,d,
           Q,n,rwork,lrwork,iwork,liwork,&mingap,&mingapi,&info,1,1);
  for (i=0;i<ds->n;i++) wr[i] = d[i];

  /* Create diagonal matrix as a result */
  if (ds->compact) {
    ierr = PetscMemzero(e,(ds->n-1)*sizeof(PetscReal));CHKERRQ(ierr);
  } else {
    for (i=0;i<ds->n;i++) {
      ierr = PetscMemzero(A+i*ld,ds->n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    for (i=0;i<ds->n;i++) A[i+i*ld] = wr[i];
  }

  /* Set zero wi */
  if (wi) for (i=0;i<ds->n;i++) wi[i] = 0.0;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "DSTruncate_HEP"
PetscErrorCode DSTruncate_HEP(DS ds,PetscInt n)
{
  PetscInt       i,ld=ds->ld,l=ds->l;
  PetscScalar    *A;

  PetscFunctionBegin;
  if (ds->state==DS_STATE_CONDENSED) ds->t = ds->n;
  A = ds->mat[DS_MAT_A];
  if (!ds->compact && ds->extrarow && ds->k==ds->n) {
    for (i=l;i<n;i++) A[n+i*ld] = A[ds->n+i*ld];
  }
  if (ds->extrarow) ds->k = n;
  else ds->k = 0;
  ds->n = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSCond_HEP"
PetscErrorCode DSCond_HEP(DS ds,PetscReal *cond)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(PETSC_MISSING_LAPACK_GETRI) || defined(SLEPC_MISSING_LAPACK_LANGE)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF/GETRI/LANGE - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscScalar    *work;
  PetscReal      *rwork;
  PetscBLASInt   *ipiv;
  PetscBLASInt   lwork,info,n,ld;
  PetscReal      hn,hin;
  PetscScalar    *A;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  lwork = 8*ld;
  ierr = DSAllocateWork_Private(ds,lwork,ld,ld);CHKERRQ(ierr);
  work  = ds->work;
  rwork = ds->rwork;
  ipiv  = ds->iwork;
  ierr = DSSwitchFormat_HEP(ds,PETSC_FALSE);CHKERRQ(ierr);

  /* use workspace matrix W to avoid overwriting A */
  ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  A = ds->mat[DS_MAT_W];
  ierr = PetscMemcpy(A,ds->mat[DS_MAT_A],sizeof(PetscScalar)*ds->ld*ds->ld);CHKERRQ(ierr);

  /* norm of A */
  hn = LAPACKlange_("I",&n,&n,A,&ld,rwork);

  /* norm of inv(A) */
  PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n,&n,A,&ld,ipiv,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGETRF %d",info);
  PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,A,&ld,ipiv,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGETRI %d",info);
  hin = LAPACKlange_("I",&n,&n,A,&ld,rwork);

  *cond = hn*hin;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSTranslateRKS_HEP"
PetscErrorCode DSTranslateRKS_HEP(DS ds,PetscScalar alpha)
{
#if defined(PETSC_MISSING_LAPACK_GEQRF) || defined(PETSC_MISSING_LAPACK_ORGQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEQRF/ORGQR - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,k=ds->k;
  PetscScalar    *Q,*A,*R,*tau,*work;
  PetscBLASInt   ld,n1,n0,lwork,info;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ds->ld,&ld);CHKERRQ(ierr);
  ierr = DSAllocateWork_Private(ds,ld*ld,0,0);CHKERRQ(ierr);
  tau = ds->work;
  work = ds->work+ld;
  ierr = PetscBLASIntCast(ld*(ld-1),&lwork);CHKERRQ(ierr);
  ierr = DSAllocateMat_Private(ds,DS_MAT_W);CHKERRQ(ierr);
  A  = ds->mat[DS_MAT_A];
  Q  = ds->mat[DS_MAT_Q];
  R  = ds->mat[DS_MAT_W];
  /* Copy I+alpha*A */
  ierr = PetscMemzero(Q,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(R,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    Q[i+i*ld] = 1.0 + alpha*A[i+i*ld];
    Q[k+i*ld] = alpha*A[k+i*ld];
  }
  /* Compute qr */
  ierr = PetscBLASIntCast(k+1,&n1);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&n0);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&n1,&n0,Q,&ld,tau,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  /* Copy R from Q */
  for (j=0;j<k;j++)
    for (i=0;i<=j;i++)
      R[i+j*ld] = Q[i+j*ld];
  /* Compute orthogonal matrix in Q */
  PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&n1,&n1,&n0,Q,&ld,tau,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
  /* Compute the updated matrix of projected problem */
  for (j=0;j<k;j++)
    for (i=0;i<k+1;i++)
      A[j*ld+i] = Q[i*ld+j];
  alpha = -1.0/alpha;
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&n1,&n0,&alpha,R,&ld,A,&ld));
  for (i=0;i<k;i++)
    A[ld*i+i]-=alpha;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "DSCreate_HEP"
PETSC_EXTERN PetscErrorCode DSCreate_HEP(DS ds)
{
  PetscFunctionBegin;
  ds->ops->allocate      = DSAllocate_HEP;
  ds->ops->view          = DSView_HEP;
  ds->ops->vectors       = DSVectors_HEP;
  ds->ops->solve[0]      = DSSolve_HEP_QR;
  ds->ops->solve[1]      = DSSolve_HEP_MRRR;
  ds->ops->solve[2]      = DSSolve_HEP_DC;
#if !defined(PETSC_USE_COMPLEX)
  ds->ops->solve[3]      = DSSolve_HEP_BDC;
#endif
  ds->ops->sort          = DSSort_HEP;
  ds->ops->truncate      = DSTruncate_HEP;
  ds->ops->update        = DSUpdateExtraRow_HEP;
  ds->ops->cond          = DSCond_HEP;
  ds->ops->transrks      = DSTranslateRKS_HEP;
  ds->ops->normalize     = DSNormalize_HEP;
  PetscFunctionReturn(0);
}

