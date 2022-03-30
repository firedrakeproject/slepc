/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Private DS routines
*/

#include <slepc/private/dsimpl.h>      /*I "slepcds.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode DSAllocateMatrix_Private(DS ds,DSMatType m,PetscBool isreal)
{
  size_t         sz;
  PetscInt       n,d,nelem;
  PetscBool      ispep,isnep;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)ds,DSPEP,&ispep));
  PetscCall(PetscObjectTypeCompare((PetscObject)ds,DSNEP,&isnep));
  if (ispep) PetscCall(DSPEPGetDegree(ds,&d));
  if (isnep) PetscCall(DSNEPGetMinimality(ds,&d));
  if ((ispep || isnep) && (m==DS_MAT_A || m==DS_MAT_B || m==DS_MAT_W || m==DS_MAT_U || m==DS_MAT_X || m==DS_MAT_Y)) n = d*ds->ld;
  else n = ds->ld;

  switch (m) {
    case DS_MAT_T:
      nelem = 3*ds->ld;
      break;
    case DS_MAT_D:
      nelem = ds->ld;
      break;
    case DS_MAT_X:
      nelem = ds->ld*n;
      break;
    case DS_MAT_Y:
      nelem = ds->ld*n;
      break;
    default:
      nelem = n*n;
  }
  if (isreal) {
    sz = nelem*sizeof(PetscReal);
    if (ds->rmat[m]) PetscCall(PetscFree(ds->rmat[m]));
    else PetscCall(PetscLogObjectMemory((PetscObject)ds,sz));
    PetscCall(PetscCalloc1(nelem,&ds->rmat[m]));
  } else {
    sz = nelem*sizeof(PetscScalar);
    if (ds->mat[m]) PetscCall(PetscFree(ds->mat[m]));
    else PetscCall(PetscLogObjectMemory((PetscObject)ds,sz));
    PetscCall(PetscCalloc1(nelem,&ds->mat[m]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSAllocateWork_Private(DS ds,PetscInt s,PetscInt r,PetscInt i)
{
  PetscFunctionBegin;
  if (s>ds->lwork) {
    PetscCall(PetscFree(ds->work));
    PetscCall(PetscMalloc1(s,&ds->work));
    PetscCall(PetscLogObjectMemory((PetscObject)ds,(s-ds->lwork)*sizeof(PetscScalar)));
    ds->lwork = s;
  }
  if (r>ds->lrwork) {
    PetscCall(PetscFree(ds->rwork));
    PetscCall(PetscMalloc1(r,&ds->rwork));
    PetscCall(PetscLogObjectMemory((PetscObject)ds,(r-ds->lrwork)*sizeof(PetscReal)));
    ds->lrwork = r;
  }
  if (i>ds->liwork) {
    PetscCall(PetscFree(ds->iwork));
    PetscCall(PetscMalloc1(i,&ds->iwork));
    PetscCall(PetscLogObjectMemory((PetscObject)ds,(i-ds->liwork)*sizeof(PetscBLASInt)));
    ds->liwork = i;
  }
  PetscFunctionReturn(0);
}

/*@C
   DSViewMat - Prints one of the internal DS matrices.

   Collective on ds

   Input Parameters:
+  ds     - the direct solver context
.  viewer - visualization context
-  m      - matrix to display

   Note:
   Works only for ascii viewers. Set the viewer in Matlab format if
   want to paste into Matlab.

   Level: developer

.seealso: DSView()
@*/
PetscErrorCode DSViewMat(DS ds,PetscViewer viewer,DSMatType m)
{
  PetscInt          i,j,rows,cols;
  PetscScalar       *v;
  PetscViewerFormat format;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         allreal = PETSC_TRUE;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ds,m,3);
  DSCheckValidMat(ds,m,3);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ds),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ds,1,viewer,2);
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
  PetscCall(DSMatGetSize(ds,m,&rows,&cols));
#if defined(PETSC_USE_COMPLEX)
  /* determine if matrix has all real values */
  v = ds->mat[m];
  for (i=0;i<rows;i++)
    for (j=0;j<cols;j++)
      if (PetscImaginaryPart(v[i+j*ds->ld])) { allreal = PETSC_FALSE; break; }
#endif
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"%% Size = %" PetscInt_FMT " %" PetscInt_FMT "\n",rows,cols));
    PetscCall(PetscViewerASCIIPrintf(viewer,"%s = [\n",DSMatName[m]));
  } else PetscCall(PetscViewerASCIIPrintf(viewer,"Matrix %s =\n",DSMatName[m]));

  for (i=0;i<rows;i++) {
    v = ds->mat[m]+i;
    for (j=0;j<cols;j++) {
#if defined(PETSC_USE_COMPLEX)
      if (allreal) PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)PetscRealPart(*v)));
      else PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e%+18.16ei ",(double)PetscRealPart(*v),(double)PetscImaginaryPart(*v)));
#else
      PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)*v));
#endif
      v += ds->ld;
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
  }

  if (format == PETSC_VIEWER_ASCII_MATLAB) PetscCall(PetscViewerASCIIPrintf(viewer,"];\n"));
  PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode DSSortEigenvalues_Private(DS ds,PetscScalar *wr,PetscScalar *wi,PetscInt *perm,PetscBool isghiep)
{
  PetscScalar    re,im,wi0;
  PetscInt       n,i,j,result,tmp1,tmp2=0,d=1;

  PetscFunctionBegin;
  n = ds->t;   /* sort only first t pairs if truncated */
  /* insertion sort */
  i=ds->l+1;
#if !defined(PETSC_USE_COMPLEX)
  if (wi && wi[perm[i-1]]!=0.0) i++; /* initial value is complex */
#else
  if (isghiep && PetscImaginaryPart(wr[perm[i-1]])!=0.0) i++;
#endif
  for (;i<n;i+=d) {
    re = wr[perm[i]];
    if (wi) im = wi[perm[i]];
    else im = 0.0;
    tmp1 = perm[i];
#if !defined(PETSC_USE_COMPLEX)
    if (im!=0.0) { d = 2; tmp2 = perm[i+1]; }
    else d = 1;
#else
    if (isghiep && PetscImaginaryPart(re)!=0.0) { d = 2; tmp2 = perm[i+1]; }
    else d = 1;
#endif
    j = i-1;
    if (wi) wi0 = wi[perm[j]];
    else wi0 = 0.0;
    PetscCall(SlepcSCCompare(ds->sc,re,im,wr[perm[j]],wi0,&result));
    while (result<0 && j>=ds->l) {
      perm[j+d] = perm[j];
      j--;
#if !defined(PETSC_USE_COMPLEX)
      if (wi && wi[perm[j+1]]!=0)
#else
      if (isghiep && PetscImaginaryPart(wr[perm[j+1]])!=0)
#endif
        { perm[j+d] = perm[j]; j--; }

      if (j>=ds->l) {
        if (wi) wi0 = wi[perm[j]];
        else wi0 = 0.0;
        PetscCall(SlepcSCCompare(ds->sc,re,im,wr[perm[j]],wi0,&result));
      }
    }
    perm[j+1] = tmp1;
    if (d==2) perm[j+2] = tmp2;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DSSortEigenvaluesReal_Private(DS ds,PetscReal *eig,PetscInt *perm)
{
  PetscScalar    re;
  PetscInt       i,j,result,tmp,l,n;

  PetscFunctionBegin;
  n = ds->t;   /* sort only first t pairs if truncated */
  l = ds->l;
  /* insertion sort */
  for (i=l+1;i<n;i++) {
    re = eig[perm[i]];
    j = i-1;
    PetscCall(SlepcSCCompare(ds->sc,re,0.0,eig[perm[j]],0.0,&result));
    while (result<0 && j>=l) {
      tmp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp; j--;
      if (j>=l) PetscCall(SlepcSCCompare(ds->sc,re,0.0,eig[perm[j]],0.0,&result));
    }
  }
  PetscFunctionReturn(0);
}

/*
  DSCopyMatrix_Private - Copies the trailing block of a matrix (from
  rows/columns l to n).
*/
PetscErrorCode DSCopyMatrix_Private(DS ds,DSMatType dst,DSMatType src)
{
  PetscInt    j,m,off,ld;
  PetscScalar *S,*D;

  PetscFunctionBegin;
  ld  = ds->ld;
  m   = ds->n-ds->l;
  off = ds->l+ds->l*ld;
  S   = ds->mat[src];
  D   = ds->mat[dst];
  for (j=0;j<m;j++) PetscCall(PetscArraycpy(D+off+j*ld,S+off+j*ld,m));
  PetscFunctionReturn(0);
}

/*
  Permute comumns [istart..iend-1] of [mat] according to perm. Columns have length n
 */
PetscErrorCode DSPermuteColumns_Private(DS ds,PetscInt istart,PetscInt iend,PetscInt n,DSMatType mat,PetscInt *perm)
{
  PetscInt    i,j,k,p,ld;
  PetscScalar *Q,rtmp;

  PetscFunctionBegin;
  ld = ds->ld;
  Q  = ds->mat[mat];
  for (i=istart;i<iend;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap columns i and j */
      for (k=0;k<n;k++) {
        rtmp = Q[k+p*ld]; Q[k+p*ld] = Q[k+i*ld]; Q[k+i*ld] = rtmp;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  The same as DSPermuteColumns_Private but for two matrices [mat1] and [mat2]
 */
PetscErrorCode DSPermuteColumnsTwo_Private(DS ds,PetscInt istart,PetscInt iend,PetscInt n,DSMatType mat1,DSMatType mat2,PetscInt *perm)
{
  PetscInt    i,j,k,p,ld;
  PetscScalar *Q,*Z,rtmp,rtmp2;

  PetscFunctionBegin;
  ld = ds->ld;
  Q  = ds->mat[mat1];
  Z  = ds->mat[mat2];
  for (i=istart;i<iend;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap columns i and j */
      for (k=0;k<n;k++) {
        rtmp  = Q[k+p*ld]; Q[k+p*ld] = Q[k+i*ld]; Q[k+i*ld] = rtmp;
        rtmp2 = Z[k+p*ld]; Z[k+p*ld] = Z[k+i*ld]; Z[k+i*ld] = rtmp2;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  Permute rows [istart..iend-1] of [mat] according to perm. Rows have length m
 */
PetscErrorCode DSPermuteRows_Private(DS ds,PetscInt istart,PetscInt iend,PetscInt m,DSMatType mat,PetscInt *perm)
{
  PetscInt    i,j,k,p,ld;
  PetscScalar *Q,rtmp;

  PetscFunctionBegin;
  ld = ds->ld;
  Q  = ds->mat[mat];
  for (i=istart;i<iend;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap rows i and j */
      for (k=0;k<m;k++) {
        rtmp = Q[p+k*ld]; Q[p+k*ld] = Q[i+k*ld]; Q[i+k*ld] = rtmp;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  Permute columns [istart..iend-1] of [mat1] and [mat2] according to perm.
  Columns of [mat1] have length n, columns of [mat2] have length m
 */
PetscErrorCode DSPermuteBoth_Private(DS ds,PetscInt istart,PetscInt iend,PetscInt n,PetscInt m,DSMatType mat1,DSMatType mat2,PetscInt *perm)
{
  PetscInt    i,j,k,p,ld;
  PetscScalar *U,*V,rtmp;

  PetscFunctionBegin;
  ld = ds->ld;
  U  = ds->mat[mat1];
  V  = ds->mat[mat2];
  for (i=istart;i<iend;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap columns i and j of U */
      for (k=0;k<n;k++) {
        rtmp = U[k+p*ld]; U[k+p*ld] = U[k+i*ld]; U[k+i*ld] = rtmp;
      }
      /* swap columns i and j of V */
      for (k=0;k<m;k++) {
        rtmp = V[k+p*ld]; V[k+p*ld] = V[k+i*ld]; V[k+i*ld] = rtmp;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
   DSSetIdentity - Copy the identity (a diagonal matrix with ones) on the
   active part of a matrix.

   Logically Collective on ds

   Input Parameters:
+  ds  - the direct solver context
-  mat - the matrix to modify

   Level: intermediate

.seealso: DSGetMat()
@*/
PetscErrorCode DSSetIdentity(DS ds,DSMatType mat)
{
  PetscScalar    *x;
  PetscInt       i,ld,n,l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ds,mat,2);
  DSCheckValidMat(ds,mat,2);

  PetscCall(DSGetDimensions(ds,&n,&l,NULL,NULL));
  PetscCall(DSGetLeadingDimension(ds,&ld));
  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(DSGetArray(ds,mat,&x));
  PetscCall(PetscArrayzero(&x[ld*l],ld*(n-l)));
  for (i=l;i<n;i++) x[i+i*ld] = 1.0;
  PetscCall(DSRestoreArray(ds,mat,&x));
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
   DSOrthogonalize - Orthogonalize the columns of a matrix.

   Logically Collective on ds

   Input Parameters:
+  ds   - the direct solver context
.  mat  - a matrix
-  cols - number of columns to orthogonalize (starting from column zero)

   Output Parameter:
.  lindcols - (optional) number of linearly independent columns of the matrix

   Level: developer

.seealso: DSPseudoOrthogonalize()
@*/
PetscErrorCode DSOrthogonalize(DS ds,DSMatType mat,PetscInt cols,PetscInt *lindcols)
{
  PetscInt       n,l,ld;
  PetscBLASInt   ld_,rA,cA,info,ltau,lw;
  PetscScalar    *A,*tau,*w,saux,dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  PetscValidLogicalCollectiveEnum(ds,mat,2);
  DSCheckValidMat(ds,mat,2);
  PetscValidLogicalCollectiveInt(ds,cols,3);

  PetscCall(DSGetDimensions(ds,&n,&l,NULL,NULL));
  PetscCall(DSGetLeadingDimension(ds,&ld));
  n = n - l;
  PetscCheck(cols<=n,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Invalid number of columns");
  if (n == 0 || cols == 0) PetscFunctionReturn(0);

  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall(DSGetArray(ds,mat,&A));
  PetscCall(PetscBLASIntCast(PetscMin(cols,n),&ltau));
  PetscCall(PetscBLASIntCast(ld,&ld_));
  PetscCall(PetscBLASIntCast(n,&rA));
  PetscCall(PetscBLASIntCast(cols,&cA));
  lw = -1;
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&rA,&cA,A,&ld_,&dummy,&saux,&lw,&info));
  SlepcCheckLapackInfo("geqrf",info);
  lw = (PetscBLASInt)PetscRealPart(saux);
  PetscCall(DSAllocateWork_Private(ds,lw+ltau,0,0));
  tau = ds->work;
  w = &tau[ltau];
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&rA,&cA,&A[ld*l+l],&ld_,tau,w,&lw,&info));
  SlepcCheckLapackInfo("geqrf",info);
  PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&rA,&ltau,&ltau,&A[ld*l+l],&ld_,tau,w,&lw,&info));
  SlepcCheckLapackInfo("orgqr",info);
  if (lindcols) *lindcols = ltau;

  PetscCall(PetscFPTrapPop());
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  PetscCall(DSRestoreArray(ds,mat,&A));
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  PetscFunctionReturn(0);
}

/*
  Compute C <- a*A*B + b*C, where
    ldC, the leading dimension of C,
    ldA, the leading dimension of A,
    rA, cA, rows and columns of A,
    At, if true use the transpose of A instead,
    ldB, the leading dimension of B,
    rB, cB, rows and columns of B,
    Bt, if true use the transpose of B instead
*/
static PetscErrorCode SlepcMatDenseMult(PetscScalar *C,PetscInt _ldC,PetscScalar b,PetscScalar a,const PetscScalar *A,PetscInt _ldA,PetscInt rA,PetscInt cA,PetscBool At,const PetscScalar *B,PetscInt _ldB,PetscInt rB,PetscInt cB,PetscBool Bt)
{
  PetscInt       tmp;
  PetscBLASInt   m, n, k, ldA = _ldA, ldB = _ldB, ldC = _ldC;
  const char     *N = "N", *T = "C", *qA = N, *qB = N;

  PetscFunctionBegin;
  if ((rA == 0) || (cB == 0)) PetscFunctionReturn(0);
  PetscValidScalarPointer(C,1);
  PetscValidScalarPointer(A,5);
  PetscValidScalarPointer(B,10);

  /* Transpose if needed */
  if (At) tmp = rA, rA = cA, cA = tmp, qA = T;
  if (Bt) tmp = rB, rB = cB, cB = tmp, qB = T;

  /* Check size */
  PetscCheck(cA==rB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix dimensions do not match");

  /* Do stub */
  if ((rA == 1) && (cA == 1) && (cB == 1)) {
    if (!At && !Bt) *C = *A * *B;
    else if (At && !Bt) *C = PetscConj(*A) * *B;
    else if (!At && Bt) *C = *A * PetscConj(*B);
    else *C = PetscConj(*A) * PetscConj(*B);
    m = n = k = 1;
  } else {
    m = rA; n = cB; k = cA;
    PetscStackCallBLAS("BLASgemm",BLASgemm_(qA,qB,&m,&n,&k,&a,(PetscScalar*)A,&ldA,(PetscScalar*)B,&ldB,&b,C,&ldC));
  }

  PetscCall(PetscLogFlops(2.0*m*n*k));
  PetscFunctionReturn(0);
}

/*@C
   DSPseudoOrthogonalize - Orthogonalize the columns of a matrix with Modified
   Gram-Schmidt in an indefinite inner product space defined by a signature.

   Logically Collective on ds

   Input Parameters:
+  ds   - the direct solver context
.  mat  - the matrix
.  cols - number of columns to orthogonalize (starting from column zero)
-  s    - the signature that defines the inner product

   Output Parameters:
+  lindcols - (optional) linearly independent columns of the matrix
-  ns   - (optional) the new signature of the vectors

   Note:
   After the call the matrix satisfies A'*s*A = ns.

   Level: developer

.seealso: DSOrthogonalize()
@*/
PetscErrorCode DSPseudoOrthogonalize(DS ds,DSMatType mat,PetscInt cols,PetscReal *s,PetscInt *lindcols,PetscReal *ns)
{
  PetscInt       i,j,k,l,n,ld;
  PetscBLASInt   info,one=1,zero=0,rA_,ld_;
  PetscScalar    *A,*A_,*m,*h,nr0;
  PetscReal      nr_o,nr,nr_abs,*ns_,done=1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  DSCheckAlloc(ds,1);
  PetscValidLogicalCollectiveEnum(ds,mat,2);
  DSCheckValidMat(ds,mat,2);
  PetscValidLogicalCollectiveInt(ds,cols,3);
  PetscValidRealPointer(s,4);
  PetscCall(DSGetDimensions(ds,&n,&l,NULL,NULL));
  PetscCall(DSGetLeadingDimension(ds,&ld));
  n = n - l;
  PetscCheck(cols<=n,PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Invalid number of columns");
  if (n == 0 || cols == 0) PetscFunctionReturn(0);
  PetscCall(PetscBLASIntCast(n,&rA_));
  PetscCall(PetscBLASIntCast(ld,&ld_));
  PetscCall(DSGetArray(ds,mat,&A_));
  A = &A_[ld*l+l];
  PetscCall(DSAllocateWork_Private(ds,n+cols,ns?0:cols,0));
  m = ds->work;
  h = &m[n];
  ns_ = ns ? ns : ds->rwork;
  PetscCall(PetscLogEventBegin(DS_Other,ds,0,0,0));
  for (i=0; i<cols; i++) {
    /* m <- diag(s)*A[i] */
    for (k=0; k<n; k++) m[k] = s[k]*A[k+i*ld];
    /* nr_o <- mynorm(A[i]'*m), mynorm(x) = sign(x)*sqrt(|x|) */
    PetscCall(SlepcMatDenseMult(&nr0,1,0.0,1.0,&A[ld*i],ld,n,1,PETSC_TRUE,m,n,n,1,PETSC_FALSE));
    nr = nr_o = PetscSign(PetscRealPart(nr0))*PetscSqrtReal(PetscAbsScalar(nr0));
    for (j=0; j<3 && i>0; j++) {
      /* h <- A[0:i-1]'*m */
      PetscCall(SlepcMatDenseMult(h,i,0.0,1.0,A,ld,n,i,PETSC_TRUE,m,n,n,1,PETSC_FALSE));
      /* h <- diag(ns)*h */
      for (k=0; k<i; k++) h[k] *= ns_[k];
      /* A[i] <- A[i] - A[0:i-1]*h */
      PetscCall(SlepcMatDenseMult(&A[ld*i],ld,1.0,-1.0,A,ld,n,i,PETSC_FALSE,h,i,i,1,PETSC_FALSE));
      /* m <- diag(s)*A[i] */
      for (k=0; k<n; k++) m[k] = s[k]*A[k+i*ld];
      /* nr_o <- mynorm(A[i]'*m) */
      PetscCall(SlepcMatDenseMult(&nr0,1,0.0,1.0,&A[ld*i],ld,n,1,PETSC_TRUE,m,n,n,1,PETSC_FALSE));
      nr = PetscSign(PetscRealPart(nr0))*PetscSqrtReal(PetscAbsScalar(nr0));
      PetscCheck(PetscAbs(nr)>PETSC_MACHINE_EPSILON,PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"Linear dependency detected");
      if (PetscAbs(nr) > 0.7*PetscAbs(nr_o)) break;
      nr_o = nr;
    }
    ns_[i] = PetscSign(nr);
    /* A[i] <- A[i]/|nr| */
    nr_abs = PetscAbs(nr);
    PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&nr_abs,&done,&rA_,&one,A+i*ld,&ld_,&info));
    SlepcCheckLapackInfo("lascl",info);
  }
  PetscCall(PetscLogEventEnd(DS_Other,ds,0,0,0));
  PetscCall(DSRestoreArray(ds,mat,&A_));
  PetscCall(PetscObjectStateIncrease((PetscObject)ds));
  if (lindcols) *lindcols = cols;
  PetscFunctionReturn(0);
}
