
#include "slepc.h" /*I "slepc.h" I*/
#include "slepceps.h" /*I "slepc.h" I*/
#include "slepcblaslapack.h"
#include "davidson.h"

PetscLogEvent SLEPC_SlepcDenseMatProd = 0;
PetscLogEvent SLEPC_SlepcDenseMatInvProd = 0;
PetscLogEvent SLEPC_SlepcDenseNorm = 0;
PetscLogEvent SLEPC_SlepcDenseOrth = 0;
PetscLogEvent SLEPC_SlepcDenseCopy = 0;
PetscLogEvent SLEPC_VecsMult = 0;

void dvd_sum_local(void *in, void *out, PetscMPIInt *cnt,MPI_Datatype *t);
PetscErrorCode VecsMultS_copy_func(PetscScalar *out, PetscInt size_out,
                                   void *ptr);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_blas_prof_init"
PetscErrorCode dvd_blas_prof_init() {
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  if (SLEPC_SlepcDenseMatProd) PetscFunctionReturn(0);
  ierr = PetscLogEventRegister("DenseMatProd", EPS_COOKIE,
                               &SLEPC_SlepcDenseMatProd); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseOrth", EPS_COOKIE,
                               &SLEPC_SlepcDenseOrth); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseMatInvProd", EPS_COOKIE,
                               &SLEPC_SlepcDenseMatInvProd); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseMatNorm", EPS_COOKIE,
                               &SLEPC_SlepcDenseNorm); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseCopy", EPS_COOKIE,
                               &SLEPC_SlepcDenseCopy); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecsMult", EPS_COOKIE,
                               &SLEPC_VecsMult); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcDenseMatProd"
PetscErrorCode SlepcDenseMatProd(PetscScalar *C, PetscInt _ldC, PetscScalar b,
  PetscScalar a,
  const PetscScalar *A, PetscInt _ldA, PetscInt rA, PetscInt cA, PetscTruth At,
  const PetscScalar *B, PetscInt _ldB, PetscInt rB, PetscInt cB, PetscTruth Bt)
{
  PetscErrorCode  ierr;
  PetscInt        tmp;
  PetscBLASInt    m, n, k, ldA = _ldA, ldB = _ldB, ldC = _ldC;
  const char      *N = "N", *T = "C", *qA = N, *qB = N;

  PetscFunctionBegin;

  if ((rA == 0) || (cB == 0)) { PetscFunctionReturn(0); }

  ierr = PetscLogEventBegin(SLEPC_SlepcDenseMatProd,0,0,0,0);CHKERRQ(ierr);

  /* Transpose if needed */
  if (At == PETSC_TRUE) tmp = rA, rA = cA, cA = tmp, qA = T;
  if (Bt == PETSC_TRUE) tmp = rB, rB = cB, cB = tmp, qB = T;
  
  /* Check size */
  if (cA != rB) {
    SETERRQ(1, "Matrix dimensions doesn't match!");
  }
  
  /* Do stub */
  if ((rA == 1) && (cA == 1) && (cB == 1)) {
    *C = *A * *B;
    m = n = k = 1;
  } else {
    m = rA; n = cB; k = cA;
    BLASgemm_(qA, qB, &m, &n, &k, &a, (PetscScalar*)A, &ldA, (PetscScalar*)B,
              &ldB, &b, C, &ldC);
  }

  ierr = PetscLogFlops(m*n*2*k);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SLEPC_SlepcDenseMatProd,0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
  Compute C <- A*B, where
    sC, structure of C,
    ldC, the leading dimension of C,
    sA, structure of A,
    ldA, the leading dimension of A,
    rA, cA, rows and columns of A,
    At, if true use the transpose of A instead,
    sB, structure of B,
    ldB, the leading dimension of B,
    rB, cB, rows and columns of B,
    Bt, if true use the transpose of B instead
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcDenseMatProdTriang"
PetscErrorCode SlepcDenseMatProdTriang(
  PetscScalar *C, MatType_t sC, PetscInt ldC,
  const PetscScalar *A, MatType_t sA, PetscInt ldA, PetscInt rA, PetscInt cA,
  PetscTruth At,
  const PetscScalar *B, MatType_t sB, PetscInt ldB, PetscInt rB, PetscInt cB,
  PetscTruth Bt)
{
  PetscErrorCode  ierr;
  PetscInt        tmp;
  PetscScalar     one=1.0, zero=0.0;
  PetscBLASInt    rC, cC, _ldA = ldA, _ldB = ldB, _ldC = ldC;

  PetscFunctionBegin;

  if ((rA == 0) || (cB == 0)) { PetscFunctionReturn(0); }

  /* Transpose if needed */
  if (At == PETSC_TRUE) tmp = rA, rA = cA, cA = tmp;
  if (Bt == PETSC_TRUE) tmp = rB, rB = cB, cB = tmp;
  
  /* Check size */
  if (cA != rB) {
    SETERRQ(1, "Matrix dimensions doesn't match!");
    PetscFunctionReturn(1);
  }

  if (sB != 0) {
    SETERRQ(1, "It doesn't support B matrix type!");
    PetscFunctionReturn(1);
  }

  /* Optimized version: trivial case */
  if ((rA == 1) && (cA == 1) && (cB == 1)) {
    if ((At == PETSC_FALSE) && (Bt == PETSC_FALSE))     *C = *A * *B;
    else if ((At == PETSC_TRUE) && (Bt == PETSC_FALSE)) *C = PetscConj(*A) * *B;
    else if ((At == PETSC_FALSE) && (Bt == PETSC_TRUE)) *C = *A * PetscConj(*B);
    else if ((At == PETSC_TRUE) && (Bt == PETSC_TRUE))  *C = PetscConj(*A) * PetscConj(*B);
    PetscFunctionReturn(0);
  }
 
  /* Optimized versions: sA == 0 && sB == 0 */
  if ((sA == 0) && (sB == 0)) {
    if (At == PETSC_TRUE) tmp = rA, rA = cA, cA = tmp;
    if (Bt == PETSC_TRUE) tmp = rB, rB = cB, cB = tmp;
    ierr = SlepcDenseMatProd(C, ldC, 0.0, 1.0, A, ldA, rA, cA, At, B, ldB, rB,
                             cB, Bt); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }

  /* Optimized versions: A hermitian && (B not triang) */
  if (DVD_IS(sA,DVD_MAT_HERMITIAN) &&
      DVD_ISNOT(sB,DVD_MAT_UTRIANG) &&
      DVD_ISNOT(sB,DVD_MAT_LTRIANG)    ) {
    ierr = PetscLogEventBegin(SLEPC_SlepcDenseMatProd,0,0,0,0);CHKERRQ(ierr);
    rC = rA; cC = cB;
    BLAShemm_("L", DVD_ISNOT(sA,DVD_MAT_LTRIANG)?"U":"L", &rC, &cC, &one,
              (PetscScalar*)A, &_ldA, (PetscScalar*)B, &_ldB, &zero, C, &_ldC);
    ierr = PetscLogFlops(rA*cB*cA); CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SLEPC_SlepcDenseMatProd,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* Optimized versions: B hermitian && (A not triang) */
  if (DVD_IS(sB,DVD_MAT_HERMITIAN) &&
      DVD_ISNOT(sA,DVD_MAT_UTRIANG) &&
      DVD_ISNOT(sA,DVD_MAT_LTRIANG)    ) {
    ierr = PetscLogEventBegin(SLEPC_SlepcDenseMatProd,0,0,0,0);CHKERRQ(ierr);
    rC = rA; cC = cB;
    BLAShemm_("R", DVD_ISNOT(sB,DVD_MAT_LTRIANG)?"U":"L", &rC, &cC, &one,
              (PetscScalar*)B, &_ldB, (PetscScalar*)A, &_ldA, &zero, C, &_ldC);
    ierr = PetscLogFlops(rA*cB*cA); CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SLEPC_SlepcDenseMatProd,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
 
  SETERRQ(1, "It doesn't support A matrix type!");
  PetscFunctionReturn(1);
}
EXTERN_C_END

/*
  Normalize the columns of the matrix A, where
    ldA, the leading dimension of A,
    rA, cA, rows and columns of A.
  if eigi is given, the pairs of contiguous columns i i+1 such as eigi[i] != 0
  are normalized as being one column.
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcDenseNorm"
PetscErrorCode SlepcDenseNorm(PetscScalar *A, PetscInt ldA, PetscInt _rA,
                              PetscInt cA, PetscScalar *eigi)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  PetscScalar     norm, norm0;
  PetscBLASInt    rA = _rA, one=1;

  PetscFunctionBegin;

  ierr = PetscLogEventBegin(SLEPC_SlepcDenseNorm,0,0,0,0);CHKERRQ(ierr);

  for(i=0; i<cA; i++) {
    if(eigi && eigi[i] != 0.0) {
      norm = BLASnrm2_(&rA, &A[i*ldA], &one);
      norm0 = BLASnrm2_(&rA, &A[(i+1)*ldA], &one);
      norm = 1.0/PetscSqrtScalar(norm*norm + norm0*norm0);
      BLASscal_(&rA, &norm, &A[i*ldA], &one);
      BLASscal_(&rA, &norm, &A[(i+1)*ldA], &one);
      i++;
    } else {
      norm = BLASnrm2_(&rA, &A[i*ldA], &one);
      norm = 1.0 / norm;
      BLASscal_(&rA, &norm, &A[i*ldA], &one);
     }
  }

  ierr = PetscLogEventEnd(SLEPC_SlepcDenseNorm,0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
  

/*
  Compute B <- A\B, where
    ldA, the leading dimension of A,
    ldB, the leading dimension of B,
    dimA, rows and columns of A,
    rB, cB, rows and columns of B,
    auxI, auxiliary vector of size dimA,
    auxS, auxiliary vector of size cB
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcDenseMatInvProd"
PetscErrorCode SlepcDenseMatInvProd(
  PetscScalar *A, PetscInt _ldA, PetscInt _dimA,
  PetscScalar *B, PetscInt _ldB, PetscInt rB, PetscInt _cB,
  PetscInt *auxI)
{
  PetscErrorCode  ierr;
  PetscBLASInt    *p = (PetscBLASInt*)auxI, dimA = _dimA, cB = _cB,
                  ldA = _ldA, ldB = _ldB, info;

  PetscFunctionBegin;

  /* Check size */
  if (_dimA != rB) {
    SETERRQ(1, "Matrix dimensions doesn't match!");
  }

  /* Quick exit */
  if ((_dimA == 0) || (cB == 0)) { PetscFunctionReturn(0); }

  ierr = PetscLogEventBegin(SLEPC_SlepcDenseMatInvProd,0,0,0,0);CHKERRQ(ierr);
  if (dimA == 1) {
    *B = *B / *A;
    ierr = PetscLogFlops(1);CHKERRQ(ierr);
  } else {
   LAPACKgesv_(&dimA, &cB, A, &ldA, p, B, &ldB, &info);
   if (info) SETERRQ1(PETSC_ERR_LIB, "Error in Lapack GESV %d", info);
  }
  ierr = PetscLogEventEnd(SLEPC_SlepcDenseMatInvProd,0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
  Compute A <- orth(A), where
    ldA, the leading dimension of A,
    rA, cA, rows and columns of A,
    auxS, auxiliary vector of more size than cA+min(rA,cA),
    lauxS, size of auxS,
    ncA, new number of columns of A
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcDenseOrth"
PetscErrorCode SlepcDenseOrth(PetscScalar *A, PetscInt _ldA, PetscInt _rA,
                              PetscInt _cA, PetscScalar *auxS, PetscInt _lauxS,
                              PetscInt *ncA)
{
  PetscErrorCode  ierr;
  PetscBLASInt    ldA = _ldA, rA = _rA, cA = _cA,
                  info, ltau = PetscMin(_cA, _rA), lw = _lauxS - ltau;
  PetscScalar     *tau = auxS, *w = tau + ltau;

  PetscFunctionBegin;

  /* Quick exit */
  if ((_rA == 0) || (cA == 0)) { PetscFunctionReturn(0); }

  /* Memory check */
  if (lw < cA) {
    SETERRQ(1, "Insufficient memory for xGEQRF!");
    PetscFunctionReturn(1);
  }
  
  ierr = PetscLogEventBegin(SLEPC_SlepcDenseOrth,0,0,0,0);CHKERRQ(ierr);
  LAPACKgeqrf_(&rA, &cA, A, &ldA, tau, w, &lw, &info);
  if (info) SETERRQ1(PETSC_ERR_LIB, "Error in Lapack xGEQRF %d", info);
  LAPACKorgqr_(&rA, &ltau, &ltau, A, &ldA, tau, w, &lw, &info);
  if (info) SETERRQ1(PETSC_ERR_LIB, "Error in Lapack xORGQR %d", info);
  ierr = PetscLogEventEnd(SLEPC_SlepcDenseOrth,0,0,0,0);CHKERRQ(ierr);

  if (ncA) *ncA = ltau;

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
  Y <- X, where
  ldX, leading dimension of X,
  rX, cX, rows and columns of X
  ldY, leading dimension of Y
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcDenseCopy"
PetscErrorCode SlepcDenseCopy(PetscScalar *Y, PetscInt ldY, PetscScalar *X,
                              PetscInt ldX, PetscInt rX, PetscInt cX)
{
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;

  if ((ldX < rX) || (ldY < rX)) {
    SETERRQ(1, "Leading dimension error!");
  }
  
  /* Quick exit */
  if (Y == X) {
    if (ldX != ldY) {
      SETERRQ(1, "Leading dimension error!");
    }
    PetscFunctionReturn(0);
  }

  ierr = PetscLogEventBegin(SLEPC_SlepcDenseCopy,0,0,0,0);CHKERRQ(ierr);
  for(i=0; i<cX; i++) {
    ierr = PetscMemcpy(&Y[ldY*i], &X[ldX*i], sizeof(PetscScalar)*rX);
    CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(SLEPC_SlepcDenseCopy,0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
  Y <- X, where
  ldX, leading dimension of X,
  rX, cX, rows and columns of X
  ldY, leading dimension of Y
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcDenseCopyTriang"
PetscErrorCode SlepcDenseCopyTriang(PetscScalar *Y, MatType_t sY, PetscInt ldY,
                                    PetscScalar *X, MatType_t sX, PetscInt ldX,
                                    PetscInt rX, PetscInt cX)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,c;

  PetscFunctionBegin;

  if ((ldX < rX) || (ldY < rX)) {
    SETERRQ(1, "Leading dimension error!");
  }

  if (rX != cX) {
    SETERRQ(1, "SlepcDenseCopyTriang doesn't support rectangular matrices!");
  }

  if (DVD_IS(sX,DVD_MAT_UTRIANG) &&
      DVD_ISNOT(sX,DVD_MAT_LTRIANG)) {        /* UpTr to ... */
    if (DVD_IS(sY,DVD_MAT_UTRIANG) &&
        DVD_ISNOT(sY,DVD_MAT_LTRIANG))        /* ... UpTr, */
      c = 0;                                      /*     so copy */
    else if(DVD_ISNOT(sY,DVD_MAT_UTRIANG) &&
            DVD_IS(sY,DVD_MAT_LTRIANG))       /* ... LoTr, */
      c = 1;                                      /*     so transpose */
    else                                          /* ... Full, */
      c = 2;                                      /*     so reflect from up */
  } else if (DVD_ISNOT(sX,DVD_MAT_UTRIANG) &&
             DVD_IS(sX,DVD_MAT_LTRIANG)) {    /* LoTr to ... */
    if (DVD_IS(sY,DVD_MAT_UTRIANG) &&
        DVD_ISNOT(sY,DVD_MAT_LTRIANG))        /* ... UpTr, */
      c = 1;                                      /*     so transpose */
    else if(DVD_ISNOT(sY,DVD_MAT_UTRIANG) &&
            DVD_IS(sY,DVD_MAT_LTRIANG))       /* ... LoTr, */
      c = 0;                                      /*     so copy */
    else                                          /* ... Full, */
      c = 3;                                      /*     so reflect fr. down */
  } else                                          /* Full to any, */
    c = 0;                                        /*     so copy */
 
  ierr = PetscLogEventBegin(SLEPC_SlepcDenseCopy,0,0,0,0);CHKERRQ(ierr);

  switch(c) {
  case 0: /* copy */
    for(i=0; i<cX; i++) {
      ierr = PetscMemcpy(&Y[ldY*i], &X[ldX*i], sizeof(PetscScalar)*rX);
      CHKERRQ(ierr);
    }
    break;

  case 1: /* transpose */
    for(i=0; i<cX; i++)
      for(j=0; j<rX; j++)
        Y[ldY*j+i] = X[ldX*i+j];
    break;

  case 2: /* reflection from up */
    for(i=0; i<cX; i++)
      for(j=0; j<PetscMin(i+1,rX); j++)
        Y[ldY*j+i] = PetscConj(Y[ldY*i+j] = X[ldX*i+j]);
    break;

  case 3: /* reflection from down */
    for(i=0; i<cX; i++)
      for(j=i; j<rX; j++)
        Y[ldY*j+i] = PetscConj(Y[ldY*i+j] = X[ldX*i+j]);
    break;
  }
  ierr = PetscLogEventEnd(SLEPC_SlepcDenseCopy,0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


/*
  Compute Y[0..cM-1] <- alpha * X[0..cX-1] * M + beta * Y[0..cM-1],
  where X and Y are contiguous global vectors.
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcUpdateVectorsZ"
PetscErrorCode SlepcUpdateVectorsZ(Vec *Y, PetscScalar beta, PetscScalar alpha,
  Vec *X, PetscInt cX, const PetscScalar *M, PetscInt ldM, PetscInt rM,
  PetscInt cM)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = SlepcUpdateVectorsS(Y, 1, beta, alpha, X, cX, 1, M, ldM, rM, cM);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


/*
  Compute Y[0:dY:cM*dY-1] <- alpha * X[0:dX:cX-1] * M + beta * Y[0:dY:cM*dY-1],
  where X and Y are contiguous global vectors.
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcUpdateVectorsS"
PetscErrorCode SlepcUpdateVectorsS(Vec *Y, PetscInt dY, PetscScalar beta,
  PetscScalar alpha, Vec *X, PetscInt cX, PetscInt dX, const PetscScalar *M,
  PetscInt ldM, PetscInt rM, PetscInt cM)
{
  PetscErrorCode  ierr;
  PetscScalar     *px, *py;
  PetscInt        rX, rY, ldX, ldY, i, rcX;

  PetscFunctionBegin;

  /* Compute the real number of columns */
  rcX = cX/dX;
  if (rcX != rM) {
    SETERRQ(1, "Matrix dimensions doesn't match!");
  }

  if ((rcX == 0) || (rM == 0) || (cM == 0)) {
    PetscFunctionReturn(0);
  } else if ((Y + cM <= X) || (X + cX <= Y) ||
             ((X != Y) && ((PetscMax(dX,dY))%(PetscMin(dX,dY))!=0))) {
    /* If Y[0..cM-1] and X[0..cX-1] are not overlapped... */

    /* Get the dense matrices and dimensions associated to Y and X */
    ierr = VecGetLocalSize(X[0], &rX); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Y[0], &rY); CHKERRQ(ierr);
    if (rX != rY) {
      SETERRQ(1, "The multivectors doesn't have the same dimension!");
    }
    ierr = VecGetArray(X[0], &px);CHKERRQ(ierr);
    ierr = VecGetArray(Y[0], &py);CHKERRQ(ierr);

    /* Update the strides */
    ldX = rX*dX; ldY= rY*dY;

    /* Do operation */ 
    ierr = SlepcDenseMatProd(py, ldY, beta, alpha, px, ldX, rX, rcX,
                    PETSC_FALSE, M, ldM, rM, cM, PETSC_FALSE); CHKERRQ(ierr);
  
    ierr = VecRestoreArray(X[0], &px);CHKERRQ(ierr);
    ierr = VecRestoreArray(Y[0], &py);CHKERRQ(ierr);

  } else if ((Y >= X) && (beta == 0.0) && (dY == dX)) {
    /* If not, call to SlepcUpdateVectors */
    ierr = SlepcUpdateStrideVectors(cX, X, Y-X, dX, Y-X+cM*dX, M-ldM*(Y-X),
                                    ldM, PETSC_FALSE); CHKERRQ(ierr);
    if (alpha != 1.0)
      for (i=0; i<cM; i++) {
        ierr = VecScale(Y[i], alpha); CHKERRQ(ierr);
      }
  } else {
    SETERRQ(1, "I don't support this case!");
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
  Compute X <- alpha * X[0:dX:cX-1] * M
  where X is a matrix with non-consecutive columns
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcUpdateVectorsD"
PetscErrorCode SlepcUpdateVectorsD(Vec *X, PetscInt cX, PetscScalar alpha,
  const PetscScalar *M, PetscInt ldM, PetscInt rM, PetscInt cM,
  PetscScalar *work, PetscInt lwork)
{
  PetscErrorCode  ierr;
  PetscScalar     **px, *Y, *Z;
  PetscInt        rX, i, j, rY, rY0, ldY;

  PetscFunctionBegin;

  if (cX != rM) {
    SETERRQ(1, "Matrix dimensions doesn't match!");
  }

  rY = (lwork/2)/cX;
  if (rY <= 0) {
    SETERRQ(1, "Insufficient work space given!");
  }
  Y = work; Z = &Y[cX*rY]; ldY = rY;

  if ((cX == 0) || (rM == 0) || (cM == 0)) {
    PetscFunctionReturn(0);
  }

  /* Get the dense vectors associated to the columns of X */
  ierr = PetscMalloc(sizeof(Vec)*cX, &px); CHKERRQ(ierr);
  for(i=0; i<cX; i++) {
    ierr = VecGetArray(X[i], &px[i]); CHKERRQ(ierr);
  }
  ierr = VecGetLocalSize(X[0], &rX); CHKERRQ(ierr);

  for(i=0, rY0=0; i<rX; i+=rY0) {
    rY0 = PetscMin(rY, rX-i);

    /* Y <- X[i0:i1,:] */
    for(j=0; j<cX; j++) {
      ierr = SlepcDenseCopy(&Y[ldY*j], ldY, px[j]+i, rX, rY0, 1);
      CHKERRQ(ierr);
    }

    /* Z <- Y * M */
    ierr = SlepcDenseMatProd(Z, ldY, 0.0, alpha, Y, ldY, rY0, cX, PETSC_FALSE,
                                                 M, ldM, rM, cM, PETSC_FALSE);
    CHKERRQ(ierr);

    /* X <- Z */
    for(j=0; j<cM; j++) {
      ierr = SlepcDenseCopy(px[j]+i, rX, &Z[j*ldY], ldY, rY0, 1);
      CHKERRQ(ierr);
    }
  }

  for(i=0; i<cX; i++) {
    ierr = VecRestoreArray(X[i], &px[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(px); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END



/* Computes M <- [ M(0:sU-1,  0:sV-1) W(0:sU-1,  sV:eV-1) ]
                 [ W(sU:eU-1, 0:sV-1) W(sU:eU-1, sV:eV-1) ]
  where W = U' * V.
  workS0 and workS1 are an auxiliary scalar vector of size
  (eU-sU)*sV+(eV-sV)*eU. But, if sU == 0, sV == 0 and eU == ldM, only workS0
  is needed, and of size eU*eV.
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecsMult"
PetscErrorCode VecsMult(PetscScalar *M, MatType_t sM, PetscInt ldM,
                        Vec *U, PetscInt sU, PetscInt eU,
                        Vec *V, PetscInt sV, PetscInt eV,
                        PetscScalar *workS0, PetscScalar *workS1)
{
  PetscErrorCode  ierr;
  PetscInt        ldU, ldV, i, j, k;
  PetscScalar     *pu, *pv, *W, *Wr;

  PetscFunctionBegin;

  /* Check if quick exit */
  if ((eU-sU == 0) || (eV-sV == 0))
    PetscFunctionReturn(0);
    
  /* Get the dense matrices and dimensions associated to U and V */
  ierr = VecGetLocalSize(U[0], &ldU); CHKERRQ(ierr);
  ierr = VecGetLocalSize(V[0], &ldV); CHKERRQ(ierr);
  if (ldU != ldV) {
    SETERRQ(1, "Matrix dimensions doesn't match!");
  }
  ierr = VecGetArray(U[0], &pu);CHKERRQ(ierr);
  ierr = VecGetArray(V[0], &pv);CHKERRQ(ierr);

  if (workS0)
    W = workS0;
  else {
    ierr = PetscMalloc(sizeof(PetscScalar)*((eU-sU)*sV+(eV-sV)*eU), &W);
    CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(SLEPC_VecsMult,0,0,0,0);CHKERRQ(ierr);

  if ((sU == 0) && (sV == 0) && (eU == ldM)) {
    /* Use the smart memory usage version */

    /* W <- U' * V */
    ierr = SlepcDenseMatProdTriang(W, sM, eU,
                                   pu, 0, ldU, ldU, eU, PETSC_TRUE,
                                   pv, 0, ldV, ldV, eV, PETSC_FALSE);
    CHKERRQ(ierr);
 
    /* ReduceAll(W, SUM) */
    ierr = MPI_Allreduce(W, M, eU*eV, MPIU_SCALAR, MPIU_SUM,
                         ((PetscObject)U[0])->comm); CHKERRQ(ierr);
  /* Full M matrix */
  } else if (DVD_ISNOT(sM,DVD_MAT_UTRIANG) && 
             DVD_ISNOT(sM,DVD_MAT_LTRIANG)) {
    if (workS1)
      Wr = workS1;
    else {
      ierr = PetscMalloc(sizeof(PetscScalar)*((eU-sU)*sV+(eV-sV)*eU), &Wr);
      CHKERRQ(ierr);
    }
 
    /* W(0:(eU-sU)*sV-1) <- U(sU:eU-1)' * V(0:sV-1) */
    ierr = SlepcDenseMatProd(W, eU-sU, 0.0, 1.0,
                             pu+ldU*sU, ldU, ldU, eU-sU, PETSC_TRUE,
                             pv       , ldV, ldV, sV,    PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* W((eU-sU)*sV:(eU-sU)*sV+(eV-sV)*eU-1) <- U(0:eU-1)' * V(sV:eV-1) */
    ierr = SlepcDenseMatProd(W+(eU-sU)*sV, eU, 0.0, 1.0,
                             pu,        ldU, ldU, eU,    PETSC_TRUE,
                             pv+ldV*sV, ldV, ldV, eV-sV, PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* ReduceAll(W, SUM) */
    ierr = MPI_Allreduce(W, Wr, (eU-sU)*sV+(eV-sV)*eU, MPIU_SCALAR,
                      MPIU_SUM, ((PetscObject)U[0])->comm); CHKERRQ(ierr);
  
    /* M(...,...) <- W */
    for (i=0,k=0; i<sV; i++)
      for (j=ldM*i+sU; j<ldM*i+eU; j++,k++) M[j] = Wr[k];
    for (i=sV; i<eV; i++)
        for (j=ldM*i; j<ldM*i+eU; j++,k++) M[j] = Wr[k];
  
    if (!workS1) {
      ierr = PetscFree(Wr); CHKERRQ(ierr);
    }

  /* Upper triangular M matrix */
  } else if (DVD_IS(sM,DVD_MAT_UTRIANG) &&
             DVD_ISNOT(sM,DVD_MAT_LTRIANG)) {
    if (workS1)
      Wr = workS1;
    else {
      ierr = PetscMalloc(sizeof(PetscScalar)*(eV-sV)*eU, &Wr);
      CHKERRQ(ierr);
    }
 
    /* W(0:(eV-sV)*eU-1) <- U(0:eU-1)' * V(sV:eV-1) */
    ierr = SlepcDenseMatProd(W,         eU,  0.0, 1.0,
                             pu,        ldU, ldU, eU,    PETSC_TRUE,
                             pv+ldV*sV, ldV, ldV, eV-sV, PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* ReduceAll(W, SUM) */
    ierr = MPI_Allreduce(W, Wr, (eV-sV)*eU, MPIU_SCALAR, MPIU_SUM,
                         ((PetscObject)U[0])->comm); CHKERRQ(ierr);
  
    /* M(...,...) <- W */
    for (i=sV,k=0; i<eV; i++)
        for (j=ldM*i; j<ldM*i+eU; j++,k++) M[j] = Wr[k];

    if (!workS1) {
      ierr = PetscFree(Wr); CHKERRQ(ierr);
    }

  /* Lower triangular M matrix */
  } else if (DVD_ISNOT(sM,DVD_MAT_UTRIANG) &&
             DVD_IS(sM,DVD_MAT_LTRIANG)) {
    if (workS1)
      Wr = workS1;
    else {
      ierr = PetscMalloc(sizeof(PetscScalar)*(eU-sU)*eV, &Wr);
      CHKERRQ(ierr);
    }
 
    /* W(0:(eU-sU)*eV-1) <- U(sU:eU-1)' * V(0:eV-1) */
    ierr = SlepcDenseMatProd(W, eU-sU, 0.0, 1.0,
                             pu+ldU*sU, ldU, ldU, eU-sU, PETSC_TRUE,
                             pv       , ldV, ldV, eV,    PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* ReduceAll(W, SUM) */
    ierr = MPI_Allreduce(W, Wr, (eU-sU)*eV, MPIU_SCALAR, MPIU_SUM,
                         ((PetscObject)U[0])->comm); CHKERRQ(ierr);
  
    /* M(...,...) <- W */
    for (i=0,k=0; i<eV; i++)
      for (j=ldM*i+sU; j<ldM*i+eU; j++,k++) M[j] = Wr[k];
  
    if (!workS1) {
      ierr = PetscFree(Wr); CHKERRQ(ierr);
    }
  }

  ierr = PetscLogEventEnd(SLEPC_VecsMult,0,0,0,0);CHKERRQ(ierr);

  if (!workS0) {
    ierr = PetscFree(W); CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(U[0], &pu); CHKERRQ(ierr);
  ierr = VecRestoreArray(V[0], &pv); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END



/* Computes M <- [ M(0:sU-1,  0:sV-1) W(0:sU-1,  sV:eV-1) ]
                 [ W(sU:eU-1, 0:sV-1) W(sU:eU-1, sV:eV-1) ]
  where W = local_U' * local_V. Needs VecsMultIb for completing the operation!
  workS0 and workS1 are an auxiliary scalar vector of size
  (eU-sU)*sV+(eV-sV)*eU. But, if sU == 0, sV == 0 and eU == ldM, only workS0
  is needed, and of size eU*eV.
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecsMultIa"
PetscErrorCode VecsMultIa(PetscScalar *M, MatType_t sM, PetscInt ldM,
                          Vec *U, PetscInt sU, PetscInt eU,
                          Vec *V, PetscInt sV, PetscInt eV)
{
  PetscErrorCode  ierr;
  PetscInt        ldU, ldV;
  PetscScalar     *pu, *pv;

  PetscFunctionBegin;

  /* Check if quick exit */
  if ((eU-sU == 0) || (eV-sV == 0))
    PetscFunctionReturn(0);
    
  /* Get the dense matrices and dimensions associated to U and V */
  ierr = VecGetLocalSize(U[0], &ldU); CHKERRQ(ierr);
  ierr = VecGetLocalSize(V[0], &ldV); CHKERRQ(ierr);
  if (ldU != ldV) {
    SETERRQ(1, "Matrix dimensions doesn't match!");
  }
  ierr = VecGetArray(U[0], &pu);CHKERRQ(ierr);
  ierr = VecGetArray(V[0], &pv);CHKERRQ(ierr);

  if ((sU == 0) && (sV == 0) && (eU == ldM)) {
    /* M <- local_U' * local_V */
    ierr = SlepcDenseMatProdTriang(M, sM, eU,
                                   pu, 0, ldU, ldU, eU, PETSC_TRUE,
                                   pv, 0, ldV, ldV, eV, PETSC_FALSE);
    CHKERRQ(ierr);
 
  /* Full M matrix */
  } else if (DVD_ISNOT(sM,DVD_MAT_UTRIANG) && 
             DVD_ISNOT(sM,DVD_MAT_LTRIANG)) {
    /* M(sU:eU-1,0:sV-1) <- U(sU:eU-1)' * V(0:sV-1) */
    ierr = SlepcDenseMatProd(&M[sU], ldM, 0.0, 1.0,
                             pu+ldU*sU, ldU, ldU, eU-sU, PETSC_TRUE,
                             pv       , ldV, ldV, sV,    PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* M(0:eU-1,sV:eV-1) <- U(0:eU-1)' * V(sV:eV-1) */
    ierr = SlepcDenseMatProd(&M[ldM*sV], ldM, 0.0, 1.0,
                             pu,        ldU, ldU, eU,    PETSC_TRUE,
                             pv+ldV*sV, ldV, ldV, eV-sV, PETSC_FALSE);
    CHKERRQ(ierr);
  
  /* Other structures */
  } else {
    SETERRQ(1, "Matrix structure doesn't support by VecsMultI!");
    PetscFunctionReturn(1);
  }

  ierr = VecRestoreArray(U[0], &pu); CHKERRQ(ierr);
  ierr = VecRestoreArray(V[0], &pv); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


/* Computes M <- nprocs*M
  where nprocs is the number of processors.
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecsMultIc"
PetscErrorCode VecsMultIc(PetscScalar *M, MatType_t sM, PetscInt ldM,
                          PetscInt rM, PetscInt cM, Vec V)
{
  PetscInt        i,j,n;

  PetscFunctionBegin;

  /* Check if quick exit */
  if ((rM == 0) || (cM == 0))
    PetscFunctionReturn(0);
    
  if (sM != 0) {
    SETERRQ(1, "Matrix structure doesn't support by VecsMultIc!");
    PetscFunctionReturn(1);
  }

  MPI_Comm_size(((PetscObject)V)->comm, &n);

  for(i=0; i<cM; i++)
    for(j=0; j<rM; j++)
      M[ldM*i+j]/= (PetscScalar)n;

  PetscFunctionReturn(0);
}
EXTERN_C_END


/* Computes N <- Allreduce( [ M(0:sU-1,  0:sV-1) W(0:sU-1,  sV:eV-1) ] )
                          ( [ W(sU:eU-1, 0:sV-1) W(sU:eU-1, sV:eV-1) ] )
  where W = U' * V.
  workS0 and workS1 are an auxiliary scalar vector of size
  (eU-sU)*sV+(eV-sV)*eU. But, if sU == 0, sV == 0 and eU == ldM, only workS0
  is needed, and of size eU*eV.
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecsMultIb"
PetscErrorCode VecsMultIb(PetscScalar *M, MatType_t sM, PetscInt ldM,
                          PetscInt rM, PetscInt cM, PetscScalar *auxS,
                          Vec V)
{
  PetscErrorCode  ierr;
  PetscScalar     *W, *Wr;

  PetscFunctionBegin;

  /* Check if quick exit */
  if ((rM == 0) || (cM == 0))
    PetscFunctionReturn(0);
    
  if (auxS)
    W = auxS;
  else {
    ierr = PetscMalloc(sizeof(PetscScalar)*rM*cM*2, &W);
    CHKERRQ(ierr);
  }
  Wr = W + rM*cM;

  ierr = PetscLogEventBegin(SLEPC_VecsMult,0,0,0,0);CHKERRQ(ierr);

  if (sM == 0) {
    /* W <- M */
    ierr = SlepcDenseCopy(W, rM, M, ldM, rM, cM); CHKERRQ(ierr);

    /* Wr <- ReduceAll(W, SUM) */
    ierr = MPI_Allreduce(W, Wr, rM*cM, MPIU_SCALAR, MPIU_SUM,
                         ((PetscObject)V)->comm); CHKERRQ(ierr);

    /* M <- Wr */
    ierr = SlepcDenseCopy(M, ldM, Wr, rM, rM, cM); CHKERRQ(ierr);

  /* Other structures */
  } else {
    SETERRQ(1, "Matrix structure doesn't support by VecsMultI!");
    PetscFunctionReturn(1);
  }

  ierr = PetscLogEventEnd(SLEPC_VecsMult,0,0,0,0);CHKERRQ(ierr);

  if (!auxS) {
    ierr = PetscFree(W); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END


/* Computes M <- [ M(0:sU-1,  0:sV-1) W(0:sU-1,  sV:eV-1) ]
                 [ W(sU:eU-1, 0:sV-1) W(sU:eU-1, sV:eV-1) ]
  where W = U' * V.
  r, a DvdReduction structure,
  sr, an structure DvdMult_copy_func.
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecsMultS"
PetscErrorCode VecsMultS(PetscScalar *M, MatType_t sM, PetscInt ldM,
                         Vec *U, PetscInt sU, PetscInt eU,
                         Vec *V, PetscInt sV, PetscInt eV, DvdReduction *r,
                         DvdMult_copy_func *sr)
{
  PetscErrorCode  ierr;
  PetscInt        ldU, ldV;
  PetscScalar     *pu, *pv, *W;

  PetscFunctionBegin;

  /* Check if quick exit */
  if ((eU-sU == 0) || (eV-sV == 0))
    PetscFunctionReturn(0);
    
  /* Get the dense matrices and dimensions associated to U and V */
  ierr = VecGetLocalSize(U[0], &ldU); CHKERRQ(ierr);
  ierr = VecGetLocalSize(V[0], &ldV); CHKERRQ(ierr);
  if (ldU != ldV) {
    SETERRQ(1, "Matrix dimensions doesn't match!");
  }
  ierr = VecGetArray(U[0], &pu);CHKERRQ(ierr);
  ierr = VecGetArray(V[0], &pv);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(SLEPC_VecsMult,0,0,0,0);CHKERRQ(ierr);

  if ((sU == 0) && (sV == 0) && (eU == ldM)) {
    /* Use the smart memory usage version */

    /* Add the reduction to r */
    ierr = SlepcAllReduceSum(r, eU*eV, VecsMultS_copy_func, sr, &W);
    CHKERRQ(ierr);

    /* W <- U' * V */
    ierr = SlepcDenseMatProdTriang(W, sM, eU,
                                   pu, 0, ldU, ldU, eU, PETSC_TRUE,
                                   pv, 0, ldV, ldV, eV, PETSC_FALSE);
    CHKERRQ(ierr);
 
    /* M <- ReduceAll(W, SUM) */
    sr->M = M;    sr->ld = ldM;
    sr->i0 = 0;   sr->i1 = eV;    sr->s0 = sU;    sr->e0 = eU;
                  sr->i2 = eV; 

  /* Full M matrix */
  } else if (DVD_ISNOT(sM,DVD_MAT_UTRIANG) && 
             DVD_ISNOT(sM,DVD_MAT_LTRIANG)) {
    /* Add the reduction to r */
    ierr = SlepcAllReduceSum(r, (eU-sU)*sV+(eV-sV)*eU, VecsMultS_copy_func,
                             sr, &W);
    CHKERRQ(ierr);

    /* W(0:(eU-sU)*sV-1) <- U(sU:eU-1)' * V(0:sV-1) */
    ierr = SlepcDenseMatProd(W, eU-sU, 0.0, 1.0,
                             pu+ldU*sU, ldU, ldU, eU-sU, PETSC_TRUE,
                             pv       , ldV, ldV, sV,    PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* W((eU-sU)*sV:(eU-sU)*sV+(eV-sV)*eU-1) <- U(0:eU-1)' * V(sV:eV-1) */
    ierr = SlepcDenseMatProd(W+(eU-sU)*sV, eU, 0.0, 1.0,
                             pu,        ldU, ldU, eU,    PETSC_TRUE,
                             pv+ldV*sV, ldV, ldV, eV-sV, PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* M <- ReduceAll(W, SUM) */
    sr->M = M;    sr->ld = ldM;
    sr->i0 = 0;   sr->i1 = sV;    sr->s0 = sU;    sr->e0 = eU;
                  sr->i2 = eV;    sr->s1 = 0;     sr->e1 = eU;

  /* Upper triangular M matrix */
  } else if (DVD_IS(sM,DVD_MAT_UTRIANG) &&
             DVD_ISNOT(sM,DVD_MAT_LTRIANG)) {
    /* Add the reduction to r */
    ierr = SlepcAllReduceSum(r, (eV-sV)*eU, VecsMultS_copy_func, sr, &W);
    CHKERRQ(ierr);
 
    /* W(0:(eV-sV)*eU-1) <- U(0:eU-1)' * V(sV:eV-1) */
    ierr = SlepcDenseMatProd(W,         eU,  0.0, 1.0,
                             pu,        ldU, ldU, eU,    PETSC_TRUE,
                             pv+ldV*sV, ldV, ldV, eV-sV, PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* M <- ReduceAll(W, SUM) */
    sr->M = M;    sr->ld = ldM;
    sr->i0 = sV;  sr->i1 = eV;    sr->s0 = 0;     sr->e0 = eU;
                  sr->i2 = eV;
  
  /* Lower triangular M matrix */
  } else if (DVD_ISNOT(sM,DVD_MAT_UTRIANG) &&
             DVD_IS(sM,DVD_MAT_LTRIANG)) {
    /* Add the reduction to r */
    ierr = SlepcAllReduceSum(r, (eU-sU)*eV, VecsMultS_copy_func, sr, &W);
    CHKERRQ(ierr);
 
    /* W(0:(eU-sU)*eV-1) <- U(sU:eU-1)' * V(0:eV-1) */
    ierr = SlepcDenseMatProd(W, eU-sU, 0.0, 1.0,
                             pu+ldU*sU, ldU, ldU, eU-sU, PETSC_TRUE,
                             pv       , ldV, ldV, eV,    PETSC_FALSE);
    CHKERRQ(ierr);
  
    /* ReduceAll(W, SUM) */
    sr->M = M;    sr->ld = ldM;
    sr->i0 = 0;   sr->i1 = eV;    sr->s0 = sU;    sr->e0 = eU;
                  sr->i2 = eV;
  }

  ierr = PetscLogEventEnd(SLEPC_VecsMult,0,0,0,0);CHKERRQ(ierr);

  ierr = VecRestoreArray(U[0], &pu); CHKERRQ(ierr);
  ierr = VecRestoreArray(V[0], &pv); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecsMultS_copy_func"
PetscErrorCode VecsMultS_copy_func(PetscScalar *out, PetscInt size_out,
                                   void *ptr)
{
  PetscInt        i, j, k;
  DvdMult_copy_func
                  *sr = (DvdMult_copy_func*)ptr;

  PetscFunctionBegin;

  for (i=sr->i0,k=0; i<sr->i1; i++)
    for (j=sr->ld*i+sr->s0; j<sr->ld*i+sr->e0; j++,k++) sr->M[j] = out[k];
  for (i=sr->i1; i<sr->i2; i++)
    for (j=sr->ld*i+sr->s1; j<sr->ld*i+sr->e1; j++,k++) sr->M[j] = out[k];

  if (k != size_out) {
    SETERRQ(1, "Error in VecsMultS_copy_func!");
    PetscFunctionReturn(1);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* Orthonormalize a chunk of parallel vector.
   NOTE: wS0 and wS1 must be of size n*n.
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecsOrthonormalize"
PetscErrorCode VecsOrthonormalize(Vec *V, PetscInt n, PetscScalar *wS0,
                                  PetscScalar *wS1)
{
  PetscErrorCode  ierr;
  PetscBLASInt    nn = n, info, ld;
  PetscInt        ldV;
  PetscScalar     *H, *T, one=1.0, *pv;
  
  PetscFunctionBegin;

  if (!wS0) {
    ierr = PetscMalloc(sizeof(PetscScalar)*n*n, &H); CHKERRQ(ierr);
  } else
    H = wS0;
  if (!wS1) {
    ierr = PetscMalloc(sizeof(PetscScalar)*n*n, &T); CHKERRQ(ierr);
  } else
    T = wS1;

  /* H <- V' * V */
  ierr = VecsMult(H, 0, n, V, 0, n, V, 0, n, T, PETSC_NULL); CHKERRQ(ierr);

  /* H <- chol(H) */
  LAPACKpbtrf_("U", &nn, &nn, H, &nn, &info);
  if (info) SETERRQ1(PETSC_ERR_LIB, "Error in Lapack PBTRF %d", info);

  /* V <- V * inv(H) */
  ierr = VecGetLocalSize(V[0], &ldV); CHKERRQ(ierr);
  ierr = VecGetArray(V[0], &pv);CHKERRQ(ierr);
  ld = ldV;
  BLAStrsm_("R", "U", "N", "N", &ld, &nn, &one, H, &nn, pv, &ld);
  ierr = VecRestoreArray(V[0], &pv);CHKERRQ(ierr);

  if (!wS0) {
    ierr = PetscFree(H); CHKERRQ(ierr);
  }
  if (!wS1) {
    ierr = PetscFree(T); CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* 
  Sum up several arrays with only one call to MPIReduce.
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcAllReduceSumBegin"
PetscErrorCode SlepcAllReduceSumBegin(DvdReductionChunk *ops,
                                      PetscInt max_size_ops,
                                      PetscScalar *in, PetscScalar *out,
                                      PetscInt max_size_in, DvdReduction *r,
                                      MPI_Comm comm)
{
  PetscFunctionBegin;

  r->in = in;
  r->out = out;
  r->size_in = 0;
  r->max_size_in = max_size_in;
  r->ops = ops;
  r->size_ops = 0;
  r->max_size_ops = max_size_ops;
  r->comm = comm;

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcAllReduceSum"
PetscErrorCode SlepcAllReduceSum(DvdReduction *r, PetscInt size_in,
                                 DvdReductionPostF f, void *ptr,
                                 PetscScalar **in)
{
  PetscFunctionBegin;

  *in = r->in + r->size_in;
  r->ops[r->size_ops].out = r->out + r->size_in;
  r->ops[r->size_ops].size_out = size_in;
  r->ops[r->size_ops].f = f;
  r->ops[r->size_ops].ptr = ptr;
  if (++(r->size_ops) > r->max_size_ops) {
    SETERRQ(1, "max_size_ops is not enought!");
  }
  if ((r->size_in+= size_in) > r->max_size_in) {
    SETERRQ(1, "max_size_in is not enought!");
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SlepcAllReduceSumEnd"
PetscErrorCode SlepcAllReduceSumEnd(DvdReduction *r)
{
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;

  /* Check if quick exit */
  if (r->size_ops == 0)
    PetscFunctionReturn(0);

  /* Call the MPIAllReduce routine */
  ierr = MPI_Allreduce(r->in, r->out, r->size_in, MPIU_SCALAR, MPIU_SUM,
                       r->comm); CHKERRQ(ierr);

  /* Call the postponed routines */
  for(i=0; i<r->size_ops; i++) {
    ierr = r->ops[i].f(r->ops[i].out, r->ops[i].size_out, r->ops[i].ptr);
    CHKERRQ(ierr);
  }

  /* Tag the operation as done */
  r->size_ops = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_orthV"
PetscErrorCode dvd_orthV(IP ip, Vec *DS, PetscInt size_DS, Vec *cX,
                         PetscInt size_cX, Vec *V, PetscInt V_new_s,
                         PetscInt V_new_e, PetscScalar *auxS, Vec auxV,
                         PetscRandom rand)
{
  PetscErrorCode  ierr;
  PetscInt        i, j;
  PetscTruth      lindep;
  PetscReal       norm;
  PetscScalar     *auxS0 = auxS;
 
  PetscFunctionBegin;
 
  /* Orthonormalize V with IP */
  for (i=V_new_s; i<V_new_e; i++) {
    for(j=0; j<3; j++) {
      if (j>0) { ierr = SlepcVecSetRandom(V[i], rand); CHKERRQ(ierr); }
      if (cX + size_cX == V) {
        /* If cX and V are contiguous, orthogonalize in one step */
        ierr = IPOrthogonalize(ip, size_DS, DS, size_cX+i, PETSC_NULL, cX,
                               V[i], auxS0, &norm, &lindep); CHKERRQ(ierr);
      } else if (DS) {
        /* Else orthogonalize first against DS, and then against cX and V */
        ierr = IPOrthogonalize(ip, size_DS, DS, size_cX, PETSC_NULL, cX,
                               V[i], auxS0, PETSC_NULL, &lindep); CHKERRQ(ierr);
        if(lindep == PETSC_FALSE) {
          ierr = IPOrthogonalize(ip, 0, PETSC_NULL, i, PETSC_NULL, V,
                                 V[i], auxS0, &norm, &lindep); CHKERRQ(ierr);
        }
      } else {
        /* Else orthogonalize first against cX and then against V */
        ierr = IPOrthogonalize(ip, size_cX, cX, i, PETSC_NULL, V,
                               V[i], auxS0, &norm, &lindep); CHKERRQ(ierr);
      }
      if((lindep == PETSC_FALSE) && (norm > PETSC_MACHINE_EPSILON)) break;
      printf("Ortho problems at %d\n", i);
    }
    if((lindep == PETSC_TRUE) || (norm < PETSC_MACHINE_EPSILON)) {
        SETERRQ(1, "Error during the orthonormalization of the eigenvectors!");
    }
    ierr = VecScale(V[i], 1.0/norm); CHKERRQ(ierr);
  }
 
  PetscFunctionReturn(0);
}
EXTERN_C_END
 
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_compute_eigenvectors"
/*
  Compute eigenvectors associated to the Schur decomposition (S, T) and
  save the left vectors in pY and the right vectors in pX, where
  n, size of the eigenproblem
  ldS, ldT, leading dimension of S and T
  ldpX, ldpY, leading dimension of pX and pY
  auxS, auxiliar scalar of length:
    double standard 3n+n*n, double generalized 11n+4n*n,
    complex standard 3n+n*n, complex generalized 4n+2n*n
  size_auxS, the length of auxS
  doProd, if true pX and pY return the eigenvectors premultiplied by the input vectors stored in pX and pY respectively
*/
PetscErrorCode dvd_compute_eigenvectors(PetscInt n_, PetscScalar *S,
  PetscInt ldS, PetscScalar *T, PetscInt ldT, PetscScalar *pX,
  PetscInt ldpX_, PetscScalar *pY, PetscInt ldpY_, PetscScalar *auxS,
  PetscInt size_auxS, PetscTruth doProd)
{
  PetscErrorCode  ierr;
  PetscBLASInt    n = PetscBLASIntCast(n_), ldpX = PetscBLASIntCast(ldpX_),
                  ldpY = PetscBLASIntCast(ldpY_), nout = n, info;
  PetscScalar     *Sc, *Tc;
  const char      *side, *howmny;
#if defined(PETSC_USE_COMPLEX)
  PetscReal       *auxR;
  PetscScalar     *diag, a;
  PetscInt        i,j;
#else
  PetscScalar     *pA,*pB;
  PetscBLASInt    n1, ldpA,ldpB;
  PetscScalar     *alphar, *alphai, *beta;
#endif
  
  PetscFunctionBegin;

  if (pX && pY) side = "B";
  else if (pX)  side = "R";
  else if (pY)  side = "L";
  else { PetscFunctionReturn(0); }

  if (!pX) ldpX = 1;
  if (!pY) ldpY = 1;

  howmny = (doProd == PETSC_TRUE)?"B":"A";

  Sc = auxS; auxS+= n*n; size_auxS-= n*n;
  if (T) Tc = auxS, auxS+= n*n, size_auxS-= n*n;
  else   Tc = PETSC_NULL;
   
  /* Sc <- S, Tc <- T */
  ierr = SlepcDenseCopy(Sc, n, S, ldS, n, n); CHKERRQ(ierr);
  if (T) {
    ierr = SlepcDenseCopy(Tc, n, T, ldT, n, n); CHKERRQ(ierr);
  }

  if (T) {
    /* [eigr, pX] = eig(S, T) */
#if defined(PETSC_USE_COMPLEX)
    auxR = (PetscReal*)auxS; auxS = (PetscScalar*)(auxR+2*n); size_auxS-= 2*n;
    diag = auxS; auxS+= n; size_auxS-= n;
    for(i=0; i<n; i++) {
      a = PetscConj(Tc[n*i+i])/PetscAbsScalar(Tc[n*i+i]);
      diag[i] = a;
      for(j=0; j<=i; j++)
        Tc[n*i+j] = PetscRealPart(Tc[n*i+j]*a),
        Sc[n*i+j]*= a;
    }
    if (size_auxS < 2*n)
      SETERRQ(PETSC_ERR_LIB,"Insufficient auxiliar memory for xTGEVC");
    LAPACKtgevc_(side,howmny,PETSC_NULL,&n,Sc,&n,Tc,&n,pY,&ldpY,pX,&ldpX,&n,&nout,auxS,auxR,&info);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTGEVC %i",info);
    for(i=0; i<n; i++)
      for(j=0; j<n; j++) pX[n*i+j]/= diag[j];
#else
    alphar = auxS; auxS+= n; size_auxS-= n;
    alphai = auxS; auxS+= n; size_auxS-= n;
    beta = auxS; auxS+= n; size_auxS-= n;
    if (doProd == PETSC_TRUE) {
      if (pX) pA = auxS, auxS+= n*n, size_auxS-= n*n, ldpA = n;
      else    pA = PETSC_NULL, ldpA = 0;
      if (pY) pB = auxS, auxS+= n*n, size_auxS-= n*n, ldpB = n;
      else    pB = PETSC_NULL, ldpB = 0;
    } else {
      pA = pX; pB = pY; ldpA = ldpX; ldpB = ldpY;
    }
    /* LAPACKtgevc_ needs the element i,i+1 in the 2-by-2 digonal blocs
       of T that represent complex conjugate eigenpairs to be zero */
    n1 = size_auxS;
    if (size_auxS < 8*n)
      SETERRQ(PETSC_ERR_LIB,"Insufficient auxiliar memory for xGGEV");
    LAPACKggev_(pY?"V":"N",pX?"V":"N",&n,Sc,&n,Tc,&n,alphar,alphai,beta,pB,&ldpB,pA,&ldpA,auxS,&n1,&info);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
    if (doProd == PETSC_TRUE) {
      if (pX) {
        /* pX <- pX * pA */
        ierr = SlepcDenseCopy(Sc, n, pX, ldpX, n, n); CHKERRQ(ierr);
        ierr = SlepcDenseMatProd(pX, ldpX, 1.0, 0.0,
                                 Sc, n, n, n, PETSC_FALSE, 
                                 pA, n, n, n, PETSC_FALSE); CHKERRQ(ierr);
      }
      if (pY) {
        /* pY <- pY * pB */
        ierr = SlepcDenseCopy(Sc, n, pY, ldpY, n, n); CHKERRQ(ierr);
        ierr = SlepcDenseMatProd(pY, ldpY, 1.0, 0.0,
                                 Sc, n, n, n, PETSC_FALSE, 
                                 pB, n, n, n, PETSC_FALSE); CHKERRQ(ierr);
      }
    }
#endif
  } else {
    /* [eigr, pX] = eig(S) */
#if defined(PETSC_USE_COMPLEX)
    auxR = (PetscReal*)auxS; auxS = (PetscScalar*)(auxR+n); size_auxS-= n;
    if (size_auxS < 2*n)
      SETERRQ(PETSC_ERR_LIB,"Insufficient auxiliar memory for xTREVC");
    LAPACKtrevc_(side,howmny,PETSC_NULL,&n,Sc,&n,pY,&ldpY,pX,&ldpX,&n,&nout,auxS,auxR,&info);
#else
    LAPACKtrevc_(side,howmny,PETSC_NULL,&n,Sc,&n,pY,&ldpY,pX,&ldpX,&n,&nout,auxS,&info);
#endif
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END
