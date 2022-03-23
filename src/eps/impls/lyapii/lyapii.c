/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "lyapii"

   Method: Lyapunov inverse iteration

   Algorithm:

       Lyapunov inverse iteration using LME solvers

   References:

       [1] H.C. Elman and M. Wu, "Lyapunov inverse iteration for computing a
           few rightmost eigenvalues of large generalized eigenvalue problems",
           SIAM J. Matrix Anal. Appl. 34(4):1685-1707, 2013.

       [2] K. Meerbergen and A. Spence, "Inverse iteration for purely imaginary
           eigenvalues with application to the detection of Hopf bifurcations in
           large-scale problems", SIAM J. Matrix Anal. Appl. 31:1982-1999, 2010.
*/

#include <slepc/private/epsimpl.h>          /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  LME      lme;      /* Lyapunov solver */
  DS       ds;       /* used to compute the SVD for compression */
  PetscInt rkl;      /* prescribed rank for the Lyapunov solver */
  PetscInt rkc;      /* the compressed rank, cannot be larger than rkl */
} EPS_LYAPII;

typedef struct {
  Mat      S;        /* the operator matrix, S=A^{-1}*B */
  BV       Q;        /* orthogonal basis of converged eigenvectors */
} EPS_LYAPII_MATSHELL;

typedef struct {
  Mat      S;        /* the matrix from which the implicit operator is built */
  PetscInt n;        /* the size of matrix S, the operator is nxn */
  LME      lme;      /* dummy LME object */
#if defined(PETSC_USE_COMPLEX)
  Mat      A,B,F;
  Vec      w;
#endif
} EPS_EIG_MATSHELL;

PetscErrorCode EPSSetUp_LyapII(EPS eps)
{
  PetscRandom    rand;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  EPSCheckSinvert(eps);
  if (eps->ncv!=PETSC_DEFAULT) {
    PetscCheck(eps->ncv>=eps->nev+1,PetscObjectComm((PetscObject)eps),PETSC_ERR_USER_INPUT,"The value of ncv must be at least nev+1");
  } else eps->ncv = eps->nev+1;
  if (eps->mpd!=PETSC_DEFAULT) CHKERRQ(PetscInfo(eps,"Warning: parameter mpd ignored\n"));
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(1000*eps->nev,100*eps->n);
  if (!eps->which) eps->which=EPS_LARGEST_REAL;
  PetscCheck(eps->which==EPS_LARGEST_REAL,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver supports only largest real eigenvalues");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_EXTRACTION | EPS_FEATURE_TWOSIDED);

  if (!ctx->rkc) ctx->rkc = 10;
  if (!ctx->rkl) ctx->rkl = 3*ctx->rkc;
  if (!ctx->lme) CHKERRQ(EPSLyapIIGetLME(eps,&ctx->lme));
  CHKERRQ(LMESetProblemType(ctx->lme,LME_LYAPUNOV));
  CHKERRQ(LMESetErrorIfNotConverged(ctx->lme,PETSC_TRUE));

  if (!ctx->ds) {
    CHKERRQ(DSCreate(PetscObjectComm((PetscObject)eps),&ctx->ds));
    CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->ds));
    CHKERRQ(DSSetType(ctx->ds,DSSVD));
  }
  CHKERRQ(DSAllocate(ctx->ds,ctx->rkl));

  CHKERRQ(DSSetType(eps->ds,DSNHEP));
  CHKERRQ(DSAllocate(eps->ds,eps->ncv));

  CHKERRQ(EPSAllocateSolution(eps,0));
  CHKERRQ(BVGetRandomContext(eps->V,&rand));  /* make sure the random context is available when duplicating */
  CHKERRQ(EPSSetWorkVecs(eps,3));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_EPSLyapIIOperator(Mat M,Vec x,Vec r)
{
  EPS_LYAPII_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  CHKERRQ(MatMult(matctx->S,x,r));
  CHKERRQ(BVOrthogonalizeVec(matctx->Q,r,NULL,NULL,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_EPSLyapIIOperator(Mat M)
{
  EPS_LYAPII_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  CHKERRQ(MatDestroy(&matctx->S));
  CHKERRQ(PetscFree(matctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_EigOperator(Mat M,Vec x,Vec y)
{
  EPS_EIG_MATSHELL  *matctx;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt          n;
  PetscScalar       *Y,*C,zero=0.0,done=1.0,dtwo=2.0;
  const PetscScalar *S,*X;
  PetscBLASInt      n_;
#endif

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));

#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(MatMult(matctx->B,x,matctx->w));
  CHKERRQ(MatSolve(matctx->F,matctx->w,y));
#else
  CHKERRQ(VecGetArrayRead(x,&X));
  CHKERRQ(VecGetArray(y,&Y));
  CHKERRQ(MatDenseGetArrayRead(matctx->S,&S));

  n = matctx->n;
  CHKERRQ(PetscCalloc1(n*n,&C));
  CHKERRQ(PetscBLASIntCast(n,&n_));

  /* C = 2*S*X*S.' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&dtwo,S,&n_,X,&n_,&zero,Y,&n_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&n_,&n_,&n_,&done,Y,&n_,S,&n_,&zero,C,&n_));

  /* Solve S*Y + Y*S' = -C */
  CHKERRQ(LMEDenseLyapunov(matctx->lme,n,(PetscScalar*)S,n,C,n,Y,n));

  CHKERRQ(PetscFree(C));
  CHKERRQ(VecRestoreArrayRead(x,&X));
  CHKERRQ(VecRestoreArray(y,&Y));
  CHKERRQ(MatDenseRestoreArrayRead(matctx->S,&S));
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_EigOperator(Mat M)
{
  EPS_EIG_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(MatDestroy(&matctx->A));
  CHKERRQ(MatDestroy(&matctx->B));
  CHKERRQ(MatDestroy(&matctx->F));
  CHKERRQ(VecDestroy(&matctx->w));
#endif
  CHKERRQ(PetscFree(matctx));
  PetscFunctionReturn(0);
}

/*
   EV2x2: solve the eigenproblem for a 2x2 matrix M
 */
static PetscErrorCode EV2x2(PetscScalar *M,PetscInt ld,PetscScalar *wr,PetscScalar *wi,PetscScalar *vec)
{
  PetscBLASInt   lwork=10,ld_;
  PetscScalar    work[10];
  PetscBLASInt   two=2,info;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      rwork[6];
#endif

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(ld,&ld_));
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","V",&two,M,&ld_,wr,wi,NULL,&ld_,vec,&ld_,work,&lwork,&info));
#else
  PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","V",&two,M,&ld_,wr,NULL,&ld_,vec,&ld_,work,&lwork,rwork,&info));
#endif
  SlepcCheckLapackInfo("geev",info);
  CHKERRQ(PetscFPTrapPop());
  PetscFunctionReturn(0);
}

/*
   LyapIIBuildRHS: prepare the right-hand side of the Lyapunov equation SY + YS' = -2*S*Z*S'
   in factored form:
      if (V)  U=sqrt(2)*S*V    (uses 1 work vector)
      else    U=sqrt(2)*S*U    (uses 2 work vectors)
   where U,V are assumed to have rk columns.
 */
static PetscErrorCode LyapIIBuildRHS(Mat S,PetscInt rk,Mat U,BV V,Vec *work)
{
  PetscScalar    *array,*uu;
  PetscInt       i,nloc;
  Vec            v,u=work[0];

  PetscFunctionBegin;
  CHKERRQ(MatGetLocalSize(U,&nloc,NULL));
  for (i=0;i<rk;i++) {
    CHKERRQ(MatDenseGetColumn(U,i,&array));
    if (V) CHKERRQ(BVGetColumn(V,i,&v));
    else {
      v = work[1];
      CHKERRQ(VecPlaceArray(v,array));
    }
    CHKERRQ(MatMult(S,v,u));
    if (V) CHKERRQ(BVRestoreColumn(V,i,&v));
    else CHKERRQ(VecResetArray(v));
    CHKERRQ(VecScale(u,PETSC_SQRT2));
    CHKERRQ(VecGetArray(u,&uu));
    CHKERRQ(PetscArraycpy(array,uu,nloc));
    CHKERRQ(VecRestoreArray(u,&uu));
    CHKERRQ(MatDenseRestoreColumn(U,&array));
  }
  PetscFunctionReturn(0);
}

/*
   LyapIIBuildEigenMat: create shell matrix Op=A\B with A = kron(I,S)+kron(S,I), B = -2*kron(S,S)
   where S is a sequential square dense matrix of order n.
   v0 is the initial vector, should have the form v0 = w*w' (for instance 1*1')
 */
static PetscErrorCode LyapIIBuildEigenMat(LME lme,Mat S,Mat *Op,Vec *v0)
{
  PetscInt          n,m;
  PetscBool         create=PETSC_FALSE;
  EPS_EIG_MATSHELL  *matctx;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar       theta,*aa,*bb;
  const PetscScalar *ss;
  PetscInt          i,j,f,c,off,ld;
  IS                perm;
#endif

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(S,&n,NULL));
  if (!*Op) create=PETSC_TRUE;
  else {
    CHKERRQ(MatGetSize(*Op,&m,NULL));
    if (m!=n*n) create=PETSC_TRUE;
  }
  if (create) {
    CHKERRQ(MatDestroy(Op));
    CHKERRQ(VecDestroy(v0));
    CHKERRQ(PetscNew(&matctx));
#if defined(PETSC_USE_COMPLEX)
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n*n,n*n,NULL,&matctx->A));
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n*n,n*n,NULL,&matctx->B));
    CHKERRQ(MatCreateVecs(matctx->A,NULL,&matctx->w));
#endif
    CHKERRQ(MatCreateShell(PETSC_COMM_SELF,n*n,n*n,PETSC_DETERMINE,PETSC_DETERMINE,matctx,Op));
    CHKERRQ(MatShellSetOperation(*Op,MATOP_MULT,(void(*)(void))MatMult_EigOperator));
    CHKERRQ(MatShellSetOperation(*Op,MATOP_DESTROY,(void(*)(void))MatDestroy_EigOperator));
    CHKERRQ(MatCreateVecs(*Op,NULL,v0));
  } else {
    CHKERRQ(MatShellGetContext(*Op,&matctx));
#if defined(PETSC_USE_COMPLEX)
    CHKERRQ(MatZeroEntries(matctx->A));
#endif
  }
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(MatDenseGetArray(matctx->A,&aa));
  CHKERRQ(MatDenseGetArray(matctx->B,&bb));
  CHKERRQ(MatDenseGetArrayRead(S,&ss));
  ld = n*n;
  for (f=0;f<n;f++) {
    off = f*n+f*n*ld;
    for (i=0;i<n;i++) for (j=0;j<n;j++) aa[off+i+j*ld] = ss[i+j*n];
    for (c=0;c<n;c++) {
      off = f*n+c*n*ld;
      theta = ss[f+c*n];
      for (i=0;i<n;i++) aa[off+i+i*ld] += theta;
      for (i=0;i<n;i++) for (j=0;j<n;j++) bb[off+i+j*ld] = -2*theta*ss[i+j*n];
    }
  }
  CHKERRQ(MatDenseRestoreArray(matctx->A,&aa));
  CHKERRQ(MatDenseRestoreArray(matctx->B,&bb));
  CHKERRQ(MatDenseRestoreArrayRead(S,&ss));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n*n,0,1,&perm));
  CHKERRQ(MatDestroy(&matctx->F));
  CHKERRQ(MatDuplicate(matctx->A,MAT_COPY_VALUES,&matctx->F));
  CHKERRQ(MatLUFactor(matctx->F,perm,perm,0));
  CHKERRQ(ISDestroy(&perm));
#endif
  matctx->lme = lme;
  matctx->S = S;
  matctx->n = n;
  CHKERRQ(VecSet(*v0,1.0));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_LyapII(EPS eps)
{
  EPS_LYAPII          *ctx = (EPS_LYAPII*)eps->data;
  PetscInt            i,ldds,rk,nloc,mloc,nv,idx,k;
  Vec                 v,w,z=eps->work[0],v0=NULL;
  Mat                 S,C,Ux[2],Y,Y1,R,U,W,X,Op=NULL;
  BV                  V;
  BVOrthogType        type;
  BVOrthogRefineType  refine;
  PetscScalar         eigr[2],eigi[2],*array,er,ei,*uu,*s,*xx,*aa,pM[4],vec[4];
  PetscReal           eta;
  EPS                 epsrr;
  PetscReal           norm;
  EPS_LYAPII_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(ctx->ds,&ldds));

  /* Operator for the Lyapunov equation */
  CHKERRQ(PetscNew(&matctx));
  CHKERRQ(STGetOperator(eps->st,&matctx->S));
  CHKERRQ(MatGetLocalSize(matctx->S,&mloc,&nloc));
  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)eps),mloc,nloc,PETSC_DETERMINE,PETSC_DETERMINE,matctx,&S));
  matctx->Q = eps->V;
  CHKERRQ(MatShellSetOperation(S,MATOP_MULT,(void(*)(void))MatMult_EPSLyapIIOperator));
  CHKERRQ(MatShellSetOperation(S,MATOP_DESTROY,(void(*)(void))MatDestroy_EPSLyapIIOperator));
  CHKERRQ(LMESetCoefficients(ctx->lme,S,NULL,NULL,NULL));

  /* Right-hand side */
  CHKERRQ(BVDuplicateResize(eps->V,ctx->rkl,&V));
  CHKERRQ(BVGetOrthogonalization(V,&type,&refine,&eta,NULL));
  CHKERRQ(BVSetOrthogonalization(V,type,refine,eta,BV_ORTHOG_BLOCK_TSQR));
  CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)eps),eps->nloc,PETSC_DECIDE,PETSC_DECIDE,1,NULL,&Ux[0]));
  CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)eps),eps->nloc,PETSC_DECIDE,PETSC_DECIDE,2,NULL,&Ux[1]));
  nv = ctx->rkl;
  CHKERRQ(PetscMalloc1(nv,&s));

  /* Initialize first column */
  CHKERRQ(EPSGetStartVector(eps,0,NULL));
  CHKERRQ(BVGetColumn(eps->V,0,&v));
  CHKERRQ(BVInsertVec(V,0,v));
  CHKERRQ(BVRestoreColumn(eps->V,0,&v));
  CHKERRQ(BVSetActiveColumns(eps->V,0,0));  /* no deflation at the beginning */
  CHKERRQ(LyapIIBuildRHS(S,1,Ux[0],V,eps->work));
  idx = 0;

  /* EPS for rank reduction */
  CHKERRQ(EPSCreate(PETSC_COMM_SELF,&epsrr));
  CHKERRQ(EPSSetOptionsPrefix(epsrr,((PetscObject)eps)->prefix));
  CHKERRQ(EPSAppendOptionsPrefix(epsrr,"eps_lyapii_"));
  CHKERRQ(EPSSetDimensions(epsrr,1,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(EPSSetTolerances(epsrr,PETSC_MACHINE_EPSILON*100,PETSC_DEFAULT));

  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Matrix for placing the solution of the Lyapunov equation (an alias of V) */
    CHKERRQ(BVSetActiveColumns(V,0,nv));
    CHKERRQ(BVGetMat(V,&Y1));
    CHKERRQ(MatZeroEntries(Y1));
    CHKERRQ(MatCreateLRC(NULL,Y1,NULL,NULL,&Y));
    CHKERRQ(LMESetSolution(ctx->lme,Y));

    /* Solve the Lyapunov equation SY + YS' = -2*S*Z*S' */
    CHKERRQ(MatCreateLRC(NULL,Ux[idx],NULL,NULL,&C));
    CHKERRQ(LMESetRHS(ctx->lme,C));
    CHKERRQ(MatDestroy(&C));
    CHKERRQ(LMESolve(ctx->lme));
    CHKERRQ(BVRestoreMat(V,&Y1));
    CHKERRQ(MatDestroy(&Y));

    /* SVD of the solution: [Q,R]=qr(V); [U,Sigma,~]=svd(R) */
    CHKERRQ(DSSetDimensions(ctx->ds,nv,0,0));
    CHKERRQ(DSSVDSetDimensions(ctx->ds,nv));
    CHKERRQ(DSGetMat(ctx->ds,DS_MAT_A,&R));
    CHKERRQ(BVOrthogonalize(V,R));
    CHKERRQ(DSRestoreMat(ctx->ds,DS_MAT_A,&R));
    CHKERRQ(DSSetState(ctx->ds,DS_STATE_RAW));
    CHKERRQ(DSSolve(ctx->ds,s,NULL));

    /* Determine rank */
    rk = nv;
    for (i=1;i<nv;i++) if (PetscAbsScalar(s[i]/s[0])<PETSC_SQRT_MACHINE_EPSILON) {rk=i; break;}
    CHKERRQ(PetscInfo(eps,"The computed solution of the Lyapunov equation has rank %" PetscInt_FMT "\n",rk));
    rk = PetscMin(rk,ctx->rkc);
    CHKERRQ(DSGetMat(ctx->ds,DS_MAT_U,&U));
    CHKERRQ(BVMultInPlace(V,U,0,rk));
    CHKERRQ(BVSetActiveColumns(V,0,rk));
    CHKERRQ(MatDestroy(&U));

    /* Rank reduction */
    CHKERRQ(DSSetDimensions(ctx->ds,rk,0,0));
    CHKERRQ(DSSVDSetDimensions(ctx->ds,rk));
    CHKERRQ(DSGetMat(ctx->ds,DS_MAT_A,&W));
    CHKERRQ(BVMatProject(V,S,V,W));
    CHKERRQ(LyapIIBuildEigenMat(ctx->lme,W,&Op,&v0)); /* Op=A\B, A=kron(I,S)+kron(S,I), B=-2*kron(S,S) */
    CHKERRQ(EPSSetOperators(epsrr,Op,NULL));
    CHKERRQ(EPSSetInitialSpace(epsrr,1,&v0));
    CHKERRQ(EPSSolve(epsrr));
    CHKERRQ(MatDestroy(&W));
    CHKERRQ(EPSComputeVectors(epsrr));
    /* Copy first eigenvector, vec(A)=x */
    CHKERRQ(BVGetArray(epsrr->V,&xx));
    CHKERRQ(DSGetArray(ctx->ds,DS_MAT_A,&aa));
    for (i=0;i<rk;i++) CHKERRQ(PetscArraycpy(aa+i*ldds,xx+i*rk,rk));
    CHKERRQ(DSRestoreArray(ctx->ds,DS_MAT_A,&aa));
    CHKERRQ(BVRestoreArray(epsrr->V,&xx));
    CHKERRQ(DSSetState(ctx->ds,DS_STATE_RAW));
    /* Compute [U,Sigma,~] = svd(A), its rank should be 1 or 2 */
    CHKERRQ(DSSolve(ctx->ds,s,NULL));
    if (PetscAbsScalar(s[1]/s[0])<PETSC_SQRT_MACHINE_EPSILON) rk=1;
    else rk = 2;
    CHKERRQ(PetscInfo(eps,"The eigenvector has rank %" PetscInt_FMT "\n",rk));
    CHKERRQ(DSGetMat(ctx->ds,DS_MAT_U,&U));
    CHKERRQ(BVMultInPlace(V,U,0,rk));
    CHKERRQ(MatDestroy(&U));

    /* Save V in Ux */
    idx = (rk==2)?1:0;
    for (i=0;i<rk;i++) {
      CHKERRQ(BVGetColumn(V,i,&v));
      CHKERRQ(VecGetArray(v,&uu));
      CHKERRQ(MatDenseGetColumn(Ux[idx],i,&array));
      CHKERRQ(PetscArraycpy(array,uu,eps->nloc));
      CHKERRQ(MatDenseRestoreColumn(Ux[idx],&array));
      CHKERRQ(VecRestoreArray(v,&uu));
      CHKERRQ(BVRestoreColumn(V,i,&v));
    }

    /* Eigenpair approximation */
    CHKERRQ(BVGetColumn(V,0,&v));
    CHKERRQ(MatMult(S,v,z));
    CHKERRQ(VecDot(z,v,pM));
    CHKERRQ(BVRestoreColumn(V,0,&v));
    if (rk>1) {
      CHKERRQ(BVGetColumn(V,1,&w));
      CHKERRQ(VecDot(z,w,pM+1));
      CHKERRQ(MatMult(S,w,z));
      CHKERRQ(VecDot(z,w,pM+3));
      CHKERRQ(BVGetColumn(V,0,&v));
      CHKERRQ(VecDot(z,v,pM+2));
      CHKERRQ(BVRestoreColumn(V,0,&v));
      CHKERRQ(BVRestoreColumn(V,1,&w));
      CHKERRQ(EV2x2(pM,2,eigr,eigi,vec));
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,2,2,vec,&X));
      CHKERRQ(BVSetActiveColumns(V,0,rk));
      CHKERRQ(BVMultInPlace(V,X,0,rk));
      CHKERRQ(MatDestroy(&X));
#if !defined(PETSC_USE_COMPLEX)
      norm = eigr[0]*eigr[0]+eigi[0]*eigi[0];
      er = eigr[0]/norm; ei = -eigi[0]/norm;
#else
      er =1.0/eigr[0]; ei = 0.0;
#endif
    } else {
      eigr[0] = pM[0]; eigi[0] = 0.0;
      er = 1.0/eigr[0]; ei = 0.0;
    }
    CHKERRQ(BVGetColumn(V,0,&v));
    if (eigi[0]!=0.0) CHKERRQ(BVGetColumn(V,1,&w));
    else w = NULL;
    eps->eigr[eps->nconv] = eigr[0]; eps->eigi[eps->nconv] = eigi[0];
    CHKERRQ(EPSComputeResidualNorm_Private(eps,PETSC_FALSE,er,ei,v,w,eps->work,&norm));
    CHKERRQ(BVRestoreColumn(V,0,&v));
    if (w) CHKERRQ(BVRestoreColumn(V,1,&w));
    CHKERRQ((*eps->converged)(eps,er,ei,norm,&eps->errest[eps->nconv],eps->convergedctx));
    k = 0;
    if (eps->errest[eps->nconv]<eps->tol) {
      k++;
      if (rk==2) {
#if !defined (PETSC_USE_COMPLEX)
        eps->eigr[eps->nconv+k] = eigr[0]; eps->eigi[eps->nconv+k] = -eigi[0];
#else
        eps->eigr[eps->nconv+k] = PetscConj(eps->eigr[eps->nconv]);
#endif
        k++;
      }
      /* Store converged eigenpairs and vectors for deflation */
      for (i=0;i<k;i++) {
        CHKERRQ(BVGetColumn(V,i,&v));
        CHKERRQ(BVInsertVec(eps->V,eps->nconv+i,v));
        CHKERRQ(BVRestoreColumn(V,i,&v));
      }
      eps->nconv += k;
      CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv-rk,eps->nconv));
      CHKERRQ(BVOrthogonalize(eps->V,NULL));
      CHKERRQ(DSSetDimensions(eps->ds,eps->nconv,0,0));
      CHKERRQ(DSGetMat(eps->ds,DS_MAT_A,&W));
      CHKERRQ(BVMatProject(eps->V,matctx->S,eps->V,W));
      CHKERRQ(DSRestoreMat(eps->ds,DS_MAT_A,&W));
      if (eps->nconv<eps->nev) {
        idx = 0;
        CHKERRQ(BVSetRandomColumn(V,0));
        CHKERRQ(BVNormColumn(V,0,NORM_2,&norm));
        CHKERRQ(BVScaleColumn(V,0,1.0/norm));
        CHKERRQ(LyapIIBuildRHS(S,1,Ux[idx],V,eps->work));
      }
    } else {
      /* Prepare right-hand side */
      CHKERRQ(LyapIIBuildRHS(S,rk,Ux[idx],NULL,eps->work));
    }
    CHKERRQ((*eps->stopping)(eps,eps->its,eps->max_it,eps->nconv,eps->nev,&eps->reason,eps->stoppingctx));
    CHKERRQ(EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1));
  }
  CHKERRQ(STRestoreOperator(eps->st,&matctx->S));
  CHKERRQ(MatDestroy(&S));
  CHKERRQ(MatDestroy(&Ux[0]));
  CHKERRQ(MatDestroy(&Ux[1]));
  CHKERRQ(MatDestroy(&Op));
  CHKERRQ(VecDestroy(&v0));
  CHKERRQ(BVDestroy(&V));
  CHKERRQ(EPSDestroy(&epsrr));
  CHKERRQ(PetscFree(s));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_LyapII(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;
  PetscInt       k,array[2]={PETSC_DEFAULT,PETSC_DEFAULT};
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"EPS Lyapunov Inverse Iteration Options"));

    k = 2;
    CHKERRQ(PetscOptionsIntArray("-eps_lyapii_ranks","Ranks for Lyapunov equation (one or two comma-separated integers)","EPSLyapIISetRanks",array,&k,&flg));
    if (flg) CHKERRQ(EPSLyapIISetRanks(eps,array[0],array[1]));

  CHKERRQ(PetscOptionsTail());

  if (!ctx->lme) CHKERRQ(EPSLyapIIGetLME(eps,&ctx->lme));
  CHKERRQ(LMESetFromOptions(ctx->lme));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLyapIISetRanks_LyapII(EPS eps,PetscInt rkc,PetscInt rkl)
{
  EPS_LYAPII *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  if (rkc==PETSC_DEFAULT) rkc = 10;
  PetscCheck(rkc>1,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The compressed rank %" PetscInt_FMT " must be larger than 1",rkc);
  if (rkl==PETSC_DEFAULT) rkl = 3*rkc;
  PetscCheck(rkl>=rkc,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The Lyapunov rank %" PetscInt_FMT " cannot be smaller than the compressed rank %" PetscInt_FMT,rkl,rkc);
  if (rkc != ctx->rkc) {
    ctx->rkc   = rkc;
    eps->state = EPS_STATE_INITIAL;
  }
  if (rkl != ctx->rkl) {
    ctx->rkl   = rkl;
    eps->state = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSLyapIISetRanks - Set the ranks used in the solution of the Lyapunov equation.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  rkc - the compressed rank
-  rkl - the Lyapunov rank

   Options Database Key:
.  -eps_lyapii_ranks <rkc,rkl> - Sets the rank parameters

   Notes:
   Lyapunov inverse iteration needs to solve a large-scale Lyapunov equation
   at each iteration of the eigensolver. For this, an iterative solver (LME)
   is used, which requires to prescribe the rank of the solution matrix X. This
   is the meaning of parameter rkl. Later, this matrix is compressed into
   another matrix of rank rkc. If not provided, rkl is a small multiple of rkc.

   Level: intermediate

.seealso: EPSLyapIIGetRanks()
@*/
PetscErrorCode EPSLyapIISetRanks(EPS eps,PetscInt rkc,PetscInt rkl)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,rkc,2);
  PetscValidLogicalCollectiveInt(eps,rkl,3);
  CHKERRQ(PetscTryMethod(eps,"EPSLyapIISetRanks_C",(EPS,PetscInt,PetscInt),(eps,rkc,rkl)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLyapIIGetRanks_LyapII(EPS eps,PetscInt *rkc,PetscInt *rkl)
{
  EPS_LYAPII *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  if (rkc) *rkc = ctx->rkc;
  if (rkl) *rkl = ctx->rkl;
  PetscFunctionReturn(0);
}

/*@
   EPSLyapIIGetRanks - Return the rank values used for the Lyapunov step.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  rkc - the compressed rank
-  rkl - the Lyapunov rank

   Level: intermediate

.seealso: EPSLyapIISetRanks()
@*/
PetscErrorCode EPSLyapIIGetRanks(EPS eps,PetscInt *rkc,PetscInt *rkl)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  CHKERRQ(PetscUseMethod(eps,"EPSLyapIIGetRanks_C",(EPS,PetscInt*,PetscInt*),(eps,rkc,rkl)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLyapIISetLME_LyapII(EPS eps,LME lme)
{
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)lme));
  CHKERRQ(LMEDestroy(&ctx->lme));
  ctx->lme = lme;
  CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->lme));
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSLyapIISetLME - Associate a linear matrix equation solver object (LME) to the
   eigenvalue solver.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  lme - the linear matrix equation solver object

   Level: advanced

.seealso: EPSLyapIIGetLME()
@*/
PetscErrorCode EPSLyapIISetLME(EPS eps,LME lme)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(lme,LME_CLASSID,2);
  PetscCheckSameComm(eps,1,lme,2);
  CHKERRQ(PetscTryMethod(eps,"EPSLyapIISetLME_C",(EPS,LME),(eps,lme)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLyapIIGetLME_LyapII(EPS eps,LME *lme)
{
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  if (!ctx->lme) {
    CHKERRQ(LMECreate(PetscObjectComm((PetscObject)eps),&ctx->lme));
    CHKERRQ(LMESetOptionsPrefix(ctx->lme,((PetscObject)eps)->prefix));
    CHKERRQ(LMEAppendOptionsPrefix(ctx->lme,"eps_lyapii_"));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->lme,(PetscObject)eps,1));
    CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->lme));
  }
  *lme = ctx->lme;
  PetscFunctionReturn(0);
}

/*@
   EPSLyapIIGetLME - Retrieve the linear matrix equation solver object (LME)
   associated with the eigenvalue solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  lme - the linear matrix equation solver object

   Level: advanced

.seealso: EPSLyapIISetLME()
@*/
PetscErrorCode EPSLyapIIGetLME(EPS eps,LME *lme)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(lme,2);
  CHKERRQ(PetscUseMethod(eps,"EPSLyapIIGetLME_C",(EPS,LME*),(eps,lme)));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_LyapII(EPS eps,PetscViewer viewer)
{
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  ranks: for Lyapunov solver=%" PetscInt_FMT ", after compression=%" PetscInt_FMT "\n",ctx->rkl,ctx->rkc));
    if (!ctx->lme) CHKERRQ(EPSLyapIIGetLME(eps,&ctx->lme));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(LMEView(ctx->lme,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_LyapII(EPS eps)
{
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  if (!ctx->lme) CHKERRQ(LMEReset(ctx->lme));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_LyapII(EPS eps)
{
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(LMEDestroy(&ctx->lme));
  CHKERRQ(DSDestroy(&ctx->ds));
  CHKERRQ(PetscFree(eps->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIISetLME_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIIGetLME_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIISetRanks_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIIGetRanks_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetDefaultST_LyapII(EPS eps)
{
  PetscFunctionBegin;
  if (!((PetscObject)eps->st)->type_name) CHKERRQ(STSetType(eps->st,STSINVERT));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_LyapII(EPS eps)
{
  EPS_LYAPII     *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
  eps->data = (void*)ctx;

  eps->useds = PETSC_TRUE;

  eps->ops->solve          = EPSSolve_LyapII;
  eps->ops->setup          = EPSSetUp_LyapII;
  eps->ops->setupsort      = EPSSetUpSort_Default;
  eps->ops->setfromoptions = EPSSetFromOptions_LyapII;
  eps->ops->reset          = EPSReset_LyapII;
  eps->ops->destroy        = EPSDestroy_LyapII;
  eps->ops->view           = EPSView_LyapII;
  eps->ops->setdefaultst   = EPSSetDefaultST_LyapII;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_Schur;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIISetLME_C",EPSLyapIISetLME_LyapII));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIIGetLME_C",EPSLyapIIGetLME_LyapII));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIISetRanks_C",EPSLyapIISetRanks_LyapII));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIIGetRanks_C",EPSLyapIIGetRanks_LyapII));
  PetscFunctionReturn(0);
}
