/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Newton refinement for polynomial eigenproblems.

   References:

       [1] T. Betcke and D. Kressner, "Perturbation, extraction and refinement
           of invariant pairs for matrix polynomials", Linear Algebra Appl.
           435(3):514-536, 2011.

       [2] C. Campos and J.E. Roman, "Parallel iterative refinement in
           polynomial eigenvalue problems", Numer. Linear Algebra Appl. 23(4):
           730-745, 2016.
*/

#include <slepc/private/pepimpl.h>
#include <slepcblaslapack.h>

typedef struct {
  Mat          *A,M1;
  BV           V,M2,M3,W;
  PetscInt     k,nmat;
  PetscScalar  *fih,*work,*M4;
  PetscBLASInt *pM4;
  PetscBool    compM1;
  Vec          t;
} PEP_REFINE_MATSHELL;

typedef struct {
  Mat          E[2],M1;
  Vec          tN,ttN,t1,vseq;
  VecScatter   scatterctx;
  PetscBool    compM1;
  PetscInt     *map0,*map1,*idxg,*idxp;
  PetscSubcomm subc;
  VecScatter   scatter_sub;
  VecScatter   *scatter_id,*scatterp_id;
  Mat          *A;
  BV           V,W,M2,M3,Wt;
  PetscScalar  *M4,*w,*wt,*d,*dt;
  Vec          t,tg,Rv,Vi,tp,tpg;
  PetscInt     idx,*cols;
} PEP_REFINE_EXPLICIT;

static PetscErrorCode MatMult_FS(Mat M ,Vec x,Vec y)
{
  PEP_REFINE_MATSHELL *ctx;
  PetscInt            k,i;
  PetscScalar         *c;
  PetscBLASInt        k_,one=1,info;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(VecCopy(x,ctx->t));
  k    = ctx->k;
  c    = ctx->work;
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(MatMult(ctx->M1,x,y));
  CHKERRQ(VecConjugate(ctx->t));
  CHKERRQ(BVDotVec(ctx->M3,ctx->t,c));
  for (i=0;i<k;i++) c[i] = PetscConj(c[i]);
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N",&k_,&one,ctx->M4,&k_,ctx->pM4,c,&k_,&info));
  CHKERRQ(PetscFPTrapPop());
  SlepcCheckLapackInfo("getrs",info);
  CHKERRQ(BVMultVec(ctx->M2,-1.0,1.0,y,c));
  PetscFunctionReturn(0);
}

/*
  Evaluates the first d elements of the polynomial basis
  on a given matrix H which is considered to be triangular
*/
static PetscErrorCode PEPEvaluateBasisforMatrix(PEP pep,PetscInt nm,PetscInt k,PetscScalar *H,PetscInt ldh,PetscScalar *fH)
{
  PetscInt       i,j,ldfh=nm*k,off,nmat=pep->nmat;
  PetscReal      *a=pep->pbc,*b=pep->pbc+nmat,*g=pep->pbc+2*nmat,t;
  PetscScalar    corr=0.0,alpha,beta;
  PetscBLASInt   k_,ldh_,ldfh_;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(ldh,&ldh_));
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(ldfh,&ldfh_));
  CHKERRQ(PetscArrayzero(fH,nm*k*k));
  if (nm>0) for (j=0;j<k;j++) fH[j+j*ldfh] = 1.0;
  if (nm>1) {
    t = b[0]/a[0];
    off = k;
    for (j=0;j<k;j++) {
      for (i=0;i<k;i++) fH[off+i+j*ldfh] = H[i+j*ldh]/a[0];
      fH[j+j*ldfh] -= t;
    }
  }
  for (i=2;i<nm;i++) {
    off = i*k;
    if (i==2) {
      for (j=0;j<k;j++) {
        fH[off+j+j*ldfh] = 1.0;
        H[j+j*ldh] -= b[1];
      }
    } else {
      for (j=0;j<k;j++) {
        CHKERRQ(PetscArraycpy(fH+off+j*ldfh,fH+(i-2)*k+j*ldfh,k));
        H[j+j*ldh] += corr-b[i-1];
      }
    }
    corr  = b[i-1];
    beta  = -g[i-1]/a[i-1];
    alpha = 1/a[i-1];
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&alpha,H,&ldh_,fH+(i-1)*k,&ldfh_,&beta,fH+off,&ldfh_));
  }
  for (j=0;j<k;j++) H[j+j*ldh] += corr;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSysSetup_shell(PEP pep,PetscInt k,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar h,PEP_REFINE_MATSHELL *ctx)
{
  PetscScalar       *DHii,*T12,*Tr,*Ts,*array,s,ss,sone=1.0,zero=0.0,*M4=ctx->M4,t,*v,*T;
  const PetscScalar *m3,*m2;
  PetscInt          i,d,j,nmat=pep->nmat,lda=nmat*k,deg=nmat-1,nloc;
  PetscReal         *a=pep->pbc,*b=pep->pbc+nmat,*g=pep->pbc+2*nmat;
  PetscBLASInt      k_,lda_,lds_,nloc_,one=1,info;
  Mat               *A=ctx->A,Mk,M1=ctx->M1,P;
  BV                V=ctx->V,M2=ctx->M2,M3=ctx->M3,W=ctx->W;
  MatStructure      str;
  Vec               vc;

  PetscFunctionBegin;
  CHKERRQ(STGetMatStructure(pep->st,&str));
  CHKERRQ(PetscMalloc3(nmat*k*k,&T12,k*k,&Tr,PetscMax(k*k,nmat),&Ts));
  DHii = T12;
  CHKERRQ(PetscArrayzero(DHii,k*k*nmat));
  for (i=0;i<k;i++) DHii[k+i+i*lda] = 1.0/a[0];
  for (d=2;d<nmat;d++) {
    for (j=0;j<k;j++) {
      for (i=0;i<k;i++) {
        DHii[d*k+i+j*lda] = ((h-b[d-1])*DHii[(d-1)*k+i+j*lda]+fH[(d-1)*k+i+j*lda]-g[d-1]*DHii[(d-2)*k+i+j*lda])/(a[d-1]);
      }
    }
  }
  /* T11 */
  if (!ctx->compM1) {
    CHKERRQ(MatCopy(A[0],M1,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(PEPEvaluateBasis(pep,h,0,Ts,NULL));
    for (j=1;j<nmat;j++) CHKERRQ(MatAXPY(M1,Ts[j],A[j],str));
  }

  /* T22 */
  CHKERRQ(PetscBLASIntCast(lds,&lds_));
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(lda,&lda_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,S,&lds_,S,&lds_,&zero,Tr,&k_));
  for (i=1;i<deg;i++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,Tr,&k_,DHii+i*k,&lda_,&zero,Ts,&k_));
    s = (i==1)?0.0:1.0;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,fH+i*k,&lda_,Ts,&k_,&s,M4,&k_));
  }
  for (i=0;i<k;i++) for (j=0;j<i;j++) { t=M4[i+j*k];M4[i+j*k]=M4[j+i*k];M4[j+i*k]=t; }

  /* T12 */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&Mk));
  for (i=1;i<nmat;i++) {
    CHKERRQ(MatDenseGetArrayWrite(Mk,&array));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,S,&lds_,DHii+i*k,&lda_,&zero,array,&k_));
    CHKERRQ(MatDenseRestoreArrayWrite(Mk,&array));
    CHKERRQ(BVSetActiveColumns(W,0,k));
    CHKERRQ(BVMult(W,1.0,0.0,V,Mk));
    if (i==1) CHKERRQ(BVMatMult(W,A[i],M2));
    else {
      CHKERRQ(BVMatMult(W,A[i],M3)); /* using M3 as work space */
      CHKERRQ(BVMult(M2,1.0,1.0,M3,NULL));
    }
  }

  /* T21 */
  CHKERRQ(MatDenseGetArrayWrite(Mk,&array));
  for (i=1;i<deg;i++) {
    s = (i==1)?0.0:1.0;
    ss = PetscConj(fh[i]);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&ss,S,&lds_,fH+i*k,&lda_,&s,array,&k_));
  }
  CHKERRQ(MatDenseRestoreArrayWrite(Mk,&array));
  CHKERRQ(BVSetActiveColumns(M3,0,k));
  CHKERRQ(BVMult(M3,1.0,0.0,V,Mk));
  for (i=0;i<k;i++) {
    CHKERRQ(BVGetColumn(M3,i,&vc));
    CHKERRQ(VecConjugate(vc));
    CHKERRQ(BVRestoreColumn(M3,i,&vc));
  }
  CHKERRQ(MatDestroy(&Mk));
  CHKERRQ(PetscFree3(T12,Tr,Ts));

  CHKERRQ(VecGetLocalSize(ctx->t,&nloc));
  CHKERRQ(PetscBLASIntCast(nloc,&nloc_));
  CHKERRQ(PetscMalloc1(nloc*k,&T));
  CHKERRQ(KSPGetOperators(pep->refineksp,NULL,&P));
  if (!ctx->compM1) CHKERRQ(MatCopy(ctx->M1,P,SAME_NONZERO_PATTERN));
  CHKERRQ(BVGetArrayRead(ctx->M2,&m2));
  CHKERRQ(BVGetArrayRead(ctx->M3,&m3));
  CHKERRQ(VecGetArray(ctx->t,&v));
  for (i=0;i<nloc;i++) for (j=0;j<k;j++) T[j+i*k] = m3[i+j*nloc];
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&k_,&nloc_,ctx->M4,&k_,ctx->pM4,T,&k_,&info));
  CHKERRQ(PetscFPTrapPop());
  SlepcCheckLapackInfo("gesv",info);
  for (i=0;i<nloc;i++) v[i] = BLASdot_(&k_,m2+i,&nloc_,T+i*k,&one);
  CHKERRQ(VecRestoreArray(ctx->t,&v));
  CHKERRQ(BVRestoreArrayRead(ctx->M2,&m2));
  CHKERRQ(BVRestoreArrayRead(ctx->M3,&m3));
  CHKERRQ(MatDiagonalSet(P,ctx->t,ADD_VALUES));
  CHKERRQ(PetscFree(T));
  CHKERRQ(KSPSetUp(pep->refineksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSysSolve_shell(KSP ksp,PetscInt nmat,Vec Rv,PetscScalar *Rh,PetscInt k,Vec dVi,PetscScalar *dHi)
{
  PetscScalar         *t0;
  PetscBLASInt        k_,one=1,info,lda_;
  PetscInt            i,lda=nmat*k;
  Mat                 M;
  PEP_REFINE_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(KSPGetOperators(ksp,&M,NULL));
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(PetscCalloc1(k,&t0));
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  CHKERRQ(PetscBLASIntCast(lda,&lda_));
  CHKERRQ(PetscBLASIntCast(k,&k_));
  for (i=0;i<k;i++) t0[i] = Rh[i];
  PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N",&k_,&one,ctx->M4,&k_,ctx->pM4,t0,&k_,&info));
  SlepcCheckLapackInfo("getrs",info);
  CHKERRQ(BVMultVec(ctx->M2,-1.0,1.0,Rv,t0));
  CHKERRQ(KSPSolve(ksp,Rv,dVi));
  CHKERRQ(VecConjugate(dVi));
  CHKERRQ(BVDotVec(ctx->M3,dVi,dHi));
  CHKERRQ(VecConjugate(dVi));
  for (i=0;i<k;i++) dHi[i] = Rh[i]-PetscConj(dHi[i]);
  PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N",&k_,&one,ctx->M4,&k_,ctx->pM4,dHi,&k_,&info));
  SlepcCheckLapackInfo("getrs",info);
  CHKERRQ(PetscFPTrapPop());
  CHKERRQ(PetscFree(t0));
  PetscFunctionReturn(0);
}

/*
   Computes the residual P(H,V*S)*e_j for the polynomial
*/
static PetscErrorCode NRefRightSide(PetscInt nmat,PetscReal *pcf,Mat *A,PetscInt k,BV V,PetscScalar *S,PetscInt lds,PetscInt j,PetscScalar *H,PetscInt ldh,PetscScalar *fH,PetscScalar *DfH,PetscScalar *dH,BV dV,PetscScalar *dVS,PetscInt rds,Vec Rv,PetscScalar *Rh,BV W,Vec t)
{
  PetscScalar    *DS0,*DS1,*F,beta=0.0,sone=1.0,none=-1.0,tt=0.0,*h,zero=0.0,*Z,*c0;
  PetscReal      *a=pcf,*b=pcf+nmat,*g=b+nmat;
  PetscInt       i,ii,jj,lda;
  PetscBLASInt   lda_,k_,ldh_,lds_,nmat_,k2_,krds_,j_,one=1;
  Mat            M0;
  Vec            w;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc4(k*nmat,&h,k*k,&DS0,k*k,&DS1,k*k,&Z));
  lda = k*nmat;
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(lds,&lds_));
  CHKERRQ(PetscBLASIntCast(lda,&lda_));
  CHKERRQ(PetscBLASIntCast(nmat,&nmat_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&nmat_,&k_,&sone,S,&lds_,fH+j*lda,&k_,&zero,h,&k_));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,nmat,h,&M0));
  CHKERRQ(BVSetActiveColumns(W,0,nmat));
  CHKERRQ(BVMult(W,1.0,0.0,V,M0));
  CHKERRQ(MatDestroy(&M0));

  CHKERRQ(BVGetColumn(W,0,&w));
  CHKERRQ(MatMult(A[0],w,Rv));
  CHKERRQ(BVRestoreColumn(W,0,&w));
  for (i=1;i<nmat;i++) {
    CHKERRQ(BVGetColumn(W,i,&w));
    CHKERRQ(MatMult(A[i],w,t));
    CHKERRQ(BVRestoreColumn(W,i,&w));
    CHKERRQ(VecAXPY(Rv,1.0,t));
  }
  /* Update right-hand side */
  if (j) {
    CHKERRQ(PetscBLASIntCast(ldh,&ldh_));
    CHKERRQ(PetscArrayzero(Z,k*k));
    CHKERRQ(PetscArrayzero(DS0,k*k));
    CHKERRQ(PetscArraycpy(Z+(j-1)*k,dH+(j-1)*k,k));
    /* Update DfH */
    for (i=1;i<nmat;i++) {
      if (i>1) {
        beta = -g[i-1];
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,fH+(i-1)*k,&lda_,Z,&k_,&beta,DS0,&k_));
        tt += -b[i-1];
        for (ii=0;ii<k;ii++) H[ii+ii*ldh] += tt;
        tt = b[i-1];
        beta = 1.0/a[i-1];
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&beta,DS1,&k_,H,&ldh_,&beta,DS0,&k_));
        F = DS0; DS0 = DS1; DS1 = F;
      } else {
        CHKERRQ(PetscArrayzero(DS1,k*k));
        for (ii=0;ii<k;ii++) DS1[ii+(j-1)*k] = Z[ii+(j-1)*k]/a[0];
      }
      for (jj=j;jj<k;jj++) {
        for (ii=0;ii<k;ii++) DfH[k*i+ii+jj*lda] += DS1[ii+jj*k];
      }
    }
    for (ii=0;ii<k;ii++) H[ii+ii*ldh] += tt;
    /* Update right-hand side */
    CHKERRQ(PetscBLASIntCast(2*k,&k2_));
    CHKERRQ(PetscBLASIntCast(j,&j_));
    CHKERRQ(PetscBLASIntCast(k+rds,&krds_));
    c0 = DS0;
    CHKERRQ(PetscArrayzero(Rh,k));
    for (i=0;i<nmat;i++) {
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&krds_,&j_,&sone,dVS,&k2_,fH+j*lda+i*k,&one,&zero,h,&one));
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&k_,&k_,&sone,S,&lds_,DfH+i*k+j*lda,&one,&sone,h,&one));
      CHKERRQ(BVMultVec(V,1.0,0.0,t,h));
      CHKERRQ(BVSetActiveColumns(dV,0,rds));
      CHKERRQ(BVMultVec(dV,1.0,1.0,t,h+k));
      CHKERRQ(BVGetColumn(W,i,&w));
      CHKERRQ(MatMult(A[i],t,w));
      CHKERRQ(BVRestoreColumn(W,i,&w));
      if (i>0 && i<nmat-1) {
        PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&k_,&k_,&sone,S,&lds_,h,&one,&zero,c0,&one));
        PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&k_,&k_,&none,fH+i*k,&lda_,c0,&one,&sone,Rh,&one));
      }
    }

    for (i=0;i<nmat;i++) h[i] = -1.0;
    CHKERRQ(BVMultVec(W,1.0,1.0,Rv,h));
  }
  CHKERRQ(PetscFree4(h,DS0,DS1,Z));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSysSolve_mbe(PetscInt k,PetscInt sz,BV W,PetscScalar *w,BV Wt,PetscScalar *wt,PetscScalar *d,PetscScalar *dt,KSP ksp,BV T2,BV T3 ,PetscScalar *T4,PetscBool trans,Vec x1,PetscScalar *x2,Vec sol1,PetscScalar *sol2,Vec vw)
{
  PetscInt       i,j,incf,incc;
  PetscScalar    *y,*g,*xx2,*ww,y2,*dd;
  Vec            v,t,xx1;
  BV             WW,T;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc3(sz,&y,sz,&g,k,&xx2));
  if (trans) {
    WW = W; ww = w; dd = d; T = T3; incf = 0; incc = 1;
  } else {
    WW = Wt; ww = wt; dd = dt; T = T2; incf = 1; incc = 0;
  }
  xx1 = vw;
  CHKERRQ(VecCopy(x1,xx1));
  CHKERRQ(PetscArraycpy(xx2,x2,sz));
  CHKERRQ(PetscArrayzero(sol2,k));
  for (i=sz-1;i>=0;i--) {
    CHKERRQ(BVGetColumn(WW,i,&v));
    CHKERRQ(VecConjugate(v));
    CHKERRQ(VecDot(xx1,v,y+i));
    CHKERRQ(VecConjugate(v));
    CHKERRQ(BVRestoreColumn(WW,i,&v));
    for (j=0;j<i;j++) y[i] += ww[j+i*k]*xx2[j];
    y[i] = -(y[i]-xx2[i])/dd[i];
    CHKERRQ(BVGetColumn(T,i,&t));
    CHKERRQ(VecAXPY(xx1,-y[i],t));
    CHKERRQ(BVRestoreColumn(T,i,&t));
    for (j=0;j<=i;j++) xx2[j] -= y[i]*T4[j*incf+incc*i+(i*incf+incc*j)*k];
    g[i] = xx2[i];
  }
  if (trans) CHKERRQ(KSPSolveTranspose(ksp,xx1,sol1));
  else CHKERRQ(KSPSolve(ksp,xx1,sol1));
  if (trans) {
    WW = Wt; ww = wt; dd = dt; T = T2; incf = 1; incc = 0;
  } else {
    WW = W; ww = w; dd = d; T = T3; incf = 0; incc = 1;
  }
  for (i=0;i<sz;i++) {
    CHKERRQ(BVGetColumn(T,i,&t));
    CHKERRQ(VecConjugate(t));
    CHKERRQ(VecDot(sol1,t,&y2));
    CHKERRQ(VecConjugate(t));
    CHKERRQ(BVRestoreColumn(T,i,&t));
    for (j=0;j<i;j++) y2 += sol2[j]*T4[j*incf+incc*i+(i*incf+incc*j)*k];
    y2 = (g[i]-y2)/dd[i];
    CHKERRQ(BVGetColumn(WW,i,&v));
    CHKERRQ(VecAXPY(sol1,-y2,v));
    for (j=0;j<i;j++) sol2[j] -= ww[j+i*k]*y2;
    sol2[i] = y[i]+y2;
    CHKERRQ(BVRestoreColumn(WW,i,&v));
  }
  CHKERRQ(PetscFree3(y,g,xx2));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSysSetup_mbe(PEP pep,PetscInt k,KSP ksp,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar h,BV V,PEP_REFINE_EXPLICIT *matctx)
{
  PetscInt       i,j,l,nmat=pep->nmat,lda=nmat*k,deg=nmat-1;
  Mat            M1=matctx->M1,*A,*At,Mk;
  PetscReal      *a=pep->pbc,*b=pep->pbc+nmat,*g=pep->pbc+2*nmat;
  PetscScalar    s,ss,*DHii,*T12,*array,*Ts,*Tr,*M4=matctx->M4,sone=1.0,zero=0.0;
  PetscScalar    *w=matctx->w,*wt=matctx->wt,*d=matctx->d,*dt=matctx->dt;
  PetscBLASInt   lds_,lda_,k_;
  MatStructure   str;
  PetscBool      flg;
  BV             M2=matctx->M2,M3=matctx->M3,W=matctx->W,Wt=matctx->Wt;
  Vec            vc,vc2;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc3(nmat*k*k,&T12,k*k,&Tr,PetscMax(k*k,nmat),&Ts));
  CHKERRQ(STGetMatStructure(pep->st,&str));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (flg) {
    CHKERRQ(PetscMalloc1(pep->nmat,&At));
    for (i=0;i<pep->nmat;i++) CHKERRQ(STGetMatrixTransformed(pep->st,i,&At[i]));
  } else At = pep->A;
  if (matctx->subc) A = matctx->A;
  else A = At;
  /* Form the explicit system matrix */
  DHii = T12;
  CHKERRQ(PetscArrayzero(DHii,k*k*nmat));
  for (i=0;i<k;i++) DHii[k+i+i*lda] = 1.0/a[0];
  for (l=2;l<nmat;l++) {
    for (j=0;j<k;j++) {
      for (i=0;i<k;i++) {
        DHii[l*k+i+j*lda] = ((h-b[l-1])*DHii[(l-1)*k+i+j*lda]+fH[(l-1)*k+i+j*lda]-g[l-1]*DHii[(l-2)*k+i+j*lda])/a[l-1];
      }
    }
  }

  /* T11 */
  if (!matctx->compM1) {
    CHKERRQ(MatCopy(A[0],M1,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(PEPEvaluateBasis(pep,h,0,Ts,NULL));
    for (j=1;j<nmat;j++) CHKERRQ(MatAXPY(M1,Ts[j],A[j],str));
  }

  /* T22 */
  CHKERRQ(PetscBLASIntCast(lds,&lds_));
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(lda,&lda_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,S,&lds_,S,&lds_,&zero,Tr,&k_));
  for (i=1;i<deg;i++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,Tr,&k_,DHii+i*k,&lda_,&zero,Ts,&k_));
    s = (i==1)?0.0:1.0;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,fH+i*k,&lda_,Ts,&k_,&s,M4,&k_));
  }

  /* T12 */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&Mk));
  for (i=1;i<nmat;i++) {
    CHKERRQ(MatDenseGetArrayWrite(Mk,&array));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,S,&lds_,DHii+i*k,&lda_,&zero,array,&k_));
    CHKERRQ(MatDenseRestoreArrayWrite(Mk,&array));
    CHKERRQ(BVSetActiveColumns(W,0,k));
    CHKERRQ(BVMult(W,1.0,0.0,V,Mk));
    if (i==1) CHKERRQ(BVMatMult(W,A[i],M2));
    else {
      CHKERRQ(BVMatMult(W,A[i],M3)); /* using M3 as work space */
      CHKERRQ(BVMult(M2,1.0,1.0,M3,NULL));
    }
  }

  /* T21 */
  CHKERRQ(MatDenseGetArrayWrite(Mk,&array));
  for (i=1;i<deg;i++) {
    s = (i==1)?0.0:1.0;
    ss = PetscConj(fh[i]);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&ss,S,&lds_,fH+i*k,&lda_,&s,array,&k_));
  }
  CHKERRQ(MatDenseRestoreArrayWrite(Mk,&array));
  CHKERRQ(BVSetActiveColumns(M3,0,k));
  CHKERRQ(BVMult(M3,1.0,0.0,V,Mk));
  for (i=0;i<k;i++) {
    CHKERRQ(BVGetColumn(M3,i,&vc));
    CHKERRQ(VecConjugate(vc));
    CHKERRQ(BVRestoreColumn(M3,i,&vc));
  }

  CHKERRQ(PEP_KSPSetOperators(ksp,M1,M1));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(MatDestroy(&Mk));

  /* Set up for BEMW */
  for (i=0;i<k;i++) {
    CHKERRQ(BVGetColumn(M2,i,&vc));
    CHKERRQ(BVGetColumn(W,i,&vc2));
    CHKERRQ(NRefSysSolve_mbe(k,i,W,w,Wt,wt,d,dt,ksp,M2,M3,M4,PETSC_FALSE,vc,M4+i*k,vc2,w+i*k,matctx->t));
    CHKERRQ(BVRestoreColumn(M2,i,&vc));
    CHKERRQ(BVGetColumn(M3,i,&vc));
    CHKERRQ(VecConjugate(vc));
    CHKERRQ(VecDot(vc2,vc,&d[i]));
    CHKERRQ(VecConjugate(vc));
    CHKERRQ(BVRestoreColumn(M3,i,&vc));
    for (j=0;j<i;j++) d[i] += M4[i+j*k]*w[j+i*k];
    d[i] = M4[i+i*k]-d[i];
    CHKERRQ(BVRestoreColumn(W,i,&vc2));

    CHKERRQ(BVGetColumn(M3,i,&vc));
    CHKERRQ(BVGetColumn(Wt,i,&vc2));
    for (j=0;j<=i;j++) Ts[j] = M4[i+j*k];
    CHKERRQ(NRefSysSolve_mbe(k,i,W,w,Wt,wt,d,dt,ksp,M2,M3,M4,PETSC_TRUE,vc,Ts,vc2,wt+i*k,matctx->t));
    CHKERRQ(BVRestoreColumn(M3,i,&vc));
    CHKERRQ(BVGetColumn(M2,i,&vc));
    CHKERRQ(VecConjugate(vc2));
    CHKERRQ(VecDot(vc,vc2,&dt[i]));
    CHKERRQ(VecConjugate(vc2));
    CHKERRQ(BVRestoreColumn(M2,i,&vc));
    for (j=0;j<i;j++) dt[i] += M4[j+i*k]*wt[j+i*k];
    dt[i] = M4[i+i*k]-dt[i];
    CHKERRQ(BVRestoreColumn(Wt,i,&vc2));
  }

  if (flg) CHKERRQ(PetscFree(At));
  CHKERRQ(PetscFree3(T12,Tr,Ts));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSysSetup_explicit(PEP pep,PetscInt k,KSP ksp,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar h,BV V,PEP_REFINE_EXPLICIT *matctx,BV W)
{
  PetscInt          i,j,d,n,n0,m0,n1,m1,nmat=pep->nmat,lda=nmat*k,deg=nmat-1;
  PetscInt          *idxg=matctx->idxg,*idxp=matctx->idxp,idx,ncols;
  Mat               M,*E=matctx->E,*A,*At,Mk,Md;
  PetscReal         *a=pep->pbc,*b=pep->pbc+nmat,*g=pep->pbc+2*nmat;
  PetscScalar       s,ss,*DHii,*T22,*T21,*T12,*Ts,*Tr,*array,*ts,sone=1.0,zero=0.0;
  PetscBLASInt      lds_,lda_,k_;
  const PetscInt    *idxmc;
  const PetscScalar *valsc,*carray;
  MatStructure      str;
  Vec               vc,vc0;
  PetscBool         flg;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc5(k*k,&T22,k*k,&T21,nmat*k*k,&T12,k*k,&Tr,k*k,&Ts));
  CHKERRQ(STGetMatStructure(pep->st,&str));
  CHKERRQ(KSPGetOperators(ksp,&M,NULL));
  CHKERRQ(MatGetOwnershipRange(E[1],&n1,&m1));
  CHKERRQ(MatGetOwnershipRange(E[0],&n0,&m0));
  CHKERRQ(MatGetOwnershipRange(M,&n,NULL));
  CHKERRQ(PetscMalloc1(nmat,&ts));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (flg) {
    CHKERRQ(PetscMalloc1(pep->nmat,&At));
    for (i=0;i<pep->nmat;i++) CHKERRQ(STGetMatrixTransformed(pep->st,i,&At[i]));
  } else At = pep->A;
  if (matctx->subc) A = matctx->A;
  else A = At;
  /* Form the explicit system matrix */
  DHii = T12;
  CHKERRQ(PetscArrayzero(DHii,k*k*nmat));
  for (i=0;i<k;i++) DHii[k+i+i*lda] = 1.0/a[0];
  for (d=2;d<nmat;d++) {
    for (j=0;j<k;j++) {
      for (i=0;i<k;i++) {
        DHii[d*k+i+j*lda] = ((h-b[d-1])*DHii[(d-1)*k+i+j*lda]+fH[(d-1)*k+i+j*lda]-g[d-1]*DHii[(d-2)*k+i+j*lda])/a[d-1];
      }
    }
  }

  /* T11 */
  if (!matctx->compM1) {
    CHKERRQ(MatCopy(A[0],E[0],DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(PEPEvaluateBasis(pep,h,0,Ts,NULL));
    for (j=1;j<nmat;j++) CHKERRQ(MatAXPY(E[0],Ts[j],A[j],str));
  }
  for (i=n0;i<m0;i++) {
    CHKERRQ(MatGetRow(E[0],i,&ncols,&idxmc,&valsc));
    idx = n+i-n0;
    for (j=0;j<ncols;j++) {
      idxg[j] = matctx->map0[idxmc[j]];
    }
    CHKERRQ(MatSetValues(M,1,&idx,ncols,idxg,valsc,INSERT_VALUES));
    CHKERRQ(MatRestoreRow(E[0],i,&ncols,&idxmc,&valsc));
  }

  /* T22 */
  CHKERRQ(PetscBLASIntCast(lds,&lds_));
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(lda,&lda_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,S,&lds_,S,&lds_,&zero,Tr,&k_));
  for (i=1;i<deg;i++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,Tr,&k_,DHii+i*k,&lda_,&zero,Ts,&k_));
    s = (i==1)?0.0:1.0;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,fH+i*k,&lda_,Ts,&k_,&s,T22,&k_));
  }
  for (j=0;j<k;j++) idxp[j] = matctx->map1[j];
  for (i=0;i<m1-n1;i++) {
    idx = n+m0-n0+i;
    for (j=0;j<k;j++) {
      Tr[j] = T22[n1+i+j*k];
    }
    CHKERRQ(MatSetValues(M,1,&idx,k,idxp,Tr,INSERT_VALUES));
  }

  /* T21 */
  for (i=1;i<deg;i++) {
    s = (i==1)?0.0:1.0;
    ss = PetscConj(fh[i]);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&ss,S,&lds_,fH+i*k,&lda_,&s,T21,&k_));
  }
  CHKERRQ(BVSetActiveColumns(W,0,k));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,T21,&Mk));
  CHKERRQ(BVMult(W,1.0,0.0,V,Mk));
  for (i=0;i<k;i++) {
    CHKERRQ(BVGetColumn(W,i,&vc));
    CHKERRQ(VecConjugate(vc));
    CHKERRQ(VecGetArrayRead(vc,&carray));
    idx = matctx->map1[i];
    CHKERRQ(MatSetValues(M,1,&idx,m0-n0,matctx->map0+n0,carray,INSERT_VALUES));
    CHKERRQ(VecRestoreArrayRead(vc,&carray));
    CHKERRQ(BVRestoreColumn(W,i,&vc));
  }

  /* T12 */
  for (i=1;i<nmat;i++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,S,&lds_,DHii+i*k,&lda_,&zero,Ts,&k_));
    for (j=0;j<k;j++) CHKERRQ(PetscArraycpy(T12+i*k+j*lda,Ts+j*k,k));
  }
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,nmat-1,NULL,&Md));
  for (i=0;i<nmat;i++) ts[i] = 1.0;
  for (j=0;j<k;j++) {
    CHKERRQ(MatDenseGetArrayWrite(Md,&array));
    CHKERRQ(PetscArraycpy(array,T12+k+j*lda,(nmat-1)*k));
    CHKERRQ(MatDenseRestoreArrayWrite(Md,&array));
    CHKERRQ(BVSetActiveColumns(W,0,nmat-1));
    CHKERRQ(BVMult(W,1.0,0.0,V,Md));
    for (i=nmat-1;i>0;i--) {
      CHKERRQ(BVGetColumn(W,i-1,&vc0));
      CHKERRQ(BVGetColumn(W,i,&vc));
      CHKERRQ(MatMult(A[i],vc0,vc));
      CHKERRQ(BVRestoreColumn(W,i-1,&vc0));
      CHKERRQ(BVRestoreColumn(W,i,&vc));
    }
    CHKERRQ(BVSetActiveColumns(W,1,nmat));
    CHKERRQ(BVGetColumn(W,0,&vc0));
    CHKERRQ(BVMultVec(W,1.0,0.0,vc0,ts));
    CHKERRQ(VecGetArrayRead(vc0,&carray));
    idx = matctx->map1[j];
    CHKERRQ(MatSetValues(M,m0-n0,matctx->map0+n0,1,&idx,carray,INSERT_VALUES));
    CHKERRQ(VecRestoreArrayRead(vc0,&carray));
    CHKERRQ(BVRestoreColumn(W,0,&vc0));
  }
  CHKERRQ(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PEP_KSPSetOperators(ksp,M,M));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(PetscFree(ts));
  CHKERRQ(MatDestroy(&Mk));
  CHKERRQ(MatDestroy(&Md));
  if (flg) CHKERRQ(PetscFree(At));
  CHKERRQ(PetscFree5(T22,T21,T12,Tr,Ts));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSysSolve_explicit(PetscInt k,KSP ksp,Vec Rv,PetscScalar *Rh,Vec dVi,PetscScalar *dHi,PEP_REFINE_EXPLICIT *matctx)
{
  PetscInt          n0,m0,n1,m1,i;
  PetscScalar       *arrayV;
  const PetscScalar *array;

  PetscFunctionBegin;
  CHKERRQ(MatGetOwnershipRange(matctx->E[1],&n1,&m1));
  CHKERRQ(MatGetOwnershipRange(matctx->E[0],&n0,&m0));

  /* Right side */
  CHKERRQ(VecGetArrayRead(Rv,&array));
  CHKERRQ(VecSetValues(matctx->tN,m0-n0,matctx->map0+n0,array,INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(Rv,&array));
  CHKERRQ(VecSetValues(matctx->tN,m1-n1,matctx->map1+n1,Rh+n1,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(matctx->tN));
  CHKERRQ(VecAssemblyEnd(matctx->tN));

  /* Solve */
  CHKERRQ(KSPSolve(ksp,matctx->tN,matctx->ttN));

  /* Retrieve solution */
  CHKERRQ(VecGetArray(dVi,&arrayV));
  CHKERRQ(VecGetArrayRead(matctx->ttN,&array));
  CHKERRQ(PetscArraycpy(arrayV,array,m0-n0));
  CHKERRQ(VecRestoreArray(dVi,&arrayV));
  if (!matctx->subc) {
    CHKERRQ(VecGetArray(matctx->t1,&arrayV));
    for (i=0;i<m1-n1;i++) arrayV[i] =  array[m0-n0+i];
    CHKERRQ(VecRestoreArray(matctx->t1,&arrayV));
    CHKERRQ(VecRestoreArrayRead(matctx->ttN,&array));
    CHKERRQ(VecScatterBegin(matctx->scatterctx,matctx->t1,matctx->vseq,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(matctx->scatterctx,matctx->t1,matctx->vseq,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecGetArrayRead(matctx->vseq,&array));
    for (i=0;i<k;i++) dHi[i] = array[i];
    CHKERRQ(VecRestoreArrayRead(matctx->vseq,&array));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSysIter(PetscInt i,PEP pep,PetscInt k,KSP ksp,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar *H,PetscInt ldh,Vec Rv,PetscScalar *Rh,BV V,Vec dVi,PetscScalar *dHi,PEP_REFINE_EXPLICIT *matctx,BV W)
{
  PetscInt            j,m,lda=pep->nmat*k,n0,m0,idx;
  PetscMPIInt         root,len;
  PetscScalar         *array2,h;
  const PetscScalar   *array;
  Vec                 R,Vi;
  PEP_REFINE_MATSHELL *ctx;
  Mat                 M;

  PetscFunctionBegin;
  if (!matctx || !matctx->subc) {
    for (j=0;j<pep->nmat;j++) fh[j] = fH[j*k+i+i*lda];
    h   = H[i+i*ldh];
    idx = i;
    R   = Rv;
    Vi  = dVi;
    switch (pep->scheme) {
    case PEP_REFINE_SCHEME_EXPLICIT:
      CHKERRQ(NRefSysSetup_explicit(pep,k,ksp,fH,S,lds,fh,h,V,matctx,W));
      matctx->compM1 = PETSC_FALSE;
      break;
    case PEP_REFINE_SCHEME_MBE:
      CHKERRQ(NRefSysSetup_mbe(pep,k,ksp,fH,S,lds,fh,h,V,matctx));
      matctx->compM1 = PETSC_FALSE;
      break;
    case PEP_REFINE_SCHEME_SCHUR:
      CHKERRQ(KSPGetOperators(ksp,&M,NULL));
      CHKERRQ(MatShellGetContext(M,&ctx));
      CHKERRQ(NRefSysSetup_shell(pep,k,fH,S,lds,fh,h,ctx));
      ctx->compM1 = PETSC_FALSE;
      break;
    }
  } else {
    if (i%matctx->subc->n==0 && (idx=i+matctx->subc->color)<k) {
      for (j=0;j<pep->nmat;j++) fh[j] = fH[j*k+idx+idx*lda];
      h = H[idx+idx*ldh];
      matctx->idx = idx;
      switch (pep->scheme) {
      case PEP_REFINE_SCHEME_EXPLICIT:
        CHKERRQ(NRefSysSetup_explicit(pep,k,ksp,fH,S,lds,fh,h,matctx->V,matctx,matctx->W));
        matctx->compM1 = PETSC_FALSE;
        break;
      case PEP_REFINE_SCHEME_MBE:
        CHKERRQ(NRefSysSetup_mbe(pep,k,ksp,fH,S,lds,fh,h,matctx->V,matctx));
        matctx->compM1 = PETSC_FALSE;
        break;
      case PEP_REFINE_SCHEME_SCHUR:
        break;
      }
    } else idx = matctx->idx;
    CHKERRQ(VecScatterBegin(matctx->scatter_id[i%matctx->subc->n],Rv,matctx->tg,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(matctx->scatter_id[i%matctx->subc->n],Rv,matctx->tg,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecGetArrayRead(matctx->tg,&array));
    CHKERRQ(VecPlaceArray(matctx->t,array));
    CHKERRQ(VecCopy(matctx->t,matctx->Rv));
    CHKERRQ(VecResetArray(matctx->t));
    CHKERRQ(VecRestoreArrayRead(matctx->tg,&array));
    R  = matctx->Rv;
    Vi = matctx->Vi;
  }
  if (idx==i && idx<k) {
    switch (pep->scheme) {
      case PEP_REFINE_SCHEME_EXPLICIT:
        CHKERRQ(NRefSysSolve_explicit(k,ksp,R,Rh,Vi,dHi,matctx));
        break;
      case PEP_REFINE_SCHEME_MBE:
        CHKERRQ(NRefSysSolve_mbe(k,k,matctx->W,matctx->w,matctx->Wt,matctx->wt,matctx->d,matctx->dt,ksp,matctx->M2,matctx->M3 ,matctx->M4,PETSC_FALSE,R,Rh,Vi,dHi,matctx->t));
        break;
      case PEP_REFINE_SCHEME_SCHUR:
        CHKERRQ(NRefSysSolve_shell(ksp,pep->nmat,R,Rh,k,Vi,dHi));
        break;
    }
  }
  if (matctx && matctx->subc) {
    CHKERRQ(VecGetLocalSize(Vi,&m));
    CHKERRQ(VecGetArrayRead(Vi,&array));
    CHKERRQ(VecGetArray(matctx->tg,&array2));
    CHKERRQ(PetscArraycpy(array2,array,m));
    CHKERRQ(VecRestoreArray(matctx->tg,&array2));
    CHKERRQ(VecRestoreArrayRead(Vi,&array));
    CHKERRQ(VecScatterBegin(matctx->scatter_id[i%matctx->subc->n],matctx->tg,dVi,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(matctx->scatter_id[i%matctx->subc->n],matctx->tg,dVi,INSERT_VALUES,SCATTER_REVERSE));
    switch (pep->scheme) {
    case PEP_REFINE_SCHEME_EXPLICIT:
      CHKERRQ(MatGetOwnershipRange(matctx->E[0],&n0,&m0));
      CHKERRQ(VecGetArrayRead(matctx->ttN,&array));
      CHKERRQ(VecPlaceArray(matctx->tp,array+m0-n0));
      CHKERRQ(VecScatterBegin(matctx->scatterp_id[i%matctx->subc->n],matctx->tp,matctx->tpg,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(matctx->scatterp_id[i%matctx->subc->n],matctx->tp,matctx->tpg,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecResetArray(matctx->tp));
      CHKERRQ(VecRestoreArrayRead(matctx->ttN,&array));
      CHKERRQ(VecGetArrayRead(matctx->tpg,&array));
      for (j=0;j<k;j++) dHi[j] = array[j];
      CHKERRQ(VecRestoreArrayRead(matctx->tpg,&array));
      break;
     case PEP_REFINE_SCHEME_MBE:
      root = 0;
      for (j=0;j<i%matctx->subc->n;j++) root += matctx->subc->subsize[j];
      CHKERRQ(PetscMPIIntCast(k,&len));
      CHKERRMPI(MPI_Bcast(dHi,len,MPIU_SCALAR,root,matctx->subc->dupparent));
      break;
    case PEP_REFINE_SCHEME_SCHUR:
      break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPNRefForwardSubstitution(PEP pep,PetscInt k,PetscScalar *S,PetscInt lds,PetscScalar *H,PetscInt ldh,PetscScalar *fH,BV dV,PetscScalar *dVS,PetscInt *rds,PetscScalar *dH,PetscInt lddh,KSP ksp,PEP_REFINE_EXPLICIT *matctx)
{
  PetscInt            i,nmat=pep->nmat,lda=nmat*k;
  PetscScalar         *fh,*Rh,*DfH;
  PetscReal           norm;
  BV                  W;
  Vec                 Rv,t,dvi;
  PEP_REFINE_MATSHELL *ctx;
  Mat                 M,*At;
  PetscBool           flg,lindep;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc2(nmat*k*k,&DfH,k,&Rh));
  *rds = 0;
  CHKERRQ(BVCreateVec(pep->V,&Rv));
  switch (pep->scheme) {
  case PEP_REFINE_SCHEME_EXPLICIT:
    CHKERRQ(BVCreateVec(pep->V,&t));
    CHKERRQ(BVDuplicateResize(pep->V,PetscMax(k,nmat),&W));
    CHKERRQ(PetscMalloc1(nmat,&fh));
    break;
  case PEP_REFINE_SCHEME_MBE:
    if (matctx->subc) {
      CHKERRQ(BVCreateVec(pep->V,&t));
      CHKERRQ(BVDuplicateResize(pep->V,PetscMax(k,nmat),&W));
    } else {
      W = matctx->W;
      CHKERRQ(PetscObjectReference((PetscObject)W));
      t = matctx->t;
      CHKERRQ(PetscObjectReference((PetscObject)t));
    }
    CHKERRQ(BVScale(matctx->W,0.0));
    CHKERRQ(BVScale(matctx->Wt,0.0));
    CHKERRQ(BVScale(matctx->M2,0.0));
    CHKERRQ(BVScale(matctx->M3,0.0));
    CHKERRQ(PetscMalloc1(nmat,&fh));
    break;
  case PEP_REFINE_SCHEME_SCHUR:
    CHKERRQ(KSPGetOperators(ksp,&M,NULL));
    CHKERRQ(MatShellGetContext(M,&ctx));
    CHKERRQ(BVCreateVec(pep->V,&t));
    CHKERRQ(BVDuplicateResize(pep->V,PetscMax(k,nmat),&W));
    fh = ctx->fih;
    break;
  }
  CHKERRQ(PetscArrayzero(dVS,2*k*k));
  CHKERRQ(PetscArrayzero(DfH,lda*k));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (flg) {
    CHKERRQ(PetscMalloc1(pep->nmat,&At));
    for (i=0;i<pep->nmat;i++) CHKERRQ(STGetMatrixTransformed(pep->st,i,&At[i]));
  } else At = pep->A;

  /* Main loop for computing the i-th columns of dX and dS */
  for (i=0;i<k;i++) {
    /* Compute and update i-th column of the right hand side */
    CHKERRQ(PetscArrayzero(Rh,k));
    CHKERRQ(NRefRightSide(nmat,pep->pbc,At,k,pep->V,S,lds,i,H,ldh,fH,DfH,dH,dV,dVS,*rds,Rv,Rh,W,t));

    /* Update and solve system */
    CHKERRQ(BVGetColumn(dV,i,&dvi));
    CHKERRQ(NRefSysIter(i,pep,k,ksp,fH,S,lds,fh,H,ldh,Rv,Rh,pep->V,dvi,dH+i*k,matctx,W));
    /* Orthogonalize computed solution */
    CHKERRQ(BVOrthogonalizeVec(pep->V,dvi,dVS+i*2*k,&norm,&lindep));
    CHKERRQ(BVRestoreColumn(dV,i,&dvi));
    if (!lindep) {
      CHKERRQ(BVOrthogonalizeColumn(dV,i,dVS+k+i*2*k,&norm,&lindep));
      if (!lindep) {
        dVS[k+i+i*2*k] = norm;
        CHKERRQ(BVScaleColumn(dV,i,1.0/norm));
        (*rds)++;
      }
    }
  }
  CHKERRQ(BVSetActiveColumns(dV,0,*rds));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(VecDestroy(&Rv));
  CHKERRQ(BVDestroy(&W));
  if (flg) CHKERRQ(PetscFree(At));
  CHKERRQ(PetscFree2(DfH,Rh));
  if (pep->scheme!=PEP_REFINE_SCHEME_SCHUR) CHKERRQ(PetscFree(fh));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefOrthogStep(PEP pep,PetscInt k,PetscScalar *H,PetscInt ldh,PetscScalar *fH,PetscScalar *S,PetscInt lds)
{
  PetscInt       j,nmat=pep->nmat,deg=nmat-1,lda=nmat*k,ldg;
  PetscScalar    *G,*tau,sone=1.0,zero=0.0,*work;
  PetscBLASInt   lds_,k_,ldh_,info,ldg_,lda_;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc3(k,&tau,k,&work,deg*k*k,&G));
  CHKERRQ(PetscBLASIntCast(lds,&lds_));
  CHKERRQ(PetscBLASIntCast(lda,&lda_));
  CHKERRQ(PetscBLASIntCast(k,&k_));

  /* Form auxiliary matrix for the orthogonalization step */
  ldg = deg*k;
  CHKERRQ(PEPEvaluateBasisforMatrix(pep,nmat,k,H,ldh,fH));
  CHKERRQ(PetscBLASIntCast(ldg,&ldg_));
  CHKERRQ(PetscBLASIntCast(ldh,&ldh_));
  for (j=0;j<deg;j++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,S,&lds_,fH+j*k,&lda_,&zero,G+j*k,&ldg_));
  }
  /* Orthogonalize and update S */
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&ldg_,&k_,G,&ldg_,tau,work,&k_,&info));
  CHKERRQ(PetscFPTrapPop());
  SlepcCheckLapackInfo("geqrf",info);
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&k_,&k_,&sone,G,&ldg_,S,&lds_));

  /* Update H */
  PetscStackCallBLAS("BLAStrmm",BLAStrmm_("L","U","N","N",&k_,&k_,&sone,G,&ldg_,H,&ldh_));
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&k_,&k_,&sone,G,&ldg_,H,&ldh_));
  CHKERRQ(PetscFree3(tau,work,G));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPNRefUpdateInvPair(PEP pep,PetscInt k,PetscScalar *H,PetscInt ldh,PetscScalar *fH,PetscScalar *dH,PetscScalar *S,PetscInt lds,BV dV,PetscScalar *dVS,PetscInt rds)
{
  PetscInt       i,j,nmat=pep->nmat,lda=nmat*k;
  PetscScalar    *tau,*array,*work;
  PetscBLASInt   lds_,k_,lda_,ldh_,kdrs_,info,k2_;
  Mat            M0;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc2(k,&tau,k,&work));
  CHKERRQ(PetscBLASIntCast(lds,&lds_));
  CHKERRQ(PetscBLASIntCast(lda,&lda_));
  CHKERRQ(PetscBLASIntCast(ldh,&ldh_));
  CHKERRQ(PetscBLASIntCast(k,&k_));
  CHKERRQ(PetscBLASIntCast(2*k,&k2_));
  CHKERRQ(PetscBLASIntCast((k+rds),&kdrs_));
  /* Update H */
  for (j=0;j<k;j++) {
    for (i=0;i<k;i++) H[i+j*ldh] -= dH[i+j*k];
  }
  /* Update V */
  for (j=0;j<k;j++) {
    for (i=0;i<k;i++) dVS[i+j*2*k] = -dVS[i+j*2*k]+S[i+j*lds];
    for (i=k;i<2*k;i++) dVS[i+j*2*k] = -dVS[i+j*2*k];
  }
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&kdrs_,&k_,dVS,&k2_,tau,work,&k_,&info));
  SlepcCheckLapackInfo("geqrf",info);
  /* Copy triangular matrix in S */
  for (j=0;j<k;j++) {
    for (i=0;i<=j;i++) S[i+j*lds] = dVS[i+j*2*k];
    for (i=j+1;i<k;i++) S[i+j*lds] = 0.0;
  }
  PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&k2_,&k_,&k_,dVS,&k2_,tau,work,&k_,&info));
  SlepcCheckLapackInfo("orgqr",info);
  CHKERRQ(PetscFPTrapPop());
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M0));
  CHKERRQ(MatDenseGetArrayWrite(M0,&array));
  for (j=0;j<k;j++) CHKERRQ(PetscArraycpy(array+j*k,dVS+j*2*k,k));
  CHKERRQ(MatDenseRestoreArrayWrite(M0,&array));
  CHKERRQ(BVMultInPlace(pep->V,M0,0,k));
  if (rds) {
    CHKERRQ(MatDenseGetArrayWrite(M0,&array));
    for (j=0;j<k;j++) CHKERRQ(PetscArraycpy(array+j*k,dVS+k+j*2*k,rds));
    CHKERRQ(MatDenseRestoreArrayWrite(M0,&array));
    CHKERRQ(BVMultInPlace(dV,M0,0,k));
    CHKERRQ(BVMult(pep->V,1.0,1.0,dV,NULL));
  }
  CHKERRQ(MatDestroy(&M0));
  CHKERRQ(NRefOrthogStep(pep,k,H,ldh,fH,S,lds));
  CHKERRQ(PetscFree2(tau,work));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPNRefSetUp(PEP pep,PetscInt k,PetscScalar *H,PetscInt ldh,PEP_REFINE_EXPLICIT *matctx,PetscBool ini)
{
  PEP_REFINE_MATSHELL *ctx;
  PetscScalar         t,*coef;
  const PetscScalar   *array;
  MatStructure        str;
  PetscInt            j,nmat=pep->nmat,n0,m0,n1,m1,n0_,m0_,n1_,m1_,N0,N1,p,*idx1,*idx2,count,si,i,l0;
  MPI_Comm            comm;
  PetscMPIInt         np;
  const PetscInt      *rgs0,*rgs1;
  Mat                 B,C,*E,*A,*At;
  IS                  is1,is2;
  Vec                 v;
  PetscBool           flg;
  Mat                 M,P;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(nmat,&coef));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (flg) {
    CHKERRQ(PetscMalloc1(pep->nmat,&At));
    for (i=0;i<pep->nmat;i++) CHKERRQ(STGetMatrixTransformed(pep->st,i,&At[i]));
  } else At = pep->A;
  switch (pep->scheme) {
  case PEP_REFINE_SCHEME_EXPLICIT:
    if (ini) {
      if (matctx->subc) {
        A = matctx->A;
        CHKERRQ(PetscSubcommGetChild(matctx->subc,&comm));
      } else {
        A = At;
        CHKERRQ(PetscObjectGetComm((PetscObject)pep,&comm));
      }
      E = matctx->E;
      CHKERRQ(STGetMatStructure(pep->st,&str));
      CHKERRQ(MatDuplicate(A[0],MAT_COPY_VALUES,&E[0]));
      j = (matctx->subc)?matctx->subc->color:0;
      CHKERRQ(PEPEvaluateBasis(pep,H[j+j*ldh],0,coef,NULL));
      for (j=1;j<nmat;j++) CHKERRQ(MatAXPY(E[0],coef[j],A[j],str));
      CHKERRQ(MatCreateDense(comm,PETSC_DECIDE,PETSC_DECIDE,k,k,NULL,&E[1]));
      CHKERRQ(MatAssemblyBegin(E[1],MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(E[1],MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatGetOwnershipRange(E[0],&n0,&m0));
      CHKERRQ(MatGetOwnershipRange(E[1],&n1,&m1));
      CHKERRQ(MatGetOwnershipRangeColumn(E[0],&n0_,&m0_));
      CHKERRQ(MatGetOwnershipRangeColumn(E[1],&n1_,&m1_));
      /* T12 and T21 are computed from V and V*, so,
         they must have the same column and row ranges */
      PetscCheck(m0_-n0_==m0-n0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent dimensions");
      CHKERRQ(MatCreateDense(comm,m0-n0,m1_-n1_,PETSC_DECIDE,PETSC_DECIDE,NULL,&B));
      CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatCreateDense(comm,m1-n1,m0_-n0_,PETSC_DECIDE,PETSC_DECIDE,NULL,&C));
      CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatCreateTile(1.0,E[0],1.0,B,1.0,C,1.0,E[1],&M));
      CHKERRQ(MatDestroy(&B));
      CHKERRQ(MatDestroy(&C));
      matctx->compM1 = PETSC_TRUE;
      CHKERRQ(MatGetSize(E[0],NULL,&N0));
      CHKERRQ(MatGetSize(E[1],NULL,&N1));
      CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)M),&np));
      CHKERRQ(MatGetOwnershipRanges(E[0],&rgs0));
      CHKERRQ(MatGetOwnershipRanges(E[1],&rgs1));
      CHKERRQ(PetscMalloc4(PetscMax(k,N1),&matctx->idxp,N0,&matctx->idxg,N0,&matctx->map0,N1,&matctx->map1));
      /* Create column (and row) mapping */
      for (p=0;p<np;p++) {
        for (j=rgs0[p];j<rgs0[p+1];j++) matctx->map0[j] = j+rgs1[p];
        for (j=rgs1[p];j<rgs1[p+1];j++) matctx->map1[j] = j+rgs0[p+1];
      }
      CHKERRQ(MatCreateVecs(M,NULL,&matctx->tN));
      CHKERRQ(MatCreateVecs(matctx->E[1],NULL,&matctx->t1));
      CHKERRQ(VecDuplicate(matctx->tN,&matctx->ttN));
      if (matctx->subc) {
        CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np));
        count = np*k;
        CHKERRQ(PetscMalloc2(count,&idx1,count,&idx2));
        CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)pep),m1-n1,PETSC_DECIDE,&matctx->tp));
        CHKERRQ(VecGetOwnershipRange(matctx->tp,&l0,NULL));
        CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)pep),k,PETSC_DECIDE,&matctx->tpg));
        for (si=0;si<matctx->subc->n;si++) {
          if (matctx->subc->color==si) {
            j=0;
            if (matctx->subc->color==si) {
              for (p=0;p<np;p++) {
                for (i=n1;i<m1;i++) {
                  idx1[j] = l0+i-n1;
                  idx2[j++] =p*k+i;
                }
              }
            }
            count = np*(m1-n1);
          } else count =0;
          CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pep),count,idx1,PETSC_COPY_VALUES,&is1));
          CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pep),count,idx2,PETSC_COPY_VALUES,&is2));
          CHKERRQ(VecScatterCreate(matctx->tp,is1,matctx->tpg,is2,&matctx->scatterp_id[si]));
          CHKERRQ(ISDestroy(&is1));
          CHKERRQ(ISDestroy(&is2));
        }
        CHKERRQ(PetscFree2(idx1,idx2));
      } else CHKERRQ(VecScatterCreateToAll(matctx->t1,&matctx->scatterctx,&matctx->vseq));
      P = M;
    } else {
      if (matctx->subc) {
        /* Scatter vectors pep->V */
        for (i=0;i<k;i++) {
          CHKERRQ(BVGetColumn(pep->V,i,&v));
          CHKERRQ(VecScatterBegin(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD));
          CHKERRQ(VecScatterEnd(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD));
          CHKERRQ(BVRestoreColumn(pep->V,i,&v));
          CHKERRQ(VecGetArrayRead(matctx->tg,&array));
          CHKERRQ(VecPlaceArray(matctx->t,(const PetscScalar*)array));
          CHKERRQ(BVInsertVec(matctx->V,i,matctx->t));
          CHKERRQ(VecResetArray(matctx->t));
          CHKERRQ(VecRestoreArrayRead(matctx->tg,&array));
        }
      }
    }
    break;
  case PEP_REFINE_SCHEME_MBE:
    if (ini) {
      if (matctx->subc) {
        A = matctx->A;
        CHKERRQ(PetscSubcommGetChild(matctx->subc,&comm));
      } else {
        matctx->V = pep->V;
        A = At;
        CHKERRQ(PetscObjectGetComm((PetscObject)pep,&comm));
        CHKERRQ(MatCreateVecs(pep->A[0],&matctx->t,NULL));
      }
      CHKERRQ(STGetMatStructure(pep->st,&str));
      CHKERRQ(MatDuplicate(A[0],MAT_COPY_VALUES,&matctx->M1));
      j = (matctx->subc)?matctx->subc->color:0;
      CHKERRQ(PEPEvaluateBasis(pep,H[j+j*ldh],0,coef,NULL));
      for (j=1;j<nmat;j++) CHKERRQ(MatAXPY(matctx->M1,coef[j],A[j],str));
      CHKERRQ(BVDuplicateResize(matctx->V,PetscMax(k,pep->nmat),&matctx->W));
      CHKERRQ(BVDuplicateResize(matctx->V,k,&matctx->M2));
      CHKERRQ(BVDuplicate(matctx->M2,&matctx->M3));
      CHKERRQ(BVDuplicate(matctx->M2,&matctx->Wt));
      CHKERRQ(PetscMalloc5(k*k,&matctx->M4,k*k,&matctx->w,k*k,&matctx->wt,k,&matctx->d,k,&matctx->dt));
      matctx->compM1 = PETSC_TRUE;
      M = matctx->M1;
      P = M;
    }
    break;
  case PEP_REFINE_SCHEME_SCHUR:
    if (ini) {
      CHKERRQ(PetscObjectGetComm((PetscObject)pep,&comm));
      CHKERRQ(MatGetSize(At[0],&m0,&n0));
      CHKERRQ(PetscMalloc1(1,&ctx));
      CHKERRQ(STGetMatStructure(pep->st,&str));
      /* Create a shell matrix to solve the linear system */
      ctx->V = pep->V;
      ctx->k = k; ctx->nmat = nmat;
      CHKERRQ(PetscMalloc5(nmat,&ctx->A,k*k,&ctx->M4,k,&ctx->pM4,2*k*k,&ctx->work,nmat,&ctx->fih));
      for (i=0;i<nmat;i++) ctx->A[i] = At[i];
      CHKERRQ(PetscArrayzero(ctx->M4,k*k));
      CHKERRQ(MatCreateShell(comm,PETSC_DECIDE,PETSC_DECIDE,m0,n0,ctx,&M));
      CHKERRQ(MatShellSetOperation(M,MATOP_MULT,(void(*)(void))MatMult_FS));
      CHKERRQ(BVDuplicateResize(ctx->V,PetscMax(k,pep->nmat),&ctx->W));
      CHKERRQ(BVDuplicateResize(ctx->V,k,&ctx->M2));
      CHKERRQ(BVDuplicate(ctx->M2,&ctx->M3));
      CHKERRQ(BVCreateVec(pep->V,&ctx->t));
      CHKERRQ(MatDuplicate(At[0],MAT_COPY_VALUES,&ctx->M1));
      CHKERRQ(PEPEvaluateBasis(pep,H[0],0,coef,NULL));
      for (j=1;j<nmat;j++) CHKERRQ(MatAXPY(ctx->M1,coef[j],At[j],str));
      CHKERRQ(MatDuplicate(At[0],MAT_COPY_VALUES,&P));
      /* Compute a precond matrix for the system */
      t = H[0];
      CHKERRQ(PEPEvaluateBasis(pep,t,0,coef,NULL));
      for (j=1;j<nmat;j++) CHKERRQ(MatAXPY(P,coef[j],At[j],str));
      ctx->compM1 = PETSC_TRUE;
    }
    break;
  }
  if (ini) {
    CHKERRQ(PEPRefineGetKSP(pep,&pep->refineksp));
    CHKERRQ(KSPSetErrorIfNotConverged(pep->refineksp,PETSC_TRUE));
    CHKERRQ(PEP_KSPSetOperators(pep->refineksp,M,P));
    CHKERRQ(KSPSetFromOptions(pep->refineksp));
  }

  if (!ini && matctx && matctx->subc) {
     /* Scatter vectors pep->V */
    for (i=0;i<k;i++) {
      CHKERRQ(BVGetColumn(pep->V,i,&v));
      CHKERRQ(VecScatterBegin(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(BVRestoreColumn(pep->V,i,&v));
      CHKERRQ(VecGetArrayRead(matctx->tg,&array));
      CHKERRQ(VecPlaceArray(matctx->t,(const PetscScalar*)array));
      CHKERRQ(BVInsertVec(matctx->V,i,matctx->t));
      CHKERRQ(VecResetArray(matctx->t));
      CHKERRQ(VecRestoreArrayRead(matctx->tg,&array));
    }
   }
  CHKERRQ(PetscFree(coef));
  if (flg) CHKERRQ(PetscFree(At));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSubcommSetup(PEP pep,PetscInt k,PEP_REFINE_EXPLICIT *matctx,PetscInt nsubc)
{
  PetscInt          i,si,j,m0,n0,nloc0,nloc_sub,*idx1,*idx2;
  IS                is1,is2;
  BVType            type;
  Vec               v;
  const PetscScalar *array;
  Mat               *A;
  PetscBool         flg;
  MPI_Comm          contpar,child;

  PetscFunctionBegin;
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (flg) {
    CHKERRQ(PetscMalloc1(pep->nmat,&A));
    for (i=0;i<pep->nmat;i++) CHKERRQ(STGetMatrixTransformed(pep->st,i,&A[i]));
  } else A = pep->A;
  CHKERRQ(PetscSubcommGetChild(matctx->subc,&child));
  CHKERRQ(PetscSubcommGetContiguousParent(matctx->subc,&contpar));

  /* Duplicate pep matrices */
  CHKERRQ(PetscMalloc3(pep->nmat,&matctx->A,nsubc,&matctx->scatter_id,nsubc,&matctx->scatterp_id));
  for (i=0;i<pep->nmat;i++) {
    CHKERRQ(MatCreateRedundantMatrix(A[i],0,child,MAT_INITIAL_MATRIX,&matctx->A[i]));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)matctx->A[i]));
  }

  /* Create Scatter */
  CHKERRQ(MatCreateVecs(matctx->A[0],&matctx->t,NULL));
  CHKERRQ(MatGetLocalSize(matctx->A[0],&nloc_sub,NULL));
  CHKERRQ(VecCreateMPI(contpar,nloc_sub,PETSC_DECIDE,&matctx->tg));
  CHKERRQ(BVGetColumn(pep->V,0,&v));
  CHKERRQ(VecGetOwnershipRange(v,&n0,&m0));
  nloc0 = m0-n0;
  CHKERRQ(PetscMalloc2(matctx->subc->n*nloc0,&idx1,matctx->subc->n*nloc0,&idx2));
  j = 0;
  for (si=0;si<matctx->subc->n;si++) {
    for (i=n0;i<m0;i++) {
      idx1[j]   = i;
      idx2[j++] = i+pep->n*si;
    }
  }
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pep),matctx->subc->n*nloc0,idx1,PETSC_COPY_VALUES,&is1));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pep),matctx->subc->n*nloc0,idx2,PETSC_COPY_VALUES,&is2));
  CHKERRQ(VecScatterCreate(v,is1,matctx->tg,is2,&matctx->scatter_sub));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  for (si=0;si<matctx->subc->n;si++) {
    j=0;
    for (i=n0;i<m0;i++) {
      idx1[j] = i;
      idx2[j++] = i+pep->n*si;
    }
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pep),nloc0,idx1,PETSC_COPY_VALUES,&is1));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pep),nloc0,idx2,PETSC_COPY_VALUES,&is2));
    CHKERRQ(VecScatterCreate(v,is1,matctx->tg,is2,&matctx->scatter_id[si]));
    CHKERRQ(ISDestroy(&is1));
    CHKERRQ(ISDestroy(&is2));
  }
  CHKERRQ(BVRestoreColumn(pep->V,0,&v));
  CHKERRQ(PetscFree2(idx1,idx2));

  /* Duplicate pep->V vecs */
  CHKERRQ(BVGetType(pep->V,&type));
  CHKERRQ(BVCreate(child,&matctx->V));
  CHKERRQ(BVSetType(matctx->V,type));
  CHKERRQ(BVSetSizesFromVec(matctx->V,matctx->t,k));
  if (pep->scheme==PEP_REFINE_SCHEME_EXPLICIT) CHKERRQ(BVDuplicateResize(matctx->V,PetscMax(k,pep->nmat),&matctx->W));
  for (i=0;i<k;i++) {
    CHKERRQ(BVGetColumn(pep->V,i,&v));
    CHKERRQ(VecScatterBegin(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(BVRestoreColumn(pep->V,i,&v));
    CHKERRQ(VecGetArrayRead(matctx->tg,&array));
    CHKERRQ(VecPlaceArray(matctx->t,(const PetscScalar*)array));
    CHKERRQ(BVInsertVec(matctx->V,i,matctx->t));
    CHKERRQ(VecResetArray(matctx->t));
    CHKERRQ(VecRestoreArrayRead(matctx->tg,&array));
  }

  CHKERRQ(VecDuplicate(matctx->t,&matctx->Rv));
  CHKERRQ(VecDuplicate(matctx->t,&matctx->Vi));
  if (flg) CHKERRQ(PetscFree(A));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRefSubcommDestroy(PEP pep,PEP_REFINE_EXPLICIT *matctx)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(VecScatterDestroy(&matctx->scatter_sub));
  for (i=0;i<matctx->subc->n;i++) CHKERRQ(VecScatterDestroy(&matctx->scatter_id[i]));
  for (i=0;i<pep->nmat;i++) CHKERRQ(MatDestroy(&matctx->A[i]));
  if (pep->scheme==PEP_REFINE_SCHEME_EXPLICIT) {
    for (i=0;i<matctx->subc->n;i++) CHKERRQ(VecScatterDestroy(&matctx->scatterp_id[i]));
    CHKERRQ(VecDestroy(&matctx->tp));
    CHKERRQ(VecDestroy(&matctx->tpg));
    CHKERRQ(BVDestroy(&matctx->W));
  }
  CHKERRQ(PetscFree3(matctx->A,matctx->scatter_id,matctx->scatterp_id));
  CHKERRQ(BVDestroy(&matctx->V));
  CHKERRQ(VecDestroy(&matctx->t));
  CHKERRQ(VecDestroy(&matctx->tg));
  CHKERRQ(VecDestroy(&matctx->Rv));
  CHKERRQ(VecDestroy(&matctx->Vi));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPNewtonRefinement_TOAR(PEP pep,PetscScalar sigma,PetscInt *maxits,PetscReal *tol,PetscInt k,PetscScalar *S,PetscInt lds)
{
  PetscScalar         *H,*work,*dH,*fH,*dVS;
  PetscInt            ldh,i,j,its=1,nmat=pep->nmat,nsubc=pep->npart,rds;
  PetscBLASInt        k_,ld_,*p,info;
  BV                  dV;
  PetscBool           sinvert,flg;
  PEP_REFINE_EXPLICIT *matctx=NULL;
  Vec                 v;
  Mat                 M,P;
  PEP_REFINE_MATSHELL *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(PEP_Refine,pep,0,0,0));
  PetscCheck(k<=pep->n,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Multiple Refinement available only for invariant pairs of dimension smaller than n=%" PetscInt_FMT,pep->n);
  /* the input tolerance is not being taken into account (by the moment) */
  its = *maxits;
  CHKERRQ(PetscMalloc3(k*k,&dH,nmat*k*k,&fH,k,&work));
  CHKERRQ(DSGetLeadingDimension(pep->ds,&ldh));
  CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&H));
  CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&H));
  CHKERRQ(PetscMalloc1(2*k*k,&dVS));
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (!flg && pep->st && pep->ops->backtransform) { /* BackTransform */
    CHKERRQ(PetscBLASIntCast(k,&k_));
    CHKERRQ(PetscBLASIntCast(ldh,&ld_));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinvert));
    if (sinvert) {
      CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&H));
      CHKERRQ(PetscMalloc1(k,&p));
      CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&k_,&k_,H,&ld_,p,&info));
      SlepcCheckLapackInfo("getrf",info);
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&k_,H,&ld_,p,work,&k_,&info));
      SlepcCheckLapackInfo("getri",info);
      CHKERRQ(PetscFPTrapPop());
      CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&H));
      pep->ops->backtransform = NULL;
    }
    if (sigma!=0.0) {
      CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&H));
      for (i=0;i<k;i++) H[i+ldh*i] += sigma;
      CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&H));
      pep->ops->backtransform = NULL;
    }
  }
  if ((pep->scale==PEP_SCALE_BOTH || pep->scale==PEP_SCALE_SCALAR) && pep->sfactor!=1.0) {
    CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&H));
    for (j=0;j<k;j++) {
      for (i=0;i<k;i++) H[i+j*ldh] *= pep->sfactor;
    }
    CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&H));
    if (!flg) {
      /* Restore original values */
      for (i=0;i<pep->nmat;i++) {
        pep->pbc[pep->nmat+i] *= pep->sfactor;
        pep->pbc[2*pep->nmat+i] *= pep->sfactor*pep->sfactor;
      }
    }
  }
  if ((pep->scale==PEP_SCALE_DIAGONAL || pep->scale==PEP_SCALE_BOTH) && pep->Dr) {
    for (i=0;i<k;i++) {
      CHKERRQ(BVGetColumn(pep->V,i,&v));
      CHKERRQ(VecPointwiseMult(v,v,pep->Dr));
      CHKERRQ(BVRestoreColumn(pep->V,i,&v));
    }
  }
  CHKERRQ(DSGetArray(pep->ds,DS_MAT_A,&H));

  CHKERRQ(NRefOrthogStep(pep,k,H,ldh,fH,S,lds));
  /* check if H is in Schur form */
  for (i=0;i<k-1;i++) {
#if !defined(PETSC_USE_COMPLEX)
    PetscCheck(H[i+1+i*ldh]==0.0,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Iterative Refinement requires the complex Schur form of the projected matrix");
#else
    PetscCheck(H[i+1+i*ldh]==0.0,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Iterative Refinement requires an upper triangular projected matrix");
#endif
  }
  PetscCheck(nsubc<=k,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Number of subcommunicators should not be larger than the invariant pair dimension");
  CHKERRQ(BVSetActiveColumns(pep->V,0,k));
  CHKERRQ(BVDuplicateResize(pep->V,k,&dV));
  CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)dV));
  if (pep->scheme!=PEP_REFINE_SCHEME_SCHUR) {
    CHKERRQ(PetscMalloc1(1,&matctx));
    if (nsubc>1) { /* splitting in subcommunicators */
      matctx->subc = pep->refinesubc;
      CHKERRQ(NRefSubcommSetup(pep,k,matctx,nsubc));
    } else matctx->subc=NULL;
  }

  /* Loop performing iterative refinements */
  for (i=0;i<its;i++) {
    /* Pre-compute the polynomial basis evaluated in H */
    CHKERRQ(PEPEvaluateBasisforMatrix(pep,nmat,k,H,ldh,fH));
    CHKERRQ(PEPNRefSetUp(pep,k,H,ldh,matctx,PetscNot(i)));
    /* Solve the linear system */
    CHKERRQ(PEPNRefForwardSubstitution(pep,k,S,lds,H,ldh,fH,dV,dVS,&rds,dH,k,pep->refineksp,matctx));
    /* Update X (=V*S) and H, and orthogonalize [X;X*fH1;...;XfH(deg-1)] */
    CHKERRQ(PEPNRefUpdateInvPair(pep,k,H,ldh,fH,dH,S,lds,dV,dVS,rds));
  }
  CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_A,&H));
  if (!flg && sinvert) CHKERRQ(PetscFree(p));
  CHKERRQ(PetscFree3(dH,fH,work));
  CHKERRQ(PetscFree(dVS));
  CHKERRQ(BVDestroy(&dV));
  switch (pep->scheme) {
  case PEP_REFINE_SCHEME_EXPLICIT:
    for (i=0;i<2;i++) CHKERRQ(MatDestroy(&matctx->E[i]));
    CHKERRQ(PetscFree4(matctx->idxp,matctx->idxg,matctx->map0,matctx->map1));
    CHKERRQ(VecDestroy(&matctx->tN));
    CHKERRQ(VecDestroy(&matctx->ttN));
    CHKERRQ(VecDestroy(&matctx->t1));
    if (nsubc>1) CHKERRQ(NRefSubcommDestroy(pep,matctx));
    else {
      CHKERRQ(VecDestroy(&matctx->vseq));
      CHKERRQ(VecScatterDestroy(&matctx->scatterctx));
    }
    CHKERRQ(PetscFree(matctx));
    CHKERRQ(KSPGetOperators(pep->refineksp,&M,NULL));
    CHKERRQ(MatDestroy(&M));
    break;
  case PEP_REFINE_SCHEME_MBE:
    CHKERRQ(BVDestroy(&matctx->W));
    CHKERRQ(BVDestroy(&matctx->Wt));
    CHKERRQ(BVDestroy(&matctx->M2));
    CHKERRQ(BVDestroy(&matctx->M3));
    CHKERRQ(MatDestroy(&matctx->M1));
    CHKERRQ(VecDestroy(&matctx->t));
    CHKERRQ(PetscFree5(matctx->M4,matctx->w,matctx->wt,matctx->d,matctx->dt));
    if (nsubc>1) CHKERRQ(NRefSubcommDestroy(pep,matctx));
    CHKERRQ(PetscFree(matctx));
    break;
  case PEP_REFINE_SCHEME_SCHUR:
    CHKERRQ(KSPGetOperators(pep->refineksp,&M,&P));
    CHKERRQ(MatShellGetContext(M,&ctx));
    CHKERRQ(PetscFree5(ctx->A,ctx->M4,ctx->pM4,ctx->work,ctx->fih));
    CHKERRQ(MatDestroy(&ctx->M1));
    CHKERRQ(BVDestroy(&ctx->M2));
    CHKERRQ(BVDestroy(&ctx->M3));
    CHKERRQ(BVDestroy(&ctx->W));
    CHKERRQ(VecDestroy(&ctx->t));
    CHKERRQ(PetscFree(ctx));
    CHKERRQ(MatDestroy(&M));
    CHKERRQ(MatDestroy(&P));
    break;
  }
  CHKERRQ(PetscLogEventEnd(PEP_Refine,pep,0,0,0));
  PetscFunctionReturn(0);
}
