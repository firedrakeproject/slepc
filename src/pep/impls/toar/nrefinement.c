/*
   Newton refinement for nonlinear eigenproblem.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/pepimpl.h>        /*I "slepcpep.h" I*/
#include <slepc-private/stimpl.h>         /*I "slepcst.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  Mat          *A;
  Vec          *V;
  PetscInt     k,nmat;
  PetscScalar  *Mm;
  PetscScalar  *fih;
  PetscScalar  *work;
  Vec           w1,w2;
} FSubctx;

typedef struct {
  Mat         E[2];
  Vec         tN,ttN,t1,vseq;
  VecScatter  scatterctx;
  PetscBool   computedt11;
} Matexplicitctx;

#undef __FUNCT__
#define __FUNCT__ "MatFSMult"
static PetscErrorCode MatFSMult(Mat M ,Vec x,Vec y)
{
  PetscErrorCode ierr;
  FSubctx        *ctx;
  PetscInt       i,k,nmat;
  PetscScalar    *fih,*c,*vals,sone=1.0,zero=0.0;
  Vec            *V;
  Mat            *A;
  PetscBLASInt   k_,lda_,one=1;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(M,&ctx);CHKERRQ(ierr);
  fih = ctx->fih;
  k = ctx->k; nmat = ctx->nmat;
  V = ctx->V; A = ctx->A;
  c = ctx->work; vals = ctx->work+k;
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nmat*k,&lda_);CHKERRQ(ierr);
  ierr =  VecMDot(x,k,V,c);CHKERRQ(ierr);
  ierr = MatMult(A[0],x,y);CHKERRQ(ierr);
  for (i=1;i<nmat;i++) {
    PetscStackCall("BLASgemv",BLASgemv_("N",&k_,&k_,&sone,ctx->Mm+i*k,&lda_,c,&one,&zero,vals,&one));
    ierr = VecCopy(x,ctx->w1);CHKERRQ(ierr);
    ierr = SlepcVecMAXPBY(ctx->w1,fih[i],-1.0,k,vals,V);CHKERRQ(ierr);
    ierr = MatMult(A[i],ctx->w1,ctx->w2);CHKERRQ(ierr);
    ierr = VecAXPY(y,1.0,ctx->w2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPEvaluateBasisforMatrix"
/*
  Evaluates the first d elements of the polynomial basis
  on a given matrix H which is considered to be triangular
*/
static PetscErrorCode PEPEvaluateBasisforMatrix(PEP pep,PetscInt nm,PetscInt k,PetscScalar *H,PetscInt ldh,PetscScalar *fH)
{
  PetscErrorCode ierr;
  PetscInt       i,j,ldfh=nm*k,off,nmat=pep->nmat;
  PetscReal      *a=pep->pbc,*b=pep->pbc+nmat,*g=pep->pbc+2*nmat,t;
  PetscScalar    corr=0.0,alpha,beta;
  PetscBLASInt   k_,ldh_,ldfh_;
  
  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldfh,&ldfh_);CHKERRQ(ierr);
  ierr = PetscMemzero(fH,nm*k*k*sizeof(PetscScalar));CHKERRQ(ierr);
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
        ierr = PetscMemcpy(fH+off+j*ldfh,fH+(i-2)*k+j*ldfh,k*sizeof(PetscScalar));CHKERRQ(ierr);
        H[j+j*ldh] += corr-b[i-1];
      }
    }
    corr = b[i-1];
    beta = -g[i-1]/a[i-1];
    alpha = 1/a[i-1];
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&alpha,H,&ldh_,fH+(i-1)*k,&ldfh_,&beta,fH+off,&ldfh_));
  }
  for (j=0;j<k;j++) H[j+j*ldh] += corr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSysSetup"
static PetscErrorCode NRefSysSetup(PetscInt nmat,PetscReal *pcf,PetscInt k,PetscInt deg,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar h,PetscScalar *Mm,PetscScalar *T22,PetscBLASInt *p22,PetscScalar *T21,PetscScalar *T12){
  PetscErrorCode ierr;
  PetscScalar    *DHii,*Tr,*Ts,s,sone=1.0,zero=0.0;
  PetscInt       i,d,j,lda=nmat*k;
  PetscReal      *a=pcf,*b=pcf+nmat,*g=pcf+2*nmat;
  PetscBLASInt   k_,lda_,lds_,info;
  
  PetscFunctionBegin;
  DHii = T12;
  ierr = PetscMemzero(DHii,k*k*nmat*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc2(k*k,&Tr,k*k,&Ts);CHKERRQ(ierr);
  for (i=0;i<k;i++) DHii[k+i+i*lda] = 1.0/a[0];
  for (d=2;d<nmat;d++) {
    for (j=0;j<k;j++) {
      for (i=0;i<k;i++) {
        DHii[d*k+i+j*lda] = ((h-b[d-1])*DHii[(d-1)*k+i+j*lda]+fH[(d-1)*k+i+j*lda]-g[d-1]*DHii[(d-2)*k+i+j*lda])/(a[d-1]);
      }
    }
  }
  /* T22 */
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  PetscStackCall("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,S,&lds_,S,&lds_,&zero,Tr,&k_));
  for (i=1;i<deg;i++) {
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,Tr,&k_,DHii+i*k,&lda_,&zero,Ts,&k_));
    s = (i==1)?0.0:1.0; 
    PetscStackCall("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,fH+i*k,&lda_,Ts,&k_,&s,T22,&k_));
  }

  /* T21 */
  for (i=1;i<deg;i++) {
    s = (i==1)?0.0:1.0; 
    PetscStackCall("BLASgemm",BLASgemm_("C","C",&k_,&k_,&k_,fh+i,fH+i*k,&lda_,S,&lds_,&s,T21,&k_));
  }
  /* Mm */
  ierr = PetscMemcpy(Tr,T21,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscStackCall("LAPACKgesv",LAPACKgesv_(&k_,&k_,T22,&k_,p22,Tr,&k_,&info));
  
  s = 0.0;
  for (i=1;i<nmat;i++) {
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,S,&lds_,DHii+i*k,&lda_,&zero,Ts,&k_));    
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,Ts,&k_,Tr,&k_,&s,Mm+i*k,&lda_));
    for (j=0;j<k;j++) {
      ierr = PetscMemcpy(T12+i*k+j*lda,Ts+j*k,k*sizeof(PetscScalar));CHKERRQ(ierr);
    }  
  }
  ierr = PetscFree2(Tr,Ts);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "NRefSysSolve"
static PetscErrorCode NRefSysSolve(Mat *A,KSP ksp,PetscInt nmat,Vec Rv,PetscScalar *Rh,PetscInt k,PetscScalar *T22,PetscBLASInt *p22,PetscScalar *T21,PetscScalar *T12,Vec *V,Vec dVi,PetscScalar *dHi,Vec *W,Vec t,PetscScalar *work,PetscInt lw)
{
  PetscErrorCode ierr;
  PetscScalar    *t0,*t1,zero=0.0,none=-1.0,sone=1.0;
  PetscBLASInt   k_,one=1,info,lda_;
  PetscInt       i,lda=nmat*k,nwu=0;

  PetscFunctionBegin;
  t0 = work+nwu;
  nwu += k;
  t1 = work+nwu;
  nwu += k;
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  for (i=0;i<k;i++) t0[i] = Rh[i];
  PetscStackCall("LAPACKgetrs",LAPACKgetrs_("N",&k_,&one,T22,&k_,p22,t0,&k_,&info));
  for (i=1;i<nmat;i++) {
    PetscStackCall("BLASgemv",BLASgemv_("N",&k_,&k_,&sone,T12+i*k,&lda_,t0,&one,&zero,t1,&one));
    ierr = SlepcVecMAXPBY(t,0.0,1.0,k,t1,V);CHKERRQ(ierr);
    ierr = MatMult(A[i],t,W[i]);CHKERRQ(ierr);
  }
  for (i=0;i<nmat-1;i++) t1[i]=-1.0;
  ierr = VecMAXPY(Rv,nmat-1,t1,W+1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,Rv,dVi);CHKERRQ(ierr);
  ierr = VecMDot(dVi,k,V,t1);CHKERRQ(ierr);
  PetscStackCall("BLASgemv",BLASgemv_("N",&k_,&k_,&none,T21,&k_,t1,&one,&zero,dHi,&one));
  for (i=0;i<k;i++) dHi[i] += Rh[i];
  PetscStackCall("LAPACKgetrs",LAPACKgetrs_("N",&k_,&one,T22,&k_,p22,dHi,&k_,&info));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefRightside"
static PetscErrorCode NRefRightside(PetscInt nmat,PetscReal *pcf,Mat *A,PetscInt k,Vec *V,PetscScalar *S,PetscInt lds,PetscInt j,PetscScalar *H,PetscInt ldh,PetscScalar *fH,PetscScalar *DfH,PetscScalar *dH,Vec *dV,PetscScalar *dVS,Vec Rv,PetscScalar *Rh,Vec *W,Vec t,PetscScalar *work,PetscInt lw)
{
  PetscErrorCode ierr;
  PetscScalar    *DS0,*DS1,*F,beta=0.0,sone=1.0,none=-1.0,tt=0.0,*h,zero=0.0,*Z,*c0;
  PetscReal      *a=pcf,*b=pcf+nmat,*g=b+nmat;
  PetscInt       i,ii,jj,nwu=0,lda;
  PetscBLASInt   lda_,k_,ldh_,lds_,nmat_,k2_,j_,one=1;
  
  PetscFunctionBegin;
  /* Computes the residual P(H,V*S)*e_j for the polynomial */
  h = work+nwu;
  nwu += k*nmat;
  lda = k*nmat;
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nmat,&nmat_);CHKERRQ(ierr);
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&nmat_,&k_,&sone,S,&lds_,fH+j*lda,&k_,&zero,h,&k_));
  for (i=0;i<nmat;i++) {
    ierr = SlepcVecMAXPBY(W[i],0.0,1.0,k,h+i*k,V);CHKERRQ(ierr);
  }
  ierr = MatMult(A[0],W[0],Rv);CHKERRQ(ierr);
  for (i=1;i<nmat;i++) {
    ierr = MatMult(A[i],W[i],t);CHKERRQ(ierr);
    ierr = VecAXPY(Rv,1.0,t);
  }
  /* Update right-hand side */
  if (j) {
    DS0 = work+nwu;
    nwu += k*k;
    DS1 = work+nwu;
    nwu += k*k;
    ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr); 
    ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr); 
    Z = work+nwu;
    nwu += k*k;
    ierr = PetscMemzero(Z,k*k*sizeof(PetscScalar));
    ierr = PetscMemzero(DS0,k*k*sizeof(PetscScalar));
    ierr = PetscMemcpy(Z+(j-1)*k,dH+(j-1)*k,k*sizeof(PetscScalar));CHKERRQ(ierr);
    /* Update DfH */
    for (i=1;i<nmat;i++) {
      if (i>1) {
        beta = -g[i-1];
        PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,fH+(i-1)*k,&lda_,Z,&k_,&beta,DS0,&k_));
        tt += -b[i-1];
        for (ii=0;ii<k;ii++) H[ii+ii*ldh] += tt;
        tt = b[i-1];
        beta = 1/a[i-1];
        PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&beta,DS1,&k_,H,&ldh_,&beta,DS0,&k_));
        F = DS0; DS0 = DS1; DS1 = F;
      } else {
        ierr = PetscMemzero(DS1,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
        for (ii=0;ii<k;ii++) DS1[ii+(j-1)*k] = Z[ii+(j-1)*k]/a[0];
      } 
      for (jj=j;jj<k;jj++) {
        for (ii=0;ii<k;ii++) 
          DfH[k*i+ii+jj*lda] += DS1[ii+jj*k];
      }
    }
    for (ii=0;ii<k;ii++) H[ii+ii*ldh] += tt;
    /* Update right-hand side */
    ierr = PetscBLASIntCast(2*k,&k2_);CHKERRQ(ierr); 
    ierr = PetscBLASIntCast(j,&j_);CHKERRQ(ierr);
    c0 = DS0;
    ierr = PetscMemzero(Rh,k*sizeof(PetscScalar));
    for (i=0;i<nmat;i++) {
      PetscStackCall("BLASgemv",BLASgemv_("N",&k2_,&j_,&sone,dVS,&k2_,fH+j*lda+i*k,&one,&zero,h,&one));
      PetscStackCall("BLASgemv",BLASgemv_("N",&k_,&k_,&sone,S,&lds_,DfH+i*k+j*lda,&one,&sone,h,&one));
      ierr = SlepcVecMAXPBY(t,0.0,1.0,k,h,V);CHKERRQ(ierr);
      ierr = SlepcVecMAXPBY(t,1.0,1.0,j,h+k,dV);CHKERRQ(ierr);
      ierr = MatMult(A[i],t,W[i]);
      if (i>0&&i<nmat-1) {
        PetscStackCall("BLASgemv",BLASgemv_("C",&k_,&k_,&sone,S,&lds_,h,&one,&zero,c0,&one));
        PetscStackCall("BLASgemv",BLASgemv_("C",&k_,&k_,&none,fH+i*k,&lda_,c0,&one,&sone,Rh,&one));
      }
    }
     
    for (i=0;i<nmat;i++) h[i] = -1.0;
    ierr = VecMAXPY(Rv,nmat,h,W);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSysIter_shell"
static PetscErrorCode NRefSysIter_shell(PEP pep,PetscInt k,KSP ksp,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar h,Vec Rv,PetscScalar *Rh,Vec *V,Vec dVi,PetscScalar *dHi,Vec *W,Vec t,PetscScalar *work,PetscInt lwork)
{
  PetscErrorCode ierr;
  PetscInt       nwu=0,nmat=pep->nmat,deg=nmat-1;
  PetscScalar    *T22,*T21,*T12;
  PetscBLASInt   *p22;
  FSubctx        *ctx;
  Mat            M,*A=pep->A;

  PetscFunctionBegin;
  ierr = PetscMalloc1(k,&p22);CHKERRQ(ierr);
  T22 = work+nwu;
  nwu += k*k;
  T21 = work+nwu;
  nwu += k*k;
  T12 = work+nwu;
  nwu += nmat*k*k;
  ierr = KSPGetOperators(ksp,&M,NULL);CHKERRQ(ierr);
  ierr = MatShellGetContext(M,&ctx);CHKERRQ(ierr);
  /* Update the matrix for the system */
  ierr = NRefSysSetup(nmat,pep->pbc,k,deg,fH,S,lds,fh,h,ctx->Mm,T22,p22,T21,T12);CHKERRQ(ierr);
  /* Solve system */
  ierr = NRefSysSolve(A,ksp,nmat,Rv,Rh,k,T22,p22,T21,T12,V,dVi,dHi,W,t,work+nwu,lwork-nwu);CHKERRQ(ierr);
  ierr = PetscFree(p22);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSysIter_explicit"
static PetscErrorCode NRefSysIter_explicit(PEP pep,PetscInt k,KSP ksp,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar h,Vec Rv,PetscScalar *Rh,Vec *V,Vec dVi,PetscScalar *dHi,Matexplicitctx *matctx,Vec *W,PetscScalar *work,PetscInt lwork)
{
  PetscErrorCode    ierr;
  PetscInt          nwu=0,i,j,d,n,n0,m0,n1,m1,n0_,m0_,p,N0,N1;
  PetscInt          *idxg,*idxp,idx,ncols,nmat=pep->nmat,lda=nmat*k,deg=nmat-1,*map0,*map1;
  PetscMPIInt       np;
  Mat               M,*A=pep->A,*E=matctx->E;
  PetscReal         *a=pep->pbc,*b=pep->pbc+nmat,*g=pep->pbc+2*nmat;
  PetscScalar       s,ss,*DHii,*T22,*T21,*T12,*Ts,*Tr,*arrayV,*array,*ts,sone=1.0,zero=0.0;
  PetscBLASInt      lds_,lda_,k_,nn_,nmat1_;
  const PetscInt    *rgs0,*rgs1,*idxmc;
  const PetscScalar *valsc;
  MatStructure      str;
  
  PetscFunctionBegin;
  T22 = work+nwu;
  nwu += k*k;
  T21 = work+nwu;
  nwu += k*k;
  T12 = work+nwu;
  nwu += nmat*k*k;
  Tr = work+nwu;
  nwu += k*k;
  Ts = work+nwu;
  nwu += k*k;
  ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(E[0],NULL,&N0);CHKERRQ(ierr);
  ierr = MatGetSize(E[1],NULL,&N1);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(E[1],&n1,&m1);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(E[0],&n0,&m0);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(M,&n,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(E[0],&n0_,&m0_);CHKERRQ(ierr);
  /* T12 and T21 are computed from V and V*, so,
   they must have the same column and row ranges */
  if (m0_-n0_ != m0-n0) SETERRQ(PETSC_COMM_SELF,1,"Inconsistent dimensions");
  MPI_Comm_size(PetscObjectComm((PetscObject)M),&np);
  ierr = MatGetOwnershipRanges(E[0],&rgs0);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(E[1],&rgs1);CHKERRQ(ierr);
  ierr = PetscMalloc5(PetscMax(k,N1),&idxp,N0,&idxg,nmat,&ts,N0,&map0,N1,&map1);CHKERRQ(ierr);
  
  /* Create column (and row) mapping */
  for (p=0;p<np;p++) {
    for (i=rgs0[p];i<rgs0[p+1];i++) map0[i] = i+rgs1[p];
    for (i=rgs1[p];i<rgs1[p+1];i++) map1[i] = i+rgs0[p+1];
  }
 
  /* Form the explicit system matrix */
  DHii = T12;
  ierr = PetscMemzero(DHii,k*k*nmat*sizeof(PetscScalar));CHKERRQ(ierr);  
  for (i=0;i<k;i++) DHii[k+i+i*lda] = 1.0/a[0];
  for (d=2;d<nmat;d++) {
    for (j=0;j<k;j++) {
      for (i=0;i<k;i++) {
        DHii[d*k+i+j*lda] = ((h-b[d-1])*DHii[(d-1)*k+i+j*lda]+fH[(d-1)*k+i+j*lda]-g[d-1]*DHii[(d-2)*k+i+j*lda])/(a[d-1]);
      }
    }
  }

  /* T11 */
  if (!matctx->computedt11) {
    ierr = MatCopy(A[0],E[0],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PEPEvaluateBasis(pep,h,0,Ts,NULL);CHKERRQ(ierr);
    for (j=1;j<nmat;j++) {
      ierr = MatAXPY(E[0],Ts[j],A[j],str);CHKERRQ(ierr);
    }
  }
  for (i=n0;i<m0;i++) {
    ierr = MatGetRow(E[0],i,&ncols,&idxmc,&valsc);CHKERRQ(ierr);
    idx = n+i-n0;
    for (j=0;j<ncols;j++) {
      idxg[j] = map0[idxmc[j]];
    }
    ierr = MatSetValues(M,1,&idx,ncols,idxg,valsc,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(E[0],i,&ncols,&idxmc,&valsc);CHKERRQ(ierr);
  }

  /* T22 */
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  PetscStackCall("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,S,&lds_,S,&lds_,&zero,Tr,&k_));
  for (i=1;i<deg;i++) {
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,Tr,&k_,DHii+i*k,&lda_,&zero,Ts,&k_));
    s = (i==1)?0.0:1.0; 
    PetscStackCall("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,fH+i*k,&lda_,Ts,&k_,&s,T22,&k_));
  }
  for (j=0;j<k;j++) idxp[j] = map1[j];
  for (i=0;i<m1-n1;i++) {
    idx = n+m0-n0+i;
    for (j=0;j<k;j++) {
      Tr[j] = T22[n1+i+j*k];
    }
    ierr = MatSetValues(M,1,&idx,k,idxp,Tr,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* T21 */
  for (i=1;i<deg;i++) {
    s = (i==1)?0.0:1.0;
    ss = PetscConj(fh[i]);
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&ss,S,&lds_,fH+i*k,&lda_,&s,T21,&k_));
  }
  ierr = VecGetArray(V[0],&arrayV);CHKERRQ(ierr);
  ierr = VecGetArray(W[0],&array);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m0-n0,&nn_);CHKERRQ(ierr);
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&nn_,&k_,&k_,&sone,arrayV,&nn_,T21,&k_,&zero,array,&nn_));

  for (i=0;i<(m0-n0)*k;i++) array[i] = PetscConj(array[i]);
  for (i=0;i<k;i++) {
    idx = map1[i];
    ierr = MatSetValues(M,1,&idx,m0-n0,map0+n0,array+i*(m0-n0),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(V[0],&arrayV);CHKERRQ(ierr);
  ierr = VecRestoreArray(W[0],&array);CHKERRQ(ierr);

  /* T12 */  
  for (i=1;i<nmat;i++) {
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,S,&lds_,DHii+i*k,&lda_,&zero,Ts,&k_));    
    for (j=0;j<k;j++) {
      ierr = PetscMemcpy(T12+i*k+j*lda,Ts+j*k,k*sizeof(PetscScalar));CHKERRQ(ierr);
    }  
  }
  ierr = VecGetArray(V[0],&arrayV);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nmat-1,&nmat1_);CHKERRQ(ierr);
  for (i=0;i<nmat;i++) ts[i] = 1.0;
  for (j=0;j<k;j++) {
    ierr = VecGetArray(W[0],&array);CHKERRQ(ierr);
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&nn_,&nmat1_,&k_,&sone,arrayV,&nn_,T12+k+j*lda,&k_,&zero,array,&nn_));    
    ierr = VecRestoreArray(W[0],&array);CHKERRQ(ierr);
    for (i=nmat-1;i>=1;i--) {
      ierr = MatMult(A[i],W[i-1],W[i]);
    }
    ierr = SlepcVecMAXPBY(W[0],0.0,1.0,nmat-1,ts,W+1);CHKERRQ(ierr);
    ierr = VecGetArray(W[0],&array);CHKERRQ(ierr);
    idx = map1[j];
    ierr = MatSetValues(M,m0-n0,map0+n0,1,&idx,array,INSERT_VALUES);CHKERRQ(ierr);   
    ierr = VecRestoreArray(W[0],&array);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
 
  /* Rigth side */  
  ierr = VecGetArray(Rv,&array);CHKERRQ(ierr);
  ierr = VecSetValues(matctx->tN,m0-n0,map0+n0,array,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(Rv,&array);CHKERRQ(ierr);
  ierr = VecSetValues(matctx->tN,m1-n1,map1+n1,Rh+n1,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(matctx->tN);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(matctx->tN);CHKERRQ(ierr);

  /* Solve */
  ierr = KSPSolve(ksp,matctx->tN,matctx->ttN);CHKERRQ(ierr);
 
 /* Retrieve solution */
  ierr = VecGetArray(dVi,&arrayV);CHKERRQ(ierr);
  ierr = VecGetArray(matctx->ttN,&array);CHKERRQ(ierr);
  ierr = PetscMemcpy(arrayV,array,(m0-n0)*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(dVi,&arrayV);CHKERRQ(ierr);
  ierr = VecGetArray(matctx->t1,&arrayV);CHKERRQ(ierr);
  for (i=0;i<m1-n1;i++) arrayV[i] =  array[m0-n0+i];
  ierr = VecRestoreArray(matctx->t1,&arrayV);CHKERRQ(ierr);
  ierr = VecRestoreArray(matctx->ttN,&array);CHKERRQ(ierr);
  ierr = VecScatterBegin(matctx->scatterctx,matctx->t1,matctx->vseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(matctx->scatterctx,matctx->t1,matctx->vseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(matctx->vseq,&array);CHKERRQ(ierr);
  for (i=0;i<k;i++) dHi[i] = array[i];
  ierr = VecRestoreArray(matctx->vseq,&array);CHKERRQ(ierr);
  ierr = PetscFree5(idxp,idxg,ts,map0,map1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNRefForwardSubstitution"
static PetscErrorCode PEPNRefForwardSubstitution(PEP pep,PetscInt k,Vec *V,PetscScalar  *S,PetscInt lds,PetscScalar *H,PetscInt ldh,PetscScalar *fH,Vec *dV,PetscScalar *dVS,PetscScalar *dH,PetscInt lddh,KSP ksp,PetscScalar *work,PetscInt lw,Matexplicitctx *matctx)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nmat=pep->nmat,lwork=0,nwu=0,lda=nmat*k,ncv;
  PetscScalar    h,*fh,*Rh,*DfH;
  PetscReal      norm;
  Vec            *W,Rv,t;
  FSubctx        *ctx;
  Mat            M;
  MatStructure   str;
  MPI_Comm       comm;

  PetscFunctionBegin;
  lwork = (7+3*nmat)*k*k+2*k+nmat;
  DfH = work+nwu;
  nwu += nmat*k*k;
  Rh = work+nwu;
  nwu += k;
  ierr = VecDuplicate(V[0],&t);CHKERRQ(ierr);
  ierr = VecDuplicate(V[0],&Rv);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&M,NULL);CHKERRQ(ierr);
  if (matctx) {
    ierr = PetscObjectGetComm((PetscObject)pep,&comm);CHKERRQ(ierr);
    ierr = MatGetVecs(M,NULL,&matctx->tN);CHKERRQ(ierr);
    ierr = MatGetVecs(matctx->E[1],NULL,&matctx->t1);CHKERRQ(ierr);
    ierr = VecDuplicate(matctx->tN,&matctx->ttN);CHKERRQ(ierr);
    ierr = VecScatterCreateToAll(matctx->t1,&matctx->scatterctx,&matctx->vseq);CHKERRQ(ierr);
    ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
    fh = work+nwu;
    nwu += nmat;
  } else {
    ierr = MatShellGetContext(M,&ctx);CHKERRQ(ierr);
    fh = ctx->fih;
  }
  ierr =  PEPGetDimensions(pep,NULL,&ncv,NULL);CHKERRQ(ierr);
  if (ncv-k<PetscMax(k,nmat)) {
    ierr = VecDuplicateVecs(pep->t,PetscMax(k,nmat),&W);CHKERRQ(ierr);
  } else W = &pep->V[k];
  ierr = PetscMemzero(dVS,2*k*k*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(DfH,lda*k*sizeof(PetscScalar));CHKERRQ(ierr);

  /* Main loop for computing the ith columns of dX and dS */
  for (i=0;i<k;i++) {
    h = H[i+i*ldh];

    /* Compute and update i-th column of the right hand side */
    ierr = PetscMemzero(Rh,k*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = NRefRightside(nmat,pep->pbc,pep->A,k,V,S,lds,i,H,ldh,fH,DfH,dH,dV,dVS,Rv,Rh,W,t,work+nwu,lwork-nwu);CHKERRQ(ierr);

    /* Update and solve system */
    for (j=0;j<nmat;j++) fh[j] = fH[j*k+i+i*lda];
    if (matctx) {
      ierr =  NRefSysIter_explicit(pep,k,ksp,fH,S,lds,fh,h,Rv,Rh,V,dV[i],dH+i*k,matctx,W,work+nwu,lwork-nwu);CHKERRQ(ierr);
      if (i==0) matctx->computedt11 = PETSC_FALSE;
    } else {
      ierr = NRefSysIter_shell(pep,k,ksp,fH,S,lds,fh,h,Rv,Rh,V,dV[i],dH+i*k,W,t,work+nwu,lwork-nwu);
    }
    /* Orthogonalize computed solution */
    ierr = IPOrthogonalize(pep->ip,0,NULL,k,NULL,V,dV[i],dVS+i*2*k,NULL,NULL);CHKERRQ(ierr);
    ierr = IPOrthogonalize(pep->ip,0,NULL,i,NULL,dV,dV[i],dVS+k+i*2*k,&norm,NULL);CHKERRQ(ierr);
    dVS[k+i+i*2*k] = norm;
    if (norm!=0) {
      ierr = VecScale(dV[i],1/norm);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = VecDestroy(&Rv);CHKERRQ(ierr);
  if (ncv-k<PetscMax(k,nmat)) {
    ierr = VecDestroyVecs(PetscMax(k,nmat),&W);CHKERRQ(ierr);
  }
  if (matctx) {
    ierr = VecDestroy(&matctx->tN);CHKERRQ(ierr);
    ierr = VecDestroy(&matctx->ttN);CHKERRQ(ierr);
    ierr = VecDestroy(&matctx->t1);CHKERRQ(ierr);
    ierr = VecDestroy(&matctx->vseq);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&matctx->scatterctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNRefUpdateInvPair"
static PetscErrorCode PEPNRefUpdateInvPair(PEP pep,PetscInt k,PetscScalar *H,PetscInt ldh,PetscScalar *fH,PetscScalar *dH,Vec *V,PetscScalar *S,PetscInt lds,Vec *dV,PetscScalar *dVS,PetscScalar *work,PetscInt lwork)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nmat=pep->nmat,deg=nmat-1,lda=nmat*k,nwu=0;
  PetscScalar    *G,sone=1.0,*tau;
  PetscBLASInt   lds_,k_,lda_,ldh_,k2_,lw_,info,rG_;

  PetscFunctionBegin;
  tau = work+nwu;
  nwu += k;
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast((nmat-1)*k,&rG_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(2*k,&k2_);CHKERRQ(ierr);
  /* Update H */
  for (j=0;j<k;j++) {
    for (i=0;i<k;i++) {
      H[i+j*ldh] -= dH[i+j*k];
    }
  }
  /* Update V */
  for (j=0;j<k;j++) {
    for (i=0;i<k;i++) {
      dVS[i+j*2*k] = -dVS[i+j*2*k]+S[i+j*lds];
    }
    for (i=k;i<2*k;i++) {
      dVS[i+j*2*k] = -dVS[i+j*2*k];
    }
  }
  ierr = PetscBLASIntCast(lwork-nwu,&lw_);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&k2_,&k_,dVS,&k2_,tau,work+nwu,&lw_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  /* Copy triangular matrix in S */
  for (j=0;j<k;j++) {
    for (i=0;i<=j;i++) S[i+j*lds] = dVS[i+j*2*k];
    for (i=j+1;i<k;i++) S[i+j*lds] = 0.0;
  }
  PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&k2_,&k_,&k_,dVS,&k2_,tau,work+nwu,&lw_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
  ierr = SlepcUpdateVectors(k,V,0,k,dVS,2*k,PETSC_FALSE);CHKERRQ(ierr);
  ierr = SlepcUpdateVectors(k,dV,0,k,dVS+k,2*k,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    ierr = VecAXPY(V[i],1.0,dV[i]);CHKERRQ(ierr);
  }

  /* Form auxiliar matrix for the orthogonalization step */
  ierr = PEPEvaluateBasisforMatrix(pep,nmat,k,H,ldh,fH);CHKERRQ(ierr);
  G = fH;
  for (j=0;j<deg;j++) {
    PetscStackCall("BLAStrmm",BLAStrmm_("L","U","N","N",&k_,&k_,&sone,S,&lds_,G+j*k,&lda_)); 
  }
  /* Orthogonalize and update S */
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&rG_,&k_,G,&lda_,tau,work+nwu,&lw_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&k_,&k_,&sone,G,&lda_,S,&lds_));
  /* Update H */
  PetscStackCall("BLAStrmm",BLAStrmm_("L","U","N","N",&k_,&k_,&sone,G,&lda_,H,&ldh_));
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&k_,&k_,&sone,G,&lda_,H,&ldh_));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNRefSetUpMatrices"
static PetscErrorCode PEPNRefSetUpMatrices(PEP pep,PetscInt k,PetscScalar *H,PetscInt ldh,Mat *M,Mat *P,Matexplicitctx *matctx,PetscBool ini)
{
  PetscErrorCode ierr;
  FSubctx        *ctx;
  PetscScalar    t,*coef; 
  MatStructure   str;
  PetscInt       j,nmat=pep->nmat,n0,m0,n1,m1,n0_,m0_,n1_,m1_;
  MPI_Comm       comm;
  Mat            B,C,*E;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nmat,&coef);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)pep,&comm);CHKERRQ(ierr);
  if (matctx) {
    if (ini) {
      E = matctx->E;
      ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
      ierr = MatDuplicate(pep->A[0],MAT_COPY_VALUES,&E[0]);CHKERRQ(ierr);
      ierr = PEPEvaluateBasis(pep,H[0],0,coef,NULL);CHKERRQ(ierr);
      for (j=1;j<nmat;j++) {
        ierr = MatAXPY(E[0],coef[j],pep->A[j],str);CHKERRQ(ierr);
      }
      ierr = MatCreateDense(comm,PETSC_DECIDE,PETSC_DECIDE,k,k,NULL,&E[1]);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(E[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(E[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(E[0],&n0,&m0);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(E[1],&n1,&m1);CHKERRQ(ierr);
      ierr = MatGetOwnershipRangeColumn(E[0],&n0_,&m0_);CHKERRQ(ierr);
      ierr = MatGetOwnershipRangeColumn(E[1],&n1_,&m1_);CHKERRQ(ierr);
      ierr = MatCreateDense(comm,m0-n0,m1_-n1_,PETSC_DECIDE,PETSC_DECIDE,NULL,&B);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatCreateDense(comm,m1-n1,m0_-n0_,PETSC_DECIDE,PETSC_DECIDE,NULL,&C);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = SlepcMatTile(1.0,E[0],1.0,B,1.0,C,1.0,E[1],M);CHKERRQ(ierr);
      *P = *M;
      ierr = MatDestroy(&B);CHKERRQ(ierr);
      ierr = MatDestroy(&C);CHKERRQ(ierr);
      matctx->computedt11 = PETSC_TRUE;
    }
  } else {
    if (ini) {
      ierr = MatGetSize(pep->A[0],&m0,&n0);CHKERRQ(ierr);
      ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
      ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
      /* Create a shell matrix to solve the linear system */
      ctx->A = pep->A;
      ctx->V = pep->V;
      ctx->k = k; ctx->nmat = nmat;
      ierr = PetscMalloc3(k*k*nmat,&ctx->Mm,2*k*k,&ctx->work,nmat,&ctx->fih);CHKERRQ(ierr);
      ierr = PetscMemzero(ctx->Mm,k*k*nmat*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecDuplicate(pep->V[0],&ctx->w1);CHKERRQ(ierr);
      ierr = VecDuplicate(pep->V[0],&ctx->w2);CHKERRQ(ierr);
      ierr = MatCreateShell(comm,PETSC_DECIDE,PETSC_DECIDE,m0,n0,ctx,M);CHKERRQ(ierr);
      ierr = MatShellSetOperation(*M,MATOP_MULT,(void(*)(void))MatFSMult);CHKERRQ(ierr);
    }
    /* Compute a precond matrix for the system */
    t = 0.0;
    for (j=0;j<k;j++) t += H[j+j*ldh];
    t /= k;
    if (ini) {
      ierr = MatDuplicate(pep->A[0],MAT_COPY_VALUES,P);CHKERRQ(ierr);
    } else {
      ierr = MatCopy(pep->A[0],*P,str);CHKERRQ(ierr);
    }
    ierr = PEPEvaluateBasis(pep,t,0,coef,NULL);CHKERRQ(ierr);
    for (j=1;j<nmat;j++) {
      ierr = MatAXPY(*P,coef[j],pep->A[j],str);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(coef);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNewtonRefinement_TOAR"
PetscErrorCode PEPNewtonRefinement_TOAR(PEP pep,PetscInt *maxits,PetscReal *tol,PetscInt k,PetscScalar *S,PetscInt lds)
{
  PetscErrorCode ierr;
  PetscScalar    *H,*work,*dH,*fH,*dVS;
  PetscInt       ldh,i,its=1,nmat=pep->nmat,nwu=0,lwa=0;
  PetscLogDouble cnt;
  PetscBLASInt   k_,ld_,*p,info,lwork=0;
  Vec            *V=pep->V,*dV;
  PetscBool      sinvert,explicitmatrix=PETSC_FALSE;
  Mat            P,M;
  MatStructure   str;
  MPI_Comm       comm;
  FSubctx        *ctx;
  KSP            ksp;
  Matexplicitctx *matctx=NULL;

  PetscFunctionBegin;
  if (maxits) its = *maxits;
  lwa = (5+3*nmat)*k*k+2*k;
  ierr = PetscMalloc4(k*k,&dH,2*k*k,&dVS,nmat*k*k,&fH,lwa,&work);CHKERRQ(ierr);
  if (pep->st && pep->st->ops->backtransform) { /* STBackTransform */
    ierr = DSGetLeadingDimension(pep->ds,&ldh);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ldh,&ld_);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinvert);CHKERRQ(ierr);
    if (sinvert){
      ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(lwa-nwu,&lwork);CHKERRQ(ierr);
      ierr = PetscMalloc1(k,&p);CHKERRQ(ierr);
      PetscStackCall("LAPACKgetrf",LAPACKgetrf_(&k_,&k_,H,&ld_,p,&info));
      PetscStackCall("LAPACKgetri",LAPACKgetri_(&k_,H,&ld_,p,work,&lwork,&info));
      ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      pep->st->ops->backtransform = NULL;
    }
    if (pep->target!=0.0) {
      ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      for (i=0;i<k;i++) H[i+ldh*i] += pep->target;
      ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      pep->st->ops->backtransform = NULL;
    }
  }
  /* the input tolerance is not being taken into account (by the moment) */
  if (!maxits) its = 1;
  ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);

  /* check if H is in Schur form */
  for (i=0;i<k-1;i++) {
    if (H[i+1+i*ldh]!=0.0) {
#if !defined(PETSC_USES_COMPLEX)
      SETERRQ(PetscObjectComm((PetscObject)pep),1,"Iterative Refinement require the complex Schur form of the projected matrix");
#else
      SETERRQ(PetscObjectComm((PetscObject)pep),1,"Iterative Refinement requires an upper triangular projected matrix");
#endif
    }
  }
  ierr = PetscOptionsGetBool(NULL,"-newton_refinement_explicitmatrix",&explicitmatrix,NULL);CHKERRQ(ierr);
  ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)pep,&comm);CHKERRQ(ierr);
  cnt = k*sizeof(PetscBLASInt)+(lwork+k*k*(nmat+3)+nmat+k)*sizeof(PetscScalar);
  ierr = PetscLogObjectMemory((PetscObject)pep,cnt);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(pep->t,k,&dV);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(pep,k,dV);CHKERRQ(ierr);  
  ierr = KSPCreate(comm,&ksp);
  if (explicitmatrix) {
    ierr = PetscMalloc1(1,&matctx);CHKERRQ(ierr);
  }
  /* Loop performing iterative refinements */
  for (i=0;i<its;i++) {
    /* Pre-compute the polynomial basis evaluated in H */
    ierr = PEPEvaluateBasisforMatrix(pep,nmat,k,H,ldh,fH);
    ierr = PEPNRefSetUpMatrices(pep,k,H,ldh,&M,&P,matctx,(i==0)?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,M,P);CHKERRQ(ierr);
    if (i==0) {
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    }
    /* Solve the linear system */
    ierr = PEPNRefForwardSubstitution(pep,k,V,S,lds,H,ldh,fH,dV,dVS,dH,k,ksp,work+nwu,lwa-nwu,matctx);CHKERRQ(ierr);
    /* Updates X (=V*S) and H, and orthogonalizes [X;X*fH1;...;XfH(deg-1)] */
    ierr = PEPNRefUpdateInvPair(pep,k,H,ldh,fH,dH,V,S,lds,dV,dVS,work+nwu,lwa-nwu);CHKERRQ(ierr);    
  }
  ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);  
  if (sinvert) {
    ierr = PetscFree(p);CHKERRQ(ierr);
  }
  ierr = PetscFree4(dH,dVS,fH,work);CHKERRQ(ierr);
  ierr = VecDestroyVecs(k,&dV);CHKERRQ(ierr);
  if (explicitmatrix) {
    for (i=0;i<2;i++) {
      ierr = MatDestroy(&matctx->E[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(matctx);CHKERRQ(ierr);
  } else {
    ierr = MatShellGetContext(M,&ctx);CHKERRQ(ierr);
    ierr = PetscFree3(ctx->Mm,ctx->work,ctx->fih);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->w1);CHKERRQ(ierr);  
    ierr = VecDestroy(&ctx->w2);CHKERRQ(ierr);  
    ierr = PetscFree(ctx);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

