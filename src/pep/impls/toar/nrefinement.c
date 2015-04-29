/*
   Newton refinement for nonlinear eigenproblem.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/pepimpl.h>
#include <slepc/private/stimpl.h>
#include <slepcblaslapack.h>

typedef struct {
  Mat          M1;
  Vec          tN,ttN,t1,vseq;
  VecScatter   scatterctx;
  PetscSubcomm subc;
  VecScatter   scatter_sub;
  VecScatter   *scatter_id;
  Mat          *A;
  BV           M2,M3,V,W,Wt;
  PetscScalar  *M4,*w,*wt,*d,*dt;
  Vec          t,tg,Rv,Vi;
  PetscInt     idx;
  PetscBool    compM1;
} MatExplicitCtx;

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
    corr  = b[i-1];
    beta  = -g[i-1]/a[i-1];
    alpha = 1/a[i-1];
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&alpha,H,&ldh_,fH+(i-1)*k,&ldfh_,&beta,fH+off,&ldfh_));
  }
  for (j=0;j<k;j++) H[j+j*ldh] += corr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefRightSide"
/*
   Computes the residual P(H,V*S)*e_j for the polynomial
*/
static PetscErrorCode NRefRightSide(PetscInt nmat,PetscReal *pcf,Mat *A,PetscInt k,BV V,PetscScalar *S,PetscInt lds,PetscInt j,PetscScalar *H,PetscInt ldh,PetscScalar *fH,PetscScalar *DfH,PetscScalar *dH,BV dV,PetscScalar *dVS,PetscInt rds,Vec Rv,PetscScalar *Rh,BV W,Vec t)
{
  PetscErrorCode ierr;
  PetscScalar    *DS0,*DS1,*F,beta=0.0,sone=1.0,none=-1.0,tt=0.0,*h,zero=0.0,*Z,*c0;
  PetscReal      *a=pcf,*b=pcf+nmat,*g=b+nmat;
  PetscInt       i,ii,jj,lda;
  PetscBLASInt   lda_,k_,ldh_,lds_,nmat_,k2_,krds_,j_,one=1;
  Mat            M0;
  Vec            w;
  
  PetscFunctionBegin;
  ierr = PetscMalloc4(k*nmat,&h,k*k,&DS0,k*k,&DS1,k*k,&Z);CHKERRQ(ierr);
  lda = k*nmat;
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nmat,&nmat_);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&nmat_,&k_,&sone,S,&lds_,fH+j*lda,&k_,&zero,h,&k_));
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,k,nmat,h,&M0);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(W,0,nmat);CHKERRQ(ierr);
  ierr = BVMult(W,1.0,0.0,V,M0);CHKERRQ(ierr);
  ierr = MatDestroy(&M0);CHKERRQ(ierr);

  ierr = BVGetColumn(W,0,&w);CHKERRQ(ierr);
  ierr = MatMult(A[0],w,Rv);CHKERRQ(ierr);
  ierr = BVRestoreColumn(W,0,&w);CHKERRQ(ierr);
  for (i=1;i<nmat;i++) {
    ierr = BVGetColumn(W,i,&w);CHKERRQ(ierr);
    ierr = MatMult(A[i],w,t);CHKERRQ(ierr);
    ierr = BVRestoreColumn(W,i,&w);CHKERRQ(ierr);
    ierr = VecAXPY(Rv,1.0,t);
  }
  /* Update right-hand side */
  if (j) {
    ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr); 
    ierr = PetscMemzero(Z,k*k*sizeof(PetscScalar));
    ierr = PetscMemzero(DS0,k*k*sizeof(PetscScalar));
    ierr = PetscMemcpy(Z+(j-1)*k,dH+(j-1)*k,k*sizeof(PetscScalar));CHKERRQ(ierr);
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
        ierr = PetscMemzero(DS1,k*k*sizeof(PetscScalar));CHKERRQ(ierr);
        for (ii=0;ii<k;ii++) DS1[ii+(j-1)*k] = Z[ii+(j-1)*k]/a[0];
      } 
      for (jj=j;jj<k;jj++) {
        for (ii=0;ii<k;ii++) DfH[k*i+ii+jj*lda] += DS1[ii+jj*k];
      }
    }
    for (ii=0;ii<k;ii++) H[ii+ii*ldh] += tt;
    /* Update right-hand side */
    ierr = PetscBLASIntCast(2*k,&k2_);CHKERRQ(ierr); 
    ierr = PetscBLASIntCast(j,&j_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(k+rds,&krds_);CHKERRQ(ierr);
    c0 = DS0;
    ierr = PetscMemzero(Rh,k*sizeof(PetscScalar));
    for (i=0;i<nmat;i++) {
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&krds_,&j_,&sone,dVS,&k2_,fH+j*lda+i*k,&one,&zero,h,&one));
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&k_,&k_,&sone,S,&lds_,DfH+i*k+j*lda,&one,&sone,h,&one));
      ierr = BVMultVec(V,1.0,0.0,t,h);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(dV,0,rds);CHKERRQ(ierr);
      ierr = BVMultVec(dV,1.0,1.0,t,h+k);CHKERRQ(ierr);
      ierr = BVGetColumn(W,i,&w);CHKERRQ(ierr);
      ierr = MatMult(A[i],t,w);CHKERRQ(ierr);
      ierr = BVRestoreColumn(W,i,&w);CHKERRQ(ierr);
      if (i>0 && i<nmat-1) {
        PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&k_,&k_,&sone,S,&lds_,h,&one,&zero,c0,&one));
        PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&k_,&k_,&none,fH+i*k,&lda_,c0,&one,&sone,Rh,&one));
      }
    }
     
    for (i=0;i<nmat;i++) h[i] = -1.0;
    ierr = BVMultVec(W,1.0,1.0,Rv,h);CHKERRQ(ierr);
  }
  ierr = PetscFree4(h,DS0,DS1,Z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSysSolveBEM"
static PetscErrorCode NRefSysSolveBEM(PetscInt k,PetscInt sz,BV W,PetscScalar *w,BV Wt,PetscScalar *wt,PetscScalar *d,PetscScalar *dt,KSP ksp,BV T2,BV T3 ,PetscScalar *T4,PetscBool trans,Vec x1,PetscScalar *x2,Vec sol1,PetscScalar *sol2,Vec vw)
{
  PetscErrorCode ierr;
  PetscInt       i,j,incf,incc;
  PetscScalar    *y,*g,*xx2,*ww,y2,*dd;
  Vec            v,t,xx1;
  BV             WW,T;

  PetscFunctionBegin;
  ierr = PetscMalloc3(sz,&y,sz,&g,k,&xx2);CHKERRQ(ierr);
  if (trans) {
    WW = W; ww = w; dd = d; T = T3; incf = 0; incc = 1;
  } else {
    WW = Wt; ww = wt; dd = dt; T = T2; incf = 1; incc = 0;
  }
  xx1 = vw;
  ierr = VecCopy(x1,xx1);CHKERRQ(ierr);
  ierr = PetscMemcpy(xx2,x2,sz*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(sol2,k*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=sz-1;i>=0;i--) {
    ierr = BVGetColumn(WW,i,&v);CHKERRQ(ierr);
    ierr = VecConjugate(v);CHKERRQ(ierr);
    ierr = VecDot(xx1,v,y+i);CHKERRQ(ierr);
    ierr = VecConjugate(v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(WW,i,&v);CHKERRQ(ierr);
    for (j=0;j<i;j++) y[i] += ww[j+i*k]*xx2[j];
    y[i] = -(y[i]-xx2[i])/dd[i];
    ierr = BVGetColumn(T,i,&t);CHKERRQ(ierr);
    ierr = VecAXPY(xx1,-y[i],t);CHKERRQ(ierr);
    ierr = BVRestoreColumn(T,i,&t);CHKERRQ(ierr);
    for(j=0;j<=i;j++) xx2[j] -= y[i]*T4[j*incf+incc*i+(i*incf+incc*j)*k];
    g[i] = xx2[i];
  }
  if (trans) {
    ierr = KSPSolveTranspose(ksp,xx1,sol1);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(ksp,xx1,sol1);CHKERRQ(ierr);
  }
  if (trans) {
    WW = Wt; ww = wt; dd = dt; T = T2; incf = 1; incc = 0;
  } else {
    WW = W; ww = w; dd = d; T = T3; incf = 0; incc = 1;
  }
  for (i=0;i<sz;i++) {
    ierr = BVGetColumn(T,i,&t);CHKERRQ(ierr);
    ierr = VecConjugate(t);CHKERRQ(ierr);
    ierr = VecDot(sol1,t,&y2);CHKERRQ(ierr);
    ierr = VecConjugate(t);CHKERRQ(ierr);
    ierr = BVRestoreColumn(T,i,&t);CHKERRQ(ierr);
    for (j=0;j<i;j++) y2 += sol2[j]*T4[j*incf+incc*i+(i*incf+incc*j)*k];
    y2 = (g[i]-y2)/dd[i];
    ierr = BVGetColumn(WW,i,&v);CHKERRQ(ierr);
    ierr = VecAXPY(sol1,-y2,v);CHKERRQ(ierr);
    for (j=0;j<i;j++) sol2[j] -= ww[j+i*k]*y2;
    sol2[i] = y[i]+y2;
    ierr = BVRestoreColumn(WW,i,&v);CHKERRQ(ierr);
  }
  ierr = PetscFree3(y,g,xx2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSysSetup_explicit"
static PetscErrorCode NRefSysSetup_explicit(PEP pep,PetscInt k,KSP ksp,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar h,BV V,MatExplicitCtx *matctx)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,l,nmat=pep->nmat,lda=nmat*k,deg=nmat-1;
  Mat               M1=matctx->M1,*A,*At,Mk;
  PetscReal         *a=pep->pbc,*b=pep->pbc+nmat,*g=pep->pbc+2*nmat;
  PetscScalar       s,ss,*DHii,*T12,*array,*Ts,*Tr,*M4=matctx->M4,sone=1.0,zero=0.0;
  PetscScalar       *w=matctx->w,*wt=matctx->wt,*d=matctx->d,*dt=matctx->dt;
  PetscBLASInt      lds_,lda_,k_;
  MatStructure      str;
  PetscBool         flg;
  BV                M2=matctx->M2,M3=matctx->M3,W=matctx->W,Wt=matctx->Wt;
  Vec               vc,vc2;
  
  PetscFunctionBegin;
  ierr = PetscMalloc3(nmat*k*k,&T12,k*k,&Tr,PetscMax(k*k,nmat),&Ts);CHKERRQ(ierr);
  ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscMalloc1(pep->nmat,&At);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = STGetTOperators(pep->st,i,&At[i]);CHKERRQ(ierr);
    }
  } else At = pep->A; 
  if (matctx->subc) A = matctx->A;
  else A = At;
  /* Form the explicit system matrix */
  DHii = T12;
  ierr = PetscMemzero(DHii,k*k*nmat*sizeof(PetscScalar));CHKERRQ(ierr);  
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
    ierr = MatCopy(A[0],M1,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PEPEvaluateBasis(pep,h,0,Ts,NULL);CHKERRQ(ierr);
    for (j=1;j<nmat;j++) {
      ierr = MatAXPY(M1,Ts[j],A[j],str);CHKERRQ(ierr);
    }
  }

  /* T22 */
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,S,&lds_,S,&lds_,&zero,Tr,&k_));
  for (i=1;i<deg;i++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,Tr,&k_,DHii+i*k,&lda_,&zero,Ts,&k_));
    s = (i==1)?0.0:1.0; 
    PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&k_,&sone,fH+i*k,&lda_,Ts,&k_,&s,M4,&k_));
  }

  /* T12 */  
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&Mk);CHKERRQ(ierr);
  for (i=1;i<nmat;i++) {
    ierr = MatDenseGetArray(Mk,&array);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,S,&lds_,DHii+i*k,&lda_,&zero,array,&k_));
    ierr = MatDenseRestoreArray(Mk,&array);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(W,0,k);CHKERRQ(ierr);
    ierr = BVMult(W,1.0,0.0,V,Mk);CHKERRQ(ierr);
    if (i==1) {
      ierr = BVMatMult(W,A[i],M2);CHKERRQ(ierr);
    } else {
      ierr = BVMatMult(W,A[i],M3);CHKERRQ(ierr); /* using M3 as work space */
      ierr = BVAXPY(M2,1.0,M3);CHKERRQ(ierr);
    }
  }

  /* T21 */
  ierr = MatDenseGetArray(Mk,&array);CHKERRQ(ierr);
  for (i=1;i<deg;i++) {
    s = (i==1)?0.0:1.0;
    ss = PetscConj(fh[i]);
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&ss,S,&lds_,fH+i*k,&lda_,&s,array,&k_));
  }
  ierr = MatDenseRestoreArray(Mk,&array);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(M3,0,k);CHKERRQ(ierr);
  ierr = BVMult(M3,1.0,0.0,V,Mk);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    ierr = BVGetColumn(M3,i,&vc);CHKERRQ(ierr);
    ierr = VecConjugate(vc);CHKERRQ(ierr);
    ierr = BVRestoreColumn(M3,i,&vc);CHKERRQ(ierr);
  }

  ierr = KSPSetOperators(ksp,M1,M1);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&Mk);CHKERRQ(ierr);
  
  /* Set up for BEMW */
  for (i=0;i<k;i++) {
    ierr = BVGetColumn(M2,i,&vc);CHKERRQ(ierr);
    ierr = BVGetColumn(W,i,&vc2);CHKERRQ(ierr);
    ierr = NRefSysSolveBEM(k,i,W,w,Wt,wt,d,dt,ksp,M2,M3,M4,PETSC_FALSE,vc,M4+i*k,vc2,w+i*k,matctx->t);CHKERRQ(ierr);
    ierr = BVRestoreColumn(M2,i,&vc);CHKERRQ(ierr);
    ierr = BVGetColumn(M3,i,&vc);CHKERRQ(ierr);
    ierr = VecConjugate(vc);CHKERRQ(ierr);
    ierr = VecDot(vc2,vc,&d[i]);CHKERRQ(ierr);
    ierr = VecConjugate(vc);CHKERRQ(ierr);
    ierr = BVRestoreColumn(M3,i,&vc);CHKERRQ(ierr);
    for (j=0;j<i;j++) d[i] += M4[i+j*k]*w[j+i*k];
    d[i] = M4[i+i*k]-d[i];
    ierr = BVRestoreColumn(W,i,&vc2);CHKERRQ(ierr);

    ierr = BVGetColumn(M3,i,&vc);CHKERRQ(ierr);
    ierr = BVGetColumn(Wt,i,&vc2);CHKERRQ(ierr);
    for (j=0;j<=i;j++) Ts[j] = M4[i+j*k];
    ierr = NRefSysSolveBEM(k,i,W,w,Wt,wt,d,dt,ksp,M2,M3,M4,PETSC_TRUE,vc,Ts,vc2,wt+i*k,matctx->t);CHKERRQ(ierr);
    ierr = BVRestoreColumn(M3,i,&vc);CHKERRQ(ierr);
    ierr = BVGetColumn(M2,i,&vc);CHKERRQ(ierr);
    ierr = VecConjugate(vc2);CHKERRQ(ierr);
    ierr = VecDot(vc,vc2,&dt[i]);CHKERRQ(ierr);
    ierr = VecConjugate(vc2);CHKERRQ(ierr);
    ierr = BVRestoreColumn(M2,i,&vc);CHKERRQ(ierr);
    for (j=0;j<i;j++) dt[i] += M4[j+i*k]*wt[j+i*k];
    dt[i] = M4[i+i*k]-dt[i];
    ierr = BVRestoreColumn(Wt,i,&vc2);CHKERRQ(ierr);
  }  

  if (flg) {
    ierr = PetscFree(At);CHKERRQ(ierr);
  }
  ierr = PetscFree3(T12,Tr,Ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSysSolve_explicit"
static PetscErrorCode NRefSysSolve_explicit(PetscInt k,KSP ksp,Vec Rv,PetscScalar *Rh,Vec dVi,PetscScalar *dHi,MatExplicitCtx *matctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
   ierr = NRefSysSolveBEM(k,k,matctx->W,matctx->w,matctx->Wt,matctx->wt,matctx->d,matctx->dt,ksp,matctx->M2,matctx->M3 ,matctx->M4,PETSC_FALSE,Rv,Rh,dVi,dHi,matctx->t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSysIter_explicit"
static PetscErrorCode NRefSysIter_explicit(PetscInt i,PEP pep,PetscInt k,KSP ksp,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscScalar *fh,PetscScalar *H,PetscInt ldh,Vec Rv,PetscScalar *Rh,BV V,Vec dVi,PetscScalar *dHi,MatExplicitCtx *matctx)
{
  PetscErrorCode    ierr;
  PetscInt          j,m,lda=pep->nmat*k,idx;
  PetscScalar       *array2,h;
  const PetscScalar *array;
  Vec               R,Vi;
 
  PetscFunctionBegin;
  if (!matctx->subc) {
    for (j=0;j<pep->nmat;j++) fh[j] = fH[j*k+i+i*lda];
    h   = H[i+i*ldh];
    idx = i;
    R   = Rv;
    Vi  = dVi;
    ierr = NRefSysSetup_explicit(pep,k,ksp,fH,S,lds,fh,h,V,matctx);CHKERRQ(ierr);
  } else {
    if (i%matctx->subc->n==0 && (idx=i+matctx->subc->color)<k) {
      for (j=0;j<pep->nmat;j++) fh[j] = fH[j*k+idx+idx*lda];
      h = H[idx+idx*ldh];
      matctx->idx = idx;
      ierr = NRefSysSetup_explicit(pep,k,ksp,fH,S,lds,fh,h,matctx->V,matctx);CHKERRQ(ierr);
    } else idx = matctx->idx;
    ierr = VecScatterBegin(matctx->scatter_id[i%matctx->subc->n],Rv,matctx->tg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matctx->scatter_id[i%matctx->subc->n],Rv,matctx->tg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArrayRead(matctx->tg,&array);CHKERRQ(ierr);
    ierr = VecPlaceArray(matctx->t,array);CHKERRQ(ierr);
    ierr = VecCopy(matctx->t,matctx->Rv);CHKERRQ(ierr);
    ierr = VecResetArray(matctx->t);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(matctx->tg,&array);CHKERRQ(ierr);
    R  = matctx->Rv;
    Vi = matctx->Vi;
  }
  if (idx==i && idx<k) {
    ierr = NRefSysSolve_explicit(k,ksp,R,Rh,Vi,dHi,matctx);CHKERRQ(ierr);
  }
  if (matctx->subc) {
    ierr = VecGetLocalSize(Vi,&m);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Vi,&array);CHKERRQ(ierr);
    ierr = VecGetArray(matctx->tg,&array2);CHKERRQ(ierr);
    ierr = PetscMemcpy(array2,array,m*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray(matctx->tg,&array2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Vi,&array);CHKERRQ(ierr);
    ierr = VecScatterBegin(matctx->scatter_id[i%matctx->subc->n],matctx->tg,dVi,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(matctx->scatter_id[i%matctx->subc->n],matctx->tg,dVi,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    m = 0;
    for (j=0;j<i%matctx->subc->n;j++) m += matctx->subc->subsize[j];
    ierr = MPI_Bcast(dHi,k,MPIU_SCALAR,m,matctx->subc->dupparent);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNRefForwardSubstitution"
static PetscErrorCode PEPNRefForwardSubstitution(PEP pep,PetscInt k,PetscScalar *S,PetscInt lds,PetscScalar *H,PetscInt ldh,PetscScalar *fH,BV dV,PetscScalar *dVS,PetscInt *rds,PetscScalar *dH,PetscInt lddh,KSP ksp,MatExplicitCtx *matctx)
{
  PetscErrorCode ierr;
  PetscInt       i,nmat=pep->nmat,lda=nmat*k;
  PetscScalar    *fh,*Rh,*DfH;
  PetscReal      norm;
  Vec            Rv,t,dvi;
  Mat            *At;
  PetscBool      flg,lindep;
  BV             W;

  PetscFunctionBegin;
  ierr = PetscMalloc3(nmat*k*k,&DfH,k,&Rh,nmat,&fh);CHKERRQ(ierr);
  *rds = 0;
  ierr = BVCreateVec(pep->V,&Rv);CHKERRQ(ierr);
  if (matctx->subc) {
    ierr = BVCreateVec(pep->V,&t);CHKERRQ(ierr);
    ierr = BVDuplicateResize(pep->V,PetscMax(k,nmat),&W);CHKERRQ(ierr);    
  } else {
    W = matctx->W;
    ierr =   PetscObjectReference((PetscObject)W);CHKERRQ(ierr);
    t = matctx->t;
    ierr =   PetscObjectReference((PetscObject)t);CHKERRQ(ierr);
  }
  ierr = BVScale(matctx->W,0.0);CHKERRQ(ierr);
  ierr = BVScale(matctx->Wt,0.0);CHKERRQ(ierr);
  ierr = BVScale(matctx->M2,0.0);CHKERRQ(ierr);
  ierr = BVScale(matctx->M3,0.0);CHKERRQ(ierr);
  ierr = PetscMemzero(dVS,2*k*k*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(DfH,lda*k*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscMalloc1(pep->nmat,&At);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = STGetTOperators(pep->st,i,&At[i]);CHKERRQ(ierr);
    }
  } else At = pep->A; 

  /* Main loop for computing the ith columns of dX and dS */
  for (i=0;i<k;i++) {
    /* Compute and update i-th column of the right hand side */
    ierr = PetscMemzero(Rh,k*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = NRefRightSide(nmat,pep->pbc,At,k,pep->V,S,lds,i,H,ldh,fH,DfH,dH,dV,dVS,*rds,Rv,Rh,W,t);CHKERRQ(ierr);

    /* Update and solve system */
    ierr = BVGetColumn(dV,i,&dvi);CHKERRQ(ierr);
    ierr = NRefSysIter_explicit(i,pep,k,ksp,fH,S,lds,fh,H,ldh,Rv,Rh,pep->V,dvi,dH+i*k,matctx);CHKERRQ(ierr);
    matctx->compM1 = PETSC_FALSE;
    /* Orthogonalize computed solution */
    ierr = BVOrthogonalizeVec(pep->V,dvi,dVS+i*2*k,&norm,&lindep);CHKERRQ(ierr);
    ierr = BVRestoreColumn(dV,i,&dvi);CHKERRQ(ierr);
    if (!lindep) {
      ierr = BVOrthogonalizeColumn(dV,i,dVS+k+i*2*k,&norm,&lindep);CHKERRQ(ierr);
      if (!lindep) {
        dVS[k+i+i*2*k] = norm;
        ierr = BVScaleColumn(dV,i,1.0/norm);CHKERRQ(ierr);
        (*rds)++;
      }
    }
  }
  ierr = BVSetActiveColumns(dV,0,*rds);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = VecDestroy(&Rv);CHKERRQ(ierr);
  ierr = BVDestroy(&W);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(At);CHKERRQ(ierr);
  }
  ierr = PetscFree3(DfH,Rh,fh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefOrthogStep"
static PetscErrorCode NRefOrthogStep(PEP pep,PetscInt k,PetscScalar *H,PetscInt ldh,PetscScalar *fH,PetscScalar *S,PetscInt lds,PetscInt *prs)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nmat=pep->nmat,deg=nmat-1,lda=nmat*k,rs=*prs,ldg;
  PetscScalar    *T,*G,*tau,*array,sone=1.0,zero=0.0,*work;
  PetscBLASInt   rs_,lds_,k_,ldh_,info,ldg_,lda_;
  Mat            M0;

  PetscFunctionBegin;
  ierr = PetscMalloc4(rs*k,&T,k,&tau,k,&work,deg*k*k,&G);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  if (rs>k) { /* Truncate S to have k columns*/
    for (j=0;j<k;j++) {
      ierr = PetscMemcpy(T+j*rs,S+j*lds,rs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = PetscBLASIntCast(rs,&rs_);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&rs_,&k_,T,&rs_,tau,work,&k_,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
    /* Copy triangular matrix in S */
    ierr = PetscMemzero(S,lds*k*sizeof(PetscScalar));CHKERRQ(ierr);
    for (j=0;j<k;j++) for (i=0;i<=j;i++) S[j*lds+i] = T[j*rs+i];
    PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&rs_,&k_,&k_,T,&rs_,tau,work,&k_,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,rs,k,NULL,&M0);CHKERRQ(ierr);
    ierr = MatDenseGetArray(M0,&array);CHKERRQ(ierr);
    for (j=0;j<k;j++) {
      ierr = PetscMemcpy(array+j*rs,T+j*rs,rs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(M0,&array);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pep->V,0,rs);CHKERRQ(ierr);
    ierr = BVMultInPlace(pep->V,M0,0,k);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pep->V,0,k);CHKERRQ(ierr);
    ierr = MatDestroy(&M0);CHKERRQ(ierr);
    *prs = rs = k;
  }
  /* Form auxiliary matrix for the orthogonalization step */
  ldg = deg*k;
  ierr = PEPEvaluateBasisforMatrix(pep,nmat,k,H,ldh,fH);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldg,&ldg_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr);
  for (j=0;j<deg;j++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&k_,&k_,&k_,&sone,S,&lds_,fH+j*k,&lda_,&zero,G+j*k,&ldg_));
  }
  /* Orthogonalize and update S */
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&ldg_,&k_,G,&ldg_,tau,work,&k_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&k_,&k_,&sone,G,&ldg_,S,&lds_));

  /* Update H */
  PetscStackCallBLAS("BLAStrmm",BLAStrmm_("L","U","N","N",&k_,&k_,&sone,G,&ldg_,H,&ldh_));
  PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&k_,&k_,&sone,G,&ldg_,H,&ldh_));
  ierr = PetscFree4(T,tau,work,G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNRefUpdateInvPair"
static PetscErrorCode PEPNRefUpdateInvPair(PEP pep,PetscInt k,PetscScalar *H,PetscInt ldh,PetscScalar *fH,PetscScalar *dH,PetscScalar *S,PetscInt lds,BV dV,PetscScalar *dVS,PetscInt rds)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nmat=pep->nmat,lda=nmat*k;
  PetscScalar    *tau,*array,*work;
  PetscBLASInt   lds_,k_,lda_,ldh_,kdrs_,info,k2_;
  Mat            M0;

  PetscFunctionBegin;
  ierr = PetscMalloc2(k,&tau,k,&work);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lda,&lda_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldh,&ldh_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(2*k,&k2_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast((k+rds),&kdrs_);CHKERRQ(ierr);
  /* Update H */
  for (j=0;j<k;j++) {
    for (i=0;i<k;i++) H[i+j*ldh] -= dH[i+j*k];
  }
  /* Update V */
  for (j=0;j<k;j++) {
    for (i=0;i<k;i++) dVS[i+j*2*k] = -dVS[i+j*2*k]+S[i+j*lds];
    for (i=k;i<2*k;i++) dVS[i+j*2*k] = -dVS[i+j*2*k];
  }
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&kdrs_,&k_,dVS,&k2_,tau,work,&k_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  /* Copy triangular matrix in S */
  for (j=0;j<k;j++) {
    for (i=0;i<=j;i++) S[i+j*lds] = dVS[i+j*2*k];
    for (i=j+1;i<k;i++) S[i+j*lds] = 0.0;
  }
  PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&k2_,&k_,&k_,dVS,&k2_,tau,work,&k_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,&M0);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M0,&array);CHKERRQ(ierr);
  for (j=0;j<k;j++) {
    ierr = PetscMemcpy(array+j*k,dVS+j*2*k,k*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(M0,&array);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,M0,0,k);CHKERRQ(ierr);
  if (rds) {
    ierr = MatDenseGetArray(M0,&array);CHKERRQ(ierr);
    for (j=0;j<k;j++) {
      ierr = PetscMemcpy(array+j*k,dVS+k+j*2*k,rds*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(M0,&array);CHKERRQ(ierr);
    ierr = BVMultInPlace(dV,M0,0,k);CHKERRQ(ierr);
    ierr = BVAXPY(pep->V,1.0,dV);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&M0);CHKERRQ(ierr);
  ierr = NRefOrthogStep(pep,k,H,ldh,fH,S,lds,&k);CHKERRQ(ierr);
  ierr = PetscFree2(tau,work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNRefSetUp"
static PetscErrorCode PEPNRefSetUp(PEP pep,PetscInt k,PetscScalar *H,PetscInt ldh,MatExplicitCtx *matctx,PetscBool ini)
{
  PetscErrorCode    ierr;
  PetscScalar       *coef,*array;
  MatStructure      str;
  PetscInt          j,nmat=pep->nmat,i;
  MPI_Comm          comm;
  Mat               *A,*At;
  Vec               v;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nmat,&coef);CHKERRQ(ierr);
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscMalloc1(pep->nmat,&At);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = STGetTOperators(pep->st,i,&At[i]);CHKERRQ(ierr);
    }
  } else At = pep->A;
  if (ini) {
    if (matctx->subc) {
      A = matctx->A;
      comm = PetscSubcommChild(matctx->subc);
    } else {
      matctx->V = pep->V;
      A = At;
      ierr = PetscObjectGetComm((PetscObject)pep,&comm);CHKERRQ(ierr);
      ierr = MatCreateVecs(pep->A[0],&matctx->t,NULL);CHKERRQ(ierr);

    }
    ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
    ierr = MatDuplicate(A[0],MAT_COPY_VALUES,&matctx->M1);CHKERRQ(ierr);
    j = (matctx->subc)?matctx->subc->color:0;
    ierr = PEPEvaluateBasis(pep,H[j+j*ldh],0,coef,NULL);CHKERRQ(ierr);
    for (j=1;j<nmat;j++) {
      ierr = MatAXPY(matctx->M1,coef[j],A[j],str);CHKERRQ(ierr);
    }
    ierr = BVDuplicateResize(matctx->V,PetscMax(k,pep->nmat),&matctx->W);CHKERRQ(ierr);
    ierr = BVDuplicateResize(matctx->V,k,&matctx->M2);CHKERRQ(ierr);
    ierr = BVDuplicate(matctx->M2,&matctx->M3);CHKERRQ(ierr);
    ierr = BVDuplicate(matctx->M2,&matctx->Wt);CHKERRQ(ierr);
    ierr = PetscMalloc5(k*k,&matctx->M4,k*k,&matctx->w,k*k,&matctx->wt,k,&matctx->d,k,&matctx->dt);CHKERRQ(ierr);
    matctx->compM1 = PETSC_TRUE;
    ierr = PEPRefineGetKSP(pep,&pep->refineksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(pep->refineksp,matctx->M1,matctx->M1);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(pep->refineksp);CHKERRQ(ierr);
  } else {
    if (matctx->subc) {
      /* Scatter vectors pep->V */
      for (i=0;i<k;i++) {
        ierr = BVGetColumn(pep->V,i,&v);CHKERRQ(ierr);
        ierr = VecScatterBegin(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = BVRestoreColumn(pep->V,i,&v);CHKERRQ(ierr);
        ierr = VecGetArray(matctx->tg,&array);CHKERRQ(ierr);
        ierr = VecPlaceArray(matctx->t,(const PetscScalar*)array);CHKERRQ(ierr);
        ierr = BVInsertVec(matctx->V,i,matctx->t);CHKERRQ(ierr);
        ierr = VecResetArray(matctx->t);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(coef);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(At);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSubcommSetup"
static PetscErrorCode NRefSubcommSetup(PEP pep,PetscInt k,MatExplicitCtx *matctx,PetscInt nsubc)
{
  PetscErrorCode    ierr;
  PetscInt          i,si,j,m0,n0,nloc0,nloc_sub,*idx1,*idx2;
  IS                is1,is2;
  BVType            type;
  Vec               v;
  const PetscScalar *array;
  Mat               *A;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscMalloc1(pep->nmat,&A);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = STGetTOperators(pep->st,i,&A[i]);CHKERRQ(ierr);
    }
  } else A = pep->A;
  

  ierr = PetscMalloc2(pep->nmat,&matctx->A,nsubc,&matctx->scatter_id);CHKERRQ(ierr);
  for (i=0;i<pep->nmat;i++) {
    ierr = MatCreateRedundantMatrix(A[i],0,PetscSubcommChild(matctx->subc),MAT_INITIAL_MATRIX,&matctx->A[i]);CHKERRQ(ierr);    
  }

  /* Create Scatter */
  ierr = MatCreateVecs(matctx->A[0],&matctx->t,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(matctx->A[0],&nloc_sub,NULL);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscSubcommContiguousParent(matctx->subc),nloc_sub,PETSC_DECIDE,&matctx->tg);CHKERRQ(ierr);
  ierr = BVGetColumn(pep->V,0,&v);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(v,&n0,&m0);CHKERRQ(ierr);
  nloc0 = m0-n0;
  ierr = PetscMalloc2(matctx->subc->n*nloc0,&idx1,matctx->subc->n*nloc0,&idx2);CHKERRQ(ierr);
  j = 0;
  for (si=0;si<matctx->subc->n;si++) {
    for (i=n0;i<m0;i++) {
      idx1[j]   = i;
      idx2[j++] = i+pep->n*si;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pep),matctx->subc->n*nloc0,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pep),matctx->subc->n*nloc0,idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
  ierr = VecScatterCreate(v,is1,matctx->tg,is2,&matctx->scatter_sub);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  for (si=0;si<matctx->subc->n;si++) {
    j=0;
    for (i=n0;i<m0;i++) {
      idx1[j] = i;
      idx2[j++] = i+pep->n*si;
    }
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pep),nloc0,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pep),nloc0,idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
    ierr = VecScatterCreate(v,is1,matctx->tg,is2,&matctx->scatter_id[si]);CHKERRQ(ierr);
    ierr = ISDestroy(&is1);CHKERRQ(ierr);
    ierr = ISDestroy(&is2);CHKERRQ(ierr);
  }
  ierr = BVRestoreColumn(pep->V,0,&v);CHKERRQ(ierr);
  ierr = PetscFree2(idx1,idx2);CHKERRQ(ierr);

  /* Duplicate pep->V vecs */
  ierr = BVGetType(pep->V,&type);CHKERRQ(ierr);
  ierr = BVCreate(PetscSubcommChild(matctx->subc),&matctx->V);CHKERRQ(ierr);
  ierr = BVSetType(matctx->V,type);CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(matctx->V,matctx->t,k);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    ierr = BVGetColumn(pep->V,i,&v);CHKERRQ(ierr);
    ierr = VecScatterBegin(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matctx->scatter_sub,v,matctx->tg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pep->V,i,&v);CHKERRQ(ierr);
    ierr = VecGetArrayRead(matctx->tg,&array);CHKERRQ(ierr);
    ierr = VecPlaceArray(matctx->t,(const PetscScalar*)array);CHKERRQ(ierr);
    ierr = BVInsertVec(matctx->V,i,matctx->t);CHKERRQ(ierr);
    ierr = VecResetArray(matctx->t);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(matctx->tg,&array);CHKERRQ(ierr);
  }

  ierr = VecDuplicate(matctx->t,&matctx->Rv);CHKERRQ(ierr);
  ierr = VecDuplicate(matctx->t,&matctx->Vi);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NRefSubcommDestroy"
static PetscErrorCode NRefSubcommDestroy(PEP pep,MatExplicitCtx *matctx)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = VecScatterDestroy(&matctx->scatter_sub);CHKERRQ(ierr);
  for (i=0;i<matctx->subc->n;i++) {
    ierr = VecScatterDestroy(&matctx->scatter_id[i]);CHKERRQ(ierr);
  }
  for (i=0;i<pep->nmat;i++) {
    ierr = MatDestroy(&matctx->A[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(matctx->A,matctx->scatter_id);CHKERRQ(ierr);
  ierr = BVDestroy(&matctx->V);CHKERRQ(ierr);
  ierr = VecDestroy(&matctx->t);CHKERRQ(ierr);
  ierr = VecDestroy(&matctx->tg);CHKERRQ(ierr);
  ierr = VecDestroy(&matctx->Rv);CHKERRQ(ierr);
  ierr = VecDestroy(&matctx->Vi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNewtonRefinement_TOAR"
PetscErrorCode PEPNewtonRefinement_TOAR(PEP pep,PetscScalar sigma,PetscInt *maxits,PetscReal *tol,PetscInt k,PetscScalar *S,PetscInt lds,PetscInt *prs)
{
  PetscErrorCode ierr;
  PetscScalar    *H,*work,*dH,*fH,*dVS;
  PetscInt       ldh,i,j,its=1,nmat=pep->nmat,nsubc=pep->npart,rds;
  PetscBLASInt   k_,ld_,*p,info;
  BV             dV;
  PetscBool      sinvert,flg;
  MatExplicitCtx *matctx=NULL;
  Vec            v;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PEP_Refine,pep,0,0,0);CHKERRQ(ierr);
  if (k > pep->n) SETERRQ1(PetscObjectComm((PetscObject)pep),1,"Multiple Refinement available only for invariant pairs of dimension smaller than n=%D",pep->n);
  /* the input tolerance is not being taken into account (by the moment) */
  its = *maxits;
  ierr = PetscMalloc3(k*k,&dH,nmat*k*k,&fH,k,&work);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(pep->ds,&ldh);CHKERRQ(ierr);
  ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
  ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*k*k,&dVS);CHKERRQ(ierr);
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (!flg && pep->st && pep->st->ops->backtransform) { /* STBackTransform */
    ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ldh,&ld_);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinvert);CHKERRQ(ierr);
    if (sinvert){
      ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      ierr = PetscMalloc1(k,&p);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&k_,&k_,H,&ld_,p,&info));
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&k_,H,&ld_,p,work,&k_,&info));
      ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      pep->st->ops->backtransform = NULL;
    }
    if (sigma!=0.0) {
      ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      for (i=0;i<k;i++) H[i+ldh*i] += sigma; 
      ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      pep->st->ops->backtransform = NULL;
    }
  }
  if ((pep->scale==PEP_SCALE_BOTH || pep->scale==PEP_SCALE_SCALAR) && pep->sfactor!=1.0) {
    ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    for (j=0;j<k;j++) {
      for (i=0;i<k;i++) H[i+j*ldh] *= pep->sfactor;
    }
    ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    if (!flg) {
      /* Restore original values */
      for (i=0;i<pep->nmat;i++){
        pep->pbc[pep->nmat+i] *= pep->sfactor;
        pep->pbc[2*pep->nmat+i] *= pep->sfactor*pep->sfactor;
      }
    }
  }
  if ((pep->scale==PEP_SCALE_DIAGONAL || pep->scale==PEP_SCALE_BOTH) && pep->Dr) {
    for (i=0;i<k;i++) {
      ierr = BVGetColumn(pep->V,i,&v);CHKERRQ(ierr);
      ierr = VecPointwiseMult(v,v,pep->Dr);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,i,&v);CHKERRQ(ierr);
    }
  }
  ierr = DSGetArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);

  ierr = NRefOrthogStep(pep,k,H,ldh,fH,S,lds,prs);CHKERRQ(ierr);
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
  if (!pep->schur && nsubc>k) SETERRQ(PetscObjectComm((PetscObject)pep),1,"Amount of subcommunicators should not be larger than the invariant pair's dimension");
  ierr = BVSetActiveColumns(pep->V,0,k);CHKERRQ(ierr);
  ierr = BVDuplicateResize(pep->V,k,&dV);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)dV);CHKERRQ(ierr);  
  ierr = PetscMalloc1(1,&matctx);CHKERRQ(ierr);
  if (nsubc>1) { /* spliting in subcommunicators */
    matctx->subc = pep->refinesubc;
    ierr = NRefSubcommSetup(pep,k,matctx,nsubc);CHKERRQ(ierr);
  } else {
    matctx->subc=NULL;
  }

  /* Loop performing iterative refinements */
  for (i=0;i<its;i++) {
    /* Pre-compute the polynomial basis evaluated in H */
    ierr = PEPEvaluateBasisforMatrix(pep,nmat,k,H,ldh,fH);
    ierr = PEPNRefSetUp(pep,k,H,ldh,matctx,i==0?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr);
    /* Solve the linear system */
    ierr = PEPNRefForwardSubstitution(pep,k,S,lds,H,ldh,fH,dV,dVS,&rds,dH,k,pep->refineksp,matctx);CHKERRQ(ierr);
    /* Update X (=V*S) and H, and orthogonalize [X;X*fH1;...;XfH(deg-1)] */
    ierr = PEPNRefUpdateInvPair(pep,k,H,ldh,fH,dH,S,lds,dV,dVS,rds);CHKERRQ(ierr);    
  }
  ierr = DSRestoreArray(pep->ds,DS_MAT_A,&H);CHKERRQ(ierr);  
  if (!flg && sinvert) {
    ierr = PetscFree(p);CHKERRQ(ierr);
  }
  ierr = PetscFree3(dH,fH,work);CHKERRQ(ierr);
  ierr = PetscFree(dVS);CHKERRQ(ierr);
  ierr = BVDestroy(&dV);CHKERRQ(ierr);
  ierr = BVDestroy(&matctx->W);CHKERRQ(ierr);
  ierr = BVDestroy(&matctx->Wt);CHKERRQ(ierr);
  ierr = BVDestroy(&matctx->M2);CHKERRQ(ierr);
  ierr = BVDestroy(&matctx->M3);CHKERRQ(ierr);
  ierr = MatDestroy(&matctx->M1);CHKERRQ(ierr);
  ierr = VecDestroy(&matctx->t);CHKERRQ(ierr);
  ierr = PetscFree5(matctx->M4,matctx->w,matctx->wt,matctx->d,matctx->dt);CHKERRQ(ierr);
  if (nsubc>1) {
    ierr = NRefSubcommDestroy(pep,matctx);CHKERRQ(ierr);
  } 
  ierr = PetscFree(matctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PEP_Refine,pep,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

