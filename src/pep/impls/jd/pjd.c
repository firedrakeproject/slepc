/*

   SLEPc polynomial eigensolver: "jd"

   Method: Jacobi-Davidson

   Algorithm:

       Jacobi-Davidson for polynomial eigenvalue problems.
       Based on code contributed by the authors of [2] below.

   References:

       [1] G.L.G. Sleijpen et al., "Jacobi-Davidson type methods for
           generalized eigenproblems and polynomial eigenproblems", BIT
           36(3):595-633, 1996.

       [2] Feng-Nan Hwang, Zih-Hao Wei, Tsung-Ming Huang, Weichung Wang,
           "A Parallel Additive Schwarz Preconditioned Jacobi-Davidson
           Algorithm for Polynomial Eigenvalue Problems in Quantum Dot
           Simulation", J. Comput. Phys. 229(8):2932-2947, 2010.

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

#include <slepc/private/pepimpl.h>    /*I "slepcpep.h" I*/
#include "pjdp.h"
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "PEPJDDuplicateBasis"
/*
   Duplicate and resize auxiliary basis
*/
static PetscErrorCode PEPJDDuplicateBasis(PEP pep,BV *basis)
{
  PetscErrorCode     ierr;
  PetscInt           nloc,m;
  PetscMPIInt        rank,nproc;
  BVType             type;
  BVOrthogType       otype;
  BVOrthogRefineType oref;
  PetscReal          oeta;
  BVOrthogBlockType  oblock;

  PetscFunctionBegin;
  if (pep->nev>1) {
    ierr = BVCreate(PetscObjectComm((PetscObject)pep),basis);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&nproc);CHKERRQ(ierr);
    ierr = BVGetSizes(pep->V,&nloc,NULL,&m);CHKERRQ(ierr);
    if (rank==nproc-1) nloc += pep->nev-1;
    ierr = BVSetSizes(*basis,nloc,PETSC_DECIDE,m);CHKERRQ(ierr);
    ierr = BVGetType(pep->V,&type);CHKERRQ(ierr);
    ierr = BVSetType(*basis,type);CHKERRQ(ierr);
    ierr = BVGetOrthogonalization(pep->V,&otype,&oref,&oeta,&oblock);CHKERRQ(ierr);
    ierr = BVSetOrthogonalization(*basis,otype,oref,oeta,oblock);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)*basis);CHKERRQ(ierr);
  } else {
    ierr = BVDuplicate(pep->V,basis);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetUp_JD"
PetscErrorCode PEPSetUp_JD(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBool      isprecond,flg;
  PetscInt       i;
  KSP            ksp;

  PetscFunctionBegin;
  pep->lineariz = PETSC_FALSE;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!pep->max_it) pep->max_it = PetscMax(100,2*pep->n/pep->ncv);
  if (!pep->which) pep->which = PEP_TARGET_MAGNITUDE;
  if (pep->which != PEP_TARGET_MAGNITUDE) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"PEPJD only supports which=target_magnitude");;

  /* Set STPRECOND as the default ST */
  if (!((PetscObject)pep->st)->type_name) {
    ierr = STSetType(pep->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STPRECOND,&isprecond);CHKERRQ(ierr);
  if (!isprecond) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"JD only works with PRECOND spectral transformation");

  if (pep->basis!=PEP_BASIS_MONOMIAL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver not implemented for non-monomial bases");
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver requires the ST transformation flag unset, see STSetTransform()");

  /* Set the default options of the KSP */
  ierr = STGetKSP(pep->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPBCGSL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,1e-5,PETSC_DEFAULT,PETSC_DEFAULT,100);CHKERRQ(ierr);
  }

  if (!pjd->keep) pjd->keep = 0.5;

  ierr = PEPAllocateSolution(pep,0);CHKERRQ(ierr);
  ierr = PEPSetWorkVecs(pep,5);CHKERRQ(ierr);
  ierr = PetscMalloc2(pep->nmat,&pjd->TV,pep->nmat,&pjd->AX);CHKERRQ(ierr);
  for (i=0;i<pep->nmat;i++) {
    ierr = PEPJDDuplicateBasis(pep,pjd->TV+i);CHKERRQ(ierr);
  }
  ierr = PEPJDDuplicateBasis(pep,&pjd->W);CHKERRQ(ierr);
  if (pep->nev>1) {
    ierr = PEPJDDuplicateBasis(pep,&pjd->V);CHKERRQ(ierr);
    ierr = BVSetFromOptions(pjd->V);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = BVDuplicateResize(pep->V,pep->nev-1,pjd->AX+i);CHKERRQ(ierr);
    }
    ierr = BVDuplicateResize(pep->V,pep->nev,&pjd->X);CHKERRQ(ierr);
    ierr = PetscCalloc3((pep->nev)*(pep->nev),&pjd->XpX,pep->nev*pep->nev,&pjd->T,pep->nev*pep->nev*pep->nmat,&pjd->Tj);CHKERRQ(ierr);
  } else pjd->V = pep->V;
  ierr = DSSetType(pep->ds,DSPEP);CHKERRQ(ierr);
  ierr = DSPEPSetDegree(pep->ds,pep->nmat-1);CHKERRQ(ierr);
  ierr = DSAllocate(pep->ds,pep->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDUpdateTV"
/*
   Updates columns (low to (high-1)) of TV[i]
*/
static PetscErrorCode PEPJDUpdateTV(PEP pep,PetscInt low,PetscInt high,Vec *w)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscInt       pp,col,i,j,nloc,nconv,deg=pep->nmat-1;
  Vec            v1,v2,t1,t2;
  PetscScalar    *array1,*array2,*x2,*tt,*xx,*y2,zero=0.0,sone=1.0;
  PetscMPIInt    rk,np,count;
  PetscBLASInt   n,ld,one=1;

  PetscFunctionBegin;
  nconv = pjd->nconv;
  ierr = PetscMalloc3(nconv,&tt,nconv,&x2,nconv,&xx);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
  ierr = BVGetSizes(pep->V,&nloc,NULL,NULL);CHKERRQ(ierr); 
  t1 = w[0];
  t2 = w[1];
  for (col=low;col<high;col++) {
    ierr = BVGetColumn(pjd->V,col,&v1);CHKERRQ(ierr);
    ierr = VecGetArray(v1,&array1);CHKERRQ(ierr);
    if (nconv>0) {
      if (rk==np-1) { for (i=0;i<nconv;i++) x2[i] = array1[nloc+i]; }
      ierr = PetscMPIIntCast(nconv,&count);CHKERRQ(ierr);
      ierr = MPI_Bcast(x2,nconv,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)pep));CHKERRQ(ierr);
    }
    ierr = VecPlaceArray(t1,array1);CHKERRQ(ierr);
    for (pp=0;pp<pep->nmat;pp++) {
      ierr = BVGetColumn(pjd->TV[pp],col,&v2);CHKERRQ(ierr);
      ierr = VecGetArray(v2,&array2);CHKERRQ(ierr);
      ierr = VecPlaceArray(t2,array2);CHKERRQ(ierr);
      ierr = MatMult(pep->A[pp],t1,t2);CHKERRQ(ierr);
      if (nconv) {
        ierr = PetscBLASIntCast(pjd->nconv,&n);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(pep->nev,&ld);CHKERRQ(ierr);
        for (j=0;j<nconv;j++) tt[j] = x2[j];
        for (i=pp+1;i<pep->nmat;i++) {
          ierr = BVMultVec(pjd->AX[i],1.0,1.0,t2,tt);CHKERRQ(ierr);
          if (i!=pep->nmat-1) PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n,pjd->T,&ld,tt,&one));
        }
        ierr = BVDotVec(pjd->X,t1,xx);CHKERRQ(ierr);
        if (rk==np-1 && pp<deg) {
          y2 = array2+nloc;
          for (j=0;j<nconv;j++) { y2[j] = xx[j]; xx[j] = x2[j]; }
          PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&n,pjd->Tj+ld*ld*pp,&ld,y2,&one));
          for (i=pp+1;i<pep->nmat-1;i++) {
            PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,pjd->XpX,&ld,xx,&one,&zero,tt,&one));
            PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&n,pjd->Tj+ld*ld*i,&ld,tt,&one));
            for (j=0;j<nconv;j++) y2[j] += tt[j];
            if (i<pep->nmat-2) PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n,pjd->T,&ld,xx,&one));
          }
        }
      }
      ierr = VecResetArray(t2);CHKERRQ(ierr);
      ierr = VecRestoreArray(v2,&array2);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->TV[pp],col,&v2);CHKERRQ(ierr);        
    }
    ierr = VecResetArray(t1);CHKERRQ(ierr);
    ierr = VecRestoreArray(v1,&array1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,col,&v1);CHKERRQ(ierr);
  }
  ierr = PetscFree3(tt,x2,xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDOrthogonalize"
/*
   RRQR of X. Xin*P=Xou*R. Rank of R is rk
*/
static PetscErrorCode PEPJDOrthogonalize(PetscInt row,PetscInt col,PetscScalar *X,PetscInt ldx,PetscInt *rk,PetscInt *P,PetscScalar *R,PetscInt ldr)
{
#if defined(SLEPC_MISSING_LAPACK_GEQP3)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEQP3 - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,n,r;
  PetscBLASInt   row_,col_,ldx_,*p,lwork,info,n_;
  PetscScalar    *tau,*work;
  PetscReal      tol,*rwork;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(row,&row_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(col,&col_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldx,&ldx_);CHKERRQ(ierr);
  n = PetscMin(row,col);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  lwork = 3*col_+1;
  ierr = PetscMalloc4(col,&p,n,&tau,lwork,&work,2*col,&rwork);CHKERRQ(ierr);
  for (i=1;i<col;i++) p[i] = 0;
  p[0] = 1;

  /* rank revealing QR */
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgeqp3",LAPACKgeqp3_(&row_,&col_,X,&ldx_,p,tau,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKgeqp3",LAPACKgeqp3_(&row_,&col_,X,&ldx_,p,tau,work,&lwork,&info));
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQP3 %d",info);
  if (P) for (i=0;i<col;i++) P[i] = p[i];

  /* rank computation */
  tol = PetscMax(row,col)*PETSC_MACHINE_EPSILON*PetscAbsScalar(X[0]);
  r = 1;
  for (i=1;i<n;i++) { 
    if (PetscAbsScalar(X[i+ldx*i])>tol) r++;
    else break;
  }
  if (rk) *rk=r;

  /* copy upper triangular matrix if requested */
  if (R) {
     for (i=0;i<r;i++) {
       ierr = PetscMemzero(R+i*ldr,r*sizeof(PetscScalar));CHKERRQ(ierr);
       for (j=0;j<=i;j++) R[i*ldr+j] = X[i*ldx+j];
     }
  }
  PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&row_,&n_,&n_,X,&ldx_,tau,work,&lwork,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
  ierr = PetscFree4(p,tau,work,rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDExtendedPCApply"
/*
   Application of extended preconditioner
*/
static PetscErrorCode PEPJDExtendedPCApply(PC pc,Vec x,Vec y)
{
  PetscInt          i,j,nloc,n,ld;
  PetscMPIInt       rk,np,count;
  Vec               tx,ty;
  PEP_JD_PCSHELL    *ctx;
  PetscErrorCode    ierr;
  const PetscScalar *array1;
  PetscScalar       *x2=NULL,*t=NULL,*ps,*array2;
  PetscBLASInt      one=1.0,ld_,n_;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);
  n  = ctx->n;
  ps = ctx->ps;
  ld = ctx->ld;
  if (n) {
    ierr = PetscMalloc2(n,&x2,n,&t);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rk);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&np);CHKERRQ(ierr);
    if (rk==np-1) {
      ierr = VecGetLocalSize(ctx->work[0],&nloc);CHKERRQ(ierr); 
      ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
      for (i=0;i<n;i++) x2[i] = array1[nloc+i];
      ierr = VecRestoreArrayRead(x,&array1);CHKERRQ(ierr);
    }
    ierr = PetscMPIIntCast(n,&count);CHKERRQ(ierr);
    ierr = MPI_Bcast(x2,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  }

  /* y = B\x apply PC */
  tx = ctx->work[0];
  ty = ctx->work[1];
  ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecPlaceArray(tx,array1);CHKERRQ(ierr);
  ierr = VecGetArray(y,&array2);CHKERRQ(ierr);
  ierr = VecPlaceArray(ty,array2);CHKERRQ(ierr);
  ierr = PCApply(ctx->pc,tx,ty);CHKERRQ(ierr);
  if (n) {
    for (j=0;j<n;j++) {
      t[j] = 0.0;
      for (i=0;i<n;i++) t[j] += ctx->M[i+j*ld]*x2[i];
    }
    if (rk==np-1) for (i=0;i<n;i++) array2[nloc+i] = t[i];
    ierr = PetscBLASIntCast(ld,&ld_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
    PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n_,ps,&ld_,t,&one));
    ierr = BVMultVec(ctx->X,-1.0,1.0,ty,t);CHKERRQ(ierr);
    ierr = PetscFree2(x2,t);CHKERRQ(ierr);
  }
  ierr = VecResetArray(tx);CHKERRQ(ierr);
  ierr = VecResetArray(ty);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&array2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCShellApply_PEPJD"
/*
   Application of shell preconditioner:
      y = B\x - eta*B\p,  with eta = (u'*B\x)/(u'*B\p)
*/
static PetscErrorCode PCShellApply_PEPJD(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscScalar    eta;
  PEP_JD_PCSHELL *ctx;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);

  /* y = B\x apply extended PC */
  ierr = PEPJDExtendedPCApply(pc,x,y);CHKERRQ(ierr);

  /* Compute eta = u'*y / u'*Bp */
  ierr = VecDot(y,ctx->u,&eta);CHKERRQ(ierr);
  eta /= ctx->gamma;
  
  /* y = y - eta*Bp */
  ierr = VecAXPY(y,-eta,ctx->Bp);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDCopyToExtendedVec"
static PetscErrorCode PEPJDCopyToExtendedVec(PEP pep,Vec v,PetscScalar *a,PetscInt na,PetscInt off,Vec vex,PetscBool back)
{
  PetscErrorCode ierr;
  PetscMPIInt    np,rk,count;
  PetscScalar    *array1,*array2;
  PetscInt       nloc;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
  ierr = BVGetSizes(pep->V,&nloc,NULL,NULL);CHKERRQ(ierr);
  if (v) {
    ierr = VecGetArray(v,&array1);CHKERRQ(ierr);
    ierr = VecGetArray(vex,&array2);CHKERRQ(ierr);
    if (back) {
      ierr = PetscMemcpy(array1,array2,nloc*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemcpy(array2,array1,nloc*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(v,&array1);CHKERRQ(ierr);
    ierr = VecRestoreArray(vex,&array2);CHKERRQ(ierr);
  }
  if (a) {
    if (rk==np-1) {
      ierr = VecGetArray(vex,&array2);CHKERRQ(ierr);
      if (back) {
        ierr = PetscMemcpy(a,array2+nloc+off,na*sizeof(PetscScalar));CHKERRQ(ierr);
      } else {
        ierr = PetscMemcpy(array2+nloc+off,a,na*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(vex,&array2);CHKERRQ(ierr);
    }
    if (back) {
      ierr = PetscMPIIntCast(na,&count);CHKERRQ(ierr);
      ierr = MPI_Bcast(a,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)pep));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDComputePResidual"
static PetscErrorCode PEPJDComputePResidual(PEP pep,Vec u,PetscScalar theta,Vec p,Vec *work)
{
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscErrorCode ierr;
  PetscMPIInt    rk,np,count;
  Vec            tu,tp,w;
  PetscScalar    *array1,*array2,*x2=NULL,*y2,fact=1.0,*q=NULL,*tt=NULL,*xx=NULL,sone=1.0,zero=0.0;
  PetscInt       i,j,nconv=pjd->nconv,nloc,deg=pep->nmat-1;
  PetscBLASInt   n,ld,one=1;

  PetscFunctionBegin;
  if (nconv>0) {
    ierr = PetscMalloc4(nconv,&xx,nconv,&tt,nconv,&x2,nconv,&q);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
    if (rk==np-1) {
      ierr = BVGetSizes(pep->V,&nloc,NULL,NULL);CHKERRQ(ierr); 
      ierr = VecGetArray(u,&array1);CHKERRQ(ierr);
      for (i=0;i<nconv;i++) x2[i] = array1[nloc+i];
      ierr = VecRestoreArray(u,&array1);CHKERRQ(ierr);
    }
    ierr = PetscMPIIntCast(nconv,&count);CHKERRQ(ierr);
    ierr = MPI_Bcast(x2,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)pep));CHKERRQ(ierr);
  }
  tu = work[0];
  tp = work[1];
  w  = work[2];
  ierr = VecGetArray(u,&array1);CHKERRQ(ierr);
  ierr = VecPlaceArray(tu,array1);CHKERRQ(ierr);
  ierr = VecGetArray(p,&array2);CHKERRQ(ierr);
  ierr = VecPlaceArray(tp,array2);CHKERRQ(ierr);
  ierr = VecSet(tp,0.0);CHKERRQ(ierr);
  for (i=1;i<pep->nmat;i++) {
    ierr = MatMult(pep->A[i],tu,w);CHKERRQ(ierr);
    ierr = VecAXPY(tp,fact*(PetscReal)i,w);CHKERRQ(ierr);
    fact *= theta;
  }
  if (nconv) {
    ierr = PetscBLASIntCast(nconv,&n);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(pep->nev,&ld);CHKERRQ(ierr);
    for (j=0;j<nconv;j++) q[j] = x2[j];
    fact = theta;
    for (i=2;i<pep->nmat;i++) {
      ierr = BVMultVec(pjd->AX[i],1.0,1.0,tp,q);CHKERRQ(ierr);
      PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n,pjd->T,&ld,q,&one));
      for (j=0;j<nconv;j++) q[j] += (PetscReal)i*fact*x2[j];
      fact *= theta;
    }
    ierr = BVSetActiveColumns(pjd->X,0,nconv);CHKERRQ(ierr);
    ierr = BVDotVec(pjd->X,tu,xx);CHKERRQ(ierr);
    if (rk==np-1) {
      y2 = array2+nloc;
      for (i=0;i<nconv;i++) { q[i] = x2[i]; y2[i] = xx[i]; }
      PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&n,pjd->Tj+ld*ld,&ld,y2,&one));
      fact = theta;
      for (j=2;j<deg;j++) {
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,pjd->XpX,&ld,q,&one,&zero,tt,&one));
        for (i=0;i<nconv;i++) tt[i] += (PetscReal)j*fact*xx[i];
        PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&n,pjd->Tj+ld*ld*j,&ld,tt,&one));
        for (i=0;i<nconv;i++) y2[i] += tt[i];
        PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n,pjd->T,&ld,q,&one));
        for (i=0;i<nconv;i++) q[i] += (PetscReal)j*fact*x2[i];
        fact *= theta;
      }
    }
    ierr = PetscFree4(xx,tt,x2,q);CHKERRQ(ierr);
  }
  ierr = VecResetArray(tu);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&array1);CHKERRQ(ierr);
  ierr = VecResetArray(tp);CHKERRQ(ierr);
  ierr = VecRestoreArray(p,&array2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDProcessInitialSpace"
static PetscErrorCode PEPJDProcessInitialSpace(PEP pep,Vec *w)
{
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscErrorCode ierr;
  PetscScalar    *tt;
  Vec            vg,wg;
  PetscInt       i;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = PetscMalloc1(pep->nev-1,&tt);CHKERRQ(ierr);
  if (pep->nini==0) {
    ierr = BVSetRandomColumn(pjd->V,0);CHKERRQ(ierr);
    for (i=0;i<pep->nev-1;i++) tt[i] = 0.0;
    ierr = BVGetColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = PEPJDCopyToExtendedVec(pep,NULL,tt,pep->nev-1,0,vg,PETSC_FALSE);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = BVNormColumn(pjd->V,0,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(pjd->V,0,1.0/norm);CHKERRQ(ierr);
    ierr = BVGetColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = BVGetColumn(pjd->W,0,&wg);CHKERRQ(ierr);
    ierr = VecSet(wg,0.0);CHKERRQ(ierr);
    ierr = PEPJDComputePResidual(pep,vg,pep->target,wg,w);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->W,0,&wg);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = BVNormColumn(pjd->W,0,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(pjd->W,0,1.0/norm);CHKERRQ(ierr);
  } else {
   SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TO DO");
  }
  ierr = PetscFree(tt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDShellMatMult"
static PetscErrorCode PEPJDShellMatMult(Mat P,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_JD_MATSHELL   *matctx;
  PEP_JD            *pjd;
  PetscMPIInt       rk,np,count;
  PetscInt          i,j,nconv,nloc,nmat,ldt,deg;
  Vec               tx,ty;
  PetscScalar       *array2,*x2=NULL,*y2,fact=1.0,*q=NULL,*tt=NULL,*xx=NULL,theta,*yy=NULL,sone=1.0,zero=0.0;
  PetscBLASInt      n,ld,one=1;
  const PetscScalar *array1;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(P,(void**)&matctx);CHKERRQ(ierr);
  pjd   = (PEP_JD*)(matctx->pep->data);
  nconv = pjd->nconv;
  theta = matctx->theta;
  nmat  = matctx->pep->nmat;
  deg   = nmat-1;
  ldt   = matctx->pep->nev;
  if (nconv>0) {
    ierr = PetscMalloc5(nconv,&tt,nconv,&x2,nconv,&q,nconv,&xx,nconv,&yy);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)P),&rk);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)P),&np);CHKERRQ(ierr);
    if (rk==np-1) {
      ierr = BVGetSizes(matctx->pep->V,&nloc,NULL,NULL);CHKERRQ(ierr); 
      ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
      for (i=0;i<nconv;i++) x2[i] = array1[nloc+i];
      ierr = VecRestoreArrayRead(x,&array1);CHKERRQ(ierr);
    }
    ierr = PetscMPIIntCast(nconv,&count);CHKERRQ(ierr);
    ierr = MPI_Bcast(x2,nconv,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)P));CHKERRQ(ierr);
  }
  tx = matctx->work[0];
  ty = matctx->work[1];
  ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecPlaceArray(tx,array1);CHKERRQ(ierr);
  ierr = VecGetArray(y,&array2);CHKERRQ(ierr);
  ierr = VecPlaceArray(ty,array2);CHKERRQ(ierr);
  ierr = VecSet(ty,0.0);CHKERRQ(ierr);
  ierr = MatMult(matctx->P,tx,ty);CHKERRQ(ierr);
  if (nconv) {
    ierr = PetscBLASIntCast(pjd->nconv,&n);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ldt,&ld);CHKERRQ(ierr);
    for (j=0;j<nconv;j++) q[j] = x2[j];
    fact = theta;
    for (i=1;i<nmat;i++) {
      ierr = BVMultVec(pjd->AX[i],1.0,1.0,ty,q);CHKERRQ(ierr);
      PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n,pjd->T,&ld,q,&one));
      for (j=0;j<nconv;j++) q[j] += fact*x2[j];
      fact *= theta;
    }
    ierr = BVSetActiveColumns(pjd->X,0,nconv);CHKERRQ(ierr);    
    ierr = BVDotVec(pjd->X,tx,xx);CHKERRQ(ierr);
    if (rk==np-1) {
      y2 = array2+nloc;
      for (i=0;i<nconv;i++) { q[i] = x2[i]; y2[i] = xx[i]; }
      fact = theta;
      for (j=1;j<deg;j++) {
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n,&n,&sone,pjd->XpX,&ld,q,&one,&zero,tt,&one));
        for (i=0;i<nconv;i++) tt[i] += fact*xx[i];
        PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&n,pjd->Tj+ld*ld*j,&ld,tt,&one));
        for (i=0;i<nconv;i++) y2[i] += tt[i];
        PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","N","N",&n,pjd->T,&ld,q,&one));
        for (i=0;i<nconv;i++) q[i] += fact*x2[i];
        fact *= theta;
      }
    }
    ierr = PetscFree5(tt,x2,q,xx,yy);CHKERRQ(ierr);
  }
  ierr = VecResetArray(tx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecResetArray(ty);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&array2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDCreateShellPC"
static PetscErrorCode PEPJDCreateShellPC(PEP pep)
{
  PEP_JD          *pjd = (PEP_JD*)pep->data;
  PEP_JD_PCSHELL  *pcctx;
  PEP_JD_MATSHELL *matctx;
  KSP             ksp;
  PetscInt        nloc,mloc;
  PetscMPIInt     np,rk;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PCCreate(PetscObjectComm((PetscObject)pep),&pjd->pcshell);CHKERRQ(ierr);
  ierr = PCSetType(pjd->pcshell,PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetName(pjd->pcshell,"PCPEPJD");CHKERRQ(ierr);
  ierr = PCShellSetApply(pjd->pcshell,PCShellApply_PEPJD);CHKERRQ(ierr);
  ierr = PetscNew(&pcctx);CHKERRQ(ierr);
  ierr = PCShellSetContext(pjd->pcshell,pcctx);CHKERRQ(ierr);
  ierr = STGetKSP(pep->st,&ksp);CHKERRQ(ierr);
  ierr = BVCreateVec(pjd->V,&pcctx->Bp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pcctx->pc);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)pcctx->pc);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pep->A[0],&mloc,&nloc);CHKERRQ(ierr);
  if (pep->nev>1) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
    if (rk==np-1) { nloc += pep->nev-1; mloc += pep->nev-1; }
  }
  ierr = PetscNew(&matctx);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)pep),nloc,mloc,PETSC_DETERMINE,PETSC_DETERMINE,matctx,&pjd->Pshell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(pjd->Pshell,MATOP_MULT,(void(*)())PEPJDShellMatMult);CHKERRQ(ierr);
  matctx->pep = pep;
  ierr = MatDuplicate(pep->A[0],MAT_DO_NOT_COPY_VALUES,&matctx->P);CHKERRQ(ierr);
  ierr = PCSetOperators(pcctx->pc,matctx->P,matctx->P);CHKERRQ(ierr);
  ierr = KSPSetPC(ksp,pjd->pcshell);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,pjd->Pshell,pjd->Pshell);CHKERRQ(ierr);
  if (pep->nev>1) {
    ierr = PetscMalloc2(pep->nev*pep->nev,&pcctx->M,pep->nev*pep->nev,&pcctx->ps);CHKERRQ(ierr);
    pcctx->X  = pjd->X;
    pcctx->ld = pep->nev;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDUpdateExtendedPC"
static PetscErrorCode PEPJDUpdateExtendedPC(PEP pep,PetscScalar theta)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PEP_JD_PCSHELL *pcctx;  
  PetscInt       i,j,k,n=pjd->nconv,ld=pep->nev,deg=pep->nmat-1;
  PetscScalar    fact,*M,*ps,*work,*U,*V,*S,sone=1.0,zero=0.0;
  PetscReal      tol,maxeig=0.0,*sg,*rwork;
  PetscBLASInt   n_,info,ld_,*p,lw_,rk=0;

  PetscFunctionBegin;
#if defined(PETSC_MISSING_LAPACK_GESVD) || defined(PETSC_MISSING_LAPACK_GETRI) || defined(PETSC_MISSING_LAPACK_GETRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESVD/GETRI/GETRF - Lapack routine is unavailable");
#else
  if (n) { 
    ierr = PCShellGetContext(pjd->pcshell,(void**)&pcctx);CHKERRQ(ierr);
    pcctx->n = n;
    M  = pcctx->M;
    ps = pcctx->ps;
                      /* h, and q are vectors containing diagonal matrices */
    ierr = PetscCalloc7(n*n,&U,n*n,&V,n*n,&S,n,&sg,10*n,&work,5*n,&rwork,n,&p);CHKERRQ(ierr);
    /* pseudo-inverse */
    for (j=0;j<n;j++) {
      for (i=0;i<j;i++) S[n*j+i] = -pjd->T[pep->nev*j+i];
      S[n*j+j] = theta-pjd->T[pep->nev*j+j];
    }
    ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ld,&ld_);CHKERRQ(ierr);
    lw_ = 10*n_;
#if !defined (PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&n_,&n_,S,&n_,sg,U,&n_,V,&n_,work,&lw_,&info));
#else
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","S",&n_,&n_,S,&n_,sg,U,&n_,V,&n_,work,&lw_,rwork,&info));
#endif
    for (i=0;i<n;i++) maxeig = PetscMax(maxeig,sg[i]);
    tol = 10*PETSC_MACHINE_EPSILON*n*maxeig;
    for (j=0;j<n;j++) {
      if (sg[j]>tol) {
        for (i=0;i<n;i++) U[j*n+i] /= sg[j];
        rk++;
      } else break;
    }
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&rk,&sone,U,&n_,V,&n_,&zero,ps,&ld_));

    /* compute M */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,pjd->XpX,&ld_,ps,&ld_,&zero,M,&ld_));
    fact = theta;
    ierr = PetscMemzero(S,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
    for (j=0;j<n;j++) S[j*(n+1)] = 1.0; /* q=S */
    for (k=0;k<deg;k++) {
      for (j=0;j<n;j++) for (i=0;i<n;i++) V[j*n+i] = S[j*n+i] + M[j*ld+i]*fact;
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,pjd->XpX,&ld_,V,&n_,&zero,U,&n_));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n_,&n_,&n_,&sone,pjd->Tj+k*ld*ld,&ld_,U,&n_,&sone,M,&ld_));
      PetscStackCallBLAS("BLAStrmm",BLAStrmm_("L","U","N","N",&n_,&n_,&sone,pjd->T,&ld_,S,&n_));
      for (j=0;j<n;j++) S[j*(n+1)] += fact;
      fact *=theta;
    }
    /* inverse */
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n_,&n_,M,&ld_,p,&info));
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n_,M,&ld_,p,work,&n_,&info));
    ierr = PetscFree7(U,V,S,sg,work,rwork,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDPCMatSetUp"
static PetscErrorCode PEPJDPCMatSetUp(PEP pep,PetscScalar theta)
{
  PetscErrorCode  ierr;
  PEP_JD          *pjd = (PEP_JD*)pep->data;
  PEP_JD_MATSHELL *matctx;
  PEP_JD_PCSHELL  *pcctx;  
  MatStructure    str;
  PetscScalar     t;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = MatShellGetContext(pjd->Pshell,(void**)&matctx);CHKERRQ(ierr);
  ierr = PCShellGetContext(pjd->pcshell,(void**)&pcctx);CHKERRQ(ierr);
  ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
  ierr = MatCopy(pep->A[0],matctx->P,str);CHKERRQ(ierr);
  t = theta;
  for (i=1;i<pep->nmat;i++) {
    if (t!=0.0) { ierr = MatAXPY(matctx->P,t,pep->A[i],str);CHKERRQ(ierr); }
    t *= theta;
  }
  ierr = PCSetOperators(pcctx->pc,matctx->P,matctx->P);CHKERRQ(ierr);
  ierr = PCSetUp(pcctx->pc);CHKERRQ(ierr);
  matctx->theta = theta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDEigenvectors"
static PetscErrorCode PEPJDEigenvectors(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBLASInt   ld,nconv,info,nc;
  PetscScalar    *Z,*w;
  PetscReal      *wr,norm;
  PetscInt       i;
  Mat            U;
 
  PetscFunctionBegin;
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable");
#else
  ierr = PetscMalloc3(pjd->nconv*pjd->nconv,&Z,3*pep->nev,&wr,2*pep->nev,&w);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(pep->nev,&ld);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(pjd->nconv,&nconv);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_("R","A",NULL,&nconv,pjd->T,&ld,NULL,&nconv,Z,&nconv,&nconv,&nc,wr,&info));
#else
  PetscStackCallBLAS("LAPACKtrevc",LAPACKtrevc_("R","A",NULL,&nconv,pjd->T,&ld,NULL,&nconv,Z,&nconv,&nconv,&nc,w,wr,&info));
#endif
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nconv,nconv,Z,&U);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pjd->X,0,pjd->nconv);CHKERRQ(ierr);
  ierr = BVMultInPlace(pjd->X,U,0,pjd->nconv);CHKERRQ(ierr);
  for (i=0;i<pjd->nconv;i++) {
    ierr = BVNormColumn(pjd->X,i,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(pjd->X,i,1.0/norm);CHKERRQ(ierr);  
  }
  ierr = MatDestroy(&U);CHKERRQ(ierr);
  ierr = PetscFree3(Z,wr,w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDLockConverged"
static PetscErrorCode PEPJDLockConverged(PEP pep,PetscInt *nv)
{
  PetscErrorCode    ierr;
  PEP_JD            *pjd = (PEP_JD*)pep->data;
  PetscInt          j,i,ldds,rk=0,*P,nvv=*nv;
  Vec               v,x;
  PetscBLASInt      n,ld,rk_,nv_,info,one=1;
  PetscScalar       sone=1.0,*Tj,*R,*r,*tt,*pX;
  Mat               X;

  PetscFunctionBegin;
#if defined(SLEPC_MISSING_LAPACK_TRTRI)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TRTRI - Lapack routine is unavailable");
#else
  /* update AX and XpX */
  ierr = BVGetColumn(pjd->X,pjd->nconv-1,&x);CHKERRQ(ierr);
  for (j=0;j<pep->nmat;j++) {
    ierr = BVGetColumn(pjd->AX[j],pjd->nconv-1,&v);CHKERRQ(ierr);
    ierr = MatMult(pep->A[j],x,v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->AX[j],pjd->nconv-1,&v);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pjd->AX[j],0,pjd->nconv);CHKERRQ(ierr);
  }
  ierr = BVRestoreColumn(pjd->X,pjd->nconv-1,&x);CHKERRQ(ierr);
  ierr = BVDotColumn(pjd->X,(pjd->nconv-1),pjd->XpX+(pjd->nconv-1)*(pep->nev));CHKERRQ(ierr);
  pjd->XpX[(pjd->nconv-1)*(1+pep->nev)] = 1.0;
  for (j=0;j<pjd->nconv-1;j++) pjd->XpX[j*(pep->nev)+pjd->nconv-1] = PetscConj(pjd->XpX[(pjd->nconv-1)*(pep->nev)+j]);
  
  /* Compute powers of T */
  ierr = PetscBLASIntCast(pjd->nconv,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(pep->nev,&ld);CHKERRQ(ierr);
  ierr = PetscMemzero(pjd->Tj,pep->nev*pep->nev*pep->nmat*sizeof(PetscScalar));CHKERRQ(ierr);
  Tj = pjd->Tj;
  for (j=0;j<pjd->nconv;j++) Tj[(pep->nev+1)*j] = 1.0;
  Tj = pjd->Tj+pep->nev*pep->nev;
  ierr = PetscMemcpy(Tj,pjd->T,pep->nev*pjd->nconv*sizeof(PetscScalar));CHKERRQ(ierr);
  for (j=2;j<pep->nmat;j++) {
    ierr = PetscMemcpy(Tj+pep->nev*pep->nev,Tj,pep->nev*pjd->nconv*sizeof(PetscScalar));CHKERRQ(ierr);
    Tj += pep->nev*pep->nev;
    PetscStackCallBLAS("BLAStrmm",BLAStrmm_("L","U","N","N",&n,&n,&sone,pjd->T,&ld,Tj,&ld));
  }

  /* Extend search space */
  ierr = PetscCalloc4(nvv,&P,nvv*nvv,&R,nvv,&r,pep->nev-1,&tt);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(pep->ds,&ldds);CHKERRQ(ierr);
  ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
  ierr = PEPJDOrthogonalize(nvv,nvv,pX,ldds,&rk,P,R,nvv);CHKERRQ(ierr);
  for (i=0;i<rk-1;i++) r[i] = PetscConj(R[nvv*i]*pep->eigr[P[i+1]]); /* first row scaled with permuted diagonal */
  ierr = PetscBLASIntCast(rk,&rk_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nvv,&nv_);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrtri",LAPACKtrtri_("U","N",&rk_,R,&nv_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,1,"Error in xTRTRI, info=%D",(PetscInt)info);
  PetscStackCallBLAS("BLAStrmv",BLAStrmv_("U","C","N",&rk_,R,&nv_,r,&one));
  for (i=0;i<rk;i++) r[i] = PetscConj(r[i]); /* revert */
  ierr = BVSetActiveColumns(pjd->V,0,nvv);CHKERRQ(ierr);
  for (j=0;j<rk-1;j++) {
    ierr = PetscMemcpy(R+j*nvv,pX+(j+1)*ldds,nvv*sizeof(PetscScalar));CHKERRQ(ierr);
  } 
  ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nvv,rk-1,R,&X);CHKERRQ(ierr);
  ierr = BVMultInPlace(pjd->V,X,0,rk-1);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pjd->V,0,rk-1);CHKERRQ(ierr);
  for (j=0;j<rk-1;j++) {
    ierr = BVGetColumn(pjd->V,j,&v);CHKERRQ(ierr);
    ierr = PEPJDCopyToExtendedVec(pep,NULL,r+j,1,pjd->nconv-1,v,PETSC_FALSE);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,j,&v);CHKERRQ(ierr);
  }
  ierr = BVOrthogonalize(pjd->V,NULL);CHKERRQ(ierr); 
  for (j=0;j<rk-1;j++) {
    ierr = BVGetColumn(pjd->W,j,&v);CHKERRQ(ierr);
    ierr = PEPJDCopyToExtendedVec(pep,NULL,tt,pep->nev-1,0,v,PETSC_FALSE);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->W,j,&v);CHKERRQ(ierr);
  }
  *nv = rk-1;
  ierr = PetscFree4(P,R,r,tt);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_JD"
PetscErrorCode PEPSolve_JD(PEP pep)
{
  PetscErrorCode  ierr;
  PEP_JD          *pjd = (PEP_JD*)pep->data;
  PetscInt        k,nv,ld,minv,low,high,dim;
  PetscScalar     theta=0.0,*pX,*eig;
  PetscReal       norm,*res;
  PetscBool       lindep,initial=PETSC_FALSE,flglk=PETSC_FALSE,flgre=PETSC_FALSE;
  Vec             t,u,p,r,*ww=pep->work,v;
  Mat             G,X,Y;
  KSP             ksp;
  PEP_JD_PCSHELL  *pcctx;
  PEP_JD_MATSHELL *matctx;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc2(pep->ncv,&eig,pep->ncv,&res);CHKERRQ(ierr);
  ierr = BVCreateVec(pjd->V,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&p);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r);CHKERRQ(ierr);
  ierr = STGetKSP(pep->st,&ksp);CHKERRQ(ierr);

  if (pep->nini) {
    nv = pep->nini; initial = PETSC_TRUE;
  } else {
    theta = pep->target;
    nv = 1;
  }
  ierr = PEPJDProcessInitialSpace(pep,ww);CHKERRQ(ierr);
  ierr = BVCopyVec(pjd->V,0,u);CHKERRQ(ierr);

  /* Restart loop */
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;

    low = (flglk || flgre)? 0: nv-1;
    high = nv;
    ierr = DSSetDimensions(pep->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pjd->V,low,high);CHKERRQ(ierr);
    ierr = PEPJDUpdateTV(pep,low,high,ww);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pjd->W,low,high);CHKERRQ(ierr);
    for (k=0;k<pep->nmat;k++) {
      ierr = BVSetActiveColumns(pjd->TV[k],low,high);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
      ierr = BVMatProject(pjd->TV[k],NULL,pjd->W,G);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(pjd->V,0,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pjd->W,0,nv);CHKERRQ(ierr);

    /* Solve projected problem */
    if (nv>1 || initial ) {
      ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
      ierr = DSSolve(pep->ds,pep->eigr+pep->nconv,pep->eigi+pep->nconv);CHKERRQ(ierr);
      ierr = DSSort(pep->ds,pep->eigr+pep->nconv,pep->eigi+pep->nconv,NULL,NULL,NULL);CHKERRQ(ierr);
      theta = pep->eigr[0];
#if !defined(PETSC_USE_COMPLEX)
      if (PetscAbsScalar(pep->eigi[pep->nconv])!=0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PJD solver not implemented for complex Ritz values in real arithmetic");
#endif

      /* Compute Ritz vector u=V*X(:,1) */
      ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(pjd->V,0,nv);CHKERRQ(ierr);
      ierr = BVMultVec(pjd->V,1.0,0.0,u,pX);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
    }
    ierr = PEPJDUpdateExtendedPC(pep,theta);CHKERRQ(ierr);

    /* Replace preconditioner with one containing projectors */
    if (!pjd->pcshell) {
      ierr = PEPJDCreateShellPC(pep);CHKERRQ(ierr);
      ierr = PCShellGetContext(pjd->pcshell,(void**)&pcctx);CHKERRQ(ierr);
      ierr = MatShellGetContext(pjd->Pshell,(void**)&matctx);CHKERRQ(ierr);
      matctx->work = ww;
      pcctx->work  = ww;
    }
    ierr = PEPJDPCMatSetUp(pep,theta);CHKERRQ(ierr);
    
    /* Compute r and r' */
    ierr = MatMult(pjd->Pshell,u,r);CHKERRQ(ierr);
    ierr = PEPJDComputePResidual(pep,u,theta,p,ww);CHKERRQ(ierr);
    pcctx->u = u;

    /* Check convergence */
    ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
    ierr = (*pep->converged)(pep,theta,0,norm,&pep->errest[pep->nconv],pep->convergedctx);CHKERRQ(ierr);
    ierr = (*pep->stopping)(pep,pep->its,pep->max_it,(pep->errest[pep->nconv]<pep->tol)?pjd->nconv+1:pjd->nconv,pep->nev,&pep->reason,pep->stoppingctx);CHKERRQ(ierr);

    if (pep->errest[pep->nconv]<pep->tol) {

      /* Ritz pair converged */
      minv = PetscMin(nv,(PetscInt)(pjd->keep*pep->ncv));
      if (pep->nev>1) {
        ierr = BVGetColumn(pjd->X,pjd->nconv,&v);CHKERRQ(ierr);
        ierr = PEPJDCopyToExtendedVec(pep,v,pjd->T+pep->nev*pjd->nconv,pep->nev-1,0,u,PETSC_TRUE);CHKERRQ(ierr);
        ierr = BVRestoreColumn(pjd->X,pjd->nconv,&v);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(pjd->X,0,pjd->nconv+1);CHKERRQ(ierr);
        ierr = BVNormColumn(pjd->X,pjd->nconv,NORM_2,&norm);CHKERRQ(ierr);
        ierr = BVScaleColumn(pjd->X,pjd->nconv,1.0/norm);CHKERRQ(ierr);
        for (k=0;k<pjd->nconv;k++) pjd->T[pep->nev*pjd->nconv+k] /= norm;
        pjd->T[(pep->nev+1)*pjd->nconv] = pep->eigr[0];
      } else {
        ierr = BVInsertVec(pep->V,pep->nconv,u);CHKERRQ(ierr);
      }
      pjd->nconv++;

      if (pep->reason==PEP_CONVERGED_ITERATING) {
        ierr = PEPJDLockConverged(pep,&nv);CHKERRQ(ierr);
        ierr = BVCopyVec(pjd->V,nv-1,u);CHKERRQ(ierr);
        if (nv==1) theta = pep->target;
      }
      flglk = PETSC_TRUE;
    } else if (nv==pep->ncv-1) {

      /* Basis full, force restart */
      minv = PetscMin(nv,(PetscInt)(pjd->keep*pep->ncv));
      ierr = DSGetDimensions(pep->ds,&dim,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = PEPJDOrthogonalize(dim,minv,pX,ld,&minv,NULL,NULL,ld);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = DSGetArray(pep->ds,DS_MAT_Y,&pX);CHKERRQ(ierr);
      ierr = PEPJDOrthogonalize(dim,minv,pX,ld,&minv,NULL,NULL,ld);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_Y,&pX);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultInPlace(pjd->V,X,pep->nconv,minv);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DS_MAT_Y,&Y);CHKERRQ(ierr);
      ierr = BVMultInPlace(pjd->W,Y,pep->nconv,minv);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DS_MAT_Y,&Y);CHKERRQ(ierr);
      nv = minv;
      flgre = PETSC_TRUE;
    } else {
      /* Solve correction equation to expand basis */
      ierr = PEPJDExtendedPCApply(pjd->pcshell,p,pcctx->Bp);CHKERRQ(ierr);
      ierr = VecDot(pcctx->Bp,u,&pcctx->gamma);CHKERRQ(ierr);
      ierr = BVGetColumn(pjd->V,nv,&t);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,r,t);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->V,nv,&t);CHKERRQ(ierr);
      ierr = BVOrthogonalizeColumn(pjd->V,nv,NULL,&norm,&lindep);CHKERRQ(ierr);
      if (lindep || norm==0.0) SETERRQ(PETSC_COMM_SELF,1,"Linearly dependent continuation vector");
      ierr = BVScaleColumn(pjd->V,nv,1.0/norm);CHKERRQ(ierr);
      ierr = BVInsertVec(pjd->W,nv,r);CHKERRQ(ierr);
      ierr = BVOrthogonalizeColumn(pjd->W,nv,NULL,&norm,&lindep);CHKERRQ(ierr);
      if (lindep) SETERRQ(PETSC_COMM_SELF,1,"Linearly dependent continuation vector");
      ierr = BVScaleColumn(pjd->W,nv,1.0/norm);CHKERRQ(ierr);
      nv++;
      flglk = PETSC_FALSE;
      flgre = PETSC_FALSE;
    }
    for (k=pjd->nconv;k<nv;k++) {
      eig[k] = pep->eigr[k-pjd->nconv];
      res[k] = pep->errest[k-pjd->nconv];
#if !defined(PETSC_USE_COMPLEX)
      pep->eigi[k-pjd->nconv] = 0.0;
#endif
    }
    ierr = PEPMonitor(pep,pep->its,pjd->nconv,eig,pep->eigi,res,pjd->nconv+1);CHKERRQ(ierr);
  }
  if (pep->nev>1) {
    if (pjd->nconv>0) { ierr = PEPJDEigenvectors(pep);CHKERRQ(ierr); }
    for (k=0;k<pjd->nconv;k++) {
      ierr = BVGetColumn(pjd->X,k,&v);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,k,v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->X,k,&v);CHKERRQ(ierr);
      pep->eigr[k] = pjd->T[(pep->nev+1)*k]; 
      pep->eigi[k] = 0.0; 
    }
    ierr = PetscFree2(pcctx->M,pcctx->ps);CHKERRQ(ierr); 
  }
  pep->nconv = pjd->nconv; 
  ierr = KSPSetPC(ksp,pcctx->pc);CHKERRQ(ierr);
  ierr = MatDestroy(&matctx->P);CHKERRQ(ierr);
  ierr = VecDestroy(&pcctx->Bp);CHKERRQ(ierr);
  ierr = MatDestroy(&pjd->Pshell);CHKERRQ(ierr);
  ierr = PCDestroy(&pcctx->pc);CHKERRQ(ierr);
  ierr = PetscFree(pcctx);CHKERRQ(ierr);
  ierr = PetscFree(matctx);CHKERRQ(ierr);
  ierr = PCDestroy(&pjd->pcshell);CHKERRQ(ierr);
  ierr = PetscFree2(eig,res);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPReset_JD"
PetscErrorCode PEPReset_JD(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0;i<pep->nmat;i++) {
    ierr = BVDestroy(pjd->TV+i);CHKERRQ(ierr);
  }
  ierr = BVDestroy(&pjd->W);CHKERRQ(ierr);
  if (pep->nev>1) {
    ierr = BVDestroy(&pjd->V);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = BVDestroy(pjd->AX+i);CHKERRQ(ierr);
    }
    ierr = BVDestroy(&pjd->X);CHKERRQ(ierr);
    ierr = PetscFree3(pjd->XpX,pjd->T,pjd->Tj);CHKERRQ(ierr);
  }
  ierr = PetscFree2(pjd->TV,pjd->AX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy_JD"
PetscErrorCode PEPDestroy_JD(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_JD"
PETSC_EXTERN PetscErrorCode PEPCreate_JD(PEP pep)
{
  PEP_JD         *pjd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&pjd);CHKERRQ(ierr);
  pep->data = (void*)pjd;

  pjd->keep = 0;
  pep->ops->solve          = PEPSolve_JD;
  pep->ops->setup          = PEPSetUp_JD;
  pep->ops->setfromoptions = PEPSetFromOptions_JD;
  pep->ops->reset          = PEPReset_JD;
  pep->ops->destroy        = PEPDestroy_JD;
  pep->ops->view           = PEPView_JD;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",PEPJDSetRestart_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",PEPJDGetRestart_JD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
