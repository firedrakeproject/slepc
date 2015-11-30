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
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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
#include <slepc/private/dsimpl.h>
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
  PetscBool      isshift,flg;
  PetscInt       i;

  PetscFunctionBegin;
  pep->lineariz = PETSC_FALSE;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!pep->max_it) pep->max_it = PetscMax(100,2*pep->n/pep->ncv);
  if (!pep->which) pep->which = PEP_LARGEST_MAGNITUDE;

  /* Set STSHIFT as the default ST */
  if (!((PetscObject)pep->st)->type_name) {
    ierr = STSetType(pep->st,STSHIFT);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSHIFT,&isshift);CHKERRQ(ierr);
  if (!isshift) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"JD only works with shift spectral transformation");

  if (pep->basis!=PEP_BASIS_MONOMIAL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver not implemented for non-monomial bases");
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver requires the ST transformation flag unset, see STSetTransform()");

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
    for (i=0;i<pep->nmat;i++) {
      ierr = BVDuplicateResize(pep->V,pep->nev-1,pjd->AX+i);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1((pep->nev-1)*(pep->nev-1),&pjd->v);CHKERRQ(ierr);
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
  PetscInt       pp,col,i,j,nloc;
  Vec            v1,v2,t1,t2;
  PetscScalar    *array1,*array2,*delta,*tt;
  PetscMPIInt    rk,np;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rk);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pep),&np);CHKERRQ(ierr);
  t1 = w[0];
  t2 = w[1];
  ierr = PetscMalloc2(pep->nconv,&delta,pep->nconv,&tt);CHKERRQ(ierr);
  for (pp=0;pp<pep->nmat;pp++) {
    for (col=low;col<high;col++) {
      ierr = BVGetColumn(pjd->V,col,&v1);CHKERRQ(ierr);
      ierr = VecGetArray(v1,&array1);CHKERRQ(ierr);
      ierr = VecPlaceArray(t1,array1);CHKERRQ(ierr);
      ierr = BVGetColumn(pjd->TV[pp],col,&v2);CHKERRQ(ierr);
      ierr = VecGetArray(v2,&array2);CHKERRQ(ierr);
      ierr = VecPlaceArray(t2,array2);CHKERRQ(ierr);
      ierr = MatMult(pep->A[pp],t1,t2);CHKERRQ(ierr);
      if (pep->nconv) {
        ierr = BVDotVec(pjd->AX[pp],t1,tt);CHKERRQ(ierr);
      }
      ierr = VecResetArray(t1);CHKERRQ(ierr);
      ierr = VecRestoreArray(v1,&array1);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->V,col,&v1);CHKERRQ(ierr);
      if (pep->nconv) {
        for (j=0;j<pep->nconv;j++) delta[j] = pjd->v[col*(pep->nev-1)+j];
        for (i=pp+1;pp<pep->nmat;i++) {
          ierr = BVMultVec(pjd->AX[i],1.0,1.0,t2,delta);CHKERRQ(ierr);
          if (pp<pep->nmat-1) for (j=0;j<pep->nconv;j++) delta[j] *= pep->eigr[j];
        }
      }
      ierr = VecResetArray(t2);CHKERRQ(ierr);
      if (pep->nconv && rk==np-1) {
        ierr = VecGetSize(t2,&nloc);CHKERRQ(ierr);
        for (j=0;j<pep->nconv;j++) array2[nloc+j] = PetscConj(delta[j]*pep->eigr[j])*(tt[j]+delta[j]);
      }
      ierr = VecRestoreArray(v2,&array2);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->TV[pp],col,&v2);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(delta,tt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDOrthogonalize"
/*
   RRQR of X. Xin*P=Xou*R. Rank of R in rk.
*/
static PetscErrorCode PEPJDOrthogonalize(PetscInt row,PetscInt col,PetscScalar *X,PetscInt ldx,PetscInt *rk,PetscInt *P,PetscScalar *R,PetscInt ldr)
{
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
  zgeqp3_(&row_,&col_,X,&ldx_,p,tau,work,&lwork,rwork,&info);
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

  /* copy uper triangular matrix if requested */
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
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDPCApplyToExtendedVec"
/*
   Application preconditioner to the top part
    of an extended Vec
*/
static PetscErrorCode PEPJDPCApplyToExtendedVec(PC pc,Vec x,Vec y,Vec *work)
{
  PetscErrorCode    ierr;
  Vec               tx,ty;
  PetscScalar       *array2;
  const PetscScalar *array1;

  PetscFunctionBegin;
  tx = work[0];
  ty = work[1];
  ierr = VecGetArrayRead(x,&array1);CHKERRQ(ierr);
  ierr = VecPlaceArray(tx,array1);CHKERRQ(ierr);
  ierr = VecGetArray(y,&array2);CHKERRQ(ierr);
  ierr = VecPlaceArray(ty,array2);CHKERRQ(ierr);

  /* y = B\x */
  ierr = PCApply(pc,tx,ty);CHKERRQ(ierr);

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
  PEP_JD_PCSHELL *pcctx;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&pcctx);CHKERRQ(ierr);

  /* y = B\x */
  ierr = PEPJDPCApplyToExtendedVec(pcctx->pc,x,y,pcctx->work);CHKERRQ(ierr);

  /* Compute eta = u'*y / u'*Bp */
  ierr = VecDot(y,pcctx->u,&eta);CHKERRQ(ierr);
  eta /= pcctx->gamma;
  
  /* y = y - eta*Bp */
  ierr = VecAXPY(y,-eta,pcctx->Bp);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDCopyToExtendedVec"
static PetscErrorCode PEPJDCopyToExtendedVec(PEP pep,Vec v,PetscScalar *a,Vec vex,PetscBool back)
{
  PetscErrorCode ierr;
  PetscMPIInt    np,rk;
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
  if (a && rk==np-1) {
    ierr = VecGetArray(vex,&array2);CHKERRQ(ierr);
    if (back) {
      ierr = PetscMemcpy(a,array2+nloc,(pep->nev-1)*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemcpy(array2+nloc,a,(pep->nev-1)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(vex,&array2);CHKERRQ(ierr);
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
  PetscScalar    *array1,*array2,*x2,*y2,fact=1.0,*q,*tt,*xx,*eigr;
  PetscInt       i,j,nconv=pep->nconv,nloc;

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
  eigr = pep->eigr;
  ierr = VecGetArray(u,&array1);CHKERRQ(ierr);
  ierr = VecPlaceArray(tu,array1);CHKERRQ(ierr);
  ierr = VecGetArray(p,&array2);CHKERRQ(ierr);
  ierr = VecPlaceArray(tp,array2);CHKERRQ(ierr);
  ierr = VecSet(tp,0.0);CHKERRQ(ierr);
  for (i=1;i<pep->nmat;i++) {
    ierr = MatMult(pep->A[i],tu,w);CHKERRQ(ierr);
    ierr = VecAXPY(tp,fact*i,w);CHKERRQ(ierr);
    fact *= theta;
  }
  if (nconv) {
    for (j=0;j<nconv;j++) q[j] = x2[j];
    fact = theta;
    for (i=2;i<pep->nmat;i++) {
      ierr = BVMultVec(pjd->AX[i],1.0,1.0,tp,q);CHKERRQ(ierr);
      for (j=0;j<nconv;j++) q[j] = q[j]*eigr[j]+ i*fact*x2[j]; /* i-1 for the next it */
      fact *= theta;
    }
    ierr = BVSetActiveColumns(pjd->V,0,nconv);CHKERRQ(ierr);
    ierr = BVDotVec(pjd->V,tu,xx);CHKERRQ(ierr);
    if (rk==np-1) {
      y2 = array2+nloc;
      for (i=0;i<nconv;i++) {
        tt[i] = PetscConj(eigr[i]);
        y2[i] = tt[i]*xx[i];
        q[i]  = x2[i];
      }
      fact = theta;
      for (j=2;j<pep->nmat;j++) {
        for (i=0;i<nconv;i++) {
          tt[i] = PetscConj(eigr[i]);
          y2[i] = tt[i]*(j*fact*xx[i]+q[i]);
          q[i]  = q[i]*eigr[i]+j*fact*x2[i];
        }
        fact *= theta;
      }
    }
    ierr = PetscFree4(xx,x2,q,tt);CHKERRQ(ierr);
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
  /* //////////// */
  PetscBool new=PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-newjd",&new,NULL);
  /* //////////// */
  ierr = PetscMalloc1(pep->nev-1,&tt);CHKERRQ(ierr);
  if (pep->nini==0) {
    ierr = BVSetRandomColumn(pjd->V,0,pep->rand);CHKERRQ(ierr);
    for (i=0;i<pep->nev-1;i++) tt[i] = 0.0;
    ierr = BVGetColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = PEPJDCopyToExtendedVec(pep,NULL,tt,vg,PETSC_FALSE);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,0,&vg);CHKERRQ(ierr);
    ierr = BVNormColumn(pjd->V,0,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(pjd->V,0,1.0/norm);CHKERRQ(ierr);
    if (new) {
      ierr = BVGetColumn(pjd->V,0,&vg);CHKERRQ(ierr);
      ierr = BVGetColumn(pjd->W,0,&wg);CHKERRQ(ierr);
      ierr = VecSet(wg,0.0);CHKERRQ(ierr);
      ierr = PEPJDComputePResidual(pep,vg,pep->target,wg,w);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->W,0,&wg);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->V,0,&vg);CHKERRQ(ierr);
      ierr = BVNormColumn(pjd->W,0,NORM_2,&norm);CHKERRQ(ierr);
      ierr = BVScaleColumn(pjd->W,0,1.0/norm);CHKERRQ(ierr);
    }
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
  PetscErrorCode  ierr;
  PEP_JD_MATSHELL *matctx;
  PEP_JD          *pjd;
  PetscMPIInt     rk,np,count;
  PetscInt        i,j,nconv,nloc,nmat;
  Vec             tx,ty;
  PetscScalar     *array2,*x2,*y2,fact=1.0,*q,*tt,*xx,*eigr,theta;
  const PetscScalar *array1;

  PetscFunctionBegin;
  ierr = MatShellGetContext(P,(void**)&matctx);CHKERRQ(ierr);
  pjd = (PEP_JD*)(matctx->pep->data);
  nconv = matctx->pep->nconv;
  eigr = matctx->pep->eigr;
  theta = matctx->theta;
  nconv = matctx->pep->nconv;
  nmat = matctx->pep->nmat;
  if (nconv>0) {
    ierr = PetscMalloc4(nconv,&tt,nconv,&x2,nconv,&q,nconv,&xx);CHKERRQ(ierr);
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
    for (j=0;j<nconv;j++) q[j] = x2[j];
    fact = theta;
    for (i=1;i<nmat;i++) {
      ierr = BVMultVec(pjd->AX[i],1.0,1.0,ty,q);CHKERRQ(ierr);
      for (j=0;j<nconv;j++) q[j] = q[j]*eigr[j]+ fact*x2[j];
      fact *= theta;
    }
    ierr = BVSetActiveColumns(pjd->V,0,nconv);CHKERRQ(ierr);
    ierr = BVDotVec(pjd->V,tx,xx);CHKERRQ(ierr);
    if (rk==np-1) {
      y2 = array2+nloc;
      for (i=0;i<nconv;i++) {
        y2[i] = xx[i];
        tt[i] = PetscConj(eigr[i]);
        q[i]  = x2[i];
      }
      fact = theta;
      for (j=1;j<nmat;j++) {
        for (i=0;i<nconv;i++) {
          y2[i] = tt[i]*(fact*xx[i]+q[i]);
          tt[i] = PetscConj(eigr[i]);
          q[i]  = q[i]*eigr[i]+fact*x2[i];
        }
        fact *= theta;
      }

      for (i=0;i<nconv;i++) {
        fact = 1.0;
        y2[i] = 1.0;
        for (j=1;j<nmat;j++) {
          fact *= theta*PetscConj(eigr[i]);
          y2[i] += fact;
        }
        y2[i] *= tt[i];
      }
      for (i=0;i<nconv;i++) tt[i] = PetscConj(eigr[i]*eigr[i])*x2[i];
      fact = theta;
      for (j=2;j<nmat;j++) {
        for (i=0;i<nconv;i++) y2[i] += tt[i];
        for (i=0;i<nconv;i++) tt[i] = tt[i]*PetscConj(eigr[i])*eigr[i]+(j-1)*fact*x2[i];
        fact *=theta;
      }
    }
    ierr = PetscFree4(tt,x2,q,xx);CHKERRQ(ierr);
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
    if (rk==np-1) {nloc += pep->nev-1; mloc += pep->nev-1;}
  }
  ierr = PetscNew(&matctx);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)pep),nloc,mloc,PETSC_DETERMINE,PETSC_DETERMINE,matctx,&pjd->Pshell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(pjd->Pshell,MATOP_MULT,(void(*)())PEPJDShellMatMult);CHKERRQ(ierr);
  matctx->pep = pep;
  ierr = MatDuplicate(pep->A[0],MAT_DO_NOT_COPY_VALUES,&matctx->P);CHKERRQ(ierr);
  ierr = PCSetOperators(pcctx->pc,matctx->P,matctx->P);CHKERRQ(ierr);
  ierr = KSPSetPC(ksp,pjd->pcshell);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,pjd->Pshell,pjd->Pshell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDPCMatSetUp"
PetscErrorCode PEPJDPCMatSetUp(PEP pep,PetscScalar theta)
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
    ierr = MatAXPY(matctx->P,t,pep->A[i],str);CHKERRQ(ierr);
    t *= theta;
  }
  ierr = PCSetOperators(pcctx->pc,matctx->P,matctx->P);CHKERRQ(ierr);
  ierr = PCSetUp(pcctx->pc);CHKERRQ(ierr);
  matctx->theta = theta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_JD"
PetscErrorCode PEPSolve_JD(PEP pep)
{
  PetscErrorCode  ierr;
  PEP_JD          *pjd = (PEP_JD*)pep->data;
  PetscInt        k,nv,ld,minv,low,high,dim,rk,*P;
  PetscScalar     theta,*pX,*R,*stt,*exu,*exr,*exp;
  PetscReal       norm;
  PetscBool       lindep,initial=PETSC_FALSE;
  Vec             t,u,p,r,*ww=pep->work,vg,v;
  Mat             G,X,Y;
  KSP             ksp;
  PEP_JD_PCSHELL  *pcctx;
  PEP_JD_MATSHELL *matctx;

  PetscFunctionBegin;
  /* //////////// */
  PetscBool new=PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-newjd",&new,NULL);
  /* //////////// */
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc6(ld*ld,&R,ld,&P,ld,&stt,pep->nev-1,&exu,pep->nev-1,&exr,pep->nev-1,&exp);CHKERRQ(ierr);
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

    low = (pjd->flglk || pjd->flgre)? 0: nv-1;
    high = nv;
    ierr = DSSetDimensions(pep->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pjd->V,low,high);CHKERRQ(ierr);
    ierr = PEPJDUpdateTV(pep,low,high,ww);CHKERRQ(ierr);
    if (new) { ierr = BVSetActiveColumns(pjd->W,low,high);CHKERRQ(ierr); }
    for (k=0;k<pep->nmat;k++) {
      ierr = BVSetActiveColumns(pjd->TV[k],low,high);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
      ierr = BVMatProject(pjd->TV[k],NULL,new?pjd->W:pjd->V,G);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(pjd->V,0,nv);CHKERRQ(ierr);
    if (new) { ierr = BVSetActiveColumns(pjd->W,0,nv);CHKERRQ(ierr); }

    /* Solve projected problem */
    if (nv>1 || initial) {
      ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
      ierr = DSSolve(pep->ds,pep->eigr+pep->nconv,pep->eigi+pep->nconv);CHKERRQ(ierr);
      ierr = DSSort(pep->ds,pep->eigr+pep->nconv,pep->eigi+pep->nconv,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = DSSort(pep->ds,pep->eigr+pep->nconv,pep->eigi+pep->nconv,NULL,NULL,NULL);CHKERRQ(ierr);
      theta = pep->eigr[pep->nconv];
#if !defined(PETSC_USE_COMPLEX)
      if (PetscAbsScalar(pep->eigi[pep->nconv])!=0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PJD solver not implemented for complex Ritz values in real arithmetic");
#endif

      /* Compute Ritz vector u=V*X(:,1) */
      ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(pep->V,0,nv);CHKERRQ(ierr);
      ierr = BVMultVec(pjd->V,1.0,0.0,u,pX);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
    }

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
    /* Compute p=P'(theta)*u  */
    ierr = PEPJDComputePResidual(pep,u,theta,p,ww);CHKERRQ(ierr);
    pcctx->u = u;

    /* Check convergence */
    ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
    ierr = (*pep->converged)(pep,theta,0,norm,&pep->errest[pep->nconv],pep->convergedctx);CHKERRQ(ierr);
    if (pep->its >= pep->max_it) pep->reason = PEP_DIVERGED_ITS;

    if (pep->errest[pep->nconv]<pep->tol) {
      /* Ritz pair converged */
      minv = PetscMin(nv,pjd->keep*pep->ncv);
      ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = DSGetDimensions(pep->ds,&dim,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = PEPJDOrthogonalize(dim,minv,pX,ld,&rk,P,R,ld);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultInPlace(pjd->V,X,pep->nconv,minv);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      pep->nconv++;
      /* if (pep->nconv >= pep->nev) pep->reason = PEP_CONVERGED_TOL; */
      if (pep->nconv == 1) pep->reason = PEP_CONVERGED_TOL;
      else nv = minv + pep->nconv;
      if (pep->reason==PEP_CONVERGED_ITERATING) {
        /* permute eig (only eigr) */
        ierr = PetscMemcpy(stt,pep->eigr+pep->nconv-1,minv*sizeof(PetscScalar));CHKERRQ(ierr);
        for (k=0;k<minv;k++) pep->eigr[k+pep->nconv-1] = stt[P[k]];
        /* extends the search space dimension */
        /* update AX */
      }
      pjd->flglk = PETSC_TRUE;
    } else if (nv==pep->ncv-1) {
      /* Basis full, force restart */
      minv = PetscMin(nv,pjd->keep*pep->ncv);
      ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = DSGetDimensions(pep->ds,&dim,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = PEPJDOrthogonalize(dim,minv,pX,ld,NULL,NULL,NULL,0);CHKERRQ(ierr);
      ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultInPlace(pep->V,X,pep->nconv,minv);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      if (new) {
        ierr = DSOrthogonalize(pep->ds,DS_MAT_Y,nv,NULL);CHKERRQ(ierr);
        ierr = DSGetMat(pep->ds,DS_MAT_Y,&Y);CHKERRQ(ierr);
        ierr = BVMultInPlace(pjd->W,Y,pep->nconv,minv);CHKERRQ(ierr);
        ierr = DSRestoreMat(pep->ds,DS_MAT_Y,&Y);CHKERRQ(ierr);
      }
      nv = minv + pep->nconv;
      pjd->flgre = PETSC_TRUE;

    } else {

      /* Solve correction equation to expand basis */
      ierr = PEPJDPCApplyToExtendedVec(pcctx->pc,p,pcctx->Bp,pcctx->work);CHKERRQ(ierr);
      if (!new) {
        ierr = VecScale(r,-1.0);CHKERRQ(ierr);
      }
      ierr = VecDot(pcctx->Bp,u,&pcctx->gamma);CHKERRQ(ierr);
      ierr = BVGetColumn(pjd->V,nv,&t);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,r,t);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pjd->V,nv,&t);CHKERRQ(ierr);
      ierr = BVOrthogonalizeColumn(pep->V,nv,NULL,&norm,&lindep);CHKERRQ(ierr);
      if (lindep) SETERRQ(PETSC_COMM_SELF,1,"Linearly dependent continuation vector");
      ierr = BVScaleColumn(pep->V,nv,1.0/norm);CHKERRQ(ierr);
      if (new) {
        ierr = BVInsertVec(pjd->W,nv,r);CHKERRQ(ierr);
        ierr = BVOrthogonalizeColumn(pjd->W,nv,NULL,&norm,&lindep);CHKERRQ(ierr);
        if (lindep) SETERRQ(PETSC_COMM_SELF,1,"Linearly dependent continuation vector");
        ierr = BVScaleColumn(pjd->W,nv,1.0/norm);CHKERRQ(ierr);
      }
      nv++;
      pjd->flglk = PETSC_FALSE;
      pjd->flgre = PETSC_FALSE;
    }

    ierr = PEPMonitor(pep,pep->its,pep->nconv,pep->eigr,pep->eigi,pep->errest,nv);CHKERRQ(ierr);
  }

  for (k=0;k<pep->nconv;k++) {
    ierr = BVGetColumn(pjd->V,k,&vg);CHKERRQ(ierr);
    ierr = BVGetColumn(pep->V,k,&v);CHKERRQ(ierr);
    ierr = PEPJDCopyToExtendedVec(pep,v,NULL,vg,PETSC_TRUE);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pjd->V,k,&vg);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pep->V,k,&v);CHKERRQ(ierr);
  }
  
  ierr = KSPSetPC(ksp,pcctx->pc);CHKERRQ(ierr);
  ierr = MatDestroy(&matctx->P);CHKERRQ(ierr);
  ierr = VecDestroy(&pcctx->Bp);CHKERRQ(ierr);
  ierr = MatDestroy(&pjd->Pshell);CHKERRQ(ierr);
  ierr = PCDestroy(&pcctx->pc);CHKERRQ(ierr);
  ierr = PetscFree(pcctx);CHKERRQ(ierr);
  ierr = PetscFree(matctx);CHKERRQ(ierr);
  ierr = PCDestroy(&pjd->pcshell);CHKERRQ(ierr);
  ierr = PetscFree6(R,P,stt,exu,exr,exp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeVectors_JD"
PetscErrorCode PEPComputeVectors_JD(PEP pep)
{
  PetscErrorCode ierr;
  PetscInt       k;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  Mat            G,X;

  PetscFunctionBegin;
  ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,0,pep->nconv);CHKERRQ(ierr);
  for (k=0;k<pep->nmat;k++) {
    ierr = BVSetActiveColumns(pjd->AX[k],0,pep->nconv);CHKERRQ(ierr);
    ierr = BVMatMult(pep->V,pep->A[k],pjd->AX[k]);CHKERRQ(ierr);
    ierr = DSGetMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
    ierr = BVMatProject(pjd->AX[k],NULL,pep->V,G);CHKERRQ(ierr);
    ierr = DSRestoreMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
  }

  /* Solve projected problem */
  ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
  ierr = DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);

  /* Compute Ritz vectors */
  ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,X,0,pep->nconv);CHKERRQ(ierr);
  ierr = DSRestoreMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
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
    ierr = PetscFree(pjd->v);CHKERRQ(ierr);
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
  pep->ops->computevectors = PEPComputeVectors_JD;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",PEPJDSetRestart_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",PEPJDGetRestart_JD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
