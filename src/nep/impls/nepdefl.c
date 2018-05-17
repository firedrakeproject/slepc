/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include "nepdefl.h"
#include <slepcblaslapack.h>

PetscErrorCode NEPDeflationGetInvariantPair(NEP_EXT_OP extop,BV *X,Mat *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X) *X = extop->X;
  if (H) {
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,extop->szd+1,extop->szd+1,extop->H,H);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationCopyToExtendedVec(NEP_EXT_OP extop,Vec v,PetscScalar *a,Vec vex,PetscBool back)
{
  PetscErrorCode ierr;
  PetscScalar    *array1,*array2;
  PetscInt       nloc;

  PetscFunctionBegin;
  if (extop->szd) { 
    ierr = BVGetSizes(extop->nep->V,&nloc,NULL,NULL);CHKERRQ(ierr);
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
      ierr = VecGetArray(vex,&array2);CHKERRQ(ierr);
      if (back) {
        ierr = PetscMemcpy(a,array2+nloc,extop->szd*sizeof(PetscScalar));CHKERRQ(ierr);
      } else {
        ierr = PetscMemcpy(array2+nloc,a,extop->szd*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(vex,&array2);CHKERRQ(ierr);
    }
  } else {
    if (back) {ierr = VecCopy(vex,v);CHKERRQ(ierr);}
    else {ierr = VecCopy(v,vex);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationReset(NEP_EXT_OP extop)
{
  PetscErrorCode    ierr;
  NEP_DEF_FUN_SOLVE solve;

  PetscFunctionBegin;
  if (!extop) PetscFunctionReturn(0);
  ierr = PetscFree(extop->H);CHKERRQ(ierr);
  ierr = BVDestroy(&extop->X);CHKERRQ(ierr);
  if (extop->szd) {
    ierr = PetscFree3(extop->Hj,extop->XpX,extop->bc);CHKERRQ(ierr);
    ierr = BVDestroy(&extop->W);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&extop->MF);CHKERRQ(ierr);
  ierr = MatDestroy(&extop->MJ);CHKERRQ(ierr);
  if (extop->solve) {
    solve = extop->solve;
    if (extop->szd) {
      if (!extop->simpU) {ierr = BVDestroy(&solve->T_1U);CHKERRQ(ierr);}
      ierr = PetscFree2(solve->M,solve->work);CHKERRQ(ierr);
      ierr = VecDestroy(&solve->w[0]);CHKERRQ(ierr);
      ierr = VecDestroy(&solve->w[1]);CHKERRQ(ierr);
    }
    ierr = PetscFree(extop->solve);CHKERRQ(ierr);
  }
  ierr = PetscFree(extop);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationInitialize(NEP nep,BV X,KSP ksp,PetscInt sz,NEP_EXT_OP *extop)
{
  PetscErrorCode    ierr;
  NEP_EXT_OP        op;
  NEP_DEF_FUN_SOLVE solve;
  PetscInt          szd;

  PetscFunctionBegin;
  ierr = NEPDeflationReset(*extop);CHKERRQ(ierr);
  ierr = PetscNew(&op);CHKERRQ(ierr);
  *extop  = op;
  op->nep = nep;
  op->n   = 0;
  op->szd = szd = sz-1;
  op->max_midx = PetscMin(MAX_MINIDX,szd);
  op->X = X;
  if (!X) { ierr = BVDuplicateResize(nep->V,sz,&op->X);CHKERRQ(ierr); }
  else { ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr); }
  ierr = PetscCalloc1(sz*sz,&(op)->H);CHKERRQ(ierr);
  if (op->szd) {
    op->simpU = PETSC_FALSE;
    if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
      ierr = PetscOptionsGetBool(NULL,NULL,"-nep_deflation_simpleu",&op->simpU,NULL);CHKERRQ(ierr);
    } else {
      op->simpU = PETSC_TRUE;
    }
    ierr = PetscCalloc3(szd*szd*op->max_midx,&(op)->Hj,szd*szd,&(op)->XpX,szd,&op->bc);CHKERRQ(ierr);
    ierr = BVDuplicateResize(op->X,op->szd,&op->W);CHKERRQ(ierr);
  }
  if (ksp) {
    ierr = PetscNew(&solve);CHKERRQ(ierr);
    op->solve  = solve;
    solve->ksp = ksp;
    solve->n   = -1;
    if (op->szd) {
      if (!op->simpU) {
        ierr = BVDuplicateResize(nep->V,szd,&solve->T_1U);CHKERRQ(ierr);
      }
      ierr = PetscMalloc2(szd*szd,&solve->M,2*szd*szd,&solve->work);CHKERRQ(ierr);
      ierr = BVCreateVec(nep->V,&solve->w[0]);CHKERRQ(ierr);
      ierr = VecDuplicate(solve->w[0],&solve->w[1]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationCreateVec(NEP_EXT_OP extop,Vec *v)
{
  PetscErrorCode ierr;
  PetscInt       nloc;
  Vec            u;
  VecType        type;

  PetscFunctionBegin;
  if (extop->szd) {
    ierr = BVGetColumn(extop->nep->V,0,&u);CHKERRQ(ierr);
    ierr = VecGetType(u,&type);CHKERRQ(ierr);
    ierr = BVRestoreColumn(extop->nep->V,0,&u);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject)extop->nep),v);CHKERRQ(ierr);
    ierr = VecSetType(*v,type);CHKERRQ(ierr);
    ierr = BVGetSizes(extop->nep->V,&nloc,NULL,NULL);CHKERRQ(ierr);
    nloc += extop->szd;
    ierr = VecSetSizes(*v,nloc,PETSC_DECIDE);CHKERRQ(ierr);
  } else {
    ierr = BVCreateVec(extop->nep->V,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationEvaluateBasisMat(NEP_EXT_OP extop,PetscInt idx,PetscBool hat,PetscScalar *bval,PetscScalar *Hj,PetscScalar *Hjprev)
{
  PetscErrorCode ierr;
  PetscInt       i,k,n=extop->n,ldhj=extop->szd,ldh=extop->szd+1;
  PetscScalar    sone=1.0,zero=0.0;
  PetscBLASInt   ldh_,ldhj_,n_;

  PetscFunctionBegin;
  i = (idx<0)?extop->szd*extop->szd*(-idx):extop->szd*extop->szd;
  ierr = PetscMemzero(Hj,i*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldhj+1,&ldh_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldhj,&ldhj_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  if (idx<1) {
    if (!hat) for (i=0;i<extop->n;i++) Hj[i+i*ldhj] = 1.0;
    else for (i=0;i<extop->n;i++) Hj[i+i*ldhj] = 0.0;
  } else {
      for (i=0;i<n;i++) extop->H[i*ldh+i] -= extop->bc[idx-1];
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,extop->H,&ldh_,Hjprev,&ldhj_,&zero,Hj,&ldhj_));
      for (i=0;i<n;i++) extop->H[i*ldh+i] += extop->bc[idx-1];
      if (hat) for (i=0;i<n;i++) Hj[i*(n+1)] += bval[idx-1];
  }
  if (idx<0) {
    idx = -idx;
    for (k=1;k<idx;k++) {
      for (i=0;i<n;i++) extop->H[i*ldh+i] -= extop->bc[k-1];
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,extop->H,&ldh_,Hj+(k-1)*ldhj*ldhj,&ldhj_,&zero,Hj+k*ldhj*ldhj,&ldhj_));
      for (i=0;i<n;i++) extop->H[i*ldh+i] += extop->bc[k-1];
      if (hat) for (i=0;i<n;i++) Hj[i*(n+1)] += bval[k-1];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationLocking(NEP_EXT_OP extop,Vec u,PetscScalar lambda)
{
  PetscErrorCode ierr;
  Vec            uu;
  PetscInt       ld,i;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = BVGetColumn(extop->X,extop->n,&uu);CHKERRQ(ierr);
  ld = extop->szd+1;
  ierr = NEPDeflationCopyToExtendedVec(extop,uu,extop->H+extop->n*ld,u,PETSC_TRUE);CHKERRQ(ierr);
  ierr = BVRestoreColumn(extop->X,extop->n,&uu);CHKERRQ(ierr);
  ierr = BVNormColumn(extop->X,extop->n,NORM_2,&norm);CHKERRQ(ierr);
  ierr = BVScaleColumn(extop->X,extop->n,1.0/norm);CHKERRQ(ierr);
  for (i=0;i<extop->n;i++) extop->H[extop->n*ld+i] /= norm;
  extop->H[extop->n*(ld+1)] = lambda;
  extop->n++;
  ierr = BVSetActiveColumns(extop->X,0,extop->n);CHKERRQ(ierr);
  if (extop->n <= extop->szd) {
    /* update XpX */
    ierr = BVDotColumn(extop->X,extop->n-1,extop->XpX+(extop->n-1)*extop->szd);CHKERRQ(ierr);
    extop->XpX[(extop->n-1)*(1+extop->szd)] = 1.0;
    for (i=0;i<extop->n-1;i++) extop->XpX[i*extop->szd+extop->n-1] = PetscConj(extop->XpX[(extop->n-1)*extop->szd+i]);
    /* determine minimality index */
    extop->midx = PetscMin(extop->max_midx,extop->n); 
    /* polynominal basis coeficients */
    for (i=0;i<extop->midx;i++) extop->bc[i] = extop->nep->target;
    /* evaluate the polynomial basis in H */
    ierr = NEPDeflationEvaluateBasisMat(extop,-extop->midx,PETSC_FALSE,NULL,extop->Hj,NULL);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationEvaluateHatFunction(NEP_EXT_OP extop, PetscInt idx,PetscScalar lambda,PetscScalar *y,PetscScalar *hfj,PetscScalar *hfjp,PetscInt ld)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,off,ini,fin,sz,ldh,n=extop->n;
  Mat            A,B;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (idx<0) {ini = 0; fin = extop->nep->nt;}
  else {ini = idx; fin = idx+1;}
  sz = hfjp?n+2:n+1;
  ldh = extop->szd+1;
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,sz,sz,NULL,&A);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,sz,sz,NULL,&B);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&array);CHKERRQ(ierr);
  for (j=0;j<n;j++) 
    for (i=0;i<n;i++) array[j*sz+i] = extop->H[j*ldh+i];
  array[extop->n*(sz+1)] = lambda;
  if (hfjp) { array[(n+1)*sz+n] = 1.0; array[(n+1)*sz+n+1] = lambda;}
  ierr = MatDenseRestoreArray(A,&array);CHKERRQ(ierr);
  if (y) {
    ierr = MatDenseGetArray(A,&array);CHKERRQ(ierr);
    for (i=0;i<n;i++) array[n*sz+i] = y[i];
    ierr = MatDenseRestoreArray(A,&array);CHKERRQ(ierr);
    for (j=ini;j<fin;j++) {
      ierr = FNEvaluateFunctionMat(extop->nep->f[j],A,B);CHKERRQ(ierr);
      ierr = MatDenseGetArray(B,&array);CHKERRQ(ierr);
      for (i=0;i<n;i++) hfj[j*ld+i] = array[n*sz+i];
      if (hfjp) for (i=0;i<n;i++) hfjp[j*ld+i] = array[(n+1)*sz+i];
      ierr = MatDenseRestoreArray(B,&array);CHKERRQ(ierr);
    }
  } else {
    off = ld*n;
    for (i=0;i<n;i++) {
      ierr = MatDenseGetArray(A,&array);CHKERRQ(ierr);
      for (k=0;k<n;k++) array[n*sz+k] = 0.0;
      array[n*sz+i] = 1.0;
      ierr = MatDenseRestoreArray(A,&array);CHKERRQ(ierr);
      for (j=ini;j<fin;j++) {
        ierr = FNEvaluateFunctionMat(extop->nep->f[j],A,B);CHKERRQ(ierr);
        ierr = MatDenseGetArray(B,&array);CHKERRQ(ierr);
        for (k=0;k<n;k++) hfj[j*off+i*ld+k] = array[n*sz+k];
        if (hfjp) for (k=0;k<n;k++) hfjp[j*off+i*ld+k] = array[(n+1)*sz+k];
        ierr = MatDenseRestoreArray(B,&array);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationMatShell_MatMult(Mat M,Vec x,Vec y)
{
  NEP_DEF_MATSHELL  *matctx;
  PetscErrorCode    ierr;
  NEP_EXT_OP        extop;
  Vec               x1,y1;
  PetscScalar       *yy,sone=1.0,zero=0.0;
  const PetscScalar *xx;
  PetscInt          nloc,i;
  PetscBLASInt      n_,one=1,szd_;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&matctx);CHKERRQ(ierr);
  extop = matctx->extop;
  if (extop->szd) {
    x1 = matctx->w[0]; y1 = matctx->w[1];
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecPlaceArray(x1,xx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    ierr = VecPlaceArray(y1,yy);CHKERRQ(ierr);
    ierr = MatMult(matctx->T,x1,y1);CHKERRQ(ierr);
    if (extop->n) {
      ierr = VecGetLocalSize(x1,&nloc);CHKERRQ(ierr);
      /* copy for avoiding warning of constant array xx */
      for (i=0;i<extop->n;i++) matctx->work[i] = xx[nloc+i];
      ierr = BVMultVec(matctx->U,1.0,1.0,y1,matctx->work);CHKERRQ(ierr);
      ierr = BVDotVec(extop->X,x1,matctx->work);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(extop->n,&n_);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(extop->szd,&szd_);CHKERRQ(ierr);
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,matctx->A,&szd_,matctx->work,&one,&zero,yy+nloc,&one));
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,matctx->B,&szd_,xx+nloc,&one,&sone,yy+nloc,&one));
    }
    ierr = VecResetArray(x1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecResetArray(y1);CHKERRQ(ierr);    
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  } else {
    ierr = MatMult(matctx->T,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationMatShell_CreateVecs(Mat M,Vec *right,Vec *left)
{
  PetscErrorCode   ierr;
  NEP_DEF_MATSHELL *matctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&matctx);CHKERRQ(ierr);
  if (right) {
    ierr = VecDuplicate(matctx->w[0],right);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecDuplicate(matctx->w[0],left);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationMatShell_Destroy(Mat M)
{
  PetscErrorCode   ierr;
  NEP_DEF_MATSHELL *matctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&matctx);CHKERRQ(ierr);
  if (matctx->extop->szd) {
    ierr = BVDestroy(&matctx->U);
    ierr = PetscFree2(matctx->hfj,matctx->work);CHKERRQ(ierr);
    ierr = PetscFree2(matctx->A,matctx->B);CHKERRQ(ierr);
    ierr = VecDestroy(&matctx->w[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&matctx->w[1]);CHKERRQ(ierr);
  }
  ierr = PetscFree(matctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationEvaluateBasis(NEP_EXT_OP extop,PetscScalar lambda,PetscInt n,PetscScalar *val,PetscBool jacobian)
{
  PetscScalar p;
  PetscInt    i;  

  PetscFunctionBegin;
  if (!jacobian) {
    val[0] = 1.0;
    for (i=1;i<extop->n;i++) val[i] = val[i-1]*(lambda-extop->bc[i-1]);
  } else {
    val[0] = 0.0;
    p = 1.0;
    for (i=1;i<extop->n;i++) {
      val[i] = val[i-1]*(lambda-extop->bc[i-1])+p;
      p *= (lambda-extop->bc[i-1]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationComputeShellMat(NEP_EXT_OP extop,PetscScalar lambda,PetscBool jacobian)
{
  PetscErrorCode   ierr;
  NEP_DEF_MATSHELL *matctx,*matctxC;
  PetscInt         nloc,mloc,n=extop->n,j,i,szd=extop->szd,ldh=szd+1,k;
  Mat              F;
  Mat              Mshell,Mcomp;
  PetscBool        ini=PETSC_FALSE;
  PetscScalar      *hf,*hfj,*hfjp,sone=1.0,*hH,*hHprev,*pts,*B,*A,*Hj=extop->Hj,*basisv,zero=0.0;
  PetscBLASInt     n_,info,szd_;

  PetscFunctionBegin;
  Mshell = jacobian?extop->MJ:extop->MF;
  Mcomp  = jacobian?extop->MF:extop->MJ;
  if (!Mshell) {
    ini = PETSC_TRUE;
    ierr = PetscNew(&matctx);CHKERRQ(ierr);
    ierr = MatGetLocalSize(extop->nep->function,&mloc,&nloc);CHKERRQ(ierr);
    nloc += szd; mloc += szd;
    ierr = MatCreateShell(PetscObjectComm((PetscObject)extop->nep),nloc,mloc,PETSC_DETERMINE,PETSC_DETERMINE,matctx,&Mshell);CHKERRQ(ierr);
    ierr = MatShellSetOperation(Mshell,MATOP_MULT,(void(*)())NEPDeflationMatShell_MatMult);CHKERRQ(ierr);
    ierr = MatShellSetOperation(Mshell,MATOP_CREATE_VECS,(void(*)())NEPDeflationMatShell_CreateVecs);CHKERRQ(ierr);
    ierr = MatShellSetOperation(Mshell,MATOP_DESTROY,(void(*)())NEPDeflationMatShell_Destroy);CHKERRQ(ierr);
    matctx->nep = extop->nep;
    matctx->extop = extop;
    if (jacobian) { matctx->jacob = PETSC_TRUE; matctx->T = extop->nep->jacobian; extop->MJ = Mshell; }
    else { matctx->jacob = PETSC_FALSE; matctx->T = extop->nep->function; extop->MF = Mshell; }
    if (szd) {
      ierr = BVCreateVec(extop->nep->V,matctx->w);CHKERRQ(ierr);
      ierr = VecDuplicate(matctx->w[0],matctx->w+1);CHKERRQ(ierr);
      ierr = BVDuplicateResize(extop->nep->V,szd,&matctx->U);CHKERRQ(ierr);
      ierr = PetscMalloc2(extop->simpU?2*(szd)*(szd):2*(szd)*(szd)*extop->nep->nt,&matctx->hfj,szd,&matctx->work);CHKERRQ(ierr);
      ierr = PetscMalloc2(szd*szd,&matctx->A,szd*szd,&matctx->B);CHKERRQ(ierr);
    }
  } else {
    ierr = MatShellGetContext(Mshell,(void**)&matctx);CHKERRQ(ierr);    
  }
  if (ini || matctx->theta != lambda || matctx->n != extop->n) {
    if (ini || matctx->theta != lambda) {
      if (jacobian) {
        ierr = NEPComputeJacobian(extop->nep,lambda,matctx->T);CHKERRQ(ierr);
      } else {
        ierr = NEPComputeFunction(extop->nep,lambda,matctx->T,matctx->T);CHKERRQ(ierr);
      }
    }
    if (n) {
      matctx->hfjset = PETSC_FALSE;
      if (!extop->simpU) {
        /* likely hfjp has been already computed */
        if (Mcomp) {
          ierr = MatShellGetContext(Mcomp,(void**)&matctxC);CHKERRQ(ierr);    
          if (matctxC->hfjset && matctxC->theta == lambda && matctxC->n == extop->n) {
            ierr = PetscMemcpy(matctx->hfj,matctxC->hfj,2*extop->szd*extop->szd*extop->nep->nt*sizeof(PetscScalar));CHKERRQ(ierr);
            matctx->hfjset = PETSC_TRUE;
          }
        }
        hfj = matctx->hfj; hfjp = matctx->hfj+extop->szd*extop->szd*extop->nep->nt;
        if (!matctx->hfjset) {
          ierr = NEPDeflationEvaluateHatFunction(extop,-1,lambda,NULL,hfj,hfjp,n);CHKERRQ(ierr);
          matctx->hfjset = PETSC_TRUE;
        }
        ierr = BVSetActiveColumns(matctx->U,0,n);CHKERRQ(ierr);
        hf = jacobian?hfjp:hfj;
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,hf,&F);CHKERRQ(ierr);
        ierr = BVMatMult(extop->X,extop->nep->A[0],matctx->U);CHKERRQ(ierr);
        ierr = BVMultInPlace(matctx->U,F,0,n);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(extop->W,0,extop->n);CHKERRQ(ierr);
        for (j=1;j<extop->nep->nt;j++) {
          ierr = BVMatMult(extop->X,extop->nep->A[j],extop->W);CHKERRQ(ierr);
          ierr = MatDensePlaceArray(F,hf+j*n*n);CHKERRQ(ierr);
          ierr = BVMult(matctx->U,1.0,1.0,extop->W,F);CHKERRQ(ierr);
          ierr = MatDenseResetArray(F);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&F);CHKERRQ(ierr);
      } else {
        hfj = matctx->hfj;
        ierr = BVSetActiveColumns(matctx->U,0,n);CHKERRQ(ierr);
        ierr = BVMatMult(extop->X,matctx->T,matctx->U);CHKERRQ(ierr);
        for (j=0;j<n;j++) {
          for (i=0;i<n;i++) hfj[j*n+i] = -extop->H[j*ldh+i];
          hfj[j*(n+1)] += lambda;
        }
        ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
        PetscStackCallBLAS("LAPACKtrtri",LAPACKtrtri_("U","N",&n_,hfj,&n_,&info));
        SlepcCheckLapackInfo("trtri",info);          
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,hfj,&F);CHKERRQ(ierr);
        ierr = BVMultInPlace(matctx->U,F,0,n);CHKERRQ(ierr);
        if (jacobian) {
          ierr = NEPDeflationComputeFunction(extop,lambda,NULL);CHKERRQ(ierr);
          ierr = PetscMemcpy(hfj+n*n,hfj,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
          PetscStackCallBLAS("BLAStrmm",BLAStrmm_("L","U","N","N",&n_,&n_,&sone,hfj,&n_,hfj+n*n,&n_));
          ierr = MatShellGetContext(extop->MF,(void**)&matctxC);CHKERRQ(ierr);
          ierr = BVSetActiveColumns(extop->W,0,n);CHKERRQ(ierr);
          ierr = BVMatMult(extop->X,matctxC->T,extop->W);CHKERRQ(ierr);
          ierr = MatDensePlaceArray(F,hfj+n*n);CHKERRQ(ierr);
          ierr = BVMult(matctxC->U,-1.0,1.0,extop->W,F);CHKERRQ(ierr);
          ierr = MatDenseResetArray(F);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&F);CHKERRQ(ierr);
      }
      ierr = PetscCalloc3(n,&basisv,szd*szd,&hH,szd*szd,&hHprev);CHKERRQ(ierr);
      ierr = NEPDeflationEvaluateBasis(extop,lambda,n,basisv,jacobian);CHKERRQ(ierr);
      A = matctx->A;
      ierr = PetscMemzero(A,szd*szd*sizeof(PetscScalar));CHKERRQ(ierr);
      if (!jacobian) for (i=0;i<n;i++) A[i*(szd+1)] = 1.0;
      for (j=0;j<n;j++) 
        for (i=0;i<n;i++) 
          for (k=1;k<extop->midx;k++) A[j*szd+i] += basisv[k]*PetscConj(Hj[k*szd*szd+i*szd+j]);
      ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(szd,&szd_);CHKERRQ(ierr);
      B = matctx->B;
      ierr = PetscMemzero(B,szd*szd*sizeof(PetscScalar));CHKERRQ(ierr);
      for (i=1;i<extop->midx;i++) {
        ierr = NEPDeflationEvaluateBasisMat(extop,i,PETSC_TRUE,basisv,hH,hHprev);CHKERRQ(ierr);
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,extop->XpX,&szd_,hH,&szd_,&zero,hHprev,&szd_));
        PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n_,&n_,&n_,&sone,extop->Hj+szd*szd*i,&szd_,hHprev,&szd_,&sone,B,&szd_));
        pts = hHprev; hHprev = hH; hH = pts;
      }
      ierr = PetscFree3(basisv,hH,hHprev);CHKERRQ(ierr);
    }
    matctx->theta = lambda;
    matctx->n = extop->n;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationComputeFunction(NEP_EXT_OP extop,PetscScalar lambda,Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = NEPDeflationComputeShellMat(extop,lambda,PETSC_FALSE);CHKERRQ(ierr);
  if (F) *F = extop->MF;
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationComputeJacobian(NEP_EXT_OP extop,PetscScalar lambda,Mat *J)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = NEPDeflationComputeShellMat(extop,lambda,PETSC_TRUE);CHKERRQ(ierr);
  if (J) *J = extop->MJ;
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationSolveSetUp(NEP_EXT_OP extop,PetscScalar lambda)
{
  PetscErrorCode    ierr;
  NEP_DEF_MATSHELL  *matctx;
  NEP_DEF_FUN_SOLVE solve;
  PetscInt          i;
  Vec               u,tu;
  Mat               F;
  PetscScalar       snone=-1.0,sone=1.0,zero=0.0;
  PetscBLASInt      n_,szd_,ldh_,*p,info;

  PetscFunctionBegin;
  solve = extop->solve;
  if (lambda!=solve->theta || extop->n!=solve->n) {
    ierr = NEPDeflationComputeFunction(extop,lambda,NULL);CHKERRQ(ierr);
    ierr = MatShellGetContext(extop->MF,(void**)&matctx);CHKERRQ(ierr);
    ierr = KSPSetOperators(solve->ksp,matctx->T,matctx->T);CHKERRQ(ierr);
    if (extop->n) {
      ierr = PetscBLASIntCast(extop->n,&n_);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(extop->szd,&szd_);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(extop->szd+1,&ldh_);CHKERRQ(ierr);
      if (!extop->simpU) {
        ierr = BVSetActiveColumns(solve->T_1U,0,extop->n);CHKERRQ(ierr);
        for (i=0;i<extop->n;i++) {
          ierr = BVGetColumn(matctx->U,i,&u);CHKERRQ(ierr);
          ierr = BVGetColumn(solve->T_1U,i,&tu);CHKERRQ(ierr);
          ierr = KSPSolve(solve->ksp,u,tu);CHKERRQ(ierr);
          ierr = BVRestoreColumn(solve->T_1U,i,&tu);CHKERRQ(ierr);
          ierr = BVRestoreColumn(matctx->U,i,&u);CHKERRQ(ierr);
        }
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,extop->n,extop->n,solve->work,&F);CHKERRQ(ierr);
        ierr = BVDot(solve->T_1U,extop->X,F);CHKERRQ(ierr);
        ierr = MatDestroy(&F);CHKERRQ(ierr);
      } else {
        for (i=0;i<extop->n;i++) extop->H[i*ldh_+i] -= lambda;
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&snone,extop->XpX,&szd_,extop->H,&ldh_,&zero,solve->work,&n_)); 
        for (i=0;i<extop->n;i++) extop->H[i*ldh_+i] += lambda;
      }
      ierr = PetscMemcpy(solve->M,matctx->B,extop->szd*extop->szd*sizeof(PetscScalar));CHKERRQ(ierr);
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&snone,matctx->A,&szd_,solve->work,&n_,&sone,solve->M,&szd_));
      ierr = PetscMalloc1(extop->n,&p);CHKERRQ(ierr);    
      PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n_,&n_,solve->M,&szd_,p,&info));
      SlepcCheckLapackInfo("getrf",info);
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n_,solve->M,&szd_,p,solve->work,&n_,&info));
      SlepcCheckLapackInfo("getri",info);
      ierr = PetscFree(p);CHKERRQ(ierr);    
    }
    solve->theta = lambda;
    solve->n = extop->n;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationFunctionSolve(NEP_EXT_OP extop,Vec b,Vec x)
{
  PetscErrorCode    ierr;
  Vec               b1,x1;
  PetscScalar       *xx,*bb,*x2,*b2,*w,*w2,snone=-1.0,sone=1.0,zero=0.0;
  NEP_DEF_MATSHELL  *matctx;
  NEP_DEF_FUN_SOLVE solve=extop->solve;
  PetscBLASInt      one=1,szd_,n_,ldh_;
  PetscInt          nloc,i;

  PetscFunctionBegin;
  if (extop->szd) {
    x1 = solve->w[0]; b1 = solve->w[1];
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    ierr = VecPlaceArray(x1,xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    ierr = VecPlaceArray(b1,bb);CHKERRQ(ierr);
  } else {
    b1 = b; x1 = x;
  }
  ierr = KSPSolve(extop->solve->ksp,b1,x1);CHKERRQ(ierr);
  if (extop->n) {
    ierr = PetscBLASIntCast(extop->szd,&szd_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(extop->n,&n_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(extop->szd+1,&ldh_);CHKERRQ(ierr);
    ierr = BVGetSizes(extop->nep->V,&nloc,NULL,NULL);CHKERRQ(ierr);
    b2 = bb+nloc; x2 = xx+nloc;
    w = solve->work; w2 = solve->work+extop->n;
    ierr = MatShellGetContext(extop->MF,(void**)&matctx);CHKERRQ(ierr);
    ierr = PetscMemcpy(w2,b2,extop->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = BVDotVec(extop->X,x1,w);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&snone,matctx->A,&szd_,w,&one,&sone,w2,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,solve->M,&szd_,w2,&one,&zero,x2,&one));
    if (extop->simpU) {
      for (i=0;i<extop->n;i++) extop->H[i+i*(extop->szd+1)] -= solve->theta;
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,extop->H,&ldh_,x2,&one,&zero,w2,&one));
      for (i=0;i<extop->n;i++) extop->H[i+i*(extop->szd+1)] += solve->theta;
      ierr = BVMultVec(extop->X,1.0,1.0,x1,x2);CHKERRQ(ierr);
    } else {
      ierr = BVMultVec(solve->T_1U,-1.0,1.0,x1,x2);CHKERRQ(ierr);
    }
  }
  if (extop->szd) {
    ierr = VecResetArray(x1);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    ierr = VecResetArray(b1);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
