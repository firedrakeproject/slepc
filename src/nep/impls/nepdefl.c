/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <slepcblaslapack.h>
#include "nepdefl.h"

PetscErrorCode NEPDeflationGetInvariantPair(NEP_EXT_OP extop,BV *X,Mat *H)
{
  PetscFunctionBegin;
  if (X) *X = extop->X;
  if (H) {
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,extop->szd+1,extop->szd+1,extop->H,H));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationExtendInvariantPair(NEP_EXT_OP extop,Vec u,PetscScalar lambda,PetscInt k)
{
  Vec            uu;
  PetscInt       ld,i;
  PetscMPIInt    np;
  PetscReal      norm;

  PetscFunctionBegin;
  CHKERRQ(BVGetColumn(extop->X,k,&uu));
  ld = extop->szd+1;
  CHKERRQ(NEPDeflationCopyToExtendedVec(extop,uu,extop->H+k*ld,u,PETSC_TRUE));
  CHKERRQ(BVRestoreColumn(extop->X,k,&uu));
  CHKERRQ(BVNormColumn(extop->X,k,NORM_2,&norm));
  CHKERRQ(BVScaleColumn(extop->X,k,1.0/norm));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)u),&np));
  for (i=0;i<k;i++) extop->H[k*ld+i] *= PetscSqrtReal(np)/norm;
  extop->H[k*(ld+1)] = lambda;
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationExtractEigenpair(NEP_EXT_OP extop,PetscInt k,Vec u,PetscScalar lambda,DS ds)
{
  PetscScalar    *Ap;
  PetscInt       ldh=extop->szd+1,ldds,i,j,k1=k+1;
  PetscScalar    *eigr,*eigi,*t,*Z;
  Vec            x;

  PetscFunctionBegin;
  CHKERRQ(NEPDeflationExtendInvariantPair(extop,u,lambda,k));
  CHKERRQ(PetscCalloc3(k1,&eigr,k1,&eigi,extop->szd,&t));
  CHKERRQ(DSReset(ds));
  CHKERRQ(DSSetType(ds,DSNHEP));
  CHKERRQ(DSAllocate(ds,ldh));
  CHKERRQ(DSGetLeadingDimension(ds,&ldds));
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&Ap));
  for (j=0;j<k1;j++)
    for (i=0;i<k1;i++) Ap[j*ldds+i] = extop->H[j*ldh+i];
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&Ap));
  CHKERRQ(DSSetDimensions(ds,k1,0,k1));
  CHKERRQ(DSSolve(ds,eigr,eigi));
  CHKERRQ(DSVectors(ds,DS_MAT_X,&k,NULL));
  CHKERRQ(DSGetArray(ds,DS_MAT_X,&Z));
  CHKERRQ(BVMultColumn(extop->X,1.0,Z[k*ldds+k],k,Z+k*ldds));
  CHKERRQ(DSRestoreArray(ds,DS_MAT_X,&Z));
  CHKERRQ(BVGetColumn(extop->X,k,&x));
  CHKERRQ(NEPDeflationCopyToExtendedVec(extop,x,t,u,PETSC_FALSE));
  CHKERRQ(BVRestoreColumn(extop->X,k,&x));
  CHKERRQ(PetscFree3(eigr,eigi,t));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationCopyToExtendedVec(NEP_EXT_OP extop,Vec v,PetscScalar *a,Vec vex,PetscBool back)
{
  PetscMPIInt    np,rk,count;
  PetscScalar    *array1,*array2;
  PetscInt       nloc;

  PetscFunctionBegin;
  if (extop->szd) {
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)vex),&rk));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)vex),&np));
    CHKERRQ(BVGetSizes(extop->nep->V,&nloc,NULL,NULL));
    if (v) {
      CHKERRQ(VecGetArray(v,&array1));
      CHKERRQ(VecGetArray(vex,&array2));
      if (back) {
        CHKERRQ(PetscArraycpy(array1,array2,nloc));
      } else {
        CHKERRQ(PetscArraycpy(array2,array1,nloc));
      }
      CHKERRQ(VecRestoreArray(v,&array1));
      CHKERRQ(VecRestoreArray(vex,&array2));
    }
    if (a) {
      CHKERRQ(VecGetArray(vex,&array2));
      if (back) {
        CHKERRQ(PetscArraycpy(a,array2+nloc,extop->szd));
        CHKERRQ(PetscMPIIntCast(extop->szd,&count));
        CHKERRMPI(MPI_Bcast(a,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)vex)));
      } else {
        CHKERRQ(PetscArraycpy(array2+nloc,a,extop->szd));
        CHKERRQ(PetscMPIIntCast(extop->szd,&count));
        CHKERRMPI(MPI_Bcast(array2+nloc,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)vex)));
      }
      CHKERRQ(VecRestoreArray(vex,&array2));
    }
  } else {
    if (back) CHKERRQ(VecCopy(vex,v));
    else CHKERRQ(VecCopy(v,vex));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationCreateVec(NEP_EXT_OP extop,Vec *v)
{
  PetscInt       nloc;
  Vec            u;
  VecType        type;

  PetscFunctionBegin;
  if (extop->szd) {
    CHKERRQ(BVGetColumn(extop->nep->V,0,&u));
    CHKERRQ(VecGetType(u,&type));
    CHKERRQ(BVRestoreColumn(extop->nep->V,0,&u));
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)extop->nep),v));
    CHKERRQ(VecSetType(*v,type));
    CHKERRQ(BVGetSizes(extop->nep->V,&nloc,NULL,NULL));
    nloc += extop->szd;
    CHKERRQ(VecSetSizes(*v,nloc,PETSC_DECIDE));
  } else {
    CHKERRQ(BVCreateVec(extop->nep->V,v));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationCreateBV(NEP_EXT_OP extop,PetscInt sz,BV *V)
{
  PetscInt           nloc;
  BVType             type;
  BVOrthogType       otype;
  BVOrthogRefineType oref;
  PetscReal          oeta;
  BVOrthogBlockType  oblock;
  NEP                nep=extop->nep;

  PetscFunctionBegin;
  if (extop->szd) {
    CHKERRQ(BVGetSizes(nep->V,&nloc,NULL,NULL));
    CHKERRQ(BVCreate(PetscObjectComm((PetscObject)nep),V));
    CHKERRQ(BVSetSizes(*V,nloc+extop->szd,PETSC_DECIDE,sz));
    CHKERRQ(BVGetType(nep->V,&type));
    CHKERRQ(BVSetType(*V,type));
    CHKERRQ(BVGetOrthogonalization(nep->V,&otype,&oref,&oeta,&oblock));
    CHKERRQ(BVSetOrthogonalization(*V,otype,oref,oeta,oblock));
    CHKERRQ(PetscObjectStateIncrease((PetscObject)*V));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)*V));
  } else {
    CHKERRQ(BVDuplicateResize(nep->V,sz,V));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationSetRandomVec(NEP_EXT_OP extop,Vec v)
{
  PetscInt       n,next,i;
  PetscRandom    rand;
  PetscScalar    *array;
  PetscMPIInt    nn,np;

  PetscFunctionBegin;
  CHKERRQ(BVGetRandomContext(extop->nep->V,&rand));
  CHKERRQ(VecSetRandom(v,rand));
  if (extop->szd) {
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v),&np));
    CHKERRQ(BVGetSizes(extop->nep->V,&n,NULL,NULL));
    CHKERRQ(VecGetLocalSize(v,&next));
    CHKERRQ(VecGetArray(v,&array));
    for (i=n+extop->n;i<next;i++) array[i] = 0.0;
    for (i=n;i<n+extop->n;i++) array[i] /= PetscSqrtReal(np);
    CHKERRQ(PetscMPIIntCast(extop->n,&nn));
    CHKERRMPI(MPI_Bcast(array+n,nn,MPIU_SCALAR,0,PetscObjectComm((PetscObject)v)));
    CHKERRQ(VecRestoreArray(v,&array));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationEvaluateBasisMat(NEP_EXT_OP extop,PetscInt idx,PetscBool hat,PetscScalar *bval,PetscScalar *Hj,PetscScalar *Hjprev)
{
  PetscInt       i,k,n=extop->n,ldhj=extop->szd,ldh=extop->szd+1;
  PetscScalar    sone=1.0,zero=0.0;
  PetscBLASInt   ldh_,ldhj_,n_;

  PetscFunctionBegin;
  i = (idx<0)?extop->szd*extop->szd*(-idx):extop->szd*extop->szd;
  CHKERRQ(PetscArrayzero(Hj,i));
  CHKERRQ(PetscBLASIntCast(ldhj+1,&ldh_));
  CHKERRQ(PetscBLASIntCast(ldhj,&ldhj_));
  CHKERRQ(PetscBLASIntCast(n,&n_));
  if (idx<1) {
    if (!hat) for (i=0;i<extop->n;i++) Hj[i+i*ldhj] = 1.0;
    else for (i=0;i<extop->n;i++) Hj[i+i*ldhj] = 0.0;
  } else {
      for (i=0;i<n;i++) extop->H[i*ldh+i] -= extop->bc[idx-1];
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,extop->H,&ldh_,Hjprev,&ldhj_,&zero,Hj,&ldhj_));
      for (i=0;i<n;i++) extop->H[i*ldh+i] += extop->bc[idx-1];
      if (hat) for (i=0;i<n;i++) Hj[i*(ldhj+1)] += bval[idx-1];
  }
  if (idx<0) {
    idx = -idx;
    for (k=1;k<idx;k++) {
      for (i=0;i<n;i++) extop->H[i*ldh+i] -= extop->bc[k-1];
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,extop->H,&ldh_,Hj+(k-1)*ldhj*ldhj,&ldhj_,&zero,Hj+k*ldhj*ldhj,&ldhj_));
      for (i=0;i<n;i++) extop->H[i*ldh+i] += extop->bc[k-1];
      if (hat) for (i=0;i<n;i++) Hj[i*(ldhj+1)] += bval[k-1];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationLocking(NEP_EXT_OP extop,Vec u,PetscScalar lambda)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(NEPDeflationExtendInvariantPair(extop,u,lambda,extop->n));
  extop->n++;
  CHKERRQ(BVSetActiveColumns(extop->X,0,extop->n));
  if (extop->n <= extop->szd) {
    /* update XpX */
    CHKERRQ(BVDotColumn(extop->X,extop->n-1,extop->XpX+(extop->n-1)*extop->szd));
    extop->XpX[(extop->n-1)*(1+extop->szd)] = 1.0;
    for (i=0;i<extop->n-1;i++) extop->XpX[i*extop->szd+extop->n-1] = PetscConj(extop->XpX[(extop->n-1)*extop->szd+i]);
    /* determine minimality index */
    extop->midx = PetscMin(extop->max_midx,extop->n);
    /* polynominal basis coefficients */
    for (i=0;i<extop->midx;i++) extop->bc[i] = extop->nep->target;
    /* evaluate the polynomial basis in H */
    CHKERRQ(NEPDeflationEvaluateBasisMat(extop,-extop->midx,PETSC_FALSE,NULL,extop->Hj,NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPDeflationEvaluateHatFunction(NEP_EXT_OP extop, PetscInt idx,PetscScalar lambda,PetscScalar *y,PetscScalar *hfj,PetscScalar *hfjp,PetscInt ld)
{
  PetscInt          i,j,k,off,ini,fin,sz,ldh,n=extop->n;
  Mat               A,B;
  PetscScalar       *array;
  const PetscScalar *barray;

  PetscFunctionBegin;
  if (idx<0) {ini = 0; fin = extop->nep->nt;}
  else {ini = idx; fin = idx+1;}
  if (y) sz = hfjp?n+2:n+1;
  else sz = hfjp?3*n:2*n;
  ldh = extop->szd+1;
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,sz,sz,NULL,&A));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,sz,sz,NULL,&B));
  CHKERRQ(MatDenseGetArray(A,&array));
  for (j=0;j<n;j++)
    for (i=0;i<n;i++) array[j*sz+i] = extop->H[j*ldh+i];
  CHKERRQ(MatDenseRestoreArrayWrite(A,&array));
  if (y) {
    CHKERRQ(MatDenseGetArray(A,&array));
    array[extop->n*(sz+1)] = lambda;
    if (hfjp) { array[(n+1)*sz+n] = 1.0; array[(n+1)*sz+n+1] = lambda;}
    for (i=0;i<n;i++) array[n*sz+i] = y[i];
    CHKERRQ(MatDenseRestoreArrayWrite(A,&array));
    for (j=ini;j<fin;j++) {
      CHKERRQ(FNEvaluateFunctionMat(extop->nep->f[j],A,B));
      CHKERRQ(MatDenseGetArrayRead(B,&barray));
      for (i=0;i<n;i++) hfj[j*ld+i] = barray[n*sz+i];
      if (hfjp) for (i=0;i<n;i++) hfjp[j*ld+i] = barray[(n+1)*sz+i];
      CHKERRQ(MatDenseRestoreArrayRead(B,&barray));
    }
  } else {
    off = idx<0?ld*n:0;
    CHKERRQ(MatDenseGetArray(A,&array));
    for (k=0;k<n;k++) {
      array[(n+k)*sz+k] = 1.0;
      array[(n+k)*sz+n+k] = lambda;
    }
    if (hfjp) for (k=0;k<n;k++) {
      array[(2*n+k)*sz+n+k] = 1.0;
      array[(2*n+k)*sz+2*n+k] = lambda;
    }
    CHKERRQ(MatDenseRestoreArray(A,&array));
    for (j=ini;j<fin;j++) {
      CHKERRQ(FNEvaluateFunctionMat(extop->nep->f[j],A,B));
      CHKERRQ(MatDenseGetArrayRead(B,&barray));
      for (i=0;i<n;i++) for (k=0;k<n;k++) hfj[j*off+i*ld+k] = barray[n*sz+i*sz+k];
      if (hfjp) for (k=0;k<n;k++) for (i=0;i<n;i++) hfjp[j*off+i*ld+k] = barray[2*n*sz+i*sz+k];
      CHKERRQ(MatDenseRestoreArrayRead(B,&barray));
    }
  }
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_NEPDeflation(Mat M,Vec x,Vec y)
{
  NEP_DEF_MATSHELL  *matctx;
  NEP_EXT_OP        extop;
  Vec               x1,y1;
  PetscScalar       *yy,sone=1.0,zero=0.0;
  const PetscScalar *xx;
  PetscInt          nloc,i;
  PetscMPIInt       np;
  PetscBLASInt      n_,one=1,szd_;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)M),&np));
  CHKERRQ(MatShellGetContext(M,&matctx));
  extop = matctx->extop;
  if (extop->ref) {
    CHKERRQ(VecZeroEntries(y));
  }
  if (extop->szd) {
    x1 = matctx->w[0]; y1 = matctx->w[1];
    CHKERRQ(VecGetArrayRead(x,&xx));
    CHKERRQ(VecPlaceArray(x1,xx));
    CHKERRQ(VecGetArray(y,&yy));
    CHKERRQ(VecPlaceArray(y1,yy));
    CHKERRQ(MatMult(matctx->T,x1,y1));
    if (!extop->ref && extop->n) {
      CHKERRQ(VecGetLocalSize(x1,&nloc));
      /* copy for avoiding warning of constant array xx */
      for (i=0;i<extop->n;i++) matctx->work[i] = xx[nloc+i]*PetscSqrtReal(np);
      CHKERRQ(BVMultVec(matctx->U,1.0,1.0,y1,matctx->work));
      CHKERRQ(BVDotVec(extop->X,x1,matctx->work));
      CHKERRQ(PetscBLASIntCast(extop->n,&n_));
      CHKERRQ(PetscBLASIntCast(extop->szd,&szd_));
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,matctx->A,&szd_,matctx->work,&one,&zero,yy+nloc,&one));
      PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,matctx->B,&szd_,xx+nloc,&one,&sone,yy+nloc,&one));
      for (i=0;i<extop->n;i++) yy[nloc+i] /= PetscSqrtReal(np);
    }
    CHKERRQ(VecResetArray(x1));
    CHKERRQ(VecRestoreArrayRead(x,&xx));
    CHKERRQ(VecResetArray(y1));
    CHKERRQ(VecRestoreArray(y,&yy));
  } else {
    CHKERRQ(MatMult(matctx->T,x,y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateVecs_NEPDeflation(Mat M,Vec *right,Vec *left)
{
  NEP_DEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  if (right) {
    CHKERRQ(VecDuplicate(matctx->w[0],right));
  }
  if (left) {
    CHKERRQ(VecDuplicate(matctx->w[0],left));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_NEPDeflation(Mat M)
{
  NEP_DEF_MATSHELL *matctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&matctx));
  if (matctx->extop->szd) {
    CHKERRQ(BVDestroy(&matctx->U));
    CHKERRQ(PetscFree2(matctx->hfj,matctx->work));
    CHKERRQ(PetscFree2(matctx->A,matctx->B));
    CHKERRQ(VecDestroy(&matctx->w[0]));
    CHKERRQ(VecDestroy(&matctx->w[1]));
  }
  if (matctx->P != matctx->T) CHKERRQ(MatDestroy(&matctx->P));
  CHKERRQ(MatDestroy(&matctx->T));
  CHKERRQ(PetscFree(matctx));
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

static PetscErrorCode NEPDeflationComputeShellMat(NEP_EXT_OP extop,PetscScalar lambda,PetscBool jacobian,Mat *M)
{
  NEP_DEF_MATSHELL *matctx,*matctxC;
  PetscInt         nloc,mloc,n=extop->n,j,i,szd=extop->szd,ldh=szd+1,k;
  Mat              F,Mshell,Mcomp;
  PetscBool        ini=PETSC_FALSE;
  PetscScalar      *hf,*hfj,*hfjp,sone=1.0,*hH,*hHprev,*pts,*B,*A,*Hj=extop->Hj,*basisv,zero=0.0;
  PetscBLASInt     n_,info,szd_;

  PetscFunctionBegin;
  if (!M) Mshell = jacobian?extop->MJ:extop->MF;
  else Mshell = *M;
  Mcomp = jacobian?extop->MF:extop->MJ;
  if (!Mshell) {
    ini = PETSC_TRUE;
    CHKERRQ(PetscNew(&matctx));
    CHKERRQ(MatGetLocalSize(extop->nep->function,&mloc,&nloc));
    nloc += szd; mloc += szd;
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)extop->nep),nloc,mloc,PETSC_DETERMINE,PETSC_DETERMINE,matctx,&Mshell));
    CHKERRQ(MatShellSetOperation(Mshell,MATOP_MULT,(void(*)(void))MatMult_NEPDeflation));
    CHKERRQ(MatShellSetOperation(Mshell,MATOP_CREATE_VECS,(void(*)(void))MatCreateVecs_NEPDeflation));
    CHKERRQ(MatShellSetOperation(Mshell,MATOP_DESTROY,(void(*)(void))MatDestroy_NEPDeflation));
    matctx->nep = extop->nep;
    matctx->extop = extop;
    if (!M) {
      if (jacobian) { matctx->jacob = PETSC_TRUE; matctx->T = extop->nep->jacobian; extop->MJ = Mshell; }
      else { matctx->jacob = PETSC_FALSE; matctx->T = extop->nep->function; extop->MF = Mshell; }
      CHKERRQ(PetscObjectReference((PetscObject)matctx->T));
      if (!jacobian) {
        if (extop->nep->function_pre && extop->nep->function_pre != extop->nep->function) {
          matctx->P = extop->nep->function_pre;
          CHKERRQ(PetscObjectReference((PetscObject)matctx->P));
        } else matctx->P = matctx->T;
      }
    } else {
      matctx->jacob = jacobian;
      CHKERRQ(MatDuplicate(jacobian?extop->nep->jacobian:extop->nep->function,MAT_DO_NOT_COPY_VALUES,&matctx->T));
      *M = Mshell;
      if (!jacobian) {
        if (extop->nep->function_pre && extop->nep->function_pre != extop->nep->function) {
          CHKERRQ(MatDuplicate(extop->nep->function_pre,MAT_DO_NOT_COPY_VALUES,&matctx->P));
        } else matctx->P = matctx->T;
      }
    }
    if (szd) {
      CHKERRQ(BVCreateVec(extop->nep->V,matctx->w));
      CHKERRQ(VecDuplicate(matctx->w[0],matctx->w+1));
      CHKERRQ(BVDuplicateResize(extop->nep->V,szd,&matctx->U));
      CHKERRQ(PetscMalloc2(extop->simpU?2*(szd)*(szd):2*(szd)*(szd)*extop->nep->nt,&matctx->hfj,szd,&matctx->work));
      CHKERRQ(PetscMalloc2(szd*szd,&matctx->A,szd*szd,&matctx->B));
    }
  } else {
    CHKERRQ(MatShellGetContext(Mshell,&matctx));
  }
  if (ini || matctx->theta != lambda || matctx->n != extop->n) {
    if (ini || matctx->theta != lambda) {
      if (jacobian) {
        CHKERRQ(NEPComputeJacobian(extop->nep,lambda,matctx->T));
      } else {
        CHKERRQ(NEPComputeFunction(extop->nep,lambda,matctx->T,matctx->P));
      }
    }
    if (n) {
      matctx->hfjset = PETSC_FALSE;
      if (!extop->simpU) {
        /* likely hfjp has been already computed */
        if (Mcomp) {
          CHKERRQ(MatShellGetContext(Mcomp,&matctxC));
          if (matctxC->hfjset && matctxC->theta == lambda && matctxC->n == extop->n) {
            CHKERRQ(PetscArraycpy(matctx->hfj,matctxC->hfj,2*extop->szd*extop->szd*extop->nep->nt));
            matctx->hfjset = PETSC_TRUE;
          }
        }
        hfj = matctx->hfj; hfjp = matctx->hfj+extop->szd*extop->szd*extop->nep->nt;
        if (!matctx->hfjset) {
          CHKERRQ(NEPDeflationEvaluateHatFunction(extop,-1,lambda,NULL,hfj,hfjp,n));
          matctx->hfjset = PETSC_TRUE;
        }
        CHKERRQ(BVSetActiveColumns(matctx->U,0,n));
        hf = jacobian?hfjp:hfj;
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,hf,&F));
        CHKERRQ(BVMatMult(extop->X,extop->nep->A[0],matctx->U));
        CHKERRQ(BVMultInPlace(matctx->U,F,0,n));
        CHKERRQ(BVSetActiveColumns(extop->W,0,extop->n));
        for (j=1;j<extop->nep->nt;j++) {
          CHKERRQ(BVMatMult(extop->X,extop->nep->A[j],extop->W));
          CHKERRQ(MatDensePlaceArray(F,hf+j*n*n));
          CHKERRQ(BVMult(matctx->U,1.0,1.0,extop->W,F));
          CHKERRQ(MatDenseResetArray(F));
        }
        CHKERRQ(MatDestroy(&F));
      } else {
        hfj = matctx->hfj;
        CHKERRQ(BVSetActiveColumns(matctx->U,0,n));
        CHKERRQ(BVMatMult(extop->X,matctx->T,matctx->U));
        for (j=0;j<n;j++) {
          for (i=0;i<n;i++) hfj[j*n+i] = -extop->H[j*ldh+i];
          hfj[j*(n+1)] += lambda;
        }
        CHKERRQ(PetscBLASIntCast(n,&n_));
        CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
        PetscStackCallBLAS("LAPACKtrtri",LAPACKtrtri_("U","N",&n_,hfj,&n_,&info));
        CHKERRQ(PetscFPTrapPop());
        SlepcCheckLapackInfo("trtri",info);
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,hfj,&F));
        CHKERRQ(BVMultInPlace(matctx->U,F,0,n));
        if (jacobian) {
          CHKERRQ(NEPDeflationComputeFunction(extop,lambda,NULL));
          CHKERRQ(MatShellGetContext(extop->MF,&matctxC));
          CHKERRQ(BVMult(matctx->U,-1.0,1.0,matctxC->U,F));
        }
        CHKERRQ(MatDestroy(&F));
      }
      CHKERRQ(PetscCalloc3(n,&basisv,szd*szd,&hH,szd*szd,&hHprev));
      CHKERRQ(NEPDeflationEvaluateBasis(extop,lambda,n,basisv,jacobian));
      A = matctx->A;
      CHKERRQ(PetscArrayzero(A,szd*szd));
      if (!jacobian) for (i=0;i<n;i++) A[i*(szd+1)] = 1.0;
      for (j=0;j<n;j++)
        for (i=0;i<n;i++)
          for (k=1;k<extop->midx;k++) A[j*szd+i] += basisv[k]*PetscConj(Hj[k*szd*szd+i*szd+j]);
      CHKERRQ(PetscBLASIntCast(n,&n_));
      CHKERRQ(PetscBLASIntCast(szd,&szd_));
      B = matctx->B;
      CHKERRQ(PetscArrayzero(B,szd*szd));
      for (i=1;i<extop->midx;i++) {
        CHKERRQ(NEPDeflationEvaluateBasisMat(extop,i,PETSC_TRUE,basisv,hH,hHprev));
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,extop->XpX,&szd_,hH,&szd_,&zero,hHprev,&szd_));
        PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n_,&n_,&n_,&sone,extop->Hj+szd*szd*i,&szd_,hHprev,&szd_,&sone,B,&szd_));
        pts = hHprev; hHprev = hH; hH = pts;
      }
      CHKERRQ(PetscFree3(basisv,hH,hHprev));
    }
    matctx->theta = lambda;
    matctx->n = extop->n;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationComputeFunction(NEP_EXT_OP extop,PetscScalar lambda,Mat *F)
{
  PetscFunctionBegin;
  CHKERRQ(NEPDeflationComputeShellMat(extop,lambda,PETSC_FALSE,NULL));
  if (F) *F = extop->MF;
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationComputeJacobian(NEP_EXT_OP extop,PetscScalar lambda,Mat *J)
{
  PetscFunctionBegin;
  CHKERRQ(NEPDeflationComputeShellMat(extop,lambda,PETSC_TRUE,NULL));
  if (J) *J = extop->MJ;
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationSolveSetUp(NEP_EXT_OP extop,PetscScalar lambda)
{
  NEP_DEF_MATSHELL  *matctx;
  NEP_DEF_FUN_SOLVE solve;
  PetscInt          i,j,n=extop->n;
  Vec               u,tu;
  Mat               F;
  PetscScalar       snone=-1.0,sone=1.0;
  PetscBLASInt      n_,szd_,ldh_,*p,info;
  Mat               Mshell;

  PetscFunctionBegin;
  solve = extop->solve;
  if (lambda!=solve->theta || n!=solve->n) {
    CHKERRQ(NEPDeflationComputeShellMat(extop,lambda,PETSC_FALSE,solve->sincf?NULL:&solve->T));
    Mshell = (solve->sincf)?extop->MF:solve->T;
    CHKERRQ(MatShellGetContext(Mshell,&matctx));
    CHKERRQ(NEP_KSPSetOperators(solve->ksp,matctx->T,matctx->P));
    if (!extop->ref && n) {
      CHKERRQ(PetscBLASIntCast(n,&n_));
      CHKERRQ(PetscBLASIntCast(extop->szd,&szd_));
      CHKERRQ(PetscBLASIntCast(extop->szd+1,&ldh_));
      if (!extop->simpU) {
        CHKERRQ(BVSetActiveColumns(solve->T_1U,0,n));
        for (i=0;i<n;i++) {
          CHKERRQ(BVGetColumn(matctx->U,i,&u));
          CHKERRQ(BVGetColumn(solve->T_1U,i,&tu));
          CHKERRQ(KSPSolve(solve->ksp,u,tu));
          CHKERRQ(BVRestoreColumn(solve->T_1U,i,&tu));
          CHKERRQ(BVRestoreColumn(matctx->U,i,&u));
        }
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,solve->work,&F));
        CHKERRQ(BVDot(solve->T_1U,extop->X,F));
        CHKERRQ(MatDestroy(&F));
      } else {
        for (j=0;j<n;j++)
          for (i=0;i<n;i++) solve->work[j*n+i] = extop->XpX[j*extop->szd+i];
        for (i=0;i<n;i++) extop->H[i*ldh_+i] -= lambda;
        PetscStackCallBLAS("BLAStrsm",BLAStrsm_("R","U","N","N",&n_,&n_,&snone,extop->H,&ldh_,solve->work,&n_));
        for (i=0;i<n;i++) extop->H[i*ldh_+i] += lambda;
      }
      CHKERRQ(PetscArraycpy(solve->M,matctx->B,extop->szd*extop->szd));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&snone,matctx->A,&szd_,solve->work,&n_,&sone,solve->M,&szd_));
      CHKERRQ(PetscMalloc1(n,&p));
      CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n_,&n_,solve->M,&szd_,p,&info));
      SlepcCheckLapackInfo("getrf",info);
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n_,solve->M,&szd_,p,solve->work,&n_,&info));
      SlepcCheckLapackInfo("getri",info);
      CHKERRQ(PetscFPTrapPop());
      CHKERRQ(PetscFree(p));
    }
    solve->theta = lambda;
    solve->n = n;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationFunctionSolve(NEP_EXT_OP extop,Vec b,Vec x)
{
  Vec               b1,x1;
  PetscScalar       *xx,*bb,*x2,*b2,*w,*w2,snone=-1.0,sone=1.0,zero=0.0;
  NEP_DEF_MATSHELL  *matctx;
  NEP_DEF_FUN_SOLVE solve=extop->solve;
  PetscBLASInt      one=1,szd_,n_,ldh_;
  PetscInt          nloc,i;
  PetscMPIInt       np,count;

  PetscFunctionBegin;
  if (extop->ref) {
    CHKERRQ(VecZeroEntries(x));
  }
  if (extop->szd) {
    x1 = solve->w[0]; b1 = solve->w[1];
    CHKERRQ(VecGetArray(x,&xx));
    CHKERRQ(VecPlaceArray(x1,xx));
    CHKERRQ(VecGetArray(b,&bb));
    CHKERRQ(VecPlaceArray(b1,bb));
  } else {
    b1 = b; x1 = x;
  }
  CHKERRQ(KSPSolve(extop->solve->ksp,b1,x1));
  if (!extop->ref && extop->n && extop->szd) {
    CHKERRQ(PetscBLASIntCast(extop->szd,&szd_));
    CHKERRQ(PetscBLASIntCast(extop->n,&n_));
    CHKERRQ(PetscBLASIntCast(extop->szd+1,&ldh_));
    CHKERRQ(BVGetSizes(extop->nep->V,&nloc,NULL,NULL));
    CHKERRQ(PetscMalloc2(extop->n,&b2,extop->n,&x2));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)b),&np));
    for (i=0;i<extop->n;i++) b2[i] = bb[nloc+i]*PetscSqrtReal(np);
    w = solve->work; w2 = solve->work+extop->n;
    CHKERRQ(MatShellGetContext(solve->sincf?extop->MF:solve->T,&matctx));
    CHKERRQ(PetscArraycpy(w2,b2,extop->n));
    CHKERRQ(BVDotVec(extop->X,x1,w));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&snone,matctx->A,&szd_,w,&one,&sone,w2,&one));
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,solve->M,&szd_,w2,&one,&zero,x2,&one));
    if (extop->simpU) {
      for (i=0;i<extop->n;i++) extop->H[i+i*(extop->szd+1)] -= solve->theta;
      for (i=0;i<extop->n;i++) w[i] = x2[i];
      PetscStackCallBLAS("BLAStrsm",BLAStrsm_("L","U","N","N",&n_,&one,&snone,extop->H,&ldh_,w,&n_));
      for (i=0;i<extop->n;i++) extop->H[i+i*(extop->szd+1)] += solve->theta;
      CHKERRQ(BVMultVec(extop->X,-1.0,1.0,x1,w));
    } else {
      CHKERRQ(BVMultVec(solve->T_1U,-1.0,1.0,x1,x2));
    }
    for (i=0;i<extop->n;i++) xx[i+nloc] = x2[i]/PetscSqrtReal(np);
    CHKERRQ(PetscMPIIntCast(extop->n,&count));
    CHKERRMPI(MPI_Bcast(xx+nloc,count,MPIU_SCALAR,np-1,PetscObjectComm((PetscObject)b)));
  }
  if (extop->szd) {
    CHKERRQ(VecResetArray(x1));
    CHKERRQ(VecRestoreArray(x,&xx));
    CHKERRQ(VecResetArray(b1));
    CHKERRQ(VecRestoreArray(b,&bb));
    if (!extop->ref && extop->n) CHKERRQ(PetscFree2(b2,x2));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationSetRefine(NEP_EXT_OP extop,PetscBool ref)
{
  PetscFunctionBegin;
  extop->ref = ref;
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationReset(NEP_EXT_OP extop)
{
  PetscInt          j;
  NEP_DEF_FUN_SOLVE solve;

  PetscFunctionBegin;
  if (!extop) PetscFunctionReturn(0);
  CHKERRQ(PetscFree(extop->H));
  CHKERRQ(BVDestroy(&extop->X));
  if (extop->szd) {
    CHKERRQ(VecDestroy(&extop->w));
    CHKERRQ(PetscFree3(extop->Hj,extop->XpX,extop->bc));
    CHKERRQ(BVDestroy(&extop->W));
  }
  CHKERRQ(MatDestroy(&extop->MF));
  CHKERRQ(MatDestroy(&extop->MJ));
  if (extop->solve) {
    solve = extop->solve;
    if (extop->szd) {
      if (!extop->simpU) CHKERRQ(BVDestroy(&solve->T_1U));
      CHKERRQ(PetscFree2(solve->M,solve->work));
      CHKERRQ(VecDestroy(&solve->w[0]));
      CHKERRQ(VecDestroy(&solve->w[1]));
    }
    if (!solve->sincf) {
      CHKERRQ(MatDestroy(&solve->T));
    }
    CHKERRQ(PetscFree(extop->solve));
  }
  if (extop->proj) {
    if (extop->szd) {
      for (j=0;j<extop->nep->nt;j++) CHKERRQ(MatDestroy(&extop->proj->V1pApX[j]));
      CHKERRQ(MatDestroy(&extop->proj->XpV1));
      CHKERRQ(PetscFree3(extop->proj->V2,extop->proj->V1pApX,extop->proj->work));
      CHKERRQ(VecDestroy(&extop->proj->w));
      CHKERRQ(BVDestroy(&extop->proj->V1));
    }
    CHKERRQ(PetscFree(extop->proj));
  }
  CHKERRQ(PetscFree(extop));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationInitialize(NEP nep,BV X,KSP ksp,PetscBool sincfun,PetscInt sz,NEP_EXT_OP *extop)
{
  NEP_EXT_OP        op;
  NEP_DEF_FUN_SOLVE solve;
  PetscInt          szd;
  Vec               x;

  PetscFunctionBegin;
  CHKERRQ(NEPDeflationReset(*extop));
  CHKERRQ(PetscNew(&op));
  *extop  = op;
  op->nep = nep;
  op->n   = 0;
  op->szd = szd = sz-1;
  op->max_midx = PetscMin(MAX_MINIDX,szd);
  op->X = X;
  if (!X) CHKERRQ(BVDuplicateResize(nep->V,sz,&op->X));
  else CHKERRQ(PetscObjectReference((PetscObject)X));
  CHKERRQ(PetscCalloc1(sz*sz,&(op)->H));
  if (op->szd) {
    CHKERRQ(BVGetColumn(op->X,0,&x));
    CHKERRQ(VecDuplicate(x,&op->w));
    CHKERRQ(BVRestoreColumn(op->X,0,&x));
    op->simpU = PETSC_FALSE;
    if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
      /* undocumented option to use the simple expression for U = T*X*inv(lambda-H) */
      CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-nep_deflation_simpleu",&op->simpU,NULL));
    } else {
      op->simpU = PETSC_TRUE;
    }
    CHKERRQ(PetscCalloc3(szd*szd*op->max_midx,&(op)->Hj,szd*szd,&(op)->XpX,szd,&op->bc));
    CHKERRQ(BVDuplicateResize(op->X,op->szd,&op->W));
  }
  if (ksp) {
    CHKERRQ(PetscNew(&solve));
    op->solve    = solve;
    solve->ksp   = ksp;
    solve->sincf = sincfun;
    solve->n     = -1;
    if (op->szd) {
      if (!op->simpU) {
        CHKERRQ(BVDuplicateResize(nep->V,szd,&solve->T_1U));
      }
      CHKERRQ(PetscMalloc2(szd*szd,&solve->M,2*szd*szd,&solve->work));
      CHKERRQ(BVCreateVec(nep->V,&solve->w[0]));
      CHKERRQ(VecDuplicate(solve->w[0],&solve->w[1]));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationDSNEPComputeMatrix(DS ds,PetscScalar lambda,PetscBool deriv,DSMatType mat,void *ctx)
{
  PetscScalar       *T,*Ei,*w1,*w2,*w=NULL,*ww,*hH,*hHprev,*pts;
  PetscScalar       alpha,alpha2,*AB,sone=1.0,zero=0.0,*basisv,s;
  const PetscScalar *E;
  PetscInt          i,ldds,nwork=0,szd,nv,j,k,n;
  PetscBLASInt      inc=1,nv_,ldds_,dim_,dim2,szdk,szd_,n_,ldh_;
  PetscMPIInt       np;
  NEP_DEF_PROJECT   proj=(NEP_DEF_PROJECT)ctx;
  NEP_EXT_OP        extop=proj->extop;
  NEP               nep=extop->nep;

  PetscFunctionBegin;
  CHKERRQ(DSGetDimensions(ds,&nv,NULL,NULL,NULL));
  CHKERRQ(DSGetLeadingDimension(ds,&ldds));
  CHKERRQ(DSGetArray(ds,mat,&T));
  CHKERRQ(PetscArrayzero(T,ldds*nv));
  CHKERRQ(PetscBLASIntCast(ldds*nv,&dim2));
  /* mat = V1^*T(lambda)V1 */
  for (i=0;i<nep->nt;i++) {
    if (deriv) {
      CHKERRQ(FNEvaluateDerivative(nep->f[i],lambda,&alpha));
    } else {
      CHKERRQ(FNEvaluateFunction(nep->f[i],lambda,&alpha));
    }
    CHKERRQ(DSGetArray(ds,DSMatExtra[i],&Ei));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&dim2,&alpha,Ei,&inc,T,&inc));
    CHKERRQ(DSRestoreArray(ds,DSMatExtra[i],&Ei));
  }
  if (!extop->ref && extop->n) {
    n = extop->n;
    szd = extop->szd;
    CHKERRQ(PetscArrayzero(proj->work,proj->lwork));
    CHKERRQ(PetscBLASIntCast(nv,&nv_));
    CHKERRQ(PetscBLASIntCast(n,&n_));
    CHKERRQ(PetscBLASIntCast(ldds,&ldds_));
    CHKERRQ(PetscBLASIntCast(szd,&szd_));
    CHKERRQ(PetscBLASIntCast(proj->dim,&dim_));
    CHKERRQ(PetscBLASIntCast(extop->szd+1,&ldh_));
    w1 = proj->work; w2 = proj->work+proj->dim*proj->dim;
    nwork += 2*proj->dim*proj->dim;

    /* mat = mat + V1^*U(lambda)V2 */
    for (i=0;i<nep->nt;i++) {
      if (extop->simpU) {
        if (deriv) {
          CHKERRQ(FNEvaluateDerivative(nep->f[i],lambda,&alpha));
        } else {
          CHKERRQ(FNEvaluateFunction(nep->f[i],lambda,&alpha));
        }
        ww = w1; w = w2;
        CHKERRQ(PetscArraycpy(ww,proj->V2,szd*nv));
        CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ds),&np));
        for (j=0;j<szd*nv;j++) ww[j] *= PetscSqrtReal(np);
        for (j=0;j<n;j++) extop->H[j*ldh_+j] -= lambda;
        alpha = -alpha;
        PetscStackCallBLAS("BLAStrsm",BLAStrsm_("L","U","N","N",&n_,&nv_,&alpha,extop->H,&ldh_,ww,&szd_));
        if (deriv) {
          CHKERRQ(PetscBLASIntCast(szd*nv,&szdk));
          CHKERRQ(FNEvaluateFunction(nep->f[i],lambda,&alpha2));
          CHKERRQ(PetscArraycpy(w,proj->V2,szd*nv));
          for (j=0;j<szd*nv;j++) w[j] *= PetscSqrtReal(np);
          alpha2 = -alpha2;
          PetscStackCallBLAS("BLAStrsm",BLAStrsm_("L","U","N","N",&n_,&nv_,&alpha2,extop->H,&ldh_,w,&szd_));
          alpha2 = 1.0;
          PetscStackCallBLAS("BLAStrsm",BLAStrsm_("L","U","N","N",&n_,&nv_,&alpha2,extop->H,&ldh_,w,&szd_));
          PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&szdk,&sone,w,&inc,ww,&inc));
        }
        for (j=0;j<n;j++) extop->H[j*ldh_+j] += lambda;
      } else {
        CHKERRQ(NEPDeflationEvaluateHatFunction(extop,i,lambda,NULL,w1,w2,szd));
        w = deriv?w2:w1; ww = deriv?w1:w2;
        CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ds),&np));
        s = PetscSqrtReal(np);
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&nv_,&n_,&s,w,&szd_,proj->V2,&szd_,&zero,ww,&szd_));
      }
      CHKERRQ(MatDenseGetArrayRead(proj->V1pApX[i],&E));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&nv_,&nv_,&n_,&sone,E,&dim_,ww,&szd_,&sone,T,&ldds_));
      CHKERRQ(MatDenseRestoreArrayRead(proj->V1pApX[i],&E));
    }

    /* mat = mat + V2^*A(lambda)V1 */
    basisv = proj->work+nwork; nwork += szd;
    hH     = proj->work+nwork; nwork += szd*szd;
    hHprev = proj->work+nwork; nwork += szd*szd;
    AB     = proj->work+nwork;
    CHKERRQ(NEPDeflationEvaluateBasis(extop,lambda,n,basisv,deriv));
    if (!deriv) for (i=0;i<n;i++) AB[i*(szd+1)] = 1.0;
    for (j=0;j<n;j++)
      for (i=0;i<n;i++)
        for (k=1;k<extop->midx;k++) AB[j*szd+i] += basisv[k]*PetscConj(extop->Hj[k*szd*szd+i*szd+j]);
    CHKERRQ(MatDenseGetArrayRead(proj->XpV1,&E));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&nv_,&n_,&sone,AB,&szd_,E,&szd_,&zero,w,&szd_));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&nv_,&nv_,&n_,&sone,proj->V2,&szd_,w,&szd_,&sone,T,&ldds_));
    CHKERRQ(MatDenseRestoreArrayRead(proj->XpV1,&E));

    /* mat = mat + V2^*B(lambda)V2 */
    CHKERRQ(PetscArrayzero(AB,szd*szd));
    for (i=1;i<extop->midx;i++) {
      CHKERRQ(NEPDeflationEvaluateBasisMat(extop,i,PETSC_TRUE,basisv,hH,hHprev));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,extop->XpX,&szd_,hH,&szd_,&zero,hHprev,&szd_));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n_,&n_,&n_,&sone,extop->Hj+szd*szd*i,&szd_,hHprev,&szd_,&sone,AB,&szd_));
      pts = hHprev; hHprev = hH; hH = pts;
    }
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&nv_,&n_,&sone,AB,&szd_,proj->V2,&szd_,&zero,w,&szd_));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&nv_,&nv_,&n_,&sone,proj->V2,&szd_,w,&szd_,&sone,T,&ldds_));
  }
  CHKERRQ(DSRestoreArray(ds,mat,&T));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDeflationProjectOperator(NEP_EXT_OP extop,BV Vext,DS ds,PetscInt j0,PetscInt j1)
{
  PetscInt        k,j,n=extop->n,dim;
  Vec             v,ve;
  BV              V1;
  Mat             G;
  NEP             nep=extop->nep;
  NEP_DEF_PROJECT proj;

  PetscFunctionBegin;
  NEPCheckSplit(extop->nep,1);
  proj = extop->proj;
  if (!proj) {
    /* Initialize the projection data structure */
    CHKERRQ(PetscNew(&proj));
    extop->proj = proj;
    proj->extop = extop;
    CHKERRQ(BVGetSizes(Vext,NULL,NULL,&dim));
    proj->dim = dim;
    if (extop->szd) {
      proj->lwork = 3*dim*dim+2*extop->szd*extop->szd+extop->szd;
      CHKERRQ(PetscMalloc3(dim*extop->szd,&proj->V2,nep->nt,&proj->V1pApX,proj->lwork,&proj->work));
      for (j=0;j<nep->nt;j++) {
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,proj->dim,extop->szd,NULL,&proj->V1pApX[j]));
      }
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,extop->szd,proj->dim,NULL,&proj->XpV1));
      CHKERRQ(BVCreateVec(extop->X,&proj->w));
      CHKERRQ(BVDuplicateResize(extop->X,proj->dim,&proj->V1));
    }
    CHKERRQ(DSNEPSetComputeMatrixFunction(ds,NEPDeflationDSNEPComputeMatrix,(void*)proj));
  }

  /* Split Vext in V1 and V2 */
  if (extop->szd) {
    for (j=j0;j<j1;j++) {
      CHKERRQ(BVGetColumn(Vext,j,&ve));
      CHKERRQ(BVGetColumn(proj->V1,j,&v));
      CHKERRQ(NEPDeflationCopyToExtendedVec(extop,v,proj->V2+j*extop->szd,ve,PETSC_TRUE));
      CHKERRQ(BVRestoreColumn(proj->V1,j,&v));
      CHKERRQ(BVRestoreColumn(Vext,j,&ve));
    }
    V1 = proj->V1;
  } else V1 = Vext;

  /* Compute matrices V1^* A_i V1 */
  CHKERRQ(BVSetActiveColumns(V1,j0,j1));
  for (k=0;k<nep->nt;k++) {
    CHKERRQ(DSGetMat(ds,DSMatExtra[k],&G));
    CHKERRQ(BVMatProject(V1,nep->A[k],V1,G));
    CHKERRQ(DSRestoreMat(ds,DSMatExtra[k],&G));
  }

  if (extop->n) {
    if (extop->szd) {
      /* Compute matrices V1^* A_i X  and V1^* X */
      CHKERRQ(BVSetActiveColumns(extop->W,0,n));
      for (k=0;k<nep->nt;k++) {
        CHKERRQ(BVMatMult(extop->X,nep->A[k],extop->W));
        CHKERRQ(BVDot(extop->W,V1,proj->V1pApX[k]));
      }
      CHKERRQ(BVDot(V1,extop->X,proj->XpV1));
    }
  }
  PetscFunctionReturn(0);
}
