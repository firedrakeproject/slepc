/*

   SLEPc eigensolver: "ciss"

   Method: Contour Integral Spectral Slicing

   Algorithm:

       Contour integral based on Sakurai-Sugiura method to construct a
       subspace, with various eigenpair extractions (Rayleigh-Ritz,
       explicit moment).

   Based on code contributed by Y. Maeda, T. Sakurai.

   References:

       [1] T. Sakurai and H. Sugiura, "A projection method for generalized
           eigenvalue problems", J. Comput. Appl. Math. 159:119-128, 2003.

       [2] T. Sakurai and H. Tadano, "CIRR: a Rayleigh-Ritz type method with
           contour integral for generalized eigenvalue problems", Hokkaido
           Math. J. 36:745-757, 2007.

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

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  /* parameters */
  PetscInt     N;          /* number of integration points (32) */
  PetscInt     L;          /* block size (16) */
  PetscInt     M;          /* moment degree (N/4 = 4) */
  PetscReal    delta;      /* threshold of singular value (1e-12) */
  PetscInt     L_max;      /* maximum number of columns of the source matrix V */
  PetscReal    spurious_threshold; /* discard spurious eigenpairs */
  PetscBool    isreal;     /* T(z) is real for real z */
  PetscInt     refine_inner;
  PetscInt     refine_blocksize;
  /* private data */
  PetscReal    *sigma;     /* threshold for numerical rank */
  PetscInt     num_subcomm;
  PetscInt     subcomm_id;
  PetscInt     num_solve_point;
  PetscScalar  *weight;
  PetscScalar  *omega;
  PetscScalar  *pp;
  BV           V;
  BV           S;
  BV           Y;
  KSP          *ksp;
  Mat          *kspMat;
  PetscBool    useconj;
  PetscReal    est_eig;
  PetscSubcomm subcomm;
  PetscBool    usest;
} NEP_CISS;

#undef __FUNCT__
#define __FUNCT__ "SetSolverComm"
static PetscErrorCode SetSolverComm(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       N = ctx->N;

  PetscFunctionBegin;
  if (ctx->useconj) N = N/2;
  if (!ctx->subcomm) {
    ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)nep),&ctx->subcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(ctx->subcomm,ctx->num_subcomm);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = PetscSubcommSetType(ctx->subcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)nep,sizeof(PetscSubcomm));CHKERRQ(ierr);
    ierr = PetscSubcommSetFromOptions(ctx->subcomm);CHKERRQ(ierr);
  }
  ctx->subcomm_id = ctx->subcomm->color;
  ctx->num_solve_point = N / ctx->num_subcomm;
  if ((N%ctx->num_subcomm) > ctx->subcomm_id) ctx->num_solve_point+=1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetPathParameter"
static PetscErrorCode SetPathParameter(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i;
  PetscScalar    center;
  PetscReal      theta,radius,vscale,rgscale;
  PetscBool      isellipse=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)nep->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  ierr = RGGetScale(nep->rg,&rgscale);CHKERRQ(ierr);
  if (isellipse) {
    ierr = RGEllipseGetParameters(nep->rg,&center,&radius,&vscale);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Region must be Ellipse");
  for (i=0;i<ctx->N;i++) {
    theta = ((2*PETSC_PI)/ctx->N)*(i+0.5);
    ctx->pp[i] = PetscCosReal(theta) + PETSC_i*vscale*PetscSinReal(theta);
    ctx->weight[i] = radius*(vscale*PetscCosReal(theta) + PETSC_i*PetscSinReal(theta))/(PetscReal)ctx->N;
    ctx->omega[i] = rgscale*(center + radius*ctx->pp[i]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CISSVecSetRandom"
static PetscErrorCode CISSVecSetRandom(BV V,PetscInt i0,PetscInt i1)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nlocal;
  PetscScalar    *vdata;
  Vec            x;

  PetscFunctionBegin;
  ierr = BVGetSizes(V,&nlocal,NULL,NULL);CHKERRQ(ierr);
  for (i=i0;i<i1;i++) {
    ierr = BVSetRandomColumn(V,i);CHKERRQ(ierr);
    ierr = BVGetColumn(V,i,&x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&vdata);CHKERRQ(ierr);
    for (j=0;j<nlocal;j++) {
      vdata[j] = PetscRealPart(vdata[j]);
      if (PetscRealPart(vdata[j]) < 0.5) vdata[j] = -1.0;
      else vdata[j] = 1.0;
    }
    ierr = VecRestoreArray(x,&vdata);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SolveLinearSystem"
static PetscErrorCode SolveLinearSystem(NEP nep,Mat T,Mat dT,BV V,PetscInt L_start,PetscInt L_end,PetscBool initksp)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,p_id;
  Mat            Fz;
  PC             pc;
  Vec            Bvj,vj,yj;
  KSP            ksp;

  PetscFunctionBegin;
  if (ctx->usest) {
    ierr = NEPComputeFunction(nep,0,T,T);CHKERRQ(ierr);
    ierr = MatDuplicate(T,MAT_DO_NOT_COPY_VALUES,&Fz);CHKERRQ(ierr);
    ierr = KSPCreate(PetscObjectComm((PetscObject)nep),&ksp);CHKERRQ(ierr);
  }
  ierr = BVCreateVec(V,&Bvj);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) {
    p_id = i*ctx->subcomm->n + ctx->subcomm_id;
    ierr = NEPComputeFunction(nep,ctx->omega[p_id],T,T);CHKERRQ(ierr);
    ierr = NEPComputeJacobian(nep,ctx->omega[p_id],dT);CHKERRQ(ierr);
    if (!ctx->usest && initksp == PETSC_TRUE) {
      ierr = MatDuplicate(T,MAT_COPY_VALUES,&ctx->kspMat[i]);CHKERRQ(ierr);
      ierr = KSPSetOperators(ctx->ksp[i],ctx->kspMat[i],ctx->kspMat[i]);CHKERRQ(ierr);
      ierr = KSPSetType(ctx->ksp[i],KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ctx->ksp[i],&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ctx->ksp[i]);CHKERRQ(ierr);
    } else if (ctx->usest) {
      ierr = MatCopy(T,Fz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,Fz,Fz);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    }
    for (j=L_start;j<L_end;j++) {
      ierr = BVGetColumn(V,j,&vj);CHKERRQ(ierr);
      ierr = BVGetColumn(ctx->Y,i*ctx->L_max+j,&yj);CHKERRQ(ierr);
      ierr = MatMult(dT,vj,Bvj);CHKERRQ(ierr);
      if (ctx->usest) {
        ierr = KSPSolve(ksp,Bvj,yj);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(ctx->ksp[i],Bvj,yj);CHKERRQ(ierr);
      }
      ierr = BVRestoreColumn(V,j,&vj);CHKERRQ(ierr);
      ierr = BVRestoreColumn(ctx->Y,i*ctx->L_max+j,&yj);CHKERRQ(ierr);
    }
    if (ctx->usest && i<ctx->num_solve_point-1) { ierr =  KSPReset(ksp);CHKERRQ(ierr); }
  }
  if (ctx->usest) {
    ierr = MatDestroy(&Fz);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&Bvj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EstimateNumberEigs"
static PetscErrorCode EstimateNumberEigs(NEP nep,PetscInt *L_add)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,p_id;
  PetscScalar    tmp,m = 1,sum = 0.0;
  PetscReal      eta;
  Vec            v,vtemp,vj,yj;

  PetscFunctionBegin;
  ierr = BVGetColumn(ctx->Y,0,&yj);CHKERRQ(ierr);
  ierr = VecDuplicate(yj,&v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(ctx->Y,0,&yj);CHKERRQ(ierr);
  ierr = BVCreateVec(ctx->V,&vtemp);CHKERRQ(ierr);
  for (j=0;j<ctx->L;j++) {
    ierr = VecSet(v,0);CHKERRQ(ierr);
    for (i=0;i<ctx->num_solve_point; i++) {
      p_id = i*ctx->subcomm->n + ctx->subcomm_id;
      ierr = BVSetActiveColumns(ctx->Y,i*ctx->L_max+j,i*ctx->L_max+j+1);CHKERRQ(ierr);
      ierr = BVMultVec(ctx->Y,ctx->weight[p_id],1,v,&m);CHKERRQ(ierr);
    }
    ierr = BVGetColumn(ctx->V,j,&vj);CHKERRQ(ierr);
    ierr = VecDot(vj,v,&tmp);CHKERRQ(ierr);
    ierr = BVRestoreColumn(ctx->V,j,&vj);CHKERRQ(ierr);
    if (ctx->useconj) sum += PetscRealPart(tmp)*2;
    else sum += tmp;
  }
  ctx->est_eig = PetscAbsScalar(sum/(PetscReal)ctx->L);
  eta = PetscPowReal(10,-PetscLog10Real(nep->tol)/ctx->N);
  ierr = PetscInfo1(nep,"Estimation_#Eig %f\n",(double)ctx->est_eig);CHKERRQ(ierr);
  *L_add = (PetscInt)PetscCeilReal((ctx->est_eig*eta)/ctx->M) - ctx->L;
  if (*L_add < 0) *L_add = 0;
  if (*L_add>ctx->L_max-ctx->L) {
    ierr = PetscInfo(nep,"Number of eigenvalues around the contour path may be too large\n");CHKERRQ(ierr);
    *L_add = ctx->L_max-ctx->L;
  }
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&vtemp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CalcMu"
static PetscErrorCode CalcMu(NEP nep, PetscScalar *Mu)
{
  PetscErrorCode ierr;
  PetscMPIInt    sub_size,len;
  PetscInt       i,j,k,s;
  PetscScalar    *m,*temp,*temp2,*ppk,alp;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  Mat            M;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscSubcommChild(ctx->subcomm),&sub_size);CHKERRQ(ierr);
  ierr = PetscMalloc3(ctx->num_solve_point*ctx->L*(ctx->L+1),&temp,2*ctx->M*ctx->L*ctx->L,&temp2,ctx->num_solve_point,&ppk);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,ctx->L,ctx->L_max*ctx->num_solve_point,NULL,&M);CHKERRQ(ierr);
  for (i=0;i<2*ctx->M*ctx->L*ctx->L;i++) temp2[i] = 0;
  ierr = BVSetActiveColumns(ctx->Y,0,ctx->L_max*ctx->num_solve_point);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
  ierr = BVDot(ctx->Y,ctx->V,M);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&m);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) {
    for (j=0;j<ctx->L;j++) {
      for (k=0;k<ctx->L;k++) {
        temp[k+j*ctx->L+i*ctx->L*ctx->L]=m[k+j*ctx->L+i*ctx->L*ctx->L_max];
      }
    }
  }
  ierr = MatDenseRestoreArray(M,&m);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) ppk[i] = 1;
  for (k=0;k<2*ctx->M;k++) {
    for (j=0;j<ctx->L;j++) {
      for (i=0;i<ctx->num_solve_point;i++) {
        alp = ppk[i]*ctx->weight[i*ctx->subcomm->n + ctx->subcomm_id];
        for (s=0;s<ctx->L;s++) {
          if (ctx->useconj) temp2[s+(j+k*ctx->L)*ctx->L] += PetscRealPart(alp*temp[s+(j+i*ctx->L)*ctx->L])*2;
          else temp2[s+(j+k*ctx->L)*ctx->L] += alp*temp[s+(j+i*ctx->L)*ctx->L];
        }
      }
    }
    for (i=0;i<ctx->num_solve_point;i++) 
      ppk[i] *= ctx->pp[i*ctx->subcomm->n + ctx->subcomm_id];
  }
  for (i=0;i<2*ctx->M*ctx->L*ctx->L;i++) temp2[i] /= sub_size;
  ierr = PetscMPIIntCast(2*ctx->M*ctx->L*ctx->L,&len);CHKERRQ(ierr);
  ierr = MPI_Allreduce(temp2,Mu,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)nep));CHKERRQ(ierr);
  ierr = PetscFree3(temp,temp2,ppk);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BlockHankel"
static PetscErrorCode BlockHankel(NEP nep,PetscScalar *Mu,PetscInt s,PetscScalar *H)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;
  PetscInt  i,j,k,L=ctx->L,M=ctx->M;

  PetscFunctionBegin;
  for (k=0;k<L*M;k++)
    for (j=0;j<M;j++) 
      for (i=0;i<L;i++) 
        H[j*L+i+k*L*M] = Mu[i+k*L+(j+s)*L*L];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVD_H0"
static PetscErrorCode SVD_H0(NEP nep,PetscScalar *S,PetscInt *K)
{
#if defined(PETSC_MISSING_LAPACK_GESVD)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESVD - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,ml=ctx->L*ctx->M;
  PetscBLASInt   m,n,lda,ldu,ldvt,lwork,info;
  PetscScalar    *work;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif

  PetscFunctionBegin;
  ierr = PetscMalloc1(5*ml,&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(5*ml,&rwork);CHKERRQ(ierr);
#endif
  ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
  n = m; lda = m; ldu = m; ldvt = m; lwork = 5*m;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&m,&n,S,&lda,ctx->sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&m,&n,S,&lda,ctx->sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,&info));
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  (*K) = 0;
  for (i=0;i<ml;i++) {
    if (ctx->sigma[i]/PetscMax(ctx->sigma[0],1)>ctx->delta) (*K)++;
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "ConstructS"
static PetscErrorCode ConstructS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,k,vec_local_size,p_id;
  Vec            v,sj,yj;
  PetscScalar    *ppk, *v_data, m = 1;

  PetscFunctionBegin;
  ierr = BVGetSizes(ctx->Y,&vec_local_size,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->num_solve_point,&ppk);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) ppk[i] = 1;
  ierr = BVGetColumn(ctx->Y,0,&yj);CHKERRQ(ierr);
  ierr = VecDuplicate(yj,&v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(ctx->Y,0,&yj);CHKERRQ(ierr);
  for (k=0;k<ctx->M;k++) {
    for (j=0;j<ctx->L;j++) {
      ierr = VecSet(v,0);CHKERRQ(ierr);
      for (i=0;i<ctx->num_solve_point;i++) {
        p_id = i*ctx->subcomm->n + ctx->subcomm_id;
        ierr = BVSetActiveColumns(ctx->Y,i*ctx->L_max+j,i*ctx->L_max+j+1);CHKERRQ(ierr);
        ierr = BVMultVec(ctx->Y,ppk[i]*ctx->weight[p_id],1,v,&m);CHKERRQ(ierr);
      }
      if (ctx->useconj) {
        ierr = VecGetArray(v,&v_data);CHKERRQ(ierr);
        for (i=0;i<vec_local_size;i++) v_data[i] = PetscRealPart(v_data[i])*2;
        ierr = VecRestoreArray(v,&v_data);CHKERRQ(ierr);
      }
      ierr = BVGetColumn(ctx->S,k*ctx->L+j,&sj);CHKERRQ(ierr);
      ierr = VecCopy(v,sj);CHKERRQ(ierr);
      ierr = BVRestoreColumn(ctx->S,k*ctx->L+j,&sj);CHKERRQ(ierr);
    }
    for (i=0;i<ctx->num_solve_point;i++) {
      p_id = i*ctx->subcomm->n + ctx->subcomm_id;
      ppk[i] *= ctx->pp[p_id];
    }
  }
  ierr = PetscFree(ppk);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "isGhost"
static PetscErrorCode isGhost(NEP nep,PetscInt ld,PetscInt nv,PetscBool *fl)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j;
  PetscScalar    *pX;
  PetscReal      *tau,s1,s2,tau_max=0.0;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nv,&tau);CHKERRQ(ierr);
  ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetArray(nep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);

  for (i=0;i<nv;i++) {
    s1 = 0;
    s2 = 0;
    for (j=0;j<nv;j++) {
      s1 += PetscAbsScalar(PetscPowScalar(pX[i*ld+j],2));
      s2 += PetscPowReal(PetscAbsScalar(pX[i*ld+j]),2)/ctx->sigma[j];
    }
    tau[i] = s1/s2;
    tau_max = PetscMax(tau_max,tau[i]);
  }
  ierr = DSRestoreArray(nep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    tau[i] /= tau_max;
  }
  for (i=0;i<nv;i++) {
    if (tau[i]>=ctx->spurious_threshold) fl[i] = PETSC_TRUE;
    else fl[i] = PETSC_FALSE;
  }
  ierr = PetscFree(tau);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_CISS"
PetscErrorCode NEPSetUp_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,nwork;
  PetscBool      istrivial,isellipse,flg;
  PetscScalar    center;

  PetscFunctionBegin;
  if (!nep->ncv) nep->ncv = ctx->L_max*ctx->M;
  else {
    ctx->L_max = nep->ncv/ctx->M;
    if (ctx->L_max == 0) {
      ctx->L_max = 1;
      nep->ncv = ctx->L_max*ctx->M;
    }
    if (ctx->L > ctx->L_max) ctx->L = ctx->L_max;
  }
  if (!nep->max_it) nep->max_it = 1;
  if (!nep->mpd) nep->mpd = nep->ncv;
  if (!nep->which) nep->which = NEP_ALL;
  if (nep->stopping!=NEPStoppingBasic) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver does not support user-defined stopping test");

  /* check region */
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"CISS requires a nontrivial region, e.g. -rg_type ellipse ...");
  ierr = RGGetComplement(nep->rg,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"A region with complement flag set is not allowed");
  ierr = PetscObjectTypeCompare((PetscObject)nep->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (!isellipse) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Currently only implemented for elliptic or arc regions");
  ierr = RGEllipseGetParameters(nep->rg,&center,NULL,NULL);CHKERRQ(ierr);
  if (ctx->isreal && PetscImaginaryPart(center) == 0.0) ctx->useconj = PETSC_TRUE;
  else ctx->useconj = PETSC_FALSE;

  /* create split comm */
  ctx->num_subcomm = 1;
  ierr = SetSolverComm(nep);CHKERRQ(ierr);

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  ierr = PetscMalloc4(ctx->N,&ctx->weight,ctx->N,&ctx->omega,ctx->N,&ctx->pp,ctx->L_max*ctx->M,&ctx->sigma);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,3*ctx->N*sizeof(PetscScalar)+ctx->L_max*ctx->N*sizeof(PetscReal));CHKERRQ(ierr);

  /* allocate basis vectors */
  ierr = BVDuplicateResize(nep->V,ctx->L_max*ctx->M,&ctx->S);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->S);CHKERRQ(ierr);
  ierr = BVDuplicateResize(nep->V,ctx->L_max,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->V);CHKERRQ(ierr);

  if (!ctx->usest) {
    ierr = PetscMalloc2(ctx->num_solve_point,&ctx->ksp,ctx->num_solve_point,&ctx->kspMat);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)nep,ctx->num_solve_point*sizeof(KSP)+ctx->num_solve_point*sizeof(Mat));CHKERRQ(ierr);
    for (i=0;i<ctx->num_solve_point;i++) {
      ierr = KSPCreate(PetscSubcommChild(ctx->subcomm),&ctx->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp[i],(PetscObject)nep,1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ctx->ksp[i],((PetscObject)nep)->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ctx->ksp[i],"nep_ciss_");CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(ctx->ksp[i],PETSC_TRUE);CHKERRQ(ierr);
    }
  }

  ierr = BVDuplicateResize(nep->V,ctx->num_solve_point*ctx->L_max,&ctx->Y);CHKERRQ(ierr);

  ierr = DSSetType(nep->ds,DSGNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->ncv);CHKERRQ(ierr);
  nwork = (nep->fui==NEP_USER_INTERFACE_SPLIT)? 2: 1;
  ierr = NEPSetWorkVecs(nep,nwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_CISS"
PetscErrorCode NEPSolve_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  Mat            X,M;
  PetscInt       i,j,ld,L_add=0,nv=0,L_base=ctx->L,inner,*inside;
  PetscScalar    *Mu,*H0,*H1,*rr,*temp,center;
  PetscReal      error,max_error,radius,rgscale;
  PetscBool      *fl1;
  Vec            si;
  SlepcSC        sc;
  PetscRandom    rand;

  PetscFunctionBegin;
  ierr = DSGetSlepcSC(nep->ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSGetLeadingDimension(nep->ds,&ld);CHKERRQ(ierr);
  ierr = SetPathParameter(nep);CHKERRQ(ierr);
  ierr = CISSVecSetRandom(ctx->V,0,ctx->L);CHKERRQ(ierr);
  ierr = BVGetRandomContext(ctx->V,&rand);CHKERRQ(ierr);

  ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  ierr = EstimateNumberEigs(nep,&L_add);CHKERRQ(ierr);
  if (L_add>0) {
    ierr = PetscInfo2(nep,"Changing L %D -> %D by Estimate #Eig\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = CISSVecSetRandom(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    ctx->L += L_add;
  }
  ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
  for (i=0;i<ctx->refine_blocksize;i++) {
    ierr = CalcMu(nep,Mu);CHKERRQ(ierr);
    ierr = BlockHankel(nep,Mu,0,H0);CHKERRQ(ierr);
    ierr = SVD_H0(nep,H0,&nv);CHKERRQ(ierr);
    if (ctx->sigma[0]<=ctx->delta || nv < ctx->L*ctx->M || ctx->L == ctx->L_max) break;
    L_add = L_base;
    if (ctx->L+L_add>ctx->L_max) L_add = ctx->L_max-ctx->L;
    ierr = PetscInfo2(nep,"Changing L %D -> %D by SVD(H0)\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = CISSVecSetRandom(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    ctx->L += L_add;
  }
  ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);

  ierr = RGGetScale(nep->rg,&rgscale);CHKERRQ(ierr);
  ierr = RGEllipseGetParameters(nep->rg,&center,&radius,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc3(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0,ctx->L*ctx->M*ctx->L*ctx->M,&H1);CHKERRQ(ierr);
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;
    for (inner=0;inner<=ctx->refine_inner;inner++) {
      ierr = CalcMu(nep,Mu);CHKERRQ(ierr);
      ierr = BlockHankel(nep,Mu,0,H0);CHKERRQ(ierr);
      ierr = SVD_H0(nep,H0,&nv);CHKERRQ(ierr);
      if (ctx->sigma[0]>ctx->delta && nv==ctx->L*ctx->M && inner!=ctx->refine_inner) {
        ierr = ConstructS(nep);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
        ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
      } else break;
    }

    nep->nconv = 0;
    if (nv == 0) break;
    ierr = BlockHankel(nep,Mu,0,H0);CHKERRQ(ierr);
    ierr = BlockHankel(nep,Mu,1,H1);CHKERRQ(ierr);
    ierr = DSSetDimensions(nep->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSGetArray(nep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
    for (j=0;j<nv;j++)
      for (i=0;i<nv;i++)
        temp[i+j*ld] = H1[i+j*ctx->L*ctx->M];
    ierr = DSRestoreArray(nep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
    ierr = DSGetArray(nep->ds,DS_MAT_B,&temp);CHKERRQ(ierr);
    for (j=0;j<nv;j++) 
      for (i=0;i<nv;i++)
        temp[i+j*ld] = H0[i+j*ctx->L*ctx->M];
    ierr = DSRestoreArray(nep->ds,DS_MAT_B,&temp);CHKERRQ(ierr);
    ierr = DSSolve(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
    ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
    for (i=0;i<nv;i++){
      nep->eigr[i] = (nep->eigr[i]*radius+center)*rgscale;
#if !defined(PETSC_USE_COMPLEX)
      nep->eigi[i] = nep->eigi[i]*radius*rgscale;
#endif
    }
    ierr = PetscMalloc3(nv,&fl1,nv,&inside,nv,&rr);CHKERRQ(ierr);
    ierr = isGhost(nep,ld,nv,fl1);CHKERRQ(ierr);
    ierr = RGCheckInside(nep->rg,nv,nep->eigr,nep->eigi,inside);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      if (fl1[i] && inside[i]>0) {
        rr[i] = 1.0;
        nep->nconv++;
      } else rr[i] = 0.0;
    }
    ierr = DSSort(nep->ds,nep->eigr,nep->eigi,rr,NULL,&nep->nconv);CHKERRQ(ierr);
    for (i=0;i<nv;i++){
      nep->eigr[i] = (nep->eigr[i]*radius+center)*rgscale;
#if !defined(PETSC_USE_COMPLEX)
      nep->eigi[i] = nep->eigi[i]*radius*rgscale;
#endif
    }
    ierr = PetscFree3(fl1,inside,rr);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(nep->V,0,nv);CHKERRQ(ierr);
    ierr = ConstructS(nep);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
    ierr = BVCopy(ctx->S,nep->V);CHKERRQ(ierr);

    ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
    ierr = DSGetMat(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
    ierr = BVMultInPlace(ctx->S,X,0,nep->nconv);CHKERRQ(ierr);
    ierr = BVMultInPlace(nep->V,X,0,nep->nconv);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    max_error = 0.0;
    for (i=0;i<nep->nconv;i++) {
      ierr = BVGetColumn(nep->V,i,&si);CHKERRQ(ierr);
      ierr = VecNormalize(si,NULL);CHKERRQ(ierr);
      ierr = NEPComputeResidualNorm_Private(nep,nep->eigr[i],si,nep->work,&error);CHKERRQ(ierr);
      ierr = (*nep->converged)(nep,nep->eigr[i],0,error,&error,nep->convergedctx);CHKERRQ(ierr);
      ierr = BVRestoreColumn(nep->V,i,&si);CHKERRQ(ierr);
      max_error = PetscMax(max_error,error);
    }
    if (max_error <= nep->tol) nep->reason = NEP_CONVERGED_TOL;
    else if (nep->its > nep->max_it) nep->reason = NEP_DIVERGED_ITS;
    else {
      if (nep->nconv > ctx->L) nv = nep->nconv;
      else if (ctx->L > nv) nv = ctx->L;
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,nv,ctx->L,NULL,&M);CHKERRQ(ierr);
      ierr = MatDenseGetArray(M,&temp);CHKERRQ(ierr);
      for (i=0;i<ctx->L*nv;i++) {
        ierr = PetscRandomGetValue(rand,&temp[i]);CHKERRQ(ierr);
        temp[i] = PetscRealPart(temp[i]);
      }
      ierr = MatDenseRestoreArray(M,&temp);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
      ierr = BVMultInPlace(ctx->S,M,0,ctx->L);CHKERRQ(ierr);
      ierr = MatDestroy(&M);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
      ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
      ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3(Mu,H0,H1);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetSizes_CISS"
static PetscErrorCode NEPCISSSetSizes_CISS(NEP nep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) {
    if (ctx->N!=32) { ctx->N =32; ctx->M = ctx->N/4; }
  } else {
    if (ip<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be > 0");
    if (ip%2) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be an even number");
    if (ctx->N!=ip) { ctx->N = ip; ctx->M = ctx->N/4; }
  }
  if (bs == PETSC_DECIDE || bs == PETSC_DEFAULT) {
    ctx->L = 16;
  } else {
    if (bs<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be > 0");
    if (bs>ctx->L_max) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be less than or equal to the maximum number of block size");
    ctx->L = bs;
  }
  if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
    ctx->M = ctx->N/4;
  } else {
    if (ms<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
    if (ms>ctx->N) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
    ctx->M = ms;
  }
  if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
    ctx->num_subcomm = 1;
  } else {
    if (npart<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The npart argument must be > 0");
    ctx->num_subcomm = npart;
  }
  if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
    ctx->L = 256;
  } else {
    if (bsmax<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
    if (bsmax<ctx->L) ctx->L_max = ctx->L;
    else ctx->L_max = bsmax;
  }
  ctx->isreal = realmats;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetSizes"
/*@
   NEPCISSSetSizes - Sets the values of various size parameters in the CISS solver.

   Logically Collective on NEP

   Input Parameters:
+  nep   - the eigenproblem solver context
.  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  realmats - T(z) is real for real z

   Options Database Keys:
+  -nep_ciss_integration_points - Sets the number of integration points
.  -nep_ciss_blocksize - Sets the block size
.  -nep_ciss_moments - Sets the moment size
.  -nep_ciss_partitions - Sets the number of partitions
.  -nep_ciss_maxblocksize - Sets the maximum block size
-  -nep_ciss_realmats - T(z) is real for real z

   Note:
   The default number of partitions is 1. This means the internal KSP object is shared
   among all processes of the NEP communicator. Otherwise, the communicator is split
   into npart communicators, so that npart KSP solves proceed simultaneously.

   The realmats flag can be set to true when T(.) is guaranteed to be real
   when the argument is a real value, for example, when all matrices in
   the split form are real. When set to true, the solver avoids some computations.

   Level: advanced

.seealso: NEPCISSGetSizes()
@*/
PetscErrorCode NEPCISSSetSizes(NEP nep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,ip,2);
  PetscValidLogicalCollectiveInt(nep,bs,3);
  PetscValidLogicalCollectiveInt(nep,ms,4);
  PetscValidLogicalCollectiveInt(nep,npart,5);
  PetscValidLogicalCollectiveInt(nep,bsmax,6);
  PetscValidLogicalCollectiveBool(nep,realmats,7);
  ierr = PetscTryMethod(nep,"NEPCISSSetSizes_C",(NEP,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool),(nep,ip,bs,ms,npart,bsmax,realmats));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetSizes_CISS"
static PetscErrorCode NEPCISSGetSizes_CISS(NEP nep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (ip) *ip = ctx->N;
  if (bs) *bs = ctx->L;
  if (ms) *ms = ctx->M;
  if (npart) *npart = ctx->num_subcomm;
  if (bsmax) *bsmax = ctx->L_max;
  if (realmats) *realmats = ctx->isreal;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetSizes"
/*@
   NEPCISSGetSizes - Gets the values of various size parameters in the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the eigenproblem solver context

   Output Parameters:
+  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  realmats - T(z) is real for real z

   Level: advanced

.seealso: NEPCISSSetSizes()
@*/
PetscErrorCode NEPCISSGetSizes(NEP nep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPCISSGetSizes_C",(NEP,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*),(nep,ip,bs,ms,npart,bsmax,realmats));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetThreshold_CISS"
static PetscErrorCode NEPCISSSetThreshold_CISS(NEP nep,PetscReal delta,PetscReal spur)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (delta == PETSC_DEFAULT) {
    ctx->delta = 1e-12;
  } else {
    if (delta<=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The delta argument must be > 0.0");
    ctx->delta = delta;
  }
  if (spur == PETSC_DEFAULT) {
    ctx->spurious_threshold = 1e-4;
  } else {
    if (spur<=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The spurious threshold argument must be > 0.0");
    ctx->spurious_threshold = spur;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetThreshold"
/*@
   NEPCISSSetThreshold - Sets the values of various threshold parameters in
   the CISS solver.

   Logically Collective on NEP

   Input Parameters:
+  nep   - the eigenproblem solver context
.  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Options Database Keys:
+  -nep_ciss_delta - Sets the delta
-  -nep_ciss_spurious_threshold - Sets the spurious threshold

   Level: advanced

.seealso: NEPCISSGetThreshold()
@*/
PetscErrorCode NEPCISSSetThreshold(NEP nep,PetscReal delta,PetscReal spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(nep,delta,2);
  PetscValidLogicalCollectiveReal(nep,spur,3);
  ierr = PetscTryMethod(nep,"NEPCISSSetThreshold_C",(NEP,PetscReal,PetscReal),(nep,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetThreshold_CISS"
static PetscErrorCode NEPCISSGetThreshold_CISS(NEP nep,PetscReal *delta,PetscReal *spur)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (delta) *delta = ctx->delta;
  if (spur)  *spur = ctx->spurious_threshold;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetThreshold"
/*@
   NEPCISSGetThreshold - Gets the values of various threshold parameters in
   the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the eigenproblem solver context

   Output Parameters:
+  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Level: advanced

.seealso: NEPCISSSetThreshold()
@*/
PetscErrorCode NEPCISSGetThreshold(NEP nep,PetscReal *delta,PetscReal *spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPCISSGetThreshold_C",(NEP,PetscReal*,PetscReal*),(nep,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetRefinement_CISS"
static PetscErrorCode NEPCISSSetRefinement_CISS(NEP nep,PetscInt inner,PetscInt blsize)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (inner == PETSC_DEFAULT) {
    ctx->refine_inner = 0;
  } else {
    if (inner<0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The refine inner argument must be >= 0");
    ctx->refine_inner = inner;
  }
  if (blsize == PETSC_DEFAULT) {
    ctx->refine_blocksize = 0;
  } else {
    if (blsize<0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The refine blocksize argument must be >= 0");
    ctx->refine_blocksize = blsize;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetRefinement"
/*@
   NEPCISSSetRefinement - Sets the values of various refinement parameters
   in the CISS solver.

   Logically Collective on NEP

   Input Parameters:
+  nep    - the eigenproblem solver context
.  inner  - number of iterative refinement iterations (inner loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Options Database Keys:
+  -nep_ciss_refine_inner - Sets number of inner iterations
-  -nep_ciss_refine_blocksize - Sets number of blocksize iterations

   Level: advanced

.seealso: NEPCISSGetRefinement()
@*/
PetscErrorCode NEPCISSSetRefinement(NEP nep,PetscInt inner,PetscInt blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,inner,2);
  PetscValidLogicalCollectiveInt(nep,blsize,3);
  ierr = PetscTryMethod(nep,"NEPCISSSetRefinement_C",(NEP,PetscInt,PetscInt),(nep,inner,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetRefinement_CISS"
static PetscErrorCode NEPCISSGetRefinement_CISS(NEP nep,PetscInt *inner,PetscInt *blsize)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (inner)  *inner = ctx->refine_inner;
  if (blsize) *blsize = ctx->refine_blocksize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetRefinement"
/*@
   NEPCISSGetRefinement - Gets the values of various refinement parameters
   in the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the eigenproblem solver context

   Output Parameters:
+  inner  - number of iterative refinement iterations (inner loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Level: advanced

.seealso: NEPCISSSetRefinement()
@*/
PetscErrorCode NEPCISSGetRefinement(NEP nep, PetscInt *inner, PetscInt *blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPCISSGetRefinement_C",(NEP,PetscInt*,PetscInt*),(nep,inner,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPReset_CISS"
PetscErrorCode NEPReset_CISS(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       i;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = PetscSubcommDestroy(&ctx->subcomm);CHKERRQ(ierr);
  ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  if (!ctx->usest) {
    for (i=0;i<ctx->num_solve_point;i++) {
      ierr = KSPDestroy(&ctx->ksp[i]);CHKERRQ(ierr);
    }
    for (i=0;i<ctx->num_solve_point;i++) {
      ierr = MatDestroy(&ctx->kspMat[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(ctx->ksp,ctx->kspMat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions_CISS"
PetscErrorCode NEPSetFromOptions_CISS(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode ierr;
  PetscReal      r1,r2;
  PetscInt       i1,i2,i3,i4,i5,i6,i7;
  PetscBool      b1;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP CISS Options");CHKERRQ(ierr);

  ierr = NEPCISSGetSizes(nep,&i1,&i2,&i3,&i4,&i5,&b1);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_integration_points","CISS number of integration points","NEPCISSSetSizes",i1,&i1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_blocksize","CISS block size","NEPCISSSetSizes",i2,&i2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_moments","CISS moment size","NEPCISSSetSizes",i3,&i3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_partitions","CISS number of partitions","NEPCISSSetSizes",i4,&i4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_maxblocksize","CISS maximum block size","NEPCISSSetSizes",i5,&i5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-nep_ciss_realmats","CISS flag indicating that T(z) is real for real z","NEPCISSSetSizes",b1,&b1,NULL);CHKERRQ(ierr);
  ierr = NEPCISSSetSizes(nep,i1,i2,i3,i4,i5,b1);CHKERRQ(ierr);

  ierr = NEPCISSGetThreshold(nep,&r1,&r2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_ciss_delta","CISS threshold for numerical rank","NEPCISSSetThreshold",r1,&r1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_ciss_spurious_threshold","CISS threshold for the spurious eigenpairs","NEPCISSSetThreshold",r2,&r2,NULL);CHKERRQ(ierr);
  ierr = NEPCISSSetThreshold(nep,r1,r2);CHKERRQ(ierr);

  ierr = NEPCISSGetRefinement(nep,&i6,&i7);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_refine_inner","CISS number of inner iterative refinement iterations","NEPCISSSetRefinement",i6,&i6,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_refine_blocksize","CISS number of blocksize iterative refinement iterations","NEPCISSSetRefinement",i7,&i7,NULL);CHKERRQ(ierr);
  ierr = NEPCISSSetRefinement(nep,i6,i7);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_CISS"
PetscErrorCode NEPDestroy_CISS(NEP nep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetRefinement_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView_CISS"
PetscErrorCode NEPView_CISS(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: sizes { integration points: %D, block size: %D, moment size: %D, partitions: %D, maximum block size: %D }\n",ctx->N,ctx->L,ctx->M,ctx->num_subcomm,ctx->L_max);CHKERRQ(ierr);
    if (ctx->isreal) {
      ierr = PetscViewerASCIIPrintf(viewer,"  CISS: exploiting symmetry of integration points\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: threshold { delta: %g, spurious threshold: %g }\n",(double)ctx->delta,(double)ctx->spurious_threshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: iterative refinement  { inner: %D, blocksize: %D }\n",ctx->refine_inner, ctx->refine_blocksize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (!ctx->usest && ctx->ksp[0]) { ierr = KSPView(ctx->ksp[0],viewer);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCreate_CISS"
PETSC_EXTERN PetscErrorCode NEPCreate_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  nep->data = ctx;
  nep->ops->solve          = NEPSolve_CISS;
  nep->ops->setup          = NEPSetUp_CISS;
  nep->ops->setfromoptions = NEPSetFromOptions_CISS;
  nep->ops->reset          = NEPReset_CISS;
  nep->ops->destroy        = NEPDestroy_CISS;
  nep->ops->view           = NEPView_CISS;
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetSizes_C",NEPCISSSetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetSizes_C",NEPCISSGetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetThreshold_C",NEPCISSSetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetThreshold_C",NEPCISSGetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetRefinement_C",NEPCISSSetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetRefinement_C",NEPCISSGetRefinement_CISS);CHKERRQ(ierr);
  /* set default values of parameters */
  ctx->N       = 32;
  ctx->L       = 16;
  ctx->M       = ctx->N/4;
  ctx->delta   = 1e-12;
  ctx->L_max   = 64;
  ctx->spurious_threshold = 1e-4;
  ctx->usest   = PETSC_FALSE;
  ctx->isreal  = PETSC_FALSE;
  ctx->refine_inner = 0;
  ctx->refine_blocksize = 0;
  ctx->num_subcomm = 1;
  PetscFunctionReturn(0);
}

