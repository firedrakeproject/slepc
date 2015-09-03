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

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode EPSSolve_CISS(EPS);

typedef struct {
  /* parameters */
  PetscInt    N;          /* number of integration points (32) */
  PetscInt    L;          /* block size (16) */
  PetscInt    M;          /* moment degree (N/4 = 4) */
  PetscReal   delta;      /* threshold of singular value (1e-12) */
  PetscInt    L_max;      /* maximum number of columns of the source matrix V */
  PetscReal   spurious_threshold; /* discard spurious eigenpairs */
  PetscBool   isreal;     /* A and B are real */
  PetscInt    refine_inner;
  PetscInt    refine_outer;
  PetscInt    refine_blocksize;
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
  BV           pV;
  BV           Y;
  Vec          xsub;
  Vec          xdup;
  KSP          *ksp;
  Mat          *kspMat;
  PetscBool    useconj;
  PetscReal    est_eig;
  VecScatter   scatterin;
  Mat          pA,pB;
  PetscSubcomm subcomm;
  PetscBool    usest;
} EPS_CISS;

#undef __FUNCT__
#define __FUNCT__ "SetSolverComm"
static PetscErrorCode SetSolverComm(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       N = ctx->N;

  PetscFunctionBegin;
  if (ctx->useconj) N = N/2;
  if (!ctx->subcomm) {
    ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)eps),&ctx->subcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(ctx->subcomm,ctx->num_subcomm);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = PetscSubcommSetType(ctx->subcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)eps,sizeof(PetscSubcomm));CHKERRQ(ierr);
    ierr = PetscSubcommSetFromOptions(ctx->subcomm);CHKERRQ(ierr);
  }
  ctx->subcomm_id = ctx->subcomm->color;
  ctx->num_solve_point = N / ctx->num_subcomm;
  if ((N%ctx->num_subcomm) > ctx->subcomm_id) ctx->num_solve_point+=1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CISSRedundantMat"
static PetscErrorCode CISSRedundantMat(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  Mat            A,B;
  PetscInt       nmat;

  PetscFunctionBegin;
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  if (ctx->subcomm->n != 1) {
    ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
    ierr = MatCreateRedundantMatrix(A,ctx->subcomm->n,PetscSubcommChild(ctx->subcomm),MAT_INITIAL_MATRIX,&ctx->pA);CHKERRQ(ierr);
    if (nmat>1) {
      ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr);
      ierr = MatCreateRedundantMatrix(B,ctx->subcomm->n,PetscSubcommChild(ctx->subcomm),MAT_INITIAL_MATRIX,&ctx->pB);CHKERRQ(ierr); 
    } else ctx->pB = NULL;
  } else {
    ctx->pA = NULL;
    ctx->pB = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CISSScatterVec"
static PetscErrorCode CISSScatterVec(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  IS             is1,is2;
  Vec            v0;
  PetscInt       i,j,k,mstart,mend,mlocal;
  PetscInt       *idx1,*idx2,mloc_sub;

  PetscFunctionBegin;
  ierr = MatCreateVecs(ctx->pA,&ctx->xsub,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->pA,&mloc_sub,NULL);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscSubcommContiguousParent(ctx->subcomm),mloc_sub,PETSC_DECIDE,&ctx->xdup);CHKERRQ(ierr);
  if (!ctx->scatterin) {
    ierr = BVGetColumn(ctx->V,0,&v0);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(v0,&mstart,&mend);CHKERRQ(ierr);
    mlocal = mend - mstart;
    ierr = PetscMalloc2(ctx->subcomm->n*mlocal,&idx1,ctx->subcomm->n*mlocal,&idx2);CHKERRQ(ierr);
    j = 0;
    for (k=0;k<ctx->subcomm->n;k++) {
      for (i=mstart;i<mend;i++) {
        idx1[j]   = i;
        idx2[j++] = i + eps->n*k;
      }
    }
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)eps),ctx->subcomm->n*mlocal,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)eps),ctx->subcomm->n*mlocal,idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
    ierr = VecScatterCreate(v0,is1,ctx->xdup,is2,&ctx->scatterin);CHKERRQ(ierr);
    ierr = ISDestroy(&is1);CHKERRQ(ierr);
    ierr = ISDestroy(&is2);CHKERRQ(ierr);
    ierr = PetscFree2(idx1,idx2);CHKERRQ(ierr);
    ierr = BVRestoreColumn(ctx->V,0,&v0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetPathParameter"
static PetscErrorCode SetPathParameter(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i;
  PetscScalar    center;
  PetscReal      theta,radius,vscale,start_ang,end_ang,width;
  PetscBool      isring=PETSC_FALSE,isellipse=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (isellipse) {
    ierr = RGEllipseGetParameters(eps->rg,&center,&radius,&vscale);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGRING,&isring);CHKERRQ(ierr);
    if (!isring) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Region must be Ellipse or Ring");
    ierr = RGRingGetParameters(eps->rg,&center,&radius,&vscale,&start_ang,&end_ang,&width);CHKERRQ(ierr);
  }
  for (i=0;i<ctx->N;i++) {
    if (isring) {
      /* Ring region only supported for complex scalars */
#if defined(PETSC_USE_COMPLEX)
      theta = (PETSC_PI/ctx->N)*(i+0.5);
      ctx->pp[i] = PetscCosReal(theta);
      ctx->weight[i] = PetscCosReal((ctx->N-1)*theta)/ctx->N;
      theta = (start_ang*2+(end_ang-start_ang)*(PetscCosReal(theta)+1.0))*PETSC_PI;
      ctx->omega[i] = center + radius*(PetscCosReal(theta)+PETSC_i*vscale*PetscSinReal(theta));
#endif
    } else {
#if defined(PETSC_USE_COMPLEX)
      theta = ((2*PETSC_PI)/ctx->N)*(i+0.5);
      ctx->pp[i] = PetscCosReal(theta) + PETSC_i*vscale*PetscSinReal(theta);
      ctx->weight[i] = radius*(vscale*PetscCosReal(theta) + PETSC_i*PetscSinReal(theta))/(PetscReal)ctx->N;
      ctx->omega[i] = center + radius*ctx->pp[i];
#else
      theta = (PETSC_PI/ctx->N)*(i+0.5);
      ctx->pp[i] = PetscCosReal(theta);
      ctx->weight[i] = PetscCosReal((ctx->N-1)*theta)/ctx->N;
      ctx->omega[i] = center + radius*ctx->pp[i];
#endif
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CISSVecSetRandom"
static PetscErrorCode CISSVecSetRandom(BV V,PetscInt i0,PetscInt i1,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nlocal;
  PetscScalar    *vdata;
  Vec            x;
 
  PetscFunctionBegin;
  ierr = BVGetSizes(V,&nlocal,NULL,NULL);CHKERRQ(ierr);
  for (i=i0;i<i1;i++) {
    ierr = BVSetRandomColumn(V,i,rctx);CHKERRQ(ierr);
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
#define __FUNCT__ "VecScatterVecs"
static PetscErrorCode VecScatterVecs(EPS eps,BV Vin,PetscInt n)
{
  PetscErrorCode    ierr;
  EPS_CISS          *ctx = (EPS_CISS*)eps->data;
  PetscInt          i;
  Vec               vi,pvi;
  const PetscScalar *array;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    ierr = BVGetColumn(Vin,i,&vi);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->scatterin,vi,ctx->xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scatterin,vi,ctx->xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = BVRestoreColumn(Vin,i,&vi);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->xdup,&array);CHKERRQ(ierr);
    ierr = VecPlaceArray(ctx->xsub,array);CHKERRQ(ierr);
    ierr = BVGetColumn(ctx->pV,i,&pvi);CHKERRQ(ierr);
    ierr = VecCopy(ctx->xsub,pvi);CHKERRQ(ierr);
    ierr = BVRestoreColumn(ctx->pV,i,&pvi);CHKERRQ(ierr);
    ierr = VecResetArray(ctx->xsub);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ctx->xdup,&array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SolveLinearSystem"
static PetscErrorCode SolveLinearSystem(EPS eps,Mat A,Mat B,BV V,PetscInt L_start,PetscInt L_end,PetscBool initksp)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j,p_id;
  Mat            Fz;
  PC             pc;
  Vec            Bvj,vj,yj;
  KSP            ksp;

  PetscFunctionBegin;
  ierr = BVCreateVec(V,&Bvj);CHKERRQ(ierr);
  if (ctx->usest) {
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Fz);CHKERRQ(ierr);
  }
  if (ctx->usest && ctx->pA) {
    ierr = KSPCreate(PetscSubcommChild(ctx->subcomm),&ksp);CHKERRQ(ierr);
  }
  for (i=0;i<ctx->num_solve_point;i++) {
    p_id = i*ctx->subcomm->n + ctx->subcomm_id;
    if (!ctx->usest && initksp == PETSC_TRUE) {
      ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&ctx->kspMat[i]);CHKERRQ(ierr);
      ierr = MatCopy(A,ctx->kspMat[i],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      if (B) {
	ierr = MatAXPY(ctx->kspMat[i],-ctx->omega[p_id],B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
	ierr = MatShift(ctx->kspMat[i],-ctx->omega[p_id]);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(ctx->ksp[i],ctx->kspMat[i],ctx->kspMat[i]);CHKERRQ(ierr);
      ierr = KSPSetType(ctx->ksp[i],KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ctx->ksp[i],&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCREDUNDANT);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ctx->ksp[i]);CHKERRQ(ierr);
    } else if (ctx->usest && ctx->pA) {
      ierr = MatCopy(A,Fz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      if (B) {
        ierr = MatAXPY(Fz,-ctx->omega[p_id],B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatShift(Fz,-ctx->omega[p_id]);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(ksp,Fz,Fz);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCREDUNDANT);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    } else if (ctx->usest && !ctx->pA) {
      ierr = STSetShift(eps->st,ctx->omega[p_id]);CHKERRQ(ierr);
      ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
    }
    
    for (j=L_start;j<L_end;j++) {
      ierr = BVGetColumn(V,j,&vj);CHKERRQ(ierr);
      ierr = BVGetColumn(ctx->Y,i*ctx->L_max+j,&yj);CHKERRQ(ierr);
      if (B) {
        ierr = MatMult(B,vj,Bvj);CHKERRQ(ierr);
        if (ctx->usest) {
	  ierr = KSPSolve(ksp,Bvj,yj);CHKERRQ(ierr);
        } else {
	  ierr = KSPSolve(ctx->ksp[i],Bvj,yj);CHKERRQ(ierr);
        }
      } else {
        if (ctx->usest) {
	  ierr = KSPSolve(ksp,vj,yj);CHKERRQ(ierr);
        } else {
	  ierr = KSPSolve(ctx->ksp[i],vj,yj);CHKERRQ(ierr);
        }
      }
      ierr = BVRestoreColumn(V,j,&vj);CHKERRQ(ierr);
      ierr = BVRestoreColumn(ctx->Y,i*ctx->L_max+j,&yj);CHKERRQ(ierr);
    }
    if (ctx->usest && i<ctx->num_solve_point-1) { ierr =  KSPReset(ksp);CHKERRQ(ierr); }
  }
  if (ctx->usest) { ierr = MatDestroy(&Fz);CHKERRQ(ierr); }
  ierr = VecDestroy(&Bvj);CHKERRQ(ierr);
  if (ctx->usest && ctx->pA) {
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
#undef __FUNCT__
#define __FUNCT__ "EstimateNumberEigs"
static PetscErrorCode EstimateNumberEigs(EPS eps,PetscInt *L_add)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
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
    if (ctx->pA) {
      ierr = VecSet(vtemp,0);CHKERRQ(ierr);
      ierr = VecScatterBegin(ctx->scatterin,v,vtemp,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ctx->scatterin,v,vtemp,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecDot(vj,vtemp,&tmp);CHKERRQ(ierr);
    } else {
      ierr = VecDot(vj,v,&tmp);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(ctx->V,j,&vj);CHKERRQ(ierr);
    if (ctx->useconj) sum += PetscRealPart(tmp)*2;
    else sum += tmp;
  }
  ctx->est_eig = PetscAbsScalar(sum/(PetscReal)ctx->L);
  eta = PetscPowReal(10.0,-PetscLog10Real(eps->tol)/ctx->N);
  ierr = PetscInfo1(eps,"Estimation_#Eig %f\n",(double)ctx->est_eig);CHKERRQ(ierr);
  *L_add = (PetscInt)PetscCeilReal((ctx->est_eig*eta)/ctx->M) - ctx->L;
  if (*L_add < 0) *L_add = 0;
  if (*L_add>ctx->L_max-ctx->L) {
    ierr = PetscInfo(eps,"Number of eigenvalues around the contour path may be too large\n");CHKERRQ(ierr);
    *L_add = ctx->L_max-ctx->L;
  }
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&vtemp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "CalcMu"
static PetscErrorCode CalcMu(EPS eps,PetscScalar *Mu)
{
  PetscErrorCode ierr;
  PetscMPIInt    sub_size;
  PetscInt       i,j,k,s;
  PetscScalar    *m,*temp,*temp2,*ppk,alp;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  Mat            M;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscSubcommChild(ctx->subcomm),&sub_size);CHKERRQ(ierr);
  ierr = PetscMalloc3(ctx->num_solve_point*ctx->L*(ctx->L+1),&temp,2*ctx->M*ctx->L*ctx->L,&temp2,ctx->num_solve_point,&ppk);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,ctx->L,ctx->L_max*ctx->num_solve_point,NULL,&M);CHKERRQ(ierr);
  for (i=0;i<2*ctx->M*ctx->L*ctx->L;i++) temp2[i] = 0;
  ierr = BVSetActiveColumns(ctx->Y,0,ctx->L_max*ctx->num_solve_point);CHKERRQ(ierr);
  if (ctx->pA) { 
    ierr = BVSetActiveColumns(ctx->pV,0,ctx->L);CHKERRQ(ierr);
    ierr = BVDot(ctx->Y,ctx->pV,M);CHKERRQ(ierr);
  } else { 
    ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
    ierr = BVDot(ctx->Y,ctx->V,M);CHKERRQ(ierr);
  }
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
  ierr = MPI_Allreduce(temp2,Mu,2*ctx->M*ctx->L*ctx->L,MPIU_SCALAR,MPIU_SUM,(PetscObjectComm((PetscObject)eps)));CHKERRQ(ierr);
  ierr = PetscFree3(temp,temp2,ppk);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BlockHankel"
static PetscErrorCode BlockHankel(EPS eps,PetscScalar *Mu,PetscInt s,PetscScalar *H)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;
  PetscInt i,j,k,L=ctx->L,M=ctx->M;

  PetscFunctionBegin;
  for (k=0;k<L*M;k++) 
    for (j=0;j<M;j++) 
      for (i=0;i<L;i++)
        H[j*L+i+k*L*M] = Mu[i+k*L+(j+s)*L*L];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVD_H0"
static PetscErrorCode SVD_H0(EPS eps,PetscScalar *S,PetscInt *K)
{
#if defined(SLEPC_MISSING_LAPACK_GESVD)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESVD - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
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
static PetscErrorCode ConstructS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
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
      if (ctx->pA) {
        ierr = VecSet(sj,0);CHKERRQ(ierr);
        ierr = VecScatterBegin(ctx->scatterin,v,sj,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(ctx->scatterin,v,sj,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(v,sj);CHKERRQ(ierr);
      }
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
#define __FUNCT__ "SVD_S"
static PetscErrorCode SVD_S(BV S,PetscInt ml,PetscReal delta,PetscReal *sigma,PetscInt *K)
{
#if defined(SLEPC_MISSING_LAPACK_GESVD)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESVD - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,k,local_size;
  PetscScalar    *work,*temp,*B,*tempB,*s_data,*Q1,*Q2,*temp2,alpha=1,beta=0;
  PetscBLASInt   l,m,n,lda,ldu,ldvt,lwork,info,ldb,ldc;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif

  PetscFunctionBegin;
  ierr = BVGetSizes(S,&local_size,NULL,NULL);CHKERRQ(ierr);    
  ierr = BVGetArray(S,&s_data);CHKERRQ(ierr);
  ierr = PetscMalloc7(ml*ml,&temp,ml*ml,&temp2,local_size*ml,&Q1,local_size*ml,&Q2,ml*ml,&B,ml*ml,&tempB,5*ml,&work);CHKERRQ(ierr);
  ierr = PetscMemzero(B,ml*ml*sizeof(PetscScalar));CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(5*ml,&rwork);CHKERRQ(ierr);
#endif
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);

  for (i=0;i<ml;i++) {
    B[i*ml+i]=1;
  }

  for (k=0;k<2;k++) {
    ierr = PetscBLASIntCast(local_size,&m);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ml,&l);CHKERRQ(ierr);
    n = l; lda = m; ldb = m; ldc = l;
    if (k == 0) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&n,&m,&alpha,s_data,&lda,s_data,&ldb,&beta,temp,&ldc));
    } else if ((k%2)==1) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&n,&m,&alpha,Q1,&lda,Q1,&ldb,&beta,temp,&ldc));
    } else {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&l,&n,&m,&alpha,Q2,&lda,Q2,&ldb,&beta,temp,&ldc));
    }
    ierr = PetscMemzero(temp2,ml*ml*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MPI_Allreduce(temp,temp2,ml*ml,MPIU_SCALAR,MPIU_SUM,(PetscObjectComm((PetscObject)S)));CHKERRQ(ierr);

    ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
    n = m; lda = m; lwork = 5*m, ldu = 1; ldvt = 1;
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&m,&n,temp2,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
#else
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&m,&n,temp2,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,&info));
#endif
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);

    ierr = PetscBLASIntCast(local_size,&l);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ml,&n);CHKERRQ(ierr);
    m = n; lda = l; ldb = m; ldc = l;
    if (k==0) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,s_data,&lda,temp2,&ldb,&beta,Q1,&ldc));
    } else if ((k%2)==1) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,Q1,&lda,temp2,&ldb,&beta,Q2,&ldc));
    } else {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,Q2,&lda,temp2,&ldb,&beta,Q1,&ldc));
    }

    ierr = PetscBLASIntCast(ml,&l);CHKERRQ(ierr);
    m = l; n = l; lda = l; ldb = m; ldc = l;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&l,&n,&m,&alpha,B,&lda,temp2,&ldb,&beta,tempB,&ldc));
    for (i=0;i<ml;i++) {
      sigma[i] = sqrt(sigma[i]);
      for (j=0;j<local_size;j++) {
        if ((k%2)==1) Q2[j+i*local_size]/=sigma[i];
        else Q1[j+i*local_size]/=sigma[i];
      }
      for (j=0;j<ml;j++) {
        B[j+i*ml]=tempB[j+i*ml]*sigma[i];
      }
    }
  }

  ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
  n = m; lda = m; ldu=1; ldvt=1;
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","O",&m,&n,B,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","O",&m,&n,B,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,&info));
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);

  ierr = PetscBLASIntCast(local_size,&l);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ml,&n);CHKERRQ(ierr);
  m = n; lda = l; ldb = m; ldc = l;
  if ((k%2)==1) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&l,&n,&m,&alpha,Q1,&lda,B,&ldb,&beta,s_data,&ldc));
  } else {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&l,&n,&m,&alpha,Q2,&lda,B,&ldb,&beta,s_data,&ldc));
  }
 
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = BVRestoreArray(S,&s_data);CHKERRQ(ierr);

  (*K) = 0;
  for (i=0;i<ml;i++) {
    if (sigma[i]/PetscMax(sigma[0],1)>delta) (*K)++;
  }
  ierr = PetscFree7(temp,temp2,Q1,Q2,B,tempB,work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "isGhost"
static PetscErrorCode isGhost(EPS eps,PetscInt ld,PetscInt nv,PetscBool *fl)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j;
  PetscScalar    *pX;
  PetscReal      *tau,s1,s2,tau_max=0.0;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nv,&tau);CHKERRQ(ierr);
  ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetArray(eps->ds,DS_MAT_X,&pX);CHKERRQ(ierr);

  for (i=0;i<nv;i++) {
    s1 = 0;
    s2 = 0;
    for (j=0;j<nv;j++) {
      s1 += PetscAbsScalar(PetscPowScalarInt(pX[i*ld+j],2));
      s2 += PetscPowRealInt(PetscAbsScalar(pX[i*ld+j]),2)/ctx->sigma[j];
    }
    tau[i] = s1/s2;
    tau_max = PetscMax(tau_max,tau[i]);
  }
  ierr = DSRestoreArray(eps->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
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
#define __FUNCT__ "EPSSetUp_CISS"
PetscErrorCode EPSSetUp_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  const char     *prefix;
  PetscInt       i;
  PetscBool      issinvert,istrivial,isring,isellipse,flg;
  PetscScalar    center;
  Mat            A;

  PetscFunctionBegin;
  eps->ncv = PetscMin(eps->n,ctx->L_max*ctx->M);
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (!eps->which) eps->which = EPS_ALL;
  if (!eps->extraction) { ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr); } 
  else if (eps->extraction!=EPS_RITZ) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  /* check region */
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"EPSCISS requires a nontrivial region, e.g. -rg_type ellipse ...");
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGRING,&isring);CHKERRQ(ierr);
  if (!isellipse && !isring) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Currently only implemented for elliptic or ring regions");
  if (isring) {
#if !defined(PETSC_USE_COMPLEX)
    SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Currently only implemented for elliptic regions");
#endif
    ctx->useconj = PETSC_FALSE;
  } else {
    ierr = RGEllipseGetParameters(eps->rg,&center,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    if (ctx->isreal && PetscImaginaryPart(center) == 0.0) ctx->useconj = PETSC_TRUE;
    else ctx->useconj = PETSC_FALSE;
#else
    ctx->useconj = PETSC_FALSE;
#endif
  }
  /* create split comm */
  ierr = SetSolverComm(eps);CHKERRQ(ierr);

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = PetscMalloc4(ctx->N,&ctx->weight,ctx->N,&ctx->omega,ctx->N,&ctx->pp,ctx->L_max*ctx->M,&ctx->sigma);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,3*ctx->N*sizeof(PetscScalar)+ctx->L_max*ctx->N*sizeof(PetscReal));CHKERRQ(ierr);

  /* allocate basis vectors */
  ierr = BVDuplicateResize(eps->V,ctx->L_max*ctx->M,&ctx->S);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->S);CHKERRQ(ierr);
  ierr = BVDuplicateResize(eps->V,ctx->L_max,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->V);CHKERRQ(ierr);

  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSHELL,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Matrix type shell not supported in this solver");

  ierr = CISSRedundantMat(eps);CHKERRQ(ierr);
  if (ctx->pA) {
    ierr = CISSScatterVec(eps);CHKERRQ(ierr);
    ierr = BVCreate(PetscObjectComm((PetscObject)ctx->xsub),&ctx->pV);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->pV,ctx->xsub,eps->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->pV);CHKERRQ(ierr);
    ierr = BVResize(ctx->pV,ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->pV);CHKERRQ(ierr);
  }

  if (ctx->usest) {
    ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSINVERT,&issinvert);CHKERRQ(ierr);
    if (!issinvert) { ierr = STSetType(eps->st,STSINVERT);CHKERRQ(ierr); }
  } else {
    ierr = PetscMalloc2(ctx->num_solve_point,&ctx->ksp,ctx->num_solve_point,&ctx->kspMat);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)eps,ctx->num_solve_point*sizeof(KSP)+ctx->num_solve_point*sizeof(Mat));CHKERRQ(ierr);
    for (i=0;i<ctx->num_solve_point;i++) {
      ierr = KSPCreate(PetscSubcommChild(ctx->subcomm),&ctx->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp[i],(PetscObject)eps,1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->ksp[i]);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ctx->ksp[i],"eps_ciss_");CHKERRQ(ierr);
      ierr = EPSGetOptionsPrefix(eps,&prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ctx->ksp[i],prefix);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(ctx->ksp[i],PETSC_TRUE);CHKERRQ(ierr);
    }
  }

  if (ctx->pA) {
    ierr = BVCreate(PetscObjectComm((PetscObject)ctx->xsub),&ctx->Y);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->Y,ctx->xsub,eps->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->Y);CHKERRQ(ierr);
    ierr = BVResize(ctx->Y,ctx->num_solve_point*ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = BVDuplicateResize(eps->V,ctx->num_solve_point*ctx->L_max,&ctx->Y);CHKERRQ(ierr);
  }
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->Y);CHKERRQ(ierr);

  if (eps->ishermitian && eps->ispositive) {
    ierr = DSSetType(eps->ds,DSGHEP);CHKERRQ(ierr);
  } else {
    ierr = DSSetType(eps->ds,DSGNHEP);CHKERRQ(ierr);
  }
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,2);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  if (!eps->ishermitian) { ierr = PetscInfo(eps,"Warning: complex eigenvalues are not calculated exactly without --with-scalar-type=complex in PETSc\n");CHKERRQ(ierr); }
#endif

  /* dispatch solve method */
  eps->ops->solve = EPSSolve_CISS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_CISS"
PetscErrorCode EPSSolve_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  Mat            A,B,X,M,pA,pB;
  PetscInt       i,ld,nmat,L_add=0,nv=0,L_base=ctx->L,inner,outer,nlocal,*inside;
  PetscScalar    *Mu,*H0,*rr,*temp;
  PetscReal      error,max_error;
  PetscBool      *fl1;
  Vec            si,w[3];
  SlepcSC        sc;
#if defined(PETSC_USE_COMPLEX)
  PetscBool      isellipse;
#endif

  PetscFunctionBegin;
  w[0] = eps->work[0];
  w[1] = NULL;
  w[2] = eps->work[1];
  /* override SC settings */
  ierr = DSGetSlepcSC(eps->ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = VecGetLocalSize(w[0],&nlocal);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
  else B = NULL;
  ierr = SetPathParameter(eps);CHKERRQ(ierr);
  ierr = CISSVecSetRandom(ctx->V,0,ctx->L,eps->rand);CHKERRQ(ierr);

  if (ctx->pA) {
    ierr = VecScatterVecs(eps,ctx->V,ctx->L);CHKERRQ(ierr);
    ierr = SolveLinearSystem(eps,ctx->pA,ctx->pB,ctx->pV,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = SolveLinearSystem(eps,A,B,ctx->V,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscObjectTypeCompare((PetscObject)eps->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (isellipse) {
    ierr = EstimateNumberEigs(eps,&L_add);CHKERRQ(ierr);
  } else {
    L_add = 0;
  }
#else
  L_add = 0;
#endif
  if (L_add>0) {
    ierr = PetscInfo2(eps,"Changing L %D -> %D by Estimate #Eig\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = CISSVecSetRandom(ctx->V,ctx->L,ctx->L+L_add,eps->rand);CHKERRQ(ierr);
    if (ctx->pA) {
      ierr = VecScatterVecs(eps,ctx->V,ctx->L+L_add);CHKERRQ(ierr);
      ierr = SolveLinearSystem(eps,ctx->pA,ctx->pB,ctx->pV,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = SolveLinearSystem(eps,A,B,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    }
    ctx->L += L_add;
  }
  ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
  for (i=0;i<ctx->refine_blocksize;i++) {
    ierr = CalcMu(eps,Mu);CHKERRQ(ierr);
    ierr = BlockHankel(eps,Mu,0,H0);CHKERRQ(ierr);
    ierr = SVD_H0(eps,H0,&nv);CHKERRQ(ierr);
    if (ctx->sigma[0]<=ctx->delta || nv < ctx->L*ctx->M || ctx->L == ctx->L_max) break;
    L_add = L_base;
    if (ctx->L+L_add>ctx->L_max) L_add = ctx->L_max-ctx->L;
    ierr = PetscInfo2(eps,"Changing L %D -> %D by SVD(H0)\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = CISSVecSetRandom(ctx->V,ctx->L,ctx->L+L_add,eps->rand);CHKERRQ(ierr);
    if (ctx->pA) {
      ierr = VecScatterVecs(eps,ctx->V,ctx->L+L_add);CHKERRQ(ierr);
      ierr = SolveLinearSystem(eps,ctx->pA,ctx->pB,ctx->pV,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = SolveLinearSystem(eps,A,B,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    }
    ctx->L += L_add;
  }
  ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);

  for (outer=0;outer<=ctx->refine_outer;outer++) {
    for (inner=0;inner<=ctx->refine_inner;inner++) {
      ierr = ConstructS(eps);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
      ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
      ierr = SVD_S(ctx->S,ctx->L*ctx->M,ctx->delta,ctx->sigma,&nv);CHKERRQ(ierr);
      if (ctx->sigma[0]>ctx->delta && nv==ctx->L*ctx->M && inner!=ctx->refine_inner) {
        if (ctx->pA) {
          ierr = VecScatterVecs(eps,ctx->V,ctx->L);CHKERRQ(ierr);
          ierr = SolveLinearSystem(eps,ctx->pA,ctx->pB,ctx->pV,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = SolveLinearSystem(eps,A,B,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        }
      } else break;
    }

    eps->nconv = 0;
    if (nv == 0) break;
    ierr = DSSetDimensions(eps->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);

    ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
    ierr = DSGetMat(eps->ds,DS_MAT_A,&pA);CHKERRQ(ierr);
    ierr = MatZeroEntries(pA);CHKERRQ(ierr);
    ierr = BVMatProject(ctx->S,A,ctx->S,pA);CHKERRQ(ierr);
    ierr = DSRestoreMat(eps->ds,DS_MAT_A,&pA);CHKERRQ(ierr);
    ierr = DSGetMat(eps->ds,DS_MAT_B,&pB);CHKERRQ(ierr);
    ierr = MatZeroEntries(pB);CHKERRQ(ierr);
    if (B) { ierr = BVMatProject(ctx->S,B,ctx->S,pB);CHKERRQ(ierr); }
    else { ierr = MatShift(pB,1);CHKERRQ(ierr); }
    ierr = DSRestoreMat(eps->ds,DS_MAT_B,&pB);CHKERRQ(ierr);

    ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);

    ierr = PetscMalloc3(nv,&fl1,nv,&inside,nv,&rr);CHKERRQ(ierr);
    ierr = isGhost(eps,ld,nv,fl1);CHKERRQ(ierr);
    ierr = RGCheckInside(eps->rg,nv,eps->eigr,eps->eigi,inside);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      if (fl1[i] && inside[i]>0) {
        rr[i] = 1.0;
        eps->nconv++;
      } else rr[i] = 0.0;
    }
    ierr = DSSort(eps->ds,eps->eigr,eps->eigi,rr,NULL,&eps->nconv);CHKERRQ(ierr);
    ierr = PetscFree3(fl1,inside,rr);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(eps->V,0,nv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
    ierr = BVCopy(ctx->S,eps->V);CHKERRQ(ierr);

    ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
    ierr = DSGetMat(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
    ierr = BVMultInPlace(ctx->S,X,0,eps->nconv);CHKERRQ(ierr);
    if (eps->ishermitian) {
      ierr = BVMultInPlace(eps->V,X,0,eps->nconv);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    max_error = 0.0;
    for (i=0;i<eps->nconv;i++) {
      ierr = BVGetColumn(ctx->S,i,&si);CHKERRQ(ierr);
      ierr = VecNormalize(si,NULL);CHKERRQ(ierr);
      ierr = EPSComputeResidualNorm_Private(eps,eps->eigr[i],0,si,NULL,w,&error);CHKERRQ(ierr);
      ierr = (*eps->converged)(eps,eps->eigr[i],0,error,&error,eps->convergedctx);CHKERRQ(ierr);
      ierr = BVRestoreColumn(ctx->S,i,&si);CHKERRQ(ierr);
      max_error = PetscMax(max_error,error);
    }

    if (max_error <= eps->tol || outer == ctx->refine_outer) break;

    if (eps->nconv > ctx->L) nv = eps->nconv;
    else if (ctx->L > nv) nv = ctx->L;
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,nv,ctx->L,NULL,&M);CHKERRQ(ierr);
    ierr = MatDenseGetArray(M,&temp);CHKERRQ(ierr);
    for (i=0;i<ctx->L*nv;i++) {
      ierr = PetscRandomGetValue(eps->rand,&temp[i]);CHKERRQ(ierr);
      temp[i] = PetscRealPart(temp[i]);
    }
    ierr = MatDenseRestoreArray(M,&temp);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
    ierr = BVMultInPlace(ctx->S,M,0,ctx->L);CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
    ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
    if (ctx->pA) {
      ierr = VecScatterVecs(eps,ctx->V,ctx->L);CHKERRQ(ierr);
      ierr = SolveLinearSystem(eps,ctx->pA,ctx->pB,ctx->pV,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = SolveLinearSystem(eps,A,B,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetSizes_CISS"
static PetscErrorCode EPSCISSSetSizes_CISS(EPS eps,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool isreal)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) {
    if (ctx->N!=32) { ctx->N =32; ctx->M = ctx->N/4; }
  } else {
    if (ip<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be > 0");
    if (ip%2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be an even number");
    if (ctx->N!=ip) { ctx->N = ip; ctx->M = ctx->N/4; }
  }
  if (bs == PETSC_DECIDE || bs == PETSC_DEFAULT) {
    ctx->L = 16;
  } else {
    if (bs<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be > 0");
    if (bs>ctx->L_max) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be less than or equal to the maximum number of block size");
    ctx->L = bs;
  }
  if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
    ctx->M = ctx->N/4;
  } else {
    if (ms<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
    if (ms>ctx->N) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
    ctx->M = ms;
  }
  if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
    ctx->num_subcomm = 1;
  } else {
    if (npart<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The npart argument must be > 0");
    ctx->num_subcomm = npart;
  }
  if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
    ctx->L = 256;
  } else {
    if (bsmax<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
    if (bsmax<ctx->L) ctx->L_max = ctx->L;
    else ctx->L_max = bsmax;
  }
  ctx->isreal = isreal;
  ierr = EPSReset(eps);CHKERRQ(ierr);   /* clean allocated arrays and force new setup */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetSizes"
/*@
   EPSCISSSetSizes - Sets the values of various size parameters in the CISS solver.

   Logically Collective on EPS

   Input Parameters:
+  eps   - the eigenproblem solver context
.  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  isreal - A and B are real

   Options Database Keys:
+  -eps_ciss_integration_points - Sets the number of integration points
.  -eps_ciss_blocksize - Sets the block size
.  -eps_ciss_moments - Sets the moment size
.  -eps_ciss_partitions - Sets the number of partitions
.  -eps_ciss_maxblocksize - Sets the maximum block size
-  -eps_ciss_realmats - A and B are real

   Note:
   The default number of partitions is 1. This means the internal KSP object is shared
   among all processes of the EPS communicator. Otherwise, the communicator is split
   into npart communicators, so that npart KSP solves proceed simultaneously.

   Level: advanced

.seealso: EPSCISSGetSizes()
@*/
PetscErrorCode EPSCISSSetSizes(EPS eps,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool isreal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,ip,2);
  PetscValidLogicalCollectiveInt(eps,bs,3);
  PetscValidLogicalCollectiveInt(eps,ms,4);
  PetscValidLogicalCollectiveInt(eps,npart,5);
  PetscValidLogicalCollectiveInt(eps,bsmax,6);
  PetscValidLogicalCollectiveBool(eps,isreal,7);
  ierr = PetscTryMethod(eps,"EPSCISSSetSizes_C",(EPS,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool),(eps,ip,bs,ms,npart,bsmax,isreal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetSizes_CISS"
static PetscErrorCode EPSCISSGetSizes_CISS(EPS eps,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *isreal)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (ip) *ip = ctx->N;
  if (bs) *bs = ctx->L;
  if (ms) *ms = ctx->M;
  if (npart) *npart = ctx->num_subcomm;
  if (bsmax) *bsmax = ctx->L_max;
  if (isreal) *isreal = ctx->isreal;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetSizes"
/*@
   EPSCISSGetSizes - Gets the values of various size parameters in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  ip    - number of integration points
.  bs    - block size
.  ms    - moment size
.  npart - number of partitions when splitting the communicator
.  bsmax - max block size
-  isreal - A and B are real

   Level: advanced

.seealso: EPSCISSSetSizes()
@*/
PetscErrorCode EPSCISSGetSizes(EPS eps,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *isreal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSCISSGetSizes_C",(EPS,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*),(eps,ip,bs,ms,npart,bsmax,isreal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetThreshold_CISS"
static PetscErrorCode EPSCISSSetThreshold_CISS(EPS eps,PetscReal delta,PetscReal spur)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (delta == PETSC_DEFAULT) {
    ctx->delta = 1e-12;
  } else {
    if (delta<=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The delta argument must be > 0.0");
    ctx->delta = delta;
  }
  if (spur == PETSC_DEFAULT) {
    ctx->spurious_threshold = 1e-4;
  } else {
    if (spur<=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The spurious threshold argument must be > 0.0");
    ctx->spurious_threshold = spur;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetThreshold"
/*@
   EPSCISSSetThreshold - Sets the values of various threshold parameters in
   the CISS solver.

   Logically Collective on EPS

   Input Parameters:
+  eps   - the eigenproblem solver context
.  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Options Database Keys:
+  -eps_ciss_delta - Sets the delta
-  -eps_ciss_spurious_threshold - Sets the spurious threshold

   Level: advanced

.seealso: EPSCISSGetThreshold()
@*/
PetscErrorCode EPSCISSSetThreshold(EPS eps,PetscReal delta,PetscReal spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,delta,2);
  PetscValidLogicalCollectiveReal(eps,spur,3);
  ierr = PetscTryMethod(eps,"EPSCISSSetThreshold_C",(EPS,PetscReal,PetscReal),(eps,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetThreshold_CISS"
static PetscErrorCode EPSCISSGetThreshold_CISS(EPS eps,PetscReal *delta,PetscReal *spur)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (delta) *delta = ctx->delta;
  if (spur)  *spur = ctx->spurious_threshold;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetThreshold"
/*@
   EPSCISSGetThreshold - Gets the values of various threshold parameters
   in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  delta - threshold for numerical rank
-  spur  - spurious threshold (to discard spurious eigenpairs)

   Level: advanced

.seealso: EPSCISSSetThreshold()
@*/
PetscErrorCode EPSCISSGetThreshold(EPS eps,PetscReal *delta,PetscReal *spur)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSCISSGetThreshold_C",(EPS,PetscReal*,PetscReal*),(eps,delta,spur));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetRefinement_CISS"
static PetscErrorCode EPSCISSSetRefinement_CISS(EPS eps,PetscInt inner,PetscInt outer,PetscInt blsize)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (inner == PETSC_DEFAULT) {
    ctx->refine_inner = 0;
  } else {
    if (inner<0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The refine inner argument must be >= 0");
    ctx->refine_inner = inner;
  }
  if (outer == PETSC_DEFAULT) {
    ctx->refine_outer = 0;
  } else {
    if (outer<0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The refine outer argument must be >= 0");
    ctx->refine_outer = outer;
  }
  if (blsize == PETSC_DEFAULT) {
    ctx->refine_blocksize = 0;
  } else {
    if (blsize<0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The refine blocksize argument must be >= 0");
    ctx->refine_blocksize = blsize;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetRefinement"
/*@
   EPSCISSSetRefinement - Sets the values of various refinement parameters
   in the CISS solver.

   Logically Collective on EPS

   Input Parameters:
+  eps    - the eigenproblem solver context
.  inner  - number of iterative refinement iterations (inner loop)
.  outer  - number of iterative refinement iterations (outer loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Options Database Keys:
+  -eps_ciss_refine_inner - Sets number of inner iterations
.  -eps_ciss_refine_outer - Sets number of outer iterations
-  -eps_ciss_refine_blocksize - Sets number of blocksize iterations

   Level: advanced

.seealso: EPSCISSGetRefinement()
@*/
PetscErrorCode EPSCISSSetRefinement(EPS eps,PetscInt inner,PetscInt outer,PetscInt blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,inner,2);
  PetscValidLogicalCollectiveInt(eps,outer,3);
  PetscValidLogicalCollectiveInt(eps,blsize,4);
  ierr = PetscTryMethod(eps,"EPSCISSSetRefinement_C",(EPS,PetscInt,PetscInt,PetscInt),(eps,inner,outer,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetRefinement_CISS"
static PetscErrorCode EPSCISSGetRefinement_CISS(EPS eps,PetscInt *inner,PetscInt *outer,PetscInt *blsize)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (inner)  *inner = ctx->refine_inner;
  if (outer)  *outer = ctx->refine_outer;
  if (blsize) *blsize = ctx->refine_blocksize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetRefinement"
/*@
   EPSCISSGetRefinement - Gets the values of various refinement parameters
   in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  inner  - number of iterative refinement iterations (inner loop)
.  outer  - number of iterative refinement iterations (outer loop)
-  blsize - number of iterative refinement iterations (blocksize loop)

   Level: advanced

.seealso: EPSCISSSetRefinement()
@*/
PetscErrorCode EPSCISSGetRefinement(EPS eps, PetscInt *inner, PetscInt *outer,PetscInt *blsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSCISSGetRefinement_C",(EPS,PetscInt*,PetscInt*,PetscInt*),(eps,inner,outer,blsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetUseST_CISS"
static PetscErrorCode EPSCISSSetUseST_CISS(EPS eps,PetscBool usest)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ctx->usest = usest;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetUseST"
/*@
   EPSCISSSetUseST - Sets a flag indicating that the CISS solver will
   use the ST object for the linear solves.

   Logically Collective on EPS

   Input Parameters:
+  eps    - the eigenproblem solver context
-  usest  - boolean flag to use the ST object or not

   Options Database Keys:
+  -eps_ciss_usest <bool> - whether the ST object will be used or not

   Level: advanced

.seealso: EPSCISSGetUseST()
@*/
PetscErrorCode EPSCISSSetUseST(EPS eps,PetscBool usest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,usest,2);
  ierr = PetscTryMethod(eps,"EPSCISSSetUseST_C",(EPS,PetscBool),(eps,usest));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetUseST_CISS"
static PetscErrorCode EPSCISSGetUseST_CISS(EPS eps,PetscBool *usest)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  *usest = ctx->usest;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetUseST"
/*@
   EPSCISSGetUseST - Gets the flag for using the ST object
   in the CISS solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  usest - boolean flag indicating if the ST object is being used

   Level: advanced

.seealso: EPSCISSSetUseST()
@*/
PetscErrorCode EPSCISSGetUseST(EPS eps, PetscBool *usest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSCISSGetUseST_C",(EPS,PetscBool*),(eps,usest));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_CISS"
PetscErrorCode EPSReset_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i;

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
  ierr = VecScatterDestroy(&ctx->scatterin);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->xsub);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->xdup);CHKERRQ(ierr);
  if (ctx->pA) {
    ierr = MatDestroy(&ctx->pA);CHKERRQ(ierr);
    ierr = MatDestroy(&ctx->pB);CHKERRQ(ierr);
    ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_CISS"
PetscErrorCode EPSSetFromOptions_CISS(PetscOptions *PetscOptionsObject,EPS eps)
{
  PetscErrorCode ierr;
  PetscReal      r3,r4;
  PetscInt       i1,i2,i3,i4,i5,i6,i7,i8;
  PetscBool      b1,b2;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS CISS Options");CHKERRQ(ierr);
  ierr = EPSCISSGetSizes(eps,&i1,&i2,&i3,&i4,&i5,&b1);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_ciss_integration_points","CISS number of integration points","EPSCISSSetSizes",i1,&i1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_ciss_blocksize","CISS block size","EPSCISSSetSizes",i2,&i2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_ciss_moments","CISS moment size","EPSCISSSetSizes",i3,&i3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_ciss_partitions","CISS number of partitions","EPSCISSSetSizes",i4,&i4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_ciss_maxblocksize","CISS maximum block size","EPSCISSSetSizes",i5,&i5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-eps_ciss_realmats","CISS A and B are real","EPSCISSSetSizes",b1,&b1,NULL);CHKERRQ(ierr);
  ierr = EPSCISSSetSizes(eps,i1,i2,i3,i4,i5,b1);CHKERRQ(ierr);

  ierr = EPSCISSGetThreshold(eps,&r3,&r4);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_ciss_delta","CISS threshold for numerical rank","EPSCISSSetThreshold",r3,&r3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_ciss_spurious_threshold","CISS threshold for the spurious eigenpairs","EPSCISSSetThreshold",r4,&r4,NULL);CHKERRQ(ierr);
  ierr = EPSCISSSetThreshold(eps,r3,r4);CHKERRQ(ierr);

  ierr = EPSCISSGetRefinement(eps,&i6,&i7,&i8);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_ciss_refine_inner","CISS number of inner iterative refinement iterations","EPSCISSSetRefinement",i6,&i6,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_ciss_refine_outer","CISS number of outer iterative refinement iterations","EPSCISSSetRefinement",i7,&i7,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_ciss_refine_blocksize","CISS number of blocksize iterative refinement iterations","EPSCISSSetRefinement",i8,&i8,NULL);CHKERRQ(ierr);
  ierr = EPSCISSSetRefinement(eps,i6,i7,i8);CHKERRQ(ierr);

  ierr = EPSCISSGetUseST(eps,&b2);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-eps_ciss_usest","CISS use ST for linear solves","EPSCISSSetUseST",b2,&b2,NULL);CHKERRQ(ierr);
  ierr = EPSCISSSetUseST(eps,b2);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_CISS"
PetscErrorCode EPSDestroy_CISS(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetUseST_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetUseST_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_CISS"
PetscErrorCode EPSView_CISS(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: sizes { integration points: %D, block size: %D, moment size: %D, partitions: %D, maximum block size: %D }\n",ctx->N,ctx->L,ctx->M,ctx->num_subcomm,ctx->L_max);CHKERRQ(ierr);
    if (ctx->isreal) {
      ierr = PetscViewerASCIIPrintf(viewer,"  CISS: exploiting symmetry of integration points\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: threshold { delta: %g, spurious threshold: %g }\n",(double)ctx->delta,(double)ctx->spurious_threshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: iterative refinement  { inner: %D, outer: %D, blocksize: %D }\n",ctx->refine_inner,ctx->refine_outer, ctx->refine_blocksize);CHKERRQ(ierr);
    if (ctx->usest) {
      ierr = PetscViewerASCIIPrintf(viewer,"  CISS: using ST for linear solves\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    /*ierr = KSPView(ctx->ksp[0],viewer);CHKERRQ(ierr);*/
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_CISS"
PETSC_EXTERN PetscErrorCode EPSCreate_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = ctx;
  eps->ops->setup          = EPSSetUp_CISS;
  eps->ops->setfromoptions = EPSSetFromOptions_CISS;
  eps->ops->destroy        = EPSDestroy_CISS;
  eps->ops->reset          = EPSReset_CISS;
  eps->ops->view           = EPSView_CISS;
  eps->ops->backtransform  = NULL;
  eps->ops->computevectors = EPSComputeVectors_Schur;
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetSizes_C",EPSCISSSetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetSizes_C",EPSCISSGetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetThreshold_C",EPSCISSSetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetThreshold_C",EPSCISSGetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRefinement_C",EPSCISSSetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRefinement_C",EPSCISSGetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetUseST_C",EPSCISSSetUseST_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetUseST_C",EPSCISSGetUseST_CISS);CHKERRQ(ierr);
  /* set default values of parameters */
  ctx->N       = 32;
  ctx->L       = 16;
  ctx->M       = ctx->N/4;
  ctx->delta   = 1e-12;
  ctx->L_max   = 64;
  ctx->spurious_threshold = 1e-4;
  ctx->usest   = PETSC_FALSE;
  ctx->isreal  = PETSC_FALSE;
  ctx->refine_outer = 1;
  ctx->refine_inner = 1;
  ctx->refine_blocksize = 1;
  ctx->num_subcomm = 1;
  PetscFunctionReturn(0);
}

