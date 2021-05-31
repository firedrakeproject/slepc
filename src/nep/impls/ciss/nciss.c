/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
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
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <slepcblaslapack.h>

typedef struct _n_nep_ciss_project *NEP_CISS_PROJECT;
typedef struct {
  /* parameters */
  PetscInt          N;             /* number of integration points (32) */
  PetscInt          L;             /* block size (16) */
  PetscInt          M;             /* moment degree (N/4 = 4) */
  PetscReal         delta;         /* threshold of singular value (1e-12) */
  PetscInt          L_max;         /* maximum number of columns of the source matrix V */
  PetscReal         spurious_threshold; /* discard spurious eigenpairs */
  PetscBool         isreal;        /* T(z) is real for real z */
  PetscInt          npart;         /* number of partitions */
  PetscInt          refine_inner;
  PetscInt          refine_blocksize;
  NEPCISSExtraction extraction;
  /* private data */
  PetscReal         *sigma;        /* threshold for numerical rank */
  PetscInt          subcomm_id;
  PetscInt          num_solve_point;
  PetscScalar       *weight;
  PetscScalar       *omega;
  PetscScalar       *pp;
  BV                V;
  BV                S;
  BV                Y;
  KSP               *ksp;           /* ksp array for storing factorizations at integration points */
  PetscBool         useconj;
  PetscReal         est_eig;
  PetscSubcomm      subcomm;
  VecScatter        scatterin;
  Mat               *pA,T,J;        /* auxiliary matrices when using subcomm */
  BV                pV;
  Vec               xsub,xdup;
  NEP_CISS_PROJECT  dsctxf;
} NEP_CISS;

struct _n_nep_ciss_project {
  NEP  nep;
  BV   Q;
};

static PetscErrorCode NEPContourDSComputeMatrix(DS ds,PetscScalar lambda,PetscBool deriv,DSMatType mat,void *ctx)
{
  NEP_CISS_PROJECT proj = (NEP_CISS_PROJECT)ctx;
  PetscErrorCode   ierr;
  Mat              M,fun;

  PetscFunctionBegin;
  if (!deriv) {
    ierr = NEPComputeFunction(proj->nep,lambda,proj->nep->function,proj->nep->function);CHKERRQ(ierr);
    fun = proj->nep->function;
  } else {
    ierr = NEPComputeJacobian(proj->nep,lambda,proj->nep->jacobian);CHKERRQ(ierr);
    fun = proj->nep->jacobian;
  }
  ierr = DSGetMat(ds,mat,&M);CHKERRQ(ierr);
  ierr = BVMatProject(proj->Q,fun,proj->Q,M);CHKERRQ(ierr);
  ierr = DSRestoreMat(ds,mat,&M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Destroy KSP objects when the number of solve points changes
*/
PETSC_STATIC_INLINE PetscErrorCode NEPCISSResetSolvers(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       i;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (ctx->ksp) {
    for (i=0;i<ctx->num_solve_point;i++) {
      ierr = KSPDestroy(&ctx->ksp[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(ctx->ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Clean PetscSubcomm object when the number of partitions changes
*/
PETSC_STATIC_INLINE PetscErrorCode NEPCISSResetSubcomm(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = NEPCISSResetSolvers(nep);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&ctx->subcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Determine whether half of integration points can be avoided (use their conjugates);
  depends on isreal and the center of the region
*/
PETSC_STATIC_INLINE PetscErrorCode NEPCISSSetUseConj(NEP nep,PetscBool *useconj)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscScalar    center;
  PetscBool      isellipse=PETSC_FALSE;

  PetscFunctionBegin;
  *useconj = PETSC_FALSE;
  ierr = PetscObjectTypeCompare((PetscObject)nep->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (isellipse) {
    ierr = RGEllipseGetParameters(nep->rg,&center,NULL,NULL);CHKERRQ(ierr);
    *useconj = (ctx->isreal && PetscImaginaryPart(center) == 0.0)? PETSC_TRUE: PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*
  Create PetscSubcomm object and determine num_solve_point (depends on npart, N, useconj)
*/
PETSC_STATIC_INLINE PetscErrorCode NEPCISSSetUpSubComm(NEP nep,PetscInt *num_solve_point)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       N = ctx->N;

  PetscFunctionBegin;
  ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)nep),&ctx->subcomm);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(ctx->subcomm,ctx->npart);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = PetscSubcommSetType(ctx->subcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,sizeof(PetscSubcomm));CHKERRQ(ierr);
  ctx->subcomm_id = ctx->subcomm->color;
  ierr = NEPCISSSetUseConj(nep,&ctx->useconj);CHKERRQ(ierr);
  if (ctx->useconj) N = N/2;
  *num_solve_point = N / ctx->npart;
  if (N%ctx->npart > ctx->subcomm_id) (*num_solve_point)++;
  PetscFunctionReturn(0);
}

static PetscErrorCode CISSRedundantMat(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (ctx->subcomm->n != 1) {
    if (!ctx->pA) {ierr = PetscCalloc1(nep->nt,&ctx->pA);CHKERRQ(ierr);}
    for (i=0;i<nep->nt;i++) {
      ierr = MatDestroy(&ctx->pA[i]);CHKERRQ(ierr);
      ierr = MatCreateRedundantMatrix(nep->A[i],ctx->subcomm->n,PetscSubcommChild(ctx->subcomm),MAT_INITIAL_MATRIX,&ctx->pA[i]);CHKERRQ(ierr);
    }
    if (!ctx->T) {
      ierr = MatDuplicate(ctx->pA[0],MAT_DO_NOT_COPY_VALUES,&ctx->T);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->T);CHKERRQ(ierr);
    }
    if (!ctx->J) {
      ierr = MatDuplicate(ctx->pA[0],MAT_DO_NOT_COPY_VALUES,&ctx->J);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->J);CHKERRQ(ierr);
    }
  } else ctx->pA = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode CISSScatterVec(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  IS             is1,is2;
  Vec            v0;
  PetscInt       i,j,k,mstart,mend,mlocal;
  PetscInt       *idx1,*idx2,mloc_sub;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->xsub);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx->pA[0],&ctx->xsub,NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&ctx->xdup);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->pA[0],&mloc_sub,NULL);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscSubcommContiguousParent(ctx->subcomm),mloc_sub,PETSC_DECIDE,&ctx->xdup);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&ctx->scatterin);CHKERRQ(ierr);
  ierr = BVGetColumn(ctx->V,0,&v0);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(v0,&mstart,&mend);CHKERRQ(ierr);
  mlocal = mend - mstart;
  ierr = PetscMalloc2(ctx->subcomm->n*mlocal,&idx1,ctx->subcomm->n*mlocal,&idx2);CHKERRQ(ierr);
  j = 0;
  for (k=0;k<ctx->subcomm->n;k++) {
    for (i=mstart;i<mend;i++) {
      idx1[j]   = i;
      idx2[j++] = i + nep->n*k;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)nep),ctx->subcomm->n*mlocal,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)nep),ctx->subcomm->n*mlocal,idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
  ierr = VecScatterCreate(v0,is1,ctx->xdup,is2,&ctx->scatterin);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = PetscFree2(idx1,idx2);CHKERRQ(ierr);
  ierr = BVRestoreColumn(ctx->V,0,&v0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
    ctx->pp[i] = PetscCMPLX(PetscCosReal(theta),vscale*PetscSinReal(theta));
    ctx->weight[i] = radius*PetscCMPLX(vscale*PetscCosReal(theta),PetscSinReal(theta))/(PetscReal)ctx->N;
    ctx->omega[i] = rgscale*(center + radius*ctx->pp[i]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterVecs(NEP nep,BV Vin,PetscInt n)
{
  PetscErrorCode    ierr;
  NEP_CISS          *ctx = (NEP_CISS*)nep->data;
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

static PetscErrorCode NEPComputeFunctionSubcomm(NEP nep,PetscScalar lambda,Mat T,PetscBool deriv)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  switch (nep->fui) {
  case NEP_USER_INTERFACE_CALLBACK:
    SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Partitions are not supported with callbacks");
  case NEP_USER_INTERFACE_SPLIT:
    ierr = MatZeroEntries(T);CHKERRQ(ierr);
    for (i=0;i<nep->nt;i++) {
      if (!deriv) {
        ierr = FNEvaluateFunction(nep->f[i],lambda,&alpha);CHKERRQ(ierr);
      } else {
        ierr = FNEvaluateDerivative(nep->f[i],lambda,&alpha);CHKERRQ(ierr);
      }
      ierr = MatAXPY(T,alpha,ctx->pA[i],nep->mstr);CHKERRQ(ierr);
    }
    break;
  }
  PetscFunctionReturn(0);
}

/*
  Y_i = F(z_i)^{-1}Fp(z_i)V for every integration point, Y=[Y_i] is in the context
*/
static PetscErrorCode SolveLinearSystem(NEP nep,Mat T,Mat dT,BV V,PetscInt L_start,PetscInt L_end,PetscBool initksp)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,p_id;
  Mat            kspMat,MV,BMV=NULL,MC;

  PetscFunctionBegin;
  if (!ctx->ksp) { ierr = NEPCISSGetKSPs(nep,&ctx->num_solve_point,&ctx->ksp);CHKERRQ(ierr); }
  ierr = BVSetActiveColumns(V,L_start,L_end);CHKERRQ(ierr);
  ierr = BVGetMat(V,&MV);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) {
    p_id = i*ctx->subcomm->n + ctx->subcomm_id;
    if (initksp) {
      if (ctx->subcomm->n == 1) {
        ierr = NEPComputeFunction(nep,ctx->omega[p_id],T,T);CHKERRQ(ierr);
      } else {
        ierr = NEPComputeFunctionSubcomm(nep,ctx->omega[p_id],T,PETSC_FALSE);CHKERRQ(ierr);
      }
      ierr = MatDuplicate(T,MAT_COPY_VALUES,&kspMat);CHKERRQ(ierr);
      ierr = KSPSetOperators(ctx->ksp[i],kspMat,kspMat);CHKERRQ(ierr);
      ierr = MatDestroy(&kspMat);CHKERRQ(ierr);
    }
    if (ctx->subcomm->n==1) {
      ierr = NEPComputeJacobian(nep,ctx->omega[p_id],dT);CHKERRQ(ierr);
    } else {
      ierr = NEPComputeFunctionSubcomm(nep,ctx->omega[p_id],dT,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(ctx->Y,i*ctx->L_max+L_start,i*ctx->L_max+L_end);CHKERRQ(ierr);
    ierr = BVGetMat(ctx->Y,&MC);CHKERRQ(ierr);
    if (!i) {
      ierr = MatProductCreate(dT,MV,NULL,&BMV);CHKERRQ(ierr);
      ierr = MatProductSetType(BMV,MATPRODUCT_AB);CHKERRQ(ierr);
      ierr = MatProductSetFromOptions(BMV);CHKERRQ(ierr);
      ierr = MatProductSymbolic(BMV);CHKERRQ(ierr);
    }
    ierr = MatProductNumeric(BMV);CHKERRQ(ierr);
    ierr = KSPMatSolve(ctx->ksp[i],BMV,MC);CHKERRQ(ierr);
    ierr = BVRestoreMat(ctx->Y,&MC);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&BMV);CHKERRQ(ierr);
  ierr = BVRestoreMat(V,&MV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  N = 1/(2*pi*i)*integral(trace(F(z)^{-1}Fp(z))dz)=Sum_i( w_i * Sum_j(v_j^*  F(z_i)^{-1}Fp(z_i)v_j)/L)
  N= 1/L Sum_j(v_j^* Sum_i w_i*Y_i(j))
  Estimation of L_add which is added to the basis L, without exceeding L_max
*/
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

static PetscErrorCode CalcMu(NEP nep, PetscScalar *Mu)
{
  PetscErrorCode ierr;
  PetscMPIInt    sub_size,len;
  PetscInt       i,j,k,s;
  PetscScalar    *m,*temp,*temp2,*ppk,alp;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  Mat            M;
  BV             V;

  PetscFunctionBegin;
  V = (ctx->pA)?ctx->pV:ctx->V;
  ierr = MPI_Comm_size(PetscSubcommChild(ctx->subcomm),&sub_size);CHKERRMPI(ierr);
  ierr = PetscMalloc3(ctx->num_solve_point*ctx->L*(ctx->L+1),&temp,2*ctx->M*ctx->L*ctx->L,&temp2,ctx->num_solve_point,&ppk);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,ctx->L,ctx->L_max*ctx->num_solve_point,NULL,&M);CHKERRQ(ierr);
  for (i=0;i<2*ctx->M*ctx->L*ctx->L;i++) temp2[i] = 0;
  ierr = BVSetActiveColumns(ctx->Y,0,ctx->L_max*ctx->num_solve_point);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(V,0,ctx->L);CHKERRQ(ierr);
  ierr = BVDot(ctx->Y,V,M);CHKERRQ(ierr);
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
  ierr = MPIU_Allreduce(temp2,Mu,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)nep));CHKERRMPI(ierr);
  ierr = PetscFree3(temp,temp2,ppk);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode BlockHankel(NEP nep,PetscScalar *Mu,PetscInt s,PetscScalar *H)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;
  PetscInt i,j,k,L=ctx->L,M=ctx->M;

  PetscFunctionBegin;
  for (k=0;k<L*M;k++)
    for (j=0;j<M;j++)
      for (i=0;i<L;i++)
        H[j*L+i+k*L*M] = Mu[i+k*L+(j+s)*L*L];
  PetscFunctionReturn(0);
}

static PetscErrorCode SVD_H0(NEP nep,PetscScalar *S,PetscInt *K)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,ml=ctx->L*ctx->M;
  PetscBLASInt   m,n,lda,ldu,ldvt,lwork,info;
  PetscScalar    *work;
  PetscReal      *rwork;

  PetscFunctionBegin;
  ierr = PetscMalloc1(5*ml,&work);CHKERRQ(ierr);
  ierr = PetscMalloc1(5*ml,&rwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
  n = m; lda = m; ldu = m; ldvt = m; lwork = 5*m;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&m,&n,S,&lda,ctx->sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
  SlepcCheckLapackInfo("gesvd",info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  (*K) = 0;
  for (i=0;i<ml;i++) {
    if (ctx->sigma[i]/PetscMax(ctx->sigma[0],1)>ctx->delta) (*K)++;
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVD_S(NEP nep,BV S,PetscScalar *pA,PetscInt *K)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,n,ml=ctx->L*ctx->M;
  PetscBLASInt   m,lda,lwork,info;
  PetscScalar    *work;
  PetscReal      *rwork;
  Mat            A;
  Vec            v;

  PetscFunctionBegin;
  /* Compute QR factorizaton of S */
  ierr = BVGetSizes(S,NULL,&n,NULL);CHKERRQ(ierr);
  n    = PetscMin(n,ml);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(S,0,n);CHKERRQ(ierr);
  ierr = PetscArrayzero(pA,ml*n);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_SELF,n,n,PETSC_DECIDE,PETSC_DECIDE,pA,&A);CHKERRQ(ierr);
  ierr = BVOrthogonalize(S,A);CHKERRQ(ierr);
  if (n<ml) {
    /* the rest of the factorization */
    for (i=n;i<ml;i++) {
      ierr = BVGetColumn(S,i,&v);CHKERRQ(ierr);
      ierr = BVOrthogonalizeVec(S,v,pA+i*n,NULL,NULL);CHKERRQ(ierr);
      ierr = BVRestoreColumn(S,i,&v);CHKERRQ(ierr);
    }
  }
  ierr = PetscBLASIntCast(n,&lda);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
  ierr = PetscMalloc2(5*ml,&work,5*ml,&rwork);CHKERRQ(ierr);
  lwork = 5*m;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&lda,&m,pA,&lda,ctx->sigma,NULL,&lda,NULL,&lda,work,&lwork,rwork,&info));
  SlepcCheckLapackInfo("gesvd",info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  (*K) = 0;
  for (i=0;i<n;i++) {
    if (ctx->sigma[i]/PetscMax(ctx->sigma[0],1)>ctx->delta) (*K)++;
  }
  /* n first columns of A have the left singular vectors */
  ierr = BVMultInPlace(S,A,0,*K);CHKERRQ(ierr);
  ierr = PetscFree2(work,rwork);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVD_S_CAA(NEP nep,BV S,PetscScalar *pA,PetscInt *K)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,n,ml=ctx->L*ctx->M;
  PetscBLASInt   m,k_,lda,lwork,info;
  PetscScalar    *work,*T,*U,*R,sone=1.0,zero=0.0;
  PetscReal      *rwork;
  Mat            A;

  PetscFunctionBegin;
  /* Compute QR factorizaton of S */
  ierr = BVGetSizes(S,NULL,&n,NULL);CHKERRQ(ierr);
  if (n<ml) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This extraction strategy does not support problem size n < m*L");
  ierr = BVSetActiveColumns(S,0,ml);CHKERRQ(ierr);
  ierr = PetscArrayzero(pA,ml*ml);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_SELF,ml,ml,PETSC_DECIDE,PETSC_DECIDE,pA,&A);CHKERRQ(ierr);
  ierr = BVOrthogonalize(S,A);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /* SVD of first (m-1)*L diagonal block */
  ierr = PetscBLASIntCast((ctx->M-1)*ctx->L,&m);CHKERRQ(ierr);
  ierr = PetscMalloc5(m*m,&T,m*m,&R,m*m,&U,5*ml,&work,5*ml,&rwork);CHKERRQ(ierr);
  for (j=0;j<m;j++) {
    ierr = PetscArraycpy(R+j*m,pA+j*ml,m);CHKERRQ(ierr);
  }
  lwork = 5*m;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("S","O",&m,&m,R,&m,ctx->sigma,U,&m,NULL,&m,work,&lwork,rwork,&info));
  SlepcCheckLapackInfo("gesvd",info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  (*K) = 0;
  for (i=0;i<m;i++) {
    if (ctx->sigma[i]/PetscMax(ctx->sigma[0],1)>ctx->delta) (*K)++;
  }
  ierr = MatCreateDense(PETSC_COMM_SELF,m,m,PETSC_DECIDE,PETSC_DECIDE,U,&A);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(S,0,m);CHKERRQ(ierr);
  ierr = BVMultInPlace(S,A,0,*K);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  /* Projected linear sistem */
  /* m first columns of A have the right singular vectors */
  ierr = PetscBLASIntCast(*K,&k_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ml,&lda);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&m,&k_,&m,&sone,pA+ctx->L*lda,&lda,R,&m,&zero,T,&m));
  ierr = PetscArrayzero(pA,ml*ml);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&k_,&k_,&m,&sone,U,&m,T,&m,&zero,pA,&k_));
  for (j=0;j<k_;j++) for (i=0;i<k_;i++) pA[j*k_+i] /= ctx->sigma[j];
  ierr = PetscFree5(T,R,U,work,rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  ierr = BVSetActiveColumns(ctx->S,0,ctx->M*ctx->L);CHKERRQ(ierr);
  ierr = PetscFree(ppk);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
      s1 += PetscAbsScalar(PetscPowScalarInt(pX[i*ld+j],2));
      s2 += PetscPowRealInt(PetscAbsScalar(pX[i*ld+j]),2)/ctx->sigma[j];
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

PetscErrorCode NEPSetUp_CISS(NEP nep)
{
  PetscErrorCode   ierr;
  NEP_CISS         *ctx = (NEP_CISS*)nep->data;
  PetscInt         nwork;
  PetscBool        istrivial,isellipse,flg,useconj;
  NEP_CISS_PROJECT dsctxf;

  PetscFunctionBegin;
  if (nep->ncv==PETSC_DEFAULT) nep->ncv = ctx->L_max*ctx->M;
  else {
    ctx->L_max = nep->ncv/ctx->M;
    if (!ctx->L_max) {
      ctx->L_max = 1;
      nep->ncv = ctx->L_max*ctx->M;
    }
  }
  ctx->L = PetscMin(ctx->L,ctx->L_max);
  if (nep->max_it==PETSC_DEFAULT) nep->max_it = 1;
  if (nep->mpd==PETSC_DEFAULT) nep->mpd = nep->ncv;
  if (!nep->which) nep->which = NEP_ALL;
  if (nep->which!=NEP_ALL) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver supports only computing all eigenvalues");
  NEPCheckUnsupported(nep,NEP_FEATURE_STOPPING | NEP_FEATURE_TWOSIDED);

  /* check region */
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"CISS requires a nontrivial region, e.g. -rg_type ellipse ...");
  ierr = RGGetComplement(nep->rg,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"A region with complement flag set is not allowed");
  ierr = PetscObjectTypeCompare((PetscObject)nep->rg,RGELLIPSE,&isellipse);CHKERRQ(ierr);
  if (!isellipse) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Currently only implemented for elliptic regions");

  /* if useconj has changed, then reset subcomm data */
  ierr = NEPCISSSetUseConj(nep,&useconj);CHKERRQ(ierr);
  if (useconj!=ctx->useconj) { ierr = NEPCISSResetSubcomm(nep);CHKERRQ(ierr); }

  /* create split comm */
  if (!ctx->subcomm) { ierr = NEPCISSSetUpSubComm(nep,&ctx->num_solve_point);CHKERRQ(ierr); }

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  if (ctx->weight) { ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr); }
  ierr = PetscMalloc4(ctx->N,&ctx->weight,ctx->N,&ctx->omega,ctx->N,&ctx->pp,ctx->L_max*ctx->M,&ctx->sigma);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,3*ctx->N*sizeof(PetscScalar)+ctx->L_max*ctx->N*sizeof(PetscReal));CHKERRQ(ierr);

  /* allocate basis vectors */
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDuplicateResize(nep->V,ctx->L_max*ctx->M,&ctx->S);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDuplicateResize(nep->V,ctx->L_max,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->V);CHKERRQ(ierr);

  ierr = CISSRedundantMat(nep);CHKERRQ(ierr);
  if (ctx->pA) {
    ierr = CISSScatterVec(nep);CHKERRQ(ierr);
    ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
    ierr = BVCreate(PetscObjectComm((PetscObject)ctx->xsub),&ctx->pV);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->pV,ctx->xsub,nep->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->pV);CHKERRQ(ierr);
    ierr = BVResize(ctx->pV,ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->pV);CHKERRQ(ierr);
  }

  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  if (ctx->pA) {
    ierr = BVCreate(PetscObjectComm((PetscObject)ctx->xsub),&ctx->Y);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(ctx->Y,ctx->xsub,nep->n);CHKERRQ(ierr);
    ierr = BVSetFromOptions(ctx->Y);CHKERRQ(ierr);
    ierr = BVResize(ctx->Y,ctx->num_solve_point*ctx->L_max,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = BVDuplicateResize(nep->V,ctx->num_solve_point*ctx->L_max,&ctx->Y);CHKERRQ(ierr);
  }

  if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
    ierr = DSSetType(nep->ds,DSGNHEP);CHKERRQ(ierr);
  } else if (ctx->extraction == NEP_CISS_EXTRACTION_CAA) {
    ierr = DSSetType(nep->ds,DSNHEP);CHKERRQ(ierr);
  } else {
    ierr = DSSetType(nep->ds,DSNEP);CHKERRQ(ierr);
    ierr = DSSetMethod(nep->ds,1);CHKERRQ(ierr);
    ierr = DSNEPSetRG(nep->ds,nep->rg);CHKERRQ(ierr);
    if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
      ierr = DSNEPSetFN(nep->ds,nep->nt,nep->f);CHKERRQ(ierr);
    } else {
      ierr = PetscNew(&dsctxf);CHKERRQ(ierr);
      ierr = DSNEPSetComputeMatrixFunction(nep->ds,NEPContourDSComputeMatrix,dsctxf);CHKERRQ(ierr);
      dsctxf->nep = nep;
      ctx->dsctxf = dsctxf;
    }
  }
  ierr = DSAllocate(nep->ds,nep->ncv);CHKERRQ(ierr);
  nwork = (nep->fui==NEP_USER_INTERFACE_SPLIT)? 2: 1;
  ierr = NEPSetWorkVecs(nep,nwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSolve_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  Mat            X,M,E;
  PetscInt       i,j,ld,L_add=0,nv=0,L_base=ctx->L,inner,*inside;
  PetscScalar    *Mu,*H0,*H1,*rr,*temp,center;
  PetscReal      error,max_error,radius,rgscale;
  PetscBool      *fl1;
  Vec            si;
  SlepcSC        sc;
  PetscRandom    rand;

  PetscFunctionBegin;
  ierr = DSSetFromOptions(nep->ds);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(nep->ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSGetLeadingDimension(nep->ds,&ld);CHKERRQ(ierr);
  ierr = SetPathParameter(nep);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
  ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
  ierr = BVGetRandomContext(ctx->V,&rand);CHKERRQ(ierr);
  if (ctx->pA) {
    ierr = VecScatterVecs(nep,ctx->V,ctx->L);CHKERRQ(ierr);
    ierr = SolveLinearSystem(nep,ctx->T,ctx->J,ctx->pV,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = EstimateNumberEigs(nep,&L_add);CHKERRQ(ierr);
  /* Updates L after estimate the number of eigenvalue */
  if (L_add>0) {
    ierr = PetscInfo2(nep,"Changing L %D -> %D by Estimate #Eig\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
    if (ctx->pA) {
      ierr = VecScatterVecs(nep,ctx->V,ctx->L+L_add);CHKERRQ(ierr);
      ierr = SolveLinearSystem(nep,ctx->T,ctx->J,ctx->pV,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    }
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
    ierr = BVSetActiveColumns(ctx->V,ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = BVSetRandomSign(ctx->V);CHKERRQ(ierr);
    if (ctx->pA) {
      ierr = VecScatterVecs(nep,ctx->V,ctx->L+L_add);CHKERRQ(ierr);
      ierr = SolveLinearSystem(nep,ctx->T,ctx->J,ctx->pV,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,ctx->L,ctx->L+L_add,PETSC_FALSE);CHKERRQ(ierr);
    }
    ctx->L += L_add;
  }
  ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);

  ierr = RGGetScale(nep->rg,&rgscale);CHKERRQ(ierr);
  ierr = RGEllipseGetParameters(nep->rg,&center,&radius,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc2(ctx->L*ctx->L*ctx->M*2,&Mu,ctx->L*ctx->M*ctx->L*ctx->M,&H0);CHKERRQ(ierr);
  if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
    ierr = PetscMalloc1(ctx->L*ctx->M*ctx->L*ctx->M,&H1);CHKERRQ(ierr);
  }

  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;
    for (inner=0;inner<=ctx->refine_inner;inner++) {
      if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
        ierr = CalcMu(nep,Mu);CHKERRQ(ierr);
        ierr = BlockHankel(nep,Mu,0,H0);CHKERRQ(ierr);
        ierr = SVD_H0(nep,H0,&nv);CHKERRQ(ierr);
      } else {
        ierr = ConstructS(nep);CHKERRQ(ierr);
        if (ctx->extraction == NEP_CISS_EXTRACTION_CAA) {
          /* compute svd of S and projected matrix problem */
          ierr = SVD_S_CAA(nep,ctx->S,H0,&nv);CHKERRQ(ierr);
        } else {
          ierr = SVD_S(nep,ctx->S,H0,&nv);CHKERRQ(ierr);
        }
      }
      if (ctx->sigma[0]>ctx->delta && nv==ctx->L*ctx->M && inner!=ctx->refine_inner) {
        ierr = ConstructS(nep);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,ctx->L);CHKERRQ(ierr);
        ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
        if (ctx->pA) {
          ierr = VecScatterVecs(nep,ctx->V,ctx->L);CHKERRQ(ierr);
          ierr = SolveLinearSystem(nep,ctx->T,ctx->J,ctx->pV,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
        }
      } else break;
    }
    nep->nconv = 0;
    if (nv == 0) { nep->reason = NEP_CONVERGED_TOL; break; }
    else {
      /* Extracting eigenpairs */
      ierr = DSSetDimensions(nep->ds,nv,0,0);CHKERRQ(ierr);
      ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);
      if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
        ierr = BlockHankel(nep,Mu,0,H0);CHKERRQ(ierr);
        ierr = BlockHankel(nep,Mu,1,H1);CHKERRQ(ierr);
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
      } else if (ctx->extraction == NEP_CISS_EXTRACTION_CAA) {
        ierr = DSSetDimensions(nep->ds,nv,0,0);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        ierr = DSGetArray(nep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
        for (i=0;i<nv;i++) {
          ierr = PetscArraycpy(temp+i*ld,H0+i*nv,nv);CHKERRQ(ierr);
        }
        ierr = DSRestoreArray(nep->ds,DS_MAT_A,&temp);CHKERRQ(ierr);
      } else {
        ierr = DSSetDimensions(nep->ds,nv,0,0);CHKERRQ(ierr);
        ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
        if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
          for (i=0;i<nep->nt;i++) {
            ierr = DSGetMat(nep->ds,DSMatExtra[i],&E);CHKERRQ(ierr);
            ierr = BVMatProject(ctx->S,nep->A[i],ctx->S,E);CHKERRQ(ierr);
            ierr = DSRestoreMat(nep->ds,DSMatExtra[i],&E);CHKERRQ(ierr);
          }
        } else { ctx->dsctxf->Q = ctx->S; }
      }
      ierr = DSSolve(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
      ierr = DSSynchronize(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
      ierr = DSGetDimensions(nep->ds,NULL,NULL,NULL,&nv);CHKERRQ(ierr);
      if (ctx->extraction == NEP_CISS_EXTRACTION_CAA || ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
        for (i=0;i<nv;i++) {
          nep->eigr[i] = (nep->eigr[i]*radius+center)*rgscale;
        }
      }
    }
    ierr = PetscMalloc3(nv,&fl1,nv,&inside,nv,&rr);CHKERRQ(ierr);
    ierr = isGhost(nep,ld,nv,fl1);CHKERRQ(ierr);
    if (nv) {
      ierr = RGCheckInside(nep->rg,nv,nep->eigr,nep->eigi,inside);CHKERRQ(ierr);
    }
    for (i=0;i<nv;i++) {
      if (fl1[i] && inside[i]>=0) {
        rr[i] = 1.0;
        nep->nconv++;
      } else rr[i] = 0.0;
    }
    ierr = DSSort(nep->ds,nep->eigr,nep->eigi,rr,NULL,&nep->nconv);CHKERRQ(ierr);
    ierr = DSSynchronize(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
    if (ctx->extraction == NEP_CISS_EXTRACTION_CAA || ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
      for (i=0;i<nv;i++) nep->eigr[i] = (nep->eigr[i]*radius+center)*rgscale;
    }
    ierr = PetscFree3(fl1,inside,rr);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(nep->V,0,nv);CHKERRQ(ierr);
    ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
    if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
      ierr = ConstructS(nep);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(ctx->S,0,nv);CHKERRQ(ierr);
      ierr = BVCopy(ctx->S,nep->V);CHKERRQ(ierr);
      ierr = DSGetMat(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultInPlace(ctx->S,X,0,nep->nconv);CHKERRQ(ierr);
      ierr = BVMultInPlace(nep->V,X,0,nep->nconv);CHKERRQ(ierr);
      ierr = MatDestroy(&X);CHKERRQ(ierr);
    } else {
      ierr = DSGetMat(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultInPlace(ctx->S,X,0,nep->nconv);CHKERRQ(ierr);
      ierr = MatDestroy(&X);CHKERRQ(ierr);
      ierr = BVCopy(ctx->S,nep->V);CHKERRQ(ierr);
    }
    max_error = 0.0;
    for (i=0;i<nep->nconv;i++) {
      ierr = BVGetColumn(nep->V,i,&si);CHKERRQ(ierr);
      ierr = VecNormalize(si,NULL);CHKERRQ(ierr);
      ierr = NEPComputeResidualNorm_Private(nep,PETSC_FALSE,nep->eigr[i],si,nep->work,&error);CHKERRQ(ierr);
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
      ierr = BVSetActiveColumns(ctx->V,0,ctx->L);CHKERRQ(ierr);
      ierr = BVCopy(ctx->S,ctx->V);CHKERRQ(ierr);
      if (ctx->pA) {
        ierr = VecScatterVecs(nep,ctx->V,ctx->L);CHKERRQ(ierr);
        ierr = SolveLinearSystem(nep,ctx->T,ctx->J,ctx->pV,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
      } else {
        ierr = SolveLinearSystem(nep,nep->function,nep->jacobian,ctx->V,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree2(Mu,H0);CHKERRQ(ierr);
  if (ctx->extraction == NEP_CISS_EXTRACTION_HANKEL) {
    ierr = PetscFree(H1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSSetSizes_CISS(NEP nep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool realmats)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       oN,onpart;

  PetscFunctionBegin;
  oN = ctx->N;
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
    ctx->L = bs;
  }
  if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
    ctx->M = ctx->N/4;
  } else {
    if (ms<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
    if (ms>ctx->N) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
    ctx->M = PetscMax(ms,2);
  }
  onpart = ctx->npart;
  if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
    ctx->npart = 1;
  } else {
    if (npart<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The npart argument must be > 0");
    ctx->npart = npart;
  }
  if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
    ctx->L_max = 64;
  } else {
    if (bsmax<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
    ctx->L_max = PetscMax(bsmax,ctx->L);
  }
  if (onpart != ctx->npart || oN != ctx->N || realmats != ctx->isreal) { ierr = NEPCISSResetSubcomm(nep);CHKERRQ(ierr); }
  ctx->isreal = realmats;
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSSetSizes - Sets the values of various size parameters in the CISS solver.

   Logically Collective on nep

   Input Parameters:
+  nep   - the nonlinear eigensolver context
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

   Notes:
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

static PetscErrorCode NEPCISSGetSizes_CISS(NEP nep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *realmats)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (ip) *ip = ctx->N;
  if (bs) *bs = ctx->L;
  if (ms) *ms = ctx->M;
  if (npart) *npart = ctx->npart;
  if (bsmax) *bsmax = ctx->L_max;
  if (realmats) *realmats = ctx->isreal;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSGetSizes - Gets the values of various size parameters in the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

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

/*@
   NEPCISSSetThreshold - Sets the values of various threshold parameters in
   the CISS solver.

   Logically Collective on nep

   Input Parameters:
+  nep   - the nonlinear eigensolver context
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

static PetscErrorCode NEPCISSGetThreshold_CISS(NEP nep,PetscReal *delta,PetscReal *spur)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (delta) *delta = ctx->delta;
  if (spur)  *spur = ctx->spurious_threshold;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSGetThreshold - Gets the values of various threshold parameters in
   the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

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

/*@
   NEPCISSSetRefinement - Sets the values of various refinement parameters
   in the CISS solver.

   Logically Collective on nep

   Input Parameters:
+  nep    - the nonlinear eigensolver context
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

static PetscErrorCode NEPCISSGetRefinement_CISS(NEP nep,PetscInt *inner,PetscInt *blsize)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (inner)  *inner = ctx->refine_inner;
  if (blsize) *blsize = ctx->refine_blocksize;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSGetRefinement - Gets the values of various refinement parameters
   in the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

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

static PetscErrorCode NEPCISSSetExtraction_CISS(NEP nep,NEPCISSExtraction extraction)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ctx->extraction = extraction;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSSetExtraction - Sets the extraction technique used in the CISS solver.

   Logically Collective on nep

   Input Parameters:
+  nep        - the nonlinear eigensolver context
-  extraction - the extraction technique

   Options Database Key:
.  -nep_ciss_extraction - Sets the extraction technique (either 'ritz' or 'hankel')

   Notes:
   By default, the Rayleigh-Ritz extraction is used (NEP_CISS_EXTRACTION_RITZ).

   If the 'hankel' or the 'caa' option is specified (NEP_CISS_EXTRACTION_HANKEL or
   NEP_CISS_EXTRACTION_CAA), then the Block Hankel method, or the Communication-avoiding
   Arnoldi method, respectively, is used for extracting eigenpairs.

   Level: advanced

.seealso: NEPCISSGetExtraction(), NEPCISSExtraction
@*/
PetscErrorCode NEPCISSSetExtraction(NEP nep,NEPCISSExtraction extraction)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(nep,extraction,2);
  ierr = PetscTryMethod(nep,"NEPCISSSetExtraction_C",(NEP,NEPCISSExtraction),(nep,extraction));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSGetExtraction_CISS(NEP nep,NEPCISSExtraction *extraction)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  *extraction = ctx->extraction;
  PetscFunctionReturn(0);
}

/*@
   NEPCISSGetExtraction - Gets the extraction technique used in the CISS solver.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
.  extraction - extraction technique

   Level: advanced

.seealso: NEPCISSSetExtraction() NEPCISSExtraction
@*/
PetscErrorCode NEPCISSGetExtraction(NEP nep,NEPCISSExtraction *extraction)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(extraction,2);
  ierr = PetscUseMethod(nep,"NEPCISSGetExtraction_C",(NEP,NEPCISSExtraction*),(nep,extraction));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPCISSGetKSPs_CISS(NEP nep,PetscInt *nsolve,KSP **ksp)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i;
  PC             pc;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    if (!ctx->subcomm) {  /* initialize subcomm first */
      ierr = NEPCISSSetUseConj(nep,&ctx->useconj);CHKERRQ(ierr);
      ierr = NEPCISSSetUpSubComm(nep,&ctx->num_solve_point);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(ctx->num_solve_point,&ctx->ksp);CHKERRQ(ierr);
    for (i=0;i<ctx->num_solve_point;i++) {
      ierr = KSPCreate(PetscSubcommChild(ctx->subcomm),&ctx->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp[i],(PetscObject)nep,1);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ctx->ksp[i],((PetscObject)nep)->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ctx->ksp[i],"nep_ciss_");CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptions((PetscObject)ctx->ksp[i],((PetscObject)nep)->options);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(ctx->ksp[i],PETSC_TRUE);CHKERRQ(ierr);
      ierr = KSPSetTolerances(ctx->ksp[i],SLEPC_DEFAULT_TOL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = KSPGetPC(ctx->ksp[i],&pc);CHKERRQ(ierr);
      ierr = KSPSetType(ctx->ksp[i],KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    }
  }
  if (nsolve) *nsolve = ctx->num_solve_point;
  if (ksp)    *ksp    = ctx->ksp;
  PetscFunctionReturn(0);
}

/*@C
   NEPCISSGetKSPs - Retrieve the array of linear solver objects associated with
   the CISS solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameters:
+  nsolve - number of solver objects
-  ksp - array of linear solver object

   Notes:
   The number of KSP solvers is equal to the number of integration points divided by
   the number of partitions. This value is halved in the case of real matrices with
   a region centered at the real axis.

   Level: advanced

.seealso: NEPCISSSetSizes()
@*/
PetscErrorCode NEPCISSGetKSPs(NEP nep,PetscInt *nsolve,KSP **ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPCISSGetKSPs_C",(NEP,PetscInt*,KSP**),(nep,nsolve,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPReset_CISS(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       i;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = BVDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->Y);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) {
    ierr = KSPReset(ctx->ksp[i]);CHKERRQ(ierr);
  }
  ierr = VecScatterDestroy(&ctx->scatterin);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->xsub);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->xdup);CHKERRQ(ierr);
  if (ctx->pA) {
    for (i=0;i<nep->nt;i++) {
      ierr = MatDestroy(&ctx->pA[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(ctx->pA);CHKERRQ(ierr);
    ierr = MatDestroy(&ctx->T);CHKERRQ(ierr);
    ierr = MatDestroy(&ctx->J);CHKERRQ(ierr);
    ierr = BVDestroy(&ctx->pV);CHKERRQ(ierr);
  }
  if (ctx->extraction == NEP_CISS_EXTRACTION_RITZ && nep->fui==NEP_USER_INTERFACE_CALLBACK) {
    ierr = PetscFree(ctx->dsctxf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSetFromOptions_CISS(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode    ierr;
  NEP_CISS          *ctx = (NEP_CISS*)nep->data;
  PetscReal         r1,r2;
  PetscInt          i,i1,i2,i3,i4,i5,i6,i7;
  PetscBool         b1,flg;
  NEPCISSExtraction extraction;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP CISS Options");CHKERRQ(ierr);

    ierr = NEPCISSGetSizes(nep,&i1,&i2,&i3,&i4,&i5,&b1);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_integration_points","Number of integration points","NEPCISSSetSizes",i1,&i1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_blocksize","Block size","NEPCISSSetSizes",i2,&i2,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_moments","Moment size","NEPCISSSetSizes",i3,&i3,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_partitions","Number of partitions","NEPCISSSetSizes",i4,&i4,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_maxblocksize","Maximum block size","NEPCISSSetSizes",i5,&i5,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-nep_ciss_realmats","True if T(z) is real for real z","NEPCISSSetSizes",b1,&b1,NULL);CHKERRQ(ierr);
    ierr = NEPCISSSetSizes(nep,i1,i2,i3,i4,i5,b1);CHKERRQ(ierr);

    ierr = NEPCISSGetThreshold(nep,&r1,&r2);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-nep_ciss_delta","Threshold for numerical rank","NEPCISSSetThreshold",r1,&r1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-nep_ciss_spurious_threshold","Threshold for the spurious eigenpairs","NEPCISSSetThreshold",r2,&r2,NULL);CHKERRQ(ierr);
    ierr = NEPCISSSetThreshold(nep,r1,r2);CHKERRQ(ierr);

    ierr = NEPCISSGetRefinement(nep,&i6,&i7);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_refine_inner","Number of inner iterative refinement iterations","NEPCISSSetRefinement",i6,&i6,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nep_ciss_refine_blocksize","Number of blocksize iterative refinement iterations","NEPCISSSetRefinement",i7,&i7,NULL);CHKERRQ(ierr);
    ierr = NEPCISSSetRefinement(nep,i6,i7);CHKERRQ(ierr);

    ierr = PetscOptionsEnum("-nep_ciss_extraction","Extraction technique","NEPCISSSetExtraction",NEPCISSExtractions,(PetscEnum)ctx->extraction,(PetscEnum*)&extraction,&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPCISSSetExtraction(nep,extraction);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!nep->rg) { ierr = NEPGetRG(nep,&nep->rg);CHKERRQ(ierr); }
  ierr = RGSetFromOptions(nep->rg);CHKERRQ(ierr); /* this is necessary here to set useconj */
  if (!ctx->ksp) { ierr = NEPCISSGetKSPs(nep,&ctx->num_solve_point,&ctx->ksp);CHKERRQ(ierr); }
  for (i=0;i<ctx->num_solve_point;i++) {
    ierr = KSPSetFromOptions(ctx->ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscSubcommSetFromOptions(ctx->subcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPDestroy_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = NEPCISSResetSubcomm(nep);CHKERRQ(ierr);
  ierr = PetscFree4(ctx->weight,ctx->omega,ctx->pp,ctx->sigma);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetExtraction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetExtraction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetKSPs_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPView_CISS(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscBool      isascii;
  PetscViewer    sviewer;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  sizes { integration points: %D, block size: %D, moment size: %D, partitions: %D, maximum block size: %D }\n",ctx->N,ctx->L,ctx->M,ctx->npart,ctx->L_max);CHKERRQ(ierr);
    if (ctx->isreal) {
      ierr = PetscViewerASCIIPrintf(viewer,"  exploiting symmetry of integration points\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  threshold { delta: %g, spurious threshold: %g }\n",(double)ctx->delta,(double)ctx->spurious_threshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  iterative refinement  { inner: %D, blocksize: %D }\n",ctx->refine_inner, ctx->refine_blocksize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  extraction: %s\n",NEPCISSExtractions[ctx->extraction]);CHKERRQ(ierr);
    if (!ctx->ksp) { ierr = NEPCISSGetKSPs(nep,&ctx->num_solve_point,&ctx->ksp);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (ctx->npart>1 && ctx->subcomm) {
      ierr = PetscViewerGetSubViewer(viewer,ctx->subcomm->child,&sviewer);CHKERRQ(ierr);
      if (!ctx->subcomm->color) {
        ierr = KSPView(ctx->ksp[0],sviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
      ierr = PetscViewerRestoreSubViewer(viewer,ctx->subcomm->child,&sviewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      /* extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    } else {
      ierr = KSPView(ctx->ksp[0],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode NEPCreate_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  nep->data = ctx;
  /* set default values of parameters */
  ctx->N                  = 32;
  ctx->L                  = 16;
  ctx->M                  = ctx->N/4;
  ctx->delta              = 1e-12;
  ctx->L_max              = 64;
  ctx->spurious_threshold = 1e-4;
  ctx->isreal             = PETSC_FALSE;
  ctx->npart              = 1;

  nep->useds = PETSC_TRUE;

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
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetExtraction_C",NEPCISSSetExtraction_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetExtraction_C",NEPCISSGetExtraction_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetKSPs_C",NEPCISSGetKSPs_CISS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

