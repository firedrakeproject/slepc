/*

   SLEPc eigensolver: "ciss"

   Method: Contour Integral Spectral Slicing

   Algorithm:

       Contour integral based on Sakurai-Sugiura method to construct a
       subspace, with various eigenpair extractions (Rayleigh-Ritz,
       explicit moment).

   Based on code contributed by Tetsuya Sakurai.

   References:

       [1] T. Sakurai and H. Sugiura, "A projection method for generalized
           eigenvalue problems", J. Comput. Appl. Math. 159:119-128, 2003.

       [2] T. Sakurai and H. Tadano, "CIRR: a Rayleigh-Ritz type method with
           contour integral for generalized eigenvalue problems", Hokkaido
           Math. J. 36:745-757, 2007.

   Last update: Jun 2013

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode EPSSolve_CISS(EPS);

typedef struct {
  /* parameters */
  PetscScalar center;     /* center of the region where to find eigenpairs (default: 0.0) */
  PetscReal   radius;     /* radius of the region (1.0) */
  PetscReal   vscale;     /* vertical scale of the region (1.0; 0.1 if spectrum real) */
  PetscInt    N;          /* number of integration points (32) */
  PetscInt    L;          /* block size (16) */
  PetscInt    M;          /* moment degree (N/4 = 4) */
  PetscReal   delta;      /* threshold of singular value (1e-12) */
  PetscReal   *sigma;     /* threshold for numerical rank */
  PetscInt    L_max;      /* maximum number of columns of the source matrix V */
  PetscReal   spurious_threshold; /* discard spurious eigenpairs */
  PetscBool   isreal;     /* A and B are real */
  PetscInt    refine_inner;
  PetscInt    refine_outer;
  PetscInt    refine_blocksize;
  /* private data */
  PetscInt     num_subcomm;
  PetscInt     subcomm_id;
  PetscInt     num_solve_point;
  PetscScalar  *weight;
  PetscScalar  *omega;
  PetscScalar  *pp;
  Vec          *V;
  Vec          *pV;
  Vec          *Y;
  Vec          *S;
  Vec          xsub;
  Vec          xdup;
  KSP          ksp;
  PetscBool    useconj;
  PetscReal    est_eig;
  VecScatter   scatterin;
  Mat          pA,pB;
  PetscSubcomm subcomm;
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
  if (!ctx->subcomm){
    ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)eps),&ctx->subcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(ctx->subcomm,ctx->num_subcomm);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = PetscSubcommSetType(ctx->subcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)eps,sizeof(PetscSubcomm));CHKERRQ(ierr);
  }
  ctx->subcomm_id =ctx->subcomm->color;
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
  if(ctx->subcomm->n != 1){
    ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
    ierr = MatGetRedundantMatrix(A,ctx->subcomm->n,ctx->subcomm->comm,MAT_INITIAL_MATRIX,&ctx->pA);CHKERRQ(ierr);
    if(nmat>1){
      ierr = STGetOperators(eps->st,0,&B);CHKERRQ(ierr);
      ierr = MatGetRedundantMatrix(B,ctx->subcomm->n,ctx->subcomm->comm,MAT_INITIAL_MATRIX,&ctx->pB);CHKERRQ(ierr); 
    }
    else ctx->pB = NULL;
  }
  else{
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
  PetscInt       i,j,k,mstart,mend,mlocal,subrank,subsize,rstart_sub,rend_sub,mloc_sub;
  PetscInt       *idx1,*idx2;
  const PetscInt *range;
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(ctx->subcomm->comm,&subrank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ctx->subcomm->comm,&subsize);CHKERRQ(ierr);
  ierr = VecGetOwnershipRanges(ctx->V[0],&range);CHKERRQ(ierr);//end index of mat A
  rstart_sub = range[ctx->subcomm->n*subrank]; 
  if (subrank+1 < subsize) rend_sub = range[ctx->subcomm->n*(subrank+1)];
  else rend_sub = eps->n;
  mloc_sub = rend_sub - rstart_sub;
  ierr = MatGetVecs(ctx->pA,&ctx->xsub,NULL);CHKERRQ(ierr);
  ierr = VecCreateMPI(ctx->subcomm->dupparent,mloc_sub,PETSC_DECIDE,&ctx->xdup);CHKERRQ(ierr);
  if (!ctx->scatterin) {
    ierr = VecGetOwnershipRange(ctx->V[0],&mstart,&mend);CHKERRQ(ierr);
    mlocal = mend - mstart;
    ierr = PetscMalloc2(ctx->subcomm->n*mlocal,PetscInt,&idx1,ctx->subcomm->n*mlocal,PetscInt,&idx2);CHKERRQ(ierr);
    j = 0;
    for (k=0; k<ctx->subcomm->n; k++) {
      for (i=mstart; i<mend; i++) {
	idx1[j]   = i;
	idx2[j++] = i + eps->n*k;
      }
    }
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)eps),ctx->subcomm->n*mlocal,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)eps),ctx->subcomm->n*mlocal,idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
    ierr = VecScatterCreate(ctx->V[0],is1,ctx->xdup,is2,&ctx->scatterin);CHKERRQ(ierr);
    ierr = ISDestroy(&is1);CHKERRQ(ierr);
    ierr = ISDestroy(&is2);CHKERRQ(ierr);
    PetscFree2(idx1,idx2);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetPathParameter"
static PetscErrorCode SetPathParameter(EPS eps)
{
  EPS_CISS  *ctx = (EPS_CISS*)eps->data;
  PetscInt  i;
  PetscReal theta;

  PetscFunctionBegin;
  for (i=0;i<ctx->N;i++){
    theta = ((2*PETSC_PI)/ctx->N)*(i+0.5);
    ctx->pp[i] = cos(theta) + PETSC_i*ctx->vscale*sin(theta);
    ctx->omega[i] = ctx->center + ctx->radius*ctx->pp[i];
    ctx->weight[i] = ctx->vscale*cos(theta) + PETSC_i*sin(theta);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CISSVecSetRandom"
static PetscErrorCode CISSVecSetRandom(Vec x,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscInt       j,nlocal;
  PetscScalar    *vdata;
 
  PetscFunctionBegin;
  ierr = SlepcVecSetRandom(x,rctx);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);
  ierr = VecGetArray(x,&vdata);CHKERRQ(ierr);
  for (j=0;j<nlocal;j++) {
    vdata[j] = PetscRealPart(vdata[j]);
    if (PetscRealPart(vdata[j]) < 0.5) vdata[j] = -1.0;
    else vdata[j] = 1.0;
  }
  ierr = VecRestoreArray(x,&vdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecScatterVecs"
PetscErrorCode VecScatterVecs(EPS eps, Vec *Vin, PetscInt n)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i;
  PetscScalar    *array;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    ierr = VecScatterBegin(ctx->scatterin,Vin[i],ctx->xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scatterin,Vin[i],ctx->xdup,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(ctx->xdup,&array);CHKERRQ(ierr);
    ierr = VecPlaceArray(ctx->xsub,(const PetscScalar*)array);CHKERRQ(ierr);
    ierr = VecCopy(ctx->xsub,ctx->pV[i]);
    ierr = VecResetArray(ctx->xsub);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SolveLinearSystem"
static PetscErrorCode SolveLinearSystem(EPS eps,Vec *V, PetscInt L_start,PetscInt L_end,PetscBool initksp)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j,p_id,nmat;
  Mat            A,B,Fz;
  PC             pc;
  Vec            BV;
  PetscFunctionBegin;
  if(ctx->pA != NULL){
    ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
    ierr = MatDuplicate(ctx->pA,MAT_DO_NOT_COPY_VALUES,&Fz);CHKERRQ(ierr);
    ierr = VecDuplicate(ctx->xsub,&BV);CHKERRQ(ierr);
    A = NULL;
    B = NULL;
    ierr = VecScatterVecs(eps,V,ctx->L);CHKERRQ(ierr);
  }
  else{
    ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
    ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
    if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
    else B = NULL;
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Fz);CHKERRQ(ierr);
    ierr = VecDuplicate(V[0],&BV);CHKERRQ(ierr);
  }
  for (i=0; i<ctx->num_solve_point; i++) {
    if(ctx->num_solve_point !=1 || initksp == PETSC_TRUE){
      p_id = i*ctx->subcomm->n + ctx->subcomm_id;
      if(ctx->pA != NULL){
	ierr = MatCopy(ctx->pA,Fz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	if(nmat>1) {
	  ierr = MatAXPY(Fz,-ctx->omega[p_id],ctx->pB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	} else {
	  ierr = MatShift(Fz,-ctx->omega[p_id]);CHKERRQ(ierr);
	}
      }
      else{
	ierr = MatCopy(A,Fz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	if(nmat>1) {
	  ierr = MatAXPY(Fz,-ctx->omega[p_id],B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	} else {
	  ierr = MatShift(Fz,-ctx->omega[p_id]);CHKERRQ(ierr);
	}
      }
      ierr = KSPSetOperators(ctx->ksp,Fz,Fz,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetType(ctx->ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ctx->ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCREDUNDANT);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
    }
    for (j=L_start;j<L_end;j++) {
      if(initksp == PETSC_TRUE) {
	ierr = VecDuplicate(BV,&ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
	ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
      }
      if(ctx->pA != NULL){
	if(nmat>1) {
	  ierr = MatMult(ctx->pB,ctx->pV[j],BV);CHKERRQ(ierr);
	  ierr = KSPSolve(ctx->ksp,BV,ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
	}
	else{
	  ierr = KSPSolve(ctx->ksp,ctx->pV[j],ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
	}
      }
      else{
	if(nmat>1) {
	  ierr = MatMult(B,V[j],BV);CHKERRQ(ierr);
	  ierr = KSPSolve(ctx->ksp,BV,ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
	}
	else{
	  ierr = KSPSolve(ctx->ksp,V[j],ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
	}
      }
    }
  }
  ierr = MatDestroy(&Fz);CHKERRQ(ierr);
  ierr = VecDestroy(&BV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EstimateNumberEigs"
static PetscErrorCode EstimateNumberEigs(EPS eps,PetscInt *L_add)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j;
  PetscScalar    tmp,sum = 0.0;
  PetscReal      eta;
  Vec            v,vtemp;

  PetscFunctionBegin;

  ierr = VecDuplicate(ctx->Y[0],&v);CHKERRQ(ierr);
  ierr = VecDuplicate(ctx->V[0],&vtemp);CHKERRQ(ierr);
  for (j=0;j<ctx->L;j++) {
    ierr = VecSet(v,0);CHKERRQ(ierr);
    for (i=0;i<ctx->num_solve_point; i++) {
      ierr = VecAXPY(v,ctx->weight[i*ctx->subcomm->n + ctx->subcomm_id]/(PetscReal)ctx->N,ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
    }
    if(ctx->pA != NULL){
      ierr = VecSet(vtemp,0);CHKERRQ(ierr);
      ierr = VecScatterBegin(ctx->scatterin,v,vtemp,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ctx->scatterin,v,vtemp,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecDot(ctx->V[j],vtemp,&tmp);CHKERRQ(ierr);
    }
    else{
      ierr = VecDot(ctx->V[j],v,&tmp);CHKERRQ(ierr);
    }
    if(ctx->useconj) sum += PetscRealPart(tmp)*2;
    else sum += tmp;
  }
  ctx->est_eig = PetscAbsScalar(ctx->radius*sum/(PetscReal)ctx->L);
  eta = PetscPowReal(10,-log10(eps->tol)/ctx->N);
  ierr = PetscInfo1(eps,"Estimation_#Eig %F\n",ctx->est_eig);CHKERRQ(ierr);
  *L_add = (PetscInt)ceil((ctx->est_eig*eta)/ctx->M) - ctx->L;
  if (*L_add < 0) *L_add = 0;
  if (*L_add>ctx->L_max-ctx->L) {
    ierr = PetscInfo(eps,"Number of eigenvalues around the contour path may be too large\n");
    *L_add = ctx->L_max-ctx->L;
  }
  ierr = VecDestroy(&v);
  ierr = VecDestroy(&vtemp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetAddVector"
static PetscErrorCode SetAddVector(EPS eps,PetscInt Ladd_end)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j,nlocal,Ladd_start=ctx->L;
  Vec            *newV;
  PetscScalar    *vdata;
  PetscFunctionBegin;
  ierr = PetscMalloc(Ladd_end*sizeof(Vec*),&newV);CHKERRQ(ierr);
  for (i=0;i<ctx->L;i++) { newV[i] = ctx->V[i]; }
  ierr = PetscFree(ctx->V);CHKERRQ(ierr);
  ctx->V = newV;
  ierr = VecGetLocalSize(ctx->V[0],&nlocal);CHKERRQ(ierr);
  for (i=Ladd_start;i<Ladd_end;i++) {
    ierr = VecDuplicate(ctx->V[0],&ctx->V[i]);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->V[i]);CHKERRQ(ierr);
    ierr = CISSVecSetRandom(ctx->V[i],eps->rand);CHKERRQ(ierr);
    ierr = VecGetArray(ctx->V[i],&vdata);CHKERRQ(ierr);
    for (j=0;j<nlocal;j++) {
      vdata[j] = PetscRealPart(vdata[j]);
      if (PetscRealPart(vdata[j]) < 0.5) vdata[j] = -1.0;
      else vdata[j] = 1.0;
    }
    ierr = VecRestoreArray(ctx->V[i],&vdata);CHKERRQ(ierr);
  }
  if(ctx->pA != NULL){
    ierr = VecDestroyVecs(ctx->L,&ctx->pV);
    ierr = VecDuplicateVecs(ctx->xsub,Ladd_end,&ctx->pV);CHKERRQ(ierr);
  }    
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "CalcMu"
static PetscErrorCode CalcMu(EPS eps,PetscScalar *Mu)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,s,sub_size;
  PetscScalar    *temp,*temp2,*ppk,alp;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscFunctionBegin;
  ierr = MPI_Comm_size(ctx->subcomm->comm,&sub_size);
  ierr = PetscMalloc(ctx->num_solve_point*ctx->L*ctx->L*sizeof(PetscScalar),&temp);CHKERRQ(ierr);
  ierr = PetscMalloc(2*ctx->M*ctx->L*ctx->L*sizeof(PetscScalar),&temp2);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->num_solve_point*sizeof(PetscScalar),&ppk);CHKERRQ(ierr);
  for (i=0;i<2*ctx->M*ctx->L*ctx->L;i++) temp2[i] = 0;
  for (i=0; i<ctx->num_solve_point;i++) {
    for (j=0;j<ctx->L;j++) {
      if(ctx->pA != NULL){
	ierr = VecMDot(ctx->Y[i*ctx->L_max+j],ctx->L,ctx->pV,&temp[(j+i*ctx->L)*ctx->L]);CHKERRQ(ierr);
      }
      else{
	ierr = VecMDot(ctx->Y[i*ctx->L_max+j],ctx->L,ctx->V,&temp[(j+i*ctx->L)*ctx->L]);CHKERRQ(ierr);
      }
    }
  }
  for (i=0;i<ctx->num_solve_point;i++) ppk[i] = 1;
  for (k=0;k<2*ctx->M;k++) {
    for (j=0;j<ctx->L;j++) {
      for (i=0;i<ctx->num_solve_point;i++) {
	alp = ppk[i]*ctx->weight[i*ctx->subcomm->n + ctx->subcomm_id]/(PetscReal)ctx->N;
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
  ierr = PetscFree(ppk);CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  ierr = PetscFree(temp2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "BlockHankel"
static PetscErrorCode BlockHankel(EPS eps,PetscScalar *Mu,PetscInt s,PetscScalar *H)
{
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j,k,L=ctx->L,M=ctx->M;

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
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,ml=ctx->L*ctx->M;
  PetscBLASInt   m,n,lda,ldu=ml,ldvt=ml,lwork,info;
  const char     jobu='N',jobvt='N';
  PetscReal      *rwork;
  PetscScalar    *work;

  PetscFunctionBegin;
  ierr = PetscMalloc(3*ml*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(5*ml*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  m = ml; n = m; lda = m; lwork = 3*m;
  zgesvd_(&jobu, &jobvt, &m, &n, S, &lda, ctx->sigma, NULL, &ldu, NULL, &ldvt, work, &lwork, rwork, &info);
  (*K) = 0;
  for (i=0;i<ml;i++) {
    if (ctx->sigma[i]/PetscMax(ctx->sigma[0],1)>ctx->delta) (*K)++;
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConstructS"
static PetscErrorCode ConstructS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j,k,vec_local_size,p_id;
  Vec            v;
  PetscScalar    *ppk, *v_data;
  PetscFunctionBegin;
  ierr = VecGetLocalSize(ctx->Y[0],&vec_local_size);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->num_solve_point*sizeof(PetscScalar),&ppk);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) ppk[i] = 1;
  ierr = VecDuplicate(ctx->Y[0],&v);CHKERRQ(ierr);
  for (k=0;k<ctx->M;k++) {
    for (j=0;j<ctx->L;j++) {
      ierr = VecSet(v,0);CHKERRQ(ierr);
      for (i=0;i<ctx->num_solve_point; i++) {
	p_id = i*ctx->subcomm->n + ctx->subcomm_id;
	ierr = VecAXPY(v,ppk[i]*ctx->weight[p_id]/(PetscReal)ctx->N,ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
	ppk[i] *= ctx->pp[p_id];
      }
      if (ctx->useconj) {
	ierr = VecGetArray(v,&v_data);CHKERRQ(ierr);
	for (i=0;i<vec_local_size; i++) v_data[i] = PetscRealPart(v_data[i])*2;
	ierr = VecRestoreArray(v,&v_data);CHKERRQ(ierr);
      }
      if(ctx->pA != NULL){
	ierr = VecSet(ctx->S[k*ctx->L+j],0);CHKERRQ(ierr);
	ierr = VecScatterBegin(ctx->scatterin,v,ctx->S[k*ctx->L+j],ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
	ierr = VecScatterEnd(ctx->scatterin,v,ctx->S[k*ctx->L+j],ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      }
      else{
	ierr = VecCopy(v,ctx->S[k*ctx->L+j]);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(ppk);CHKERRQ(ierr);
  ierr = VecDestroy(&v);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVD_S"
static PetscErrorCode SVD_S(EPS eps,PetscInt *K)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscInt       i,j,k;
  const PetscInt ml=ctx->L*ctx->M;
  PetscReal      *rwork;
  PetscScalar    *work, *temp, *B, *tempB;
  PetscScalar    alpha = 1,beta = 0;
  Vec            *tempQ1, *tempQ2;
  char           jobu,jobvt,transa,transb;
  PetscBLASInt   l,m,n,lda,ldu,ldvt,lwork,info,ldb,ldc;
  PetscFunctionBegin;
  ierr = PetscMalloc(3*ml*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(5*ml*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
  ierr = PetscMalloc(ml*ml*sizeof(PetscScalar),&temp);CHKERRQ(ierr);
  ierr = PetscMalloc(ml*ml*sizeof(PetscScalar),&B);CHKERRQ(ierr);
  ierr = PetscMemzero(B,ml*ml*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ml*ml*sizeof(PetscScalar),&tempB);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ctx->V[0],ctx->L*ctx->M,&tempQ1);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ctx->V[0],ctx->L*ctx->M,&tempQ2);CHKERRQ(ierr);

  for (i=0;i<ml;i++) {
    B[i*ml+i]=1;
  }
  for(k=0;k<2;k++){
    ierr = PetscMemzero(temp,ml*ml*sizeof(PetscScalar));CHKERRQ(ierr);
    for (j=0;j<ml;j++) {
      if(k == 0){ ierr = VecMDot(ctx->S[j],j+1,ctx->S,&temp[j*ml]);CHKERRQ(ierr); }
      else{ ierr = VecMDot(tempQ1[j],j+1,tempQ1,&temp[j*ml]);CHKERRQ(ierr); }
      for (i=0;i<j;i++)
	temp[j+i*ml] = PetscConj(temp[i+j*ml]);
    }
    jobu='O'; jobvt='N';
    m = (PetscBLASInt)ml; n = m; lda = m; lwork = 3*m, ldu = (PetscBLASInt)ml; ldvt = (PetscBLASInt)ml;
    zgesvd_(&jobu, &jobvt, &m, &n, temp, &lda, ctx->sigma, NULL, &ldu, NULL, &ldvt, work, &lwork, rwork, &info);
    transa='N'; transb='N'; l = ml; m = ml; n = ml; lda = l; ldb = m; ldc = l; alpha = 1; beta = 0;
    zgemm_(&transa, &transb, &l, &n, &m, &alpha, B, &lda, temp, &ldb, &beta, tempB, &ldc);
    for (j=0;j<ml;j++) {
      ctx->sigma[j] = sqrt(ctx->sigma[j]);
      if(k==0){
	ierr = VecSet(tempQ1[j],0);CHKERRQ(ierr);
	ierr = VecMAXPY(tempQ1[j],ml,&temp[j*ml],ctx->S);CHKERRQ(ierr);
	ierr = VecScale(tempQ1[j],1/ctx->sigma[j]);CHKERRQ(ierr);
      }
      else{
	ierr = VecSet(tempQ2[j],0);CHKERRQ(ierr);
	ierr = VecMAXPY(tempQ2[j],ml,&temp[j*ml],tempQ1);CHKERRQ(ierr);
	ierr = VecScale(tempQ2[j],1/ctx->sigma[j]);CHKERRQ(ierr);
      }
      for (i=0;i<ml;i++) {
	B[i+j*ml]=tempB[i+j*ml]*ctx->sigma[j];
      }
    }
  }
  jobu='N' ,jobvt='O'; m = ml; n = m; lda = m; ldu=1; ldvt=1;
  zgesvd_(&jobu, &jobvt, &m, &n, B, &lda, ctx->sigma, NULL, &ldu, NULL, &ldvt, work, &lwork, rwork, &info);
  for (j=0;j<ml;j++) {
    ierr = VecSet(ctx->S[j],0);CHKERRQ(ierr);
    ierr = VecMAXPY(ctx->S[j],ml,&B[j*ml],tempQ2);CHKERRQ(ierr);
  }
  (*K) = 0;

  PetscInt rank;
  ierr = MPI_Comm_rank((PetscObjectComm((PetscObject)eps)),&rank);CHKERRQ(ierr);
  for (i=0;i<ml;i++) {
    if (ctx->sigma[i]/PetscMax(ctx->sigma[0],1)>ctx->delta) (*K)++;
  }
  ierr = PetscFree(temp);CHKERRQ(ierr);
  ierr = PetscFree(B);CHKERRQ(ierr);
  ierr = PetscFree(tempB);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ctx->L*ctx->M,&tempQ1);
  ierr = VecDestroyVecs(ctx->L*ctx->M,&tempQ2);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProjectMatrix"
static PetscErrorCode ProjectMatrix(Mat A,PetscInt nv,PetscInt ld,Vec *Q,PetscScalar *H,Vec w,PetscBool isherm)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  if (isherm) {
    for (j=0;j<nv;j++) {
      ierr = MatMult(A,Q[j],w);CHKERRQ(ierr);
      ierr = VecMDot(w,j+1,Q,H+j*ld);CHKERRQ(ierr);
      for (i=0;i<j;i++)
        H[j+i*ld] = PetscConj(H[i+j*ld]);
    }
  } else {
    for (j=0;j<nv;j++) {
      ierr = MatMult(A,Q[j],w);CHKERRQ(ierr);
      ierr = VecMDot(w,nv,Q,H+j*ld);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
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
  ierr = PetscMalloc(nv*sizeof(PetscReal),&tau);CHKERRQ(ierr);
  ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetArray(eps->ds,DS_MAT_X,&pX);CHKERRQ(ierr);

  for (i=0;i<nv;i++) {
    s1 = 0;
    s2 = 0;
    for (j=0;j<nv;j++) {
      s1 += PetscAbsScalar(PetscPowScalar(pX[i*ld+j],2));
      s2 += PetscPowScalar(PetscAbsScalar(pX[i*ld+j]),2)/ctx->sigma[j];
    }
    tau[i] = s1/s2;
    tau_max = PetscMax(tau_max,tau[i]);
  }
  ierr = DSRestoreArray(eps->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
  for (i=0;i<nv;i++){
    tau[i] /= tau_max;
  }
  for (i=0;i<nv;i++) {
    if (tau[i]>=ctx->spurious_threshold) {
      fl[i] = PETSC_TRUE;
    } else fl[i] = PETSC_FALSE;
  }
  ierr = PetscFree(tau);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "isInsideGamma"
static PetscErrorCode isInsideGamma(EPS eps,PetscInt nv,PetscBool *fl)
{
  EPS_CISS    *ctx = (EPS_CISS*)eps->data;
  PetscInt    i;
  PetscScalar d;
  PetscReal   dx,dy;
  PetscFunctionBegin;
  for (i=0;i<nv;i++) {
    d = (eps->eigr[i]-ctx->center)/ctx->radius;
    dx = PetscRealPart(d);
    dy = PetscImaginaryPart(d);
    if ((dx*dx+(dy*dy)/(ctx->vscale*ctx->vscale))<=1) fl[i] = PETSC_TRUE;
    else fl[i] = PETSC_FALSE;
    if(fl[i]) PetscPrintf(PETSC_COMM_WORLD,"[%d]%f+%fi\n",i,eps->eigr[i]);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_CISS"
PetscErrorCode EPSSetUp_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  Vec            stemp;
  const char     *prefix;


  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"CISS only works for complex scalars");
#endif
  eps->ncv = PetscMin(eps->n,ctx->L*ctx->M);
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (!eps->which) eps->which = EPS_ALL;
  if (!eps->extraction) { ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr); } 
  else if (eps->extraction!=EPS_RITZ) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  if (ctx->isreal && PetscImaginaryPart(ctx->center) == 0.0) ctx->useconj = PETSC_TRUE;
  else ctx->useconj = PETSC_FALSE;

  if (!ctx->delta) ctx->delta = PetscMin((eps->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL*1e-1:eps->tol*1e-1),1e-12);

  if (!ctx->vscale) {
    if (eps->ishermitian && (eps->ispositive || !eps->isgeneralized) && PetscImaginaryPart(ctx->center) == 0.0) ctx->vscale = 0.1;
    else ctx->vscale = 1.0;
  }

  /* create split comm */
  ierr = SetSolverComm(eps);CHKERRQ(ierr);

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->N*sizeof(PetscScalar),&ctx->weight);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->N*sizeof(PetscScalar),&ctx->omega);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->N*sizeof(PetscScalar),&ctx->pp);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,3*ctx->N*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->L*ctx->M*sizeof(PetscReal),&ctx->sigma);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,ctx->L*ctx->N*sizeof(PetscReal));CHKERRQ(ierr);

  /* create a template vector for Vecs on solver communicator */
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)eps),PETSC_DECIDE,eps->n,&stemp); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(stemp,ctx->L,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(eps,ctx->L,ctx->V);CHKERRQ(ierr);
  ierr = VecDestroy(&stemp);CHKERRQ(ierr);

  ierr = CISSRedundantMat(eps);CHKERRQ(ierr);
  if(ctx->pA != NULL){
    ierr = CISSScatterVec(eps);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ctx->xsub,ctx->L,&ctx->pV);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)eps,ctx->L*sizeof(Vec));CHKERRQ(ierr);
  }
  ierr = KSPCreate(ctx->subcomm->comm,&ctx->ksp);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,sizeof(KSP));CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)eps,1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->ksp);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(ctx->ksp,"eps_ciss_");CHKERRQ(ierr);
  ierr = EPSGetOptionsPrefix(eps,&prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(ctx->ksp,prefix);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->num_solve_point*ctx->L_max*sizeof(Vec),&ctx->Y);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx->Y,ctx->num_solve_point*ctx->L_max*sizeof(Vec));CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,ctx->num_solve_point*ctx->L_max*sizeof(Vec));CHKERRQ(ierr);

  if (eps->isgeneralized) {
    if (eps->ishermitian && eps->ispositive) {
      ierr = DSSetType(eps->ds,DSGHEP);CHKERRQ(ierr);
    } else {
      ierr = DSSetType(eps->ds,DSGNHEP);CHKERRQ(ierr);
    }
    } else {
    if (eps->ishermitian) {
      ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
    } else {
      ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
    }
  }
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,2);CHKERRQ(ierr);
  
  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_CISS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_CISS"
PetscErrorCode EPSSolve_CISS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  Mat            A,B;
  PetscInt       i,j,ld,nmat,L_add=0,nv,L_base=ctx->L,inner,outer,nlocal;
  PetscScalar    *Mu,*H0,*H,*rr,*pX,*temp;
  PetscReal      error,max_error,tempr;
  PetscBool      *fl1,*fl2;
  Vec            w=eps->work[0];

  PetscFunctionBegin;
  ierr = VecGetLocalSize(w,&nlocal);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
  else B = NULL;
  ierr = SetPathParameter(eps);CHKERRQ(ierr);
  for (i=0;i<ctx->L;i++) {
    ierr = CISSVecSetRandom(ctx->V[i],eps->rand);CHKERRQ(ierr);
  }

  ierr = SolveLinearSystem(eps,ctx->V,0,ctx->L,PETSC_TRUE);CHKERRQ(ierr);
  ierr = EstimateNumberEigs(eps,&L_add);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"[Estimate]%f\n",ctx->est_eig);
  if (L_add>0) {
    ierr = PetscInfo2(eps,"Changing L %d -> %d by Estimate #Eig\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = SetAddVector(eps,ctx->L+L_add);CHKERRQ(ierr);
    ierr = SolveLinearSystem(eps,ctx->V,ctx->L,ctx->L+L_add,PETSC_TRUE);CHKERRQ(ierr);
    ctx->L += L_add;
    ierr = PetscFree(ctx->sigma);CHKERRQ(ierr);
    ierr = PetscMalloc(ctx->L*ctx->M*sizeof(PetscReal),&ctx->sigma);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(ctx->L*ctx->L*ctx->M*2*sizeof(PetscScalar),&Mu);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->L*ctx->M*ctx->L*ctx->M*sizeof(PetscScalar),&H0);CHKERRQ(ierr);
  for (i=0;i<ctx->refine_blocksize;i++) {
    ierr = CalcMu(eps,Mu);CHKERRQ(ierr);
    ierr = BlockHankel(eps,Mu,0,H0);CHKERRQ(ierr);
    ierr = SVD_H0(eps,H0,&nv);CHKERRQ(ierr);
    if (ctx->sigma[0]<=ctx->delta || nv < ctx->L*ctx->M || ctx->L == ctx->L_max) break;
    L_add = L_base;
    if (ctx->L+L_add>ctx->L_max) L_add = ctx->L_max-ctx->L;
    ierr = PetscInfo2(eps,"Changing L %d -> %d by SVD(H0)\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = SetAddVector(eps,ctx->L+L_add);CHKERRQ(ierr);
    ierr = SolveLinearSystem(eps,ctx->V,ctx->L,ctx->L+L_add,PETSC_TRUE);CHKERRQ(ierr);
    ctx->L += L_add;
    ierr = PetscFree(ctx->sigma);CHKERRQ(ierr);
    ierr = PetscMalloc(ctx->L*ctx->M*sizeof(PetscReal),&ctx->sigma);CHKERRQ(ierr);
  }
  ierr = PetscFree(Mu);CHKERRQ(ierr);
  ierr = PetscFree(H0);CHKERRQ(ierr);
  if (ctx->L != L_base) {
    eps->ncv = PetscMin(eps->n,ctx->L*ctx->M);
    eps->mpd = eps->ncv;
    ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
    ierr = DSReset(eps->ds);CHKERRQ(ierr);
    ierr = DSSetEigenvalueComparison(eps->ds,eps->comparison,eps->comparisonctx);CHKERRQ(ierr);
    if (eps->isgeneralized) {
      if (eps->ishermitian && eps->ispositive) {
        ierr = DSSetType(eps->ds,DSGHEP);CHKERRQ(ierr);
      } else {
        ierr = DSSetType(eps->ds,DSGNHEP);CHKERRQ(ierr);
      }
    } else {
      if (eps->ishermitian) {
        ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
      } else {
        ierr = DSSetType(eps->ds,DSNHEP);CHKERRQ(ierr);
      }
    }
    ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
    ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(ctx->V[0],ctx->M*ctx->L,&ctx->S);CHKERRQ(ierr);

  for (outer=0;outer<=ctx->refine_outer;outer++) {
    for (inner=0;inner<=ctx->refine_inner;inner++) {
      ierr = ConstructS(eps);CHKERRQ(ierr);
      ierr = SVD_S(eps,&nv);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"%d\n",nv);
      if (ctx->sigma[0]>ctx->delta && nv==ctx->L*ctx->M && inner!=ctx->refine_inner) {
	ierr = SolveLinearSystem(eps,ctx->S,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
      } else break;
    }

    eps->nconv = 0;
    if (nv == 0) break;
    ierr = DSSetDimensions(eps->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);

    if (nmat>1) {
      ierr = DSGetArray(eps->ds,DS_MAT_B,&H);CHKERRQ(ierr);
      ierr = ProjectMatrix(B,nv,ld,ctx->S,H,w,eps->ishermitian);CHKERRQ(ierr);
    }
    ierr = DSGetArray(eps->ds,DS_MAT_A,&H0);CHKERRQ(ierr);
    ierr = ProjectMatrix(A,nv,ld,ctx->S,H0,w,eps->ishermitian);CHKERRQ(ierr);
    for(i = 0; i < ld; i++){
      if (nmat>1) {
	for(j = 0; j < ld; j++){
	  H0[i*ld+j] = H0[i*ld+j] - ctx->center*H[i*ld+j];
	}
      }
      else{
	H0[i*ld+i] = H0[i*ld+i] - ctx->center*H[i*ld+j];
      }
    }
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&H0);CHKERRQ(ierr);
    if (nmat>1) {
      ierr = DSRestoreArray(eps->ds,DS_MAT_B,&H);CHKERRQ(ierr);
    }

    ierr = DSSolve(eps->ds,eps->eigr,NULL);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Eig\n");
    for (i=0;i<nv;i++) {
      eps->eigr[i]=ctx->center + ctx->radius*eps->eigr[i];
      PetscPrintf(PETSC_COMM_WORLD,"%f+%fi\n",eps->eigr[i]);
    }

    ierr = PetscMalloc(nv*sizeof(PetscBool),&fl1);CHKERRQ(ierr);
    ierr = PetscMalloc(nv*sizeof(PetscBool),&fl2);CHKERRQ(ierr);
    ierr = isGhost(eps,ld,nv,fl1);CHKERRQ(ierr);
    ierr = isInsideGamma(eps,nv,fl2);CHKERRQ(ierr);
    ierr = PetscMalloc(nv*sizeof(PetscScalar),&rr);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      if (fl1[i] && fl2[i]) {
	rr[i] = 1.0;
	eps->nconv++;
      } else rr[i] = 0.0;
    }
    ierr = PetscFree(fl1);CHKERRQ(ierr);
    ierr = PetscFree(fl2);CHKERRQ(ierr);
    ierr = DSSetEigenvalueComparison(eps->ds,SlepcCompareLargestMagnitude,NULL);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,eps->eigr,NULL,rr,NULL,&eps->nconv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"[test1]\n");
    for (i=0;i<nv;i++) {
      eps->eigr[i]=ctx->center + ctx->radius*eps->eigr[i];
      PetscPrintf(PETSC_COMM_WORLD,"%f+%fi\n",eps->eigr[i]);
    }
    ierr = DSSetEigenvalueComparison(eps->ds,eps->comparison,eps->comparisonctx);CHKERRQ(ierr);
    ierr = PetscFree(rr);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      ierr = VecCopy(ctx->S[i],eps->V[i]);CHKERRQ(ierr);
    }

    ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
    ierr = DSGetArray(eps->ds,DS_MAT_X,&pX);CHKERRQ(ierr);

    ierr = SlepcUpdateVectors(nv,ctx->S,0,eps->nconv,pX,ld,PETSC_FALSE);CHKERRQ(ierr);
    if (eps->ishermitian) {
      ierr = SlepcUpdateVectors(nv,eps->V,0,eps->nconv,pX,ld,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = DSRestoreArray(eps->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
    max_error = 0.0;
    for (i=0;i<eps->nconv;i++) {
      ierr = VecNormalize(ctx->S[i],NULL);CHKERRQ(ierr);
      ierr = EPSComputeRelativeError_Private(eps,eps->eigr[i],0,ctx->S[i],NULL,&error);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"%e\n",error);
      max_error = PetscMax(max_error,error);
    }

    PetscPrintf(PETSC_COMM_WORLD,"[test]\n");
    for (i=0;i<eps->nconv;i++) {
      PetscPrintf(PETSC_COMM_WORLD,"%f+%fi\n",eps->eigr[i]);
    }

    if (max_error <= eps->tol || outer == ctx->refine_outer) break;

    if(nv<ctx->L) nv=ctx->L;
    ierr = PetscMalloc(ctx->L*nv*sizeof(PetscScalar),&temp);CHKERRQ(ierr);
    for (i=0;i<ctx->L*nv;i++) {
      ierr = PetscRandomGetValueReal(eps->rand,&tempr);CHKERRQ(ierr);
      temp[i] = 2*tempr-1;
    }
    ierr = SlepcUpdateVectors(nv,ctx->S,0,ctx->L,temp,nv,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscFree(temp);CHKERRQ(ierr);
    ierr = SolveLinearSystem(eps,ctx->S,0,ctx->L,PETSC_FALSE);CHKERRQ(ierr);
  }
  eps->reason = EPS_CONVERGED_TOL;
  
  ierr = VecDestroyVecs(ctx->L*ctx->M,&ctx->S);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetRegion_CISS"
static PetscErrorCode EPSCISSSetRegion_CISS(EPS eps,PetscScalar center,PetscReal radius,PetscReal vscale)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ctx->center = center;
  if (radius) {
    if (radius == PETSC_DEFAULT) {
      ctx->radius = 1.0;
    } else {
      if (radius<0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The radius argument must be > 0.0");
      ctx->radius = radius;
    }
  }
  if (vscale) {
    if (vscale<0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The vscale argument must be > 0.0");
    ctx->vscale = vscale;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetRegion"
/*@
   EPSCISSSetRegion - Sets the parameters defining the region where eigenvalues
   must be computed.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  center - center of the region
.  radius - radius of the region
-  vscale - vertical scale of the region

   Options Database Keys:
+  -eps_ciss_center - Sets the center
.  -eps_ciss_radius - Sets the radius
-  -eps_ciss_vscale - Sets the vertical scale

   Level: advanced

.seealso: EPSCISSGetRegion()
@*/
PetscErrorCode EPSCISSSetRegion(EPS eps,PetscScalar center,PetscReal radius,PetscReal vscale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveScalar(eps,center,2);
  PetscValidLogicalCollectiveReal(eps,radius,3);
  PetscValidLogicalCollectiveReal(eps,vscale,4);
  ierr = PetscTryMethod(eps,"EPSCISSSetRegion_C",(EPS,PetscScalar,PetscReal,PetscReal),(eps,center,radius,vscale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetRegion_CISS"
static PetscErrorCode EPSCISSGetRegion_CISS(EPS eps,PetscScalar *center,PetscReal *radius,PetscReal *vscale)
{
  EPS_CISS *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (center) *center = ctx->center;
  if (radius) *radius = ctx->radius;
  if (vscale) *vscale = ctx->vscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSGetRegion"
/*@
   EPSCISSGetRegion - Gets the parameters that define the region where eigenvalues
   must be computed.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
+  center - center of the region
.  radius - radius of the region
-  vscale - vertical scale of the region

   Level: advanced

.seealso: EPSCISSSetRegion()
@*/
PetscErrorCode EPSCISSGetRegion(EPS eps,PetscScalar *center,PetscReal *radius,PetscReal *vscale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSCISSGetRegion_C",(EPS,PetscScalar*,PetscReal*,PetscReal*),(eps,center,radius,vscale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCISSSetSizes_CISS"
static PetscErrorCode EPSCISSSetSizes_CISS(EPS eps,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool isreal)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  if (ip) {
    if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) {
      if (ctx->N!=32) { ctx->N =32; ctx->M = ctx->N/4; }
    } else {
      if (ip<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be > 0");
      if (ip%2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be an even number");
      if (ctx->N!=ip) { ctx->N = ip; ctx->M = ctx->N/4; }
    }
  }
  if (bs) {
    if (bs == PETSC_DECIDE || bs == PETSC_DEFAULT) {
      ctx->L = 16;
    } else {
      if (bs<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be > 0");
      if (bs>ctx->L_max) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be less than or equal to the maximum number of block size");
      ctx->L = bs;
    }
  }
  if (ms) {
    if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
      ctx->M = ctx->N/4;
    } else {
      if (ms<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
      if (ms>ctx->N) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
      ctx->M = ms;
    }
  }
  if (npart) {
    if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
      ctx->num_subcomm = 1;
    } else {
      if (npart<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The npart argument must be > 0");
      ctx->num_subcomm = npart;
    }
  }
  if (bsmax) {
    if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
      ctx->L = 256;
    } else {
      if (bsmax<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
      if (bsmax<ctx->L) ctx->L_max = ctx->L;
      else ctx->L_max = bsmax;
    }
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
  if (delta) {
    if (delta == PETSC_DEFAULT) {
      ctx->delta = 1e-12;
    } else {
      if (delta<=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The delta argument must be > 0.0");
      ctx->delta = delta;
    }
  }
  if (spur) {
    if (spur == PETSC_DEFAULT) {
      ctx->spurious_threshold = 1e-4;
    } else {
      if (spur<=0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The spurious threshold argument must be > 0.0");
      ctx->spurious_threshold = spur;
    }
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
#define __FUNCT__ "EPSReset_CISS"
PetscErrorCode EPSReset_CISS(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;

  PetscFunctionBegin;
  ierr = PetscSubcommDestroy(&ctx->subcomm);CHKERRQ(ierr);
  ierr = PetscFree(ctx->weight);CHKERRQ(ierr);
  ierr = PetscFree(ctx->omega);CHKERRQ(ierr);
  ierr = PetscFree(ctx->pp);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ctx->L,&ctx->V);CHKERRQ(ierr);
  ierr = PetscFree(ctx->sigma);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point*ctx->L_max;i++) {
    ierr = VecDestroy(&ctx->Y[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->Y);CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx->scatterin);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->xsub);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->xdup);CHKERRQ(ierr);
  if(ctx->pA != NULL){
    ierr = MatDestroy(&ctx->pA);CHKERRQ(ierr);
    ierr = MatDestroy(&ctx->pB);CHKERRQ(ierr);
    ierr = VecDestroyVecs(ctx->L,&ctx->pV);CHKERRQ(ierr);
  }
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_CISS"
PetscErrorCode EPSSetFromOptions_CISS(EPS eps)
{
  PetscErrorCode ierr;
  PetscScalar    s;
  PetscReal      r1,r2,r3,r4;
  PetscInt       i1=0,i2=0,i3=0,i4=0,i5=0,i6=0,i7=0,i8=0;
  PetscBool      b1=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS CISS Options");CHKERRQ(ierr);
  ierr = EPSCISSGetRegion(eps,&s,&r1,&r2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_ciss_radius","CISS radius of region","EPSCISSSetRegion",r1,&r1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-eps_ciss_center","CISS center of region","EPSCISSSetRegion",s,&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_ciss_vscale","CISS vertical scale of region","EPSCISSSetRegion",r2,&r2,NULL);CHKERRQ(ierr);
  ierr = EPSCISSSetRegion(eps,s,r1,r2);CHKERRQ(ierr);

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
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRegion_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRegion_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRefinement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRefinement_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_CISS"
PetscErrorCode EPSView_CISS(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_CISS       *ctx = (EPS_CISS*)eps->data;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = SlepcSNPrintfScalar(str,50,ctx->center,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: region { center: %s, radius: %G, vscale: %G }\n",str,ctx->radius,ctx->vscale);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: sizes { integration points: %D, block size: %D, moment size: %D, maximum block size: %D, partitions: %D}\n",ctx->N,ctx->L,ctx->M,ctx->L_max,ctx->num_subcomm);CHKERRQ(ierr);
    if (ctx->isreal) {
      ierr = PetscViewerASCIIPrintf(viewer,"  CISS: exploiting symmetry of integration points\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: threshold { delta: %G, spurious threshold: %G }\n",ctx->delta,ctx->spurious_threshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: iterative refinement  { inner: %D, outer: %D, blocksize: %D }\n",ctx->refine_inner,ctx->refine_outer, ctx->refine_blocksize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    //ierr = KSPView(ctx->ksp,viewer);CHKERRQ(ierr);
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
  ierr = PetscNewLog(eps,EPS_CISS,&ctx);CHKERRQ(ierr);
  eps->data = ctx;
  eps->ops->setup          = EPSSetUp_CISS;
  eps->ops->setfromoptions = EPSSetFromOptions_CISS;
  eps->ops->destroy        = EPSDestroy_CISS;
  eps->ops->reset          = EPSReset_CISS;
  eps->ops->view           = EPSView_CISS;
  eps->ops->backtransform  = PETSC_NULL;
  eps->ops->computevectors = EPSComputeVectors_Schur;
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRegion_C",EPSCISSSetRegion_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRegion_C",EPSCISSGetRegion_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetSizes_C",EPSCISSSetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetSizes_C",EPSCISSGetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetThreshold_C",EPSCISSSetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetThreshold_C",EPSCISSGetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSSetRefinement_C",EPSCISSSetRefinement_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSCISSGetRefinement_C",EPSCISSGetRefinement_CISS);CHKERRQ(ierr);
  /* set default values of parameters */
  ctx->center  = 0.0;
  ctx->radius  = 1.0;
  ctx->vscale  = 0.0;
  ctx->N       = 32;
  ctx->L       = 16;
  ctx->M       = ctx->N/4;
  ctx->delta   = 0;
  ctx->L_max   = 128;
  ctx->spurious_threshold = 1e-4;
  ctx->isreal  = PETSC_FALSE;
  ctx->refine_outer = 1;
  ctx->refine_inner = 1;
  ctx->refine_blocksize = 1;
  ctx->num_subcomm = 1;
  PetscFunctionReturn(0);
}

