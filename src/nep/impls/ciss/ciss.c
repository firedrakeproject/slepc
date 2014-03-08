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

#include <slepc-private/nepimpl.h>                /*I "slepcnep.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode NEPSolve_CISS(NEP);

typedef struct {
  /* parameters */
  PetscScalar center;     /* center of the region where to find eigenpairs (default: 0.0) */
  PetscReal   radius;     /* radius of the region (1.0) */
  PetscReal   vscale;     /* vertical scale of the region (1.0; 0.1 if spectrum real) */
  PetscInt    N;          /* number of integration points (32) */
  PetscInt    L;          /* block size (16) */
  PetscInt    M;          /* moment degree (N/4 = 4) */
  PetscReal   delta;      /* threshold of singular value (1e-12) */
  PetscInt    npart;      /* number of partitions of the matrix (1) */
  PetscReal   *sigma;     /* threshold for numerical rank */
  PetscInt    L_max;      /* maximum number of columns of the source matrix V */
  PetscReal   spurious_threshold; /* discard spurious eigenpairs */
  PetscBool   isreal;     /* A and B are real */
  /* private data */
  MPI_Comm    scomm;
  PetscInt    solver_comm_id;
  PetscInt    num_solve_point;
  PetscScalar *weight;
  PetscScalar *omega;
  PetscScalar *pp;
  Vec         *V;
  Vec         *Y;
  Vec         *S;
  KSP         *ksp;
  PetscBool   useconj;
} NEP_CISS;

#undef __FUNCT__
#define __FUNCT__ "SetSolverComm"
static PetscErrorCode SetSolverComm(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       N = ctx->N;
  PetscInt       size_solver = ctx->npart;
  PetscInt       size_region,rank_region,icolor,ikey,num_solver_comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)nep),&size_region);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)nep),&rank_region);CHKERRQ(ierr);

  if (ctx->useconj) N = N/2;
  num_solver_comm = size_region / size_solver;
  if (num_solver_comm > N) {
    size_solver = size_region / N;
    ierr = PetscInfo2(nep,"CISS changed size_solver %D --> %D\n",ctx->npart,size_solver);CHKERRQ(ierr);
    ctx->npart = size_solver;
  }

  icolor = rank_region / size_solver;
  ikey = rank_region % size_solver;
  ierr = MPI_Comm_split(PetscObjectComm((PetscObject)nep),icolor,ikey,&ctx->scomm);CHKERRQ(ierr);

  ctx->solver_comm_id = icolor;
  num_solver_comm = size_region / size_solver;
  ctx->num_solve_point = N / num_solver_comm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetPathParameter"
static PetscErrorCode SetPathParameter(NEP nep)
{
  NEP_CISS  *ctx = (NEP_CISS*)nep->data;
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
#define __FUNCT__ "SolveLinearSystem"
static PetscErrorCode SolveLinearSystem(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,p_id;
  Mat            Fz,dFz;
  PC             pc;
  Vec            BV;
  Mat            T=nep->function, dT=nep->jacobian;
  MatStructure   mats;

  PetscFunctionBegin;
  ierr = VecDuplicate(ctx->V[0],&BV);CHKERRQ(ierr);

  for (i=0;i<ctx->num_solve_point;i++) {
    p_id = ctx->solver_comm_id * ctx->num_solve_point + i;

    ierr = NEPComputeFunction(nep,ctx->omega[p_id],&T,&T,&mats);CHKERRQ(ierr);
    if (i == 0){ ierr = MatDuplicate(T,MAT_COPY_VALUES,&Fz);CHKERRQ(ierr); }
    else{ ierr = MatCopy(T,Fz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); }

    ierr = NEPComputeJacobian(nep,ctx->omega[p_id],&dT,&mats);CHKERRQ(ierr);
    if (i == 0){ ierr = MatDuplicate(dT,MAT_COPY_VALUES,&dFz);CHKERRQ(ierr); }
    else{ ierr = MatCopy(dT,dFz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); }

    ierr = KSPSetOperators(ctx->ksp[i],Fz,Fz);CHKERRQ(ierr);
    ierr = KSPSetType(ctx->ksp[i],KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ctx->ksp[i],&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ctx->ksp[i]);CHKERRQ(ierr);
    for (j=0;j<ctx->L;j++) {
      ierr = VecDuplicate(ctx->V[0],&ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
      ierr = MatMult(dFz,ctx->V[j],BV);CHKERRQ(ierr);
      ierr = KSPSolve(ctx->ksp[i],BV,ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&Fz);CHKERRQ(ierr);
  ierr = MatDestroy(&dFz);CHKERRQ(ierr);
  ierr = VecDestroy(&BV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConstructS"
static PetscErrorCode ConstructS(NEP nep,PetscInt M,Vec **S)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,k,s,t;
  PetscInt       rank_region,icolor,ikey,rank_row_comm,size_row_comm;
  PetscInt       vec_local_size,row_vec_local_size;
  PetscInt       *array_row_vec_local_size,*array_start_id;
  MPI_Comm       Row_Comm,Vec_Comm;
  Vec            v,row_vec,z;
  PetscScalar    *ppk,*S_data,*v_data,*send;

  PetscFunctionBegin;
  /* 1 */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)nep),&rank_region);CHKERRQ(ierr);
  icolor = rank_region % ctx->npart;
  ikey = rank_region / ctx->npart;
  ierr = MPI_Comm_split(PetscObjectComm((PetscObject)nep),icolor,ikey,&Row_Comm);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(Row_Comm,&rank_row_comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(Row_Comm,&size_row_comm);CHKERRQ(ierr);

  ikey = size_row_comm * icolor + rank_row_comm;
  icolor = 0;
  ierr = MPI_Comm_split(PetscObjectComm((PetscObject)nep),icolor,ikey,&Vec_Comm);CHKERRQ(ierr);

  /* 2 */
  ierr = VecGetLocalSize(ctx->Y[0],&vec_local_size);CHKERRQ(ierr);
  ierr = VecCreateMPI(Row_Comm,PETSC_DECIDE,vec_local_size,&row_vec);CHKERRQ(ierr);
  ierr = VecGetLocalSize(row_vec,&row_vec_local_size);CHKERRQ(ierr);
  ierr = VecDestroy(&row_vec);CHKERRQ(ierr);

  ierr = VecCreateMPI(Vec_Comm,row_vec_local_size,nep->n,&z);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(z,M*ctx->L,S);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);

  ierr = PetscMalloc(size_row_comm*sizeof(PetscInt),&array_row_vec_local_size);CHKERRQ(ierr);
  ierr = PetscMalloc((size_row_comm+1)*sizeof(PetscInt),&array_start_id);CHKERRQ(ierr);
  ierr = MPI_Allgather(&row_vec_local_size,1,MPI_INT,array_row_vec_local_size,1,MPI_INT,Row_Comm);CHKERRQ(ierr);
  array_start_id[0] = 0;
  for (i=1;i<size_row_comm+1;i++) {
    array_start_id[i] = array_start_id[i-1] + array_row_vec_local_size[i-1];
  }

  /* 3 */
  ierr = PetscMalloc(ctx->num_solve_point*sizeof(PetscScalar),&ppk);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) ppk[i] = 1;
  ierr = VecDuplicate(ctx->Y[0],&v);CHKERRQ(ierr);
  for (k=0;k<M;k++) {
    for (j=0;j<ctx->L;j++) {
      ierr = VecSet(v,0);CHKERRQ(ierr);
      for (i=0;i<ctx->num_solve_point; i++) {
        ierr = VecAXPY(v,ppk[i]*ctx->weight[ctx->solver_comm_id*ctx->num_solve_point+i]/(PetscReal)ctx->N,ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
      }
      ierr = VecGetArray((*S)[k*ctx->L+j],&S_data);CHKERRQ(ierr);
      ierr = VecGetArray(v,&v_data);CHKERRQ(ierr);
      for (i=0;i<size_row_comm;i++) {
        ierr = PetscMalloc(array_row_vec_local_size[i]*sizeof(PetscScalar),&send);CHKERRQ(ierr);
        for (s=array_start_id[i],t=0;s<array_start_id[i+1];s++,t++) {
          if (ctx->useconj) send[t] = PetscRealPart(v_data[s])*2;
          else send[t] = v_data[s];
        }
        ierr = MPI_Reduce(send,S_data,array_row_vec_local_size[i],MPIU_SCALAR,MPIU_SUM,i,Row_Comm);CHKERRQ(ierr);
        ierr = PetscFree(send);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray((*S)[k*ctx->L+j],&S_data);CHKERRQ(ierr);
      ierr = VecRestoreArray(v,&v_data);CHKERRQ(ierr);
    }
    for (i=0;i<ctx->num_solve_point;i++) {
      ppk[i] *= ctx->pp[ctx->solver_comm_id*ctx->num_solve_point+i];
    }
  }

  ierr = PetscFree(array_row_vec_local_size);CHKERRQ(ierr);
  ierr = PetscFree(array_start_id);CHKERRQ(ierr);
  ierr = PetscFree(ppk);CHKERRQ(ierr);
  ierr = VecDestroy(&v);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EstimateNumberEigs"
static PetscErrorCode EstimateNumberEigs(NEP nep,Vec *S1,PetscInt *L_add)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,istart,p_start,p_end;
  PetscScalar    *data,*p_data,tmp,sum = 0.0;
  Vec            V_p;
  PetscReal      eta,est_eig;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(ctx->V[0],&istart,NULL);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(S1[0],&p_start,&p_end);CHKERRQ(ierr);

  ierr = VecDuplicate(S1[0],&V_p);CHKERRQ(ierr);
  for (i=0;i<ctx->L;i++) {
    ierr = VecGetArray(ctx->V[i],&data);CHKERRQ(ierr);
    ierr = VecGetArray(V_p,&p_data);CHKERRQ(ierr);
    for (j=p_start;j<p_end;j++) p_data[j-p_start] = data[j-istart];
    ierr = VecRestoreArray(ctx->V[i],&data);CHKERRQ(ierr);
    ierr = VecRestoreArray(V_p,&p_data);CHKERRQ(ierr);
    ierr = VecDot(V_p,S1[i],&tmp);CHKERRQ(ierr);
    sum += tmp;
  }
  ierr = VecDestroy(&V_p);CHKERRQ(ierr);
  est_eig = PetscAbsScalar(ctx->radius*sum/(PetscReal)ctx->L);
  eta = PetscPowReal(10,-log10(nep->rtol)/ctx->N);
  ierr = PetscInfo1(nep,"Estimation_#Eig %F\n",est_eig);CHKERRQ(ierr);
  *L_add = (PetscInt)ceil((est_eig*eta)/ctx->M) - ctx->L;
  if (*L_add < 0) *L_add = 0;
  if (*L_add>ctx->L_max-ctx->L) {
    ierr = PetscInfo(nep,"Number of eigenvalues around the contour path may be too large\n");
    *L_add = ctx->L_max-ctx->L;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetAddVector"
static PetscErrorCode SetAddVector(NEP nep,PetscInt Ladd_end)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
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
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->V[i]);CHKERRQ(ierr);
    ierr = CISSVecSetRandom(ctx->V[i],nep->rand);CHKERRQ(ierr);
    ierr = VecGetArray(ctx->V[i],&vdata);CHKERRQ(ierr);
    for (j=0;j<nlocal;j++) vdata[j] = PetscRealPart(vdata[j]);
    ierr = VecRestoreArray(ctx->V[i],&vdata);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SolveAddLinearSystem"
static PetscErrorCode SolveAddLinearSystem(NEP nep,PetscInt Ladd_end)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,Ladd_start=ctx->L;

  PetscFunctionBegin;
  for (i=0;i<ctx->num_solve_point;i++) {
    for (j=Ladd_start;j<Ladd_end;j++) {
      ierr = VecDuplicate(ctx->V[0],&ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
      ierr = KSPSolve(ctx->ksp[i],ctx->V[j],ctx->Y[i*ctx->L_max+j]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "CalcMu"
static PetscErrorCode CalcMu(NEP nep, PetscScalar *Mu)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,s;
  PetscInt       rank_region,icolor,ikey;
  PetscScalar    *temp,*temp2,*ppk,alp;
  MPI_Comm       Row_Comm;
  NEP_CISS      *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)nep),&rank_region);CHKERRQ(ierr);
  icolor = rank_region % ctx->npart;
  ikey = rank_region / ctx->npart;
  ierr = MPI_Comm_split(PetscObjectComm((PetscObject)nep),icolor,ikey,&Row_Comm);CHKERRQ(ierr);


  ierr = PetscMalloc(ctx->num_solve_point*ctx->L*ctx->L*sizeof(PetscScalar),&temp);CHKERRQ(ierr);
  ierr = PetscMalloc(2*ctx->M*ctx->L*ctx->L*sizeof(PetscScalar),&temp2);CHKERRQ(ierr);
  for (i=0;i<2*ctx->M*ctx->L*ctx->L; i++) temp2[i] = 0;
  ierr = PetscMalloc(ctx->num_solve_point*sizeof(PetscScalar),&ppk);CHKERRQ(ierr);
  for (i=0; i<ctx->num_solve_point; i++) {
    for (j=0;j<ctx->L;j++) {
      ierr = VecMDot(ctx->Y[i*ctx->L_max+j],ctx->L,ctx->V,&temp[(j+i*ctx->L)*ctx->L]);CHKERRQ(ierr);
    }
  }
  for (i=0;i<ctx->num_solve_point;i++) ppk[i] = 1;
  for (k=0;k<2*ctx->M;k++) {
    for (j=0;j<ctx->L;j++) {
      for (i=0;i<ctx->num_solve_point; i++) {
	alp = ppk[i]*ctx->weight[ctx->solver_comm_id*ctx->num_solve_point+i]/(PetscReal)ctx->N;
	for (s=0;s<ctx->L;s++){
	  if (ctx->useconj) temp2[s+(j+k*ctx->L)*ctx->L] += PetscRealPart(alp*temp[s+(j+i*ctx->L)*ctx->L])*2;
	  else temp2[s+(j+k*ctx->L)*ctx->L] += alp*temp[s+(j+i*ctx->L)*ctx->L];
	}
      }
    }
    for (i=0;i<ctx->num_solve_point;i++) 
      ppk[i] *= ctx->pp[ctx->solver_comm_id*ctx->num_solve_point+i];
  }
  ierr = MPI_Allreduce(temp2,Mu,2*ctx->M*ctx->L*ctx->L,MPIU_SCALAR,MPIU_SUM,Row_Comm);CHKERRQ(ierr);

  ierr = PetscFree(ppk);CHKERRQ(ierr);
  ierr = PetscFree(temp);CHKERRQ(ierr);
  ierr = PetscFree(temp2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BlockHankel"
static PetscErrorCode BlockHankel(NEP nep, PetscScalar *Mu,PetscInt s,Vec *H)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;
  PetscInt  i,j,k,L=ctx->L,M=ctx->M;
  PetscScalar *H_data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (k=0; k<L*M; k++) {
    ierr = VecGetArray(H[k],&H_data);CHKERRQ(ierr);
    for (j=0; j<M; j++) 
      for (i=0; i<L; i++) 
	H_data[j*L+i]=Mu[i+k*L+(j+s)*L*L];
    ierr = VecRestoreArray(H[k],&H_data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SVD"
static PetscErrorCode SVD(NEP nep,Vec *Q,PetscInt *K,PetscScalar *V,PetscBool isqr)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,ld,ml=ctx->M*ctx->L,n=nep->n;
  PetscScalar    *R,*w,*s;
  DS             ds;

  PetscFunctionBegin;

  if (isqr){
    ierr = PetscMalloc(ml*ml*sizeof(PetscScalar),&s);CHKERRQ(ierr);
    ierr = IPQRDecomposition(nep->ip,Q,0,ml,s,ml);CHKERRQ(ierr);
  }

  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSSVD);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);
  ld = ml;
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  if(n<ml){ ierr = DSSetDimensions(ds,n,ml,0,0);CHKERRQ(ierr);}
  else{ ierr = DSSetDimensions(ds,ml,ml,0,0);CHKERRQ(ierr);}
  ierr = DSGetArray(ds,DS_MAT_A,&R);CHKERRQ(ierr);

  if (isqr){
    for (i=0;i<ml;i++) 
      for (j=0;j<ml;j++) 
	R[i*ld+j] = s[i*ml+j];
  }else{
    for (i=0;i<ml;i++) {
      ierr = VecGetArray(Q[i],&s);CHKERRQ(ierr);
      for (j=0;j<ml;j++) {
	R[i*ld+j] = s[j];
      }
      ierr = VecRestoreArray(Q[i],&s);CHKERRQ(ierr);
    }
  }
  ierr = DSRestoreArray(ds,DS_MAT_A,&R);CHKERRQ(ierr);
  if (isqr) ierr = PetscFree(s);CHKERRQ(ierr);
  ierr = DSSetState(ds,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = PetscMalloc(ml*sizeof(PetscScalar),&w);CHKERRQ(ierr);
  ierr = DSSetEigenvalueComparison(ds,SlepcCompareLargestReal,NULL);CHKERRQ(ierr);
  ierr = DSSolve(ds,w,NULL);CHKERRQ(ierr);
  ierr = DSSort(ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  (*K) = 0;
  for (i=0;i<ml;i++) {
    ctx->sigma[i] = PetscRealPart(w[i]);
    if (ctx->sigma[i]/PetscMax(ctx->sigma[0],1)>ctx->delta) (*K)++;
  }

  ierr = DSGetArray(ds,DS_MAT_VT,&R);CHKERRQ(ierr);
  for (i=0;i<(*K);i++) 
    for (j=0;j<(*K);j++) 
      V[i*ml+j] = R[i*ld+j];
  ierr = DSRestoreArray(ds,DS_MAT_VT,&R);CHKERRQ(ierr);

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "isInsideGamma"
static PetscErrorCode isInsideGamma(NEP nep,PetscInt nv,PetscBool *fl)
{
  NEP_CISS    *ctx = (NEP_CISS*)nep->data;
  PetscInt    i;
  PetscScalar d;
  PetscReal   dx,dy;
  for (i=0;i<nv;i++) {
    d = (nep->eig[i]-ctx->center)/ctx->radius;
    dx = PetscRealPart(d);
    dy = PetscImaginaryPart(d);
    if ((dx*dx+(dy*dy)/(ctx->vscale*ctx->vscale))<=1) fl[i] = PETSC_TRUE;
    else fl[i] = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_CISS"
PetscErrorCode NEPSetUp_CISS(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            stemp;
  NEP_CISS      *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"CISS only works for complex scalars");
#endif
  nep->ncv = ctx->L*ctx->M;
  if (!nep->mpd) nep->mpd = nep->ncv;
  if (!nep->which) nep->which = NEP_LARGEST_MAGNITUDE;
  //if (ctx->vscale == PETSC_DEFAULT) {
  //  if (nep->ishermitian && (nep->ispositive || !nep->isgeneralized) && PetscImaginaryPart(ctx->center) == 0.0) ctx->vscale = 0.1;
  //  else ctx->vscale = 1.0;
  //}
  if (ctx->isreal && PetscImaginaryPart(ctx->center) == 0.0) ctx->useconj = PETSC_TRUE;
  else ctx->useconj = PETSC_FALSE;

  /* create split comm */
  ierr = SetSolverComm(nep);CHKERRQ(ierr);

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->N*sizeof(PetscScalar),&ctx->weight);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->N*sizeof(PetscScalar),&ctx->omega);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->N*sizeof(PetscScalar),&ctx->pp);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,3*ctx->N*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->L*ctx->M*sizeof(PetscReal),&ctx->sigma);CHKERRQ(ierr);

  /* create a template vector for Vecs on solver communicator */
  ierr = VecCreateMPI(ctx->scomm,PETSC_DECIDE,nep->n,&stemp); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(stemp,ctx->L,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(nep,ctx->L,ctx->V);CHKERRQ(ierr);
  ierr = VecDestroy(&stemp);CHKERRQ(ierr);

  ierr = PetscMalloc(ctx->num_solve_point*sizeof(KSP),&ctx->ksp);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,ctx->num_solve_point*sizeof(KSP));CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) {
    ierr = KSPCreate(ctx->scomm,&ctx->ksp[i]);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(ctx->num_solve_point*ctx->L_max*sizeof(Vec),&ctx->Y);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,ctx->num_solve_point*ctx->L_max*sizeof(Vec));CHKERRQ(ierr);
  ierr = DSSetType(nep->ds,DSGNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->ncv);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,1);CHKERRQ(ierr);

  /* dispatch solve method */
  //if (nep->leftvecs) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Left vectors not supported in this solver");
  nep->ops->solve = NEPSolve_CISS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_CISS"
PetscErrorCode NEPSolve_CISS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscInt       i,j,k,ld,nv,L_add=0;
  PetscScalar    *H,*rr,*pX,*Mu,*H0d,*SVD_V,*alpha,*temp;
  PetscReal      *tau,s1,s2,tau_max=1.0,norm;
  PetscBool      *fl;
  Vec            *S1,*H0,*H1,tempv;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(nep->ds,&ld);CHKERRQ(ierr);

  ierr = SetPathParameter(nep);CHKERRQ(ierr);
  for (i=0;i<ctx->L;i++) {
    ierr = CISSVecSetRandom(ctx->V[i],nep->rand);CHKERRQ(ierr);
  }
  ierr = SolveLinearSystem(nep);CHKERRQ(ierr);
  ierr = ConstructS(nep,1,&S1);CHKERRQ(ierr);
  ierr = EstimateNumberEigs(nep,S1,&L_add);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ctx->L,&S1);CHKERRQ(ierr);

  if (L_add>0) {
    ierr = PetscInfo2(nep,"Changing L %d -> %d\n",ctx->L,ctx->L+L_add);CHKERRQ(ierr);
    ierr = SetAddVector(nep,ctx->L+L_add);CHKERRQ(ierr);
    ierr = SolveAddLinearSystem(nep,ctx->L+L_add);CHKERRQ(ierr);

    ctx->L += L_add;
    nep->ncv = ctx->L*ctx->M;
    nep->mpd = nep->ncv;
    ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
    ierr = DSReset(nep->ds);CHKERRQ(ierr);
    ierr = DSSetEigenvalueComparison(nep->ds,nep->comparison,nep->comparisonctx);CHKERRQ(ierr);
    ierr = DSSetType(nep->ds,DSGNHEP);CHKERRQ(ierr);
    ierr = DSAllocate(nep->ds,nep->ncv);CHKERRQ(ierr);
    ierr = NEPSetWorkVecs(nep,1);CHKERRQ(ierr);
    ierr = DSGetLeadingDimension(nep->ds,&ld);CHKERRQ(ierr);
    ierr = PetscFree(ctx->sigma);CHKERRQ(ierr);
    ierr = PetscMalloc(ctx->L*ctx->M*sizeof(PetscReal),&ctx->sigma);CHKERRQ(ierr);
  }
  ierr = ConstructS(nep,ctx->M,&ctx->S);CHKERRQ(ierr);

  ierr = PetscMalloc(ctx->L*ctx->L*ctx->M*2*sizeof(PetscScalar),&Mu);CHKERRQ(ierr);
  ierr = CalcMu(nep,Mu);CHKERRQ(ierr);

  ierr = VecCreateMPI(ctx->scomm,PETSC_DECIDE,ctx->L*ctx->M,&tempv); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(tempv,ctx->L*ctx->M,&H0);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(tempv,ctx->L*ctx->M,&H1);CHKERRQ(ierr);
  ierr = VecDestroy(&tempv);CHKERRQ(ierr);

  ierr = PetscMalloc(ctx->L*ctx->M*ctx->L*ctx->M*sizeof(PetscScalar),&H0d);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->L*ctx->M*ctx->L*ctx->M*sizeof(PetscScalar),&SVD_V);CHKERRQ(ierr);
  ierr = BlockHankel(nep,Mu,0,H0);CHKERRQ(ierr);
  ierr = BlockHankel(nep,Mu,1,H1);CHKERRQ(ierr);

  for (j=0;j<ctx->M*ctx->L;j++){ 
    ierr = VecGetArray(H0[j],&temp);CHKERRQ(ierr);
    for (i=0;i<ctx->L*ctx->M;i++){
      H0d[i+j*ctx->L*ctx->M] = temp[i];
    }
    ierr = VecRestoreArray(H0[j],&temp);CHKERRQ(ierr);
  }

  ierr = SVD(nep,H0,&nv,SVD_V,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscFree(Mu);CHKERRQ(ierr);

  nep->nconv = 0;
  if (nv>0){
    ierr = DSSetDimensions(nep->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);

    ierr = DSGetArray(nep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    for (j=0;j<nv;j++){ 
      ierr = VecGetArray(H1[j],&temp);CHKERRQ(ierr);
      for (i=0;i<nv;i++){
	H[i+j*ld] = temp[i];
      }
      ierr = VecRestoreArray(H1[j],&temp);CHKERRQ(ierr);
    }
    ierr = DSRestoreArray(nep->ds,DS_MAT_A,&H);CHKERRQ(ierr);

    ierr = DSGetArray(nep->ds,DS_MAT_B,&H);CHKERRQ(ierr);
    
    for (j=0;j<nv;j++) 
      for (i=0;i<nv;i++)
	H[i+j*ld] = H0d[i+j*ctx->L*ctx->M];
    ierr = DSRestoreArray(nep->ds,DS_MAT_B,&H);CHKERRQ(ierr);

    ierr = DSSolve(nep->ds,nep->eig,NULL);CHKERRQ(ierr);

    ierr = PetscMalloc(nv*sizeof(PetscReal),&tau);CHKERRQ(ierr);    
    ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
    ierr = DSGetArray(nep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
    ierr = PetscMalloc(nv*sizeof(PetscScalar),&alpha);CHKERRQ(ierr);
    
    for (k=0;k<nv;k++) {
      for (i=0;i<nv;i++) {
	alpha[i] = 0;
	for (j=0;j<nv;j++) {
	  alpha[i] += PetscConj(SVD_V[j*ctx->M*ctx->L+i])*pX[j+k*ld];
	}
	alpha[i] = PetscPowScalar(PetscAbsScalar(alpha[i]),2);
      }
      s1 = 0; s2 = 0;
      for (j=0;j<nv;j++) {
	s1 += ctx->sigma[j]*PetscRealPart(alpha[j]);
	s2 += PetscRealPart(alpha[j]);
      }
      tau[k] = s1/s2;
      tau_max = PetscMax(tau_max,tau[k]);
    }
    ierr = PetscFree(alpha);CHKERRQ(ierr);
    ierr = DSRestoreArray(nep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
    
    for (i=0;i<nv;i++){
      tau[i]/=tau_max; 
      nep->eig[i] = nep->eig[i]*ctx->radius+ctx->center;
    }

    ierr = PetscMalloc(nv*sizeof(PetscBool),&fl);CHKERRQ(ierr);
    ierr = isInsideGamma(nep,nv,fl);CHKERRQ(ierr);
    ierr = PetscMalloc(nv*sizeof(PetscScalar),&rr);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      if (fl[i] && tau[i] >= ctx->spurious_threshold*tau_max) {
	rr[i] = 1.0;
	nep->nconv++;
      } else rr[i] = 0.0;
    }
    ierr = PetscFree(fl);CHKERRQ(ierr);
    
    ierr = DSSetEigenvalueComparison(nep->ds,SlepcCompareLargestMagnitude,NULL);CHKERRQ(ierr);
    ierr = DSSort(nep->ds,nep->eig,NULL,rr,NULL,&nep->nconv);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      nep->eig[i] = nep->eig[i]*ctx->radius+ctx->center;
      ierr = VecCopy(ctx->S[i],nep->V[i]);CHKERRQ(ierr);
    }
    ierr = DSSetEigenvalueComparison(nep->ds,nep->comparison,nep->comparisonctx);CHKERRQ(ierr);
    ierr = PetscFree(rr);CHKERRQ(ierr);
    ierr = PetscFree(tau);CHKERRQ(ierr);
    
    /* compute eigenvectors */
    ierr = DSVectors(nep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
    ierr = DSGetArray(nep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,nep->V,0,nep->nconv,pX,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(nep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
    for (i=0;i<nep->nconv;i++) {
      ierr = VecNorm(nep->V[i],NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecScale(nep->V[i],1.0/norm);CHKERRQ(ierr);
    }
  }
  
  ierr = VecDestroyVecs(ctx->L*ctx->M,&H0);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ctx->L*ctx->M,&H1);CHKERRQ(ierr);
  ierr = PetscFree(H0d);CHKERRQ(ierr);
  ierr = PetscFree(SVD_V);CHKERRQ(ierr);
  
  nep->reason = NEP_CONVERGED_FNORM_RELATIVE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetRegion_CISS"
static PetscErrorCode NEPCISSSetRegion_CISS(NEP nep,PetscScalar center,PetscReal radius,PetscReal vscale)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  ctx->center = center;
  if (radius) {
    if (radius == PETSC_DEFAULT) {
      ctx->radius = 1.0;
    } else {
      if (radius<0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The radius argument must be > 0.0");
      ctx->radius = radius;
    }
  }
  if (vscale) {
    if (vscale == PETSC_DEFAULT) {
      ctx->vscale = 1.0;
    } else {
      if (vscale<0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The vscale argument must be > 0.0");
      ctx->vscale = vscale;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetRegion"
/*@
   NEPCISSSetRegion - Sets the parameters defining the region where eigenvalues
   must be computed.

   Logically Collective on NEP

   Input Parameters:
+  nep - the eigenproblem solver context
.  center - center of the region
.  radius - radius of the region
-  vscale - vertical scale of the region

   Options Database Keys:
+  -nep_ciss_center - Sets the center
.  -nep_ciss_radius - Sets the radius
-  -nep_ciss_vscale - Sets the vertical scale

   Level: advanced

.seealso: NEPCISSGetRegion()
@*/
PetscErrorCode NEPCISSSetRegion(NEP nep,PetscScalar center,PetscReal radius,PetscReal vscale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(nep,center,2);
  PetscValidLogicalCollectiveReal(nep,radius,3);
  PetscValidLogicalCollectiveReal(nep,vscale,4);
  ierr = PetscTryMethod(nep,"NEPCISSSetRegion_C",(NEP,PetscScalar,PetscReal,PetscReal),(nep,center,radius,vscale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetRegion_CISS"
static PetscErrorCode NEPCISSGetRegion_CISS(NEP nep,PetscScalar *center,PetscReal *radius,PetscReal *vscale)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (center) *center = ctx->center;
  if (radius) *radius = ctx->radius;
  if (vscale) *vscale = ctx->vscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetRegion"
/*@
   NEPCISSGetRegion - Gets the parameters that define the region where eigenvalues
   must be computed.

   Not Collective

   Input Parameter:
.  nep - the eigenproblem solver context

   Output Parameters:
+  center - center of the region
.  radius - radius of the region
-  vscale - vertical scale of the region

   Level: advanced

.seealso: NEPCISSSetRegion()
@*/
PetscErrorCode NEPCISSGetRegion(NEP nep,PetscScalar *center,PetscReal *radius,PetscReal *vscale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscTryMethod(nep,"NEPCISSGetRegion_C",(NEP,PetscScalar*,PetscReal*,PetscReal*),(nep,center,radius,vscale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetSizes_CISS"
static PetscErrorCode NEPCISSSetSizes_CISS(NEP nep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool isreal)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (ip) {
    if (ip == PETSC_DECIDE || ip == PETSC_DEFAULT) {
      if (ctx->N!=32) { ctx->N =32; ctx->M = ctx->N/4; }
    } else {
      if (ip<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be > 0");
      if (ip%2) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ip argument must be an even number");
      if (ctx->N!=ip) { ctx->N = ip; ctx->M = ctx->N/4; }
    }
  }
  if (bs) {
    if (bs == PETSC_DECIDE || bs == PETSC_DEFAULT) {
      ctx->L = 16;
    } else {
      if (bs<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be > 0");
      if (bs>ctx->L_max) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bs argument must be less than or equal to the maximum number of block size");
      ctx->L = bs;
    }
  }
  if (ms) {
    if (ms == PETSC_DECIDE || ms == PETSC_DEFAULT) {
      ctx->M = ctx->N/4;
    } else {
      if (ms<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be > 0");
      if (ms>ctx->N) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The ms argument must be less than or equal to the number of integration points");
      ctx->M = ms;
    }
  }
  if (npart) {
    if (npart == PETSC_DECIDE || npart == PETSC_DEFAULT) {
      ctx->npart = 1;
    } else {
      if (npart<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The npart argument must be > 0");
      ctx->npart = npart;
    }
  }
  if (bsmax) {
    if (bsmax == PETSC_DECIDE || bsmax == PETSC_DEFAULT) {
      ctx->L = 256;
    } else {
      if (bsmax<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The bsmax argument must be > 0");
      if (bsmax<ctx->L) ctx->L_max = ctx->L;
      else ctx->L_max = bsmax;
    }
  }
  ctx->isreal = isreal;
  ierr = NEPReset(nep);CHKERRQ(ierr);   /* clean allocated arrays and force new setup */
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
-  isreal - A and B are real

   Options Database Keys:
+  -nep_ciss_integration_points - Sets the number of integration points
.  -nep_ciss_blocksize - Sets the block size
.  -nep_ciss_moments - Sets the moment size
.  -nep_ciss_partitions - Sets the number of partitions
.  -nep_ciss_maxblocksize - Sets the maximum block size
-  -nep_ciss_realmats - A and B are real

   Note:
   The default number of partitions is 1. This means the internal KSP object is shared
   among all processes of the NEP communicator. Otherwise, the communicator is split
   into npart communicators, so that npart KSP solves proceed simultaneously.

   Level: advanced

.seealso: NEPCISSGetSizes()
@*/
PetscErrorCode NEPCISSSetSizes(NEP nep,PetscInt ip,PetscInt bs,PetscInt ms,PetscInt npart,PetscInt bsmax,PetscBool isreal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,ip,2);
  PetscValidLogicalCollectiveInt(nep,bs,3);
  PetscValidLogicalCollectiveInt(nep,ms,4);
  PetscValidLogicalCollectiveInt(nep,npart,5);
  PetscValidLogicalCollectiveInt(nep,bsmax,6);
  PetscValidLogicalCollectiveBool(nep,isreal,7);
  ierr = PetscTryMethod(nep,"NEPCISSSetSizes_C",(NEP,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool),(nep,ip,bs,ms,npart,bsmax,isreal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSGetSizes_CISS"
static PetscErrorCode NEPCISSGetSizes_CISS(NEP nep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *isreal)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;

  PetscFunctionBegin;
  if (ip) *ip = ctx->N;
  if (bs) *bs = ctx->L;
  if (ms) *ms = ctx->M;
  if (npart) *npart = ctx->npart;
  if (bsmax) *bsmax = ctx->L_max;
  if (isreal) *isreal = ctx->isreal;
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
-  isreal - A and B are real

   Level: advanced

.seealso: NEPCISSSetSizes()
@*/
PetscErrorCode NEPCISSGetSizes(NEP nep,PetscInt *ip,PetscInt *bs,PetscInt *ms,PetscInt *npart,PetscInt *bsmax,PetscBool *isreal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscTryMethod(nep,"NEPCISSGetSizes_C",(NEP,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*),(nep,ip,bs,ms,npart,bsmax,isreal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetThreshold_CISS"
static PetscErrorCode NEPCISSSetThreshold_CISS(NEP nep,PetscReal delta,PetscReal spur)
{
  NEP_CISS *ctx = (NEP_CISS*)nep->data;
  if (delta) {
    if (delta == PETSC_DEFAULT) {
      ctx->delta = 1e-12;
    } else {
      if (delta<=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The delta argument must be > 0.0");
      ctx->delta = delta;
    }
  }
  if (spur) {
    if (spur == PETSC_DEFAULT) {
      ctx->spurious_threshold = 1e-4;
    } else {
      if (spur<=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The spurious threshold argument must be > 0.0");
      ctx->spurious_threshold = spur;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCISSSetThreshold"
/*@
   NEPCISSSetThreshold - Sets the values of various threshold parameters in the CISS solver.

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
   NEPCISSGetThreshold - Gets the values of various threshold parameters in the CISS solver.

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
  ierr = PetscTryMethod(nep,"NEPCISSGetThreshold_C",(NEP,PetscReal*,PetscReal*),(nep,delta,spur));CHKERRQ(ierr);
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
  if (nep->setupcalled) { ierr = MPI_Comm_free(&ctx->scomm);CHKERRQ(ierr); }
  ierr = PetscFree(ctx->weight);CHKERRQ(ierr);
  ierr = PetscFree(ctx->omega);CHKERRQ(ierr);
  ierr = PetscFree(ctx->pp);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ctx->L,&ctx->V);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point;i++) {
    ierr = KSPDestroy(&ctx->ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->ksp);CHKERRQ(ierr);
  ierr = PetscFree(ctx->sigma);CHKERRQ(ierr);
  for (i=0;i<ctx->num_solve_point*ctx->L_max;i++) {
    ierr = VecDestroy(&ctx->Y[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->Y);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ctx->M*ctx->L,&ctx->S);CHKERRQ(ierr);
  //ierr = VecDestroyVecs(ctx->M*ctx->L*2,&ctx->S);CHKERRQ(ierr);
  ierr = NEPReset_Default(nep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions_CISS"
PetscErrorCode NEPSetFromOptions_CISS(NEP nep)
{
  PetscErrorCode ierr;
  PetscScalar    s;
  PetscReal      r1,r2,r3,r4;
  PetscInt       i1=0,i2=0,i3=0,i4=0,i5=0;
  PetscBool      b1=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("NEP CISS Options");CHKERRQ(ierr);
  ierr = NEPCISSGetRegion(nep,&s,&r1,&r2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_ciss_radius","CISS radius of region","NEPCISSSetRegion",r1,&r1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-nep_ciss_center","CISS center of region","NEPCISSSetRegion",s,&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_ciss_vscale","CISS vertical scale of region","NEPCISSSetRegion",r2,&r2,NULL);CHKERRQ(ierr);
  ierr = NEPCISSSetRegion(nep,s,r1,r2);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-nep_ciss_integration_points","CISS number of integration points","NEPCISSSetSizes",i1,&i1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_blocksize","CISS block size","NEPCISSSetSizes",i2,&i2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_moments","CISS moment size","NEPCISSSetSizes",i3,&i3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_partitions","CISS number of partitions","NEPCISSSetSizes",i4,&i4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_ciss_maxblocksize","CISS maximum block size","NEPCISSSetSizes",i5,&i5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-nep_ciss_realmats","CISS A and B are real","NEPCISSSetSizes",b1,&b1,NULL);CHKERRQ(ierr);
  ierr = NEPCISSSetSizes(nep,i1,i2,i3,i4,i5,b1);CHKERRQ(ierr);

  ierr = NEPCISSGetThreshold(nep,&r3,&r4);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_ciss_delta","CISS threshold for numerical rank","NEPCISSSetThreshold",r3,&r3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_ciss_spurious_threshold","CISS threshold for the spurious eigenpairs","NEPCISSSetThreshold",r4,&r4,NULL);CHKERRQ(ierr);
  ierr = NEPCISSSetThreshold(nep,r3,r4);CHKERRQ(ierr);

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
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetRegion_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetRegion_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetThreshold_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetThreshold_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView_CISS"
PetscErrorCode NEPView_CISS(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_CISS       *ctx = (NEP_CISS*)nep->data;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = SlepcSNPrintfScalar(str,50,ctx->center,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: region { center: %s, radius: %G, vscale: %G }\n",str,ctx->radius,ctx->vscale);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: sizes { integration points: %D, block size: %D, moment size: %D, partitions: %D, maximum block size: %D }\n",ctx->N,ctx->L,ctx->M,ctx->npart,ctx->L_max);CHKERRQ(ierr);
    if (ctx->isreal) {
      ierr = PetscViewerASCIIPrintf(viewer,"  CISS: exploiting symmetry of integration points\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  CISS: threshold { delta: %G, spurious threshold: %G }\n",ctx->delta,ctx->spurious_threshold);CHKERRQ(ierr);
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
  nep->ops->setup          = NEPSetUp_CISS;
  nep->ops->setfromoptions = NEPSetFromOptions_CISS;
  nep->ops->destroy        = NEPDestroy_CISS;
  nep->ops->reset          = NEPReset_CISS;
  nep->ops->view           = NEPView_CISS;
  //nep->ops->backtransform  = PETSC_NULL;
  //nep->ops->computevectors = NEPComputeVectors_Schur;
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetRegion_C",NEPCISSSetRegion_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetRegion_C",NEPCISSGetRegion_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetSizes_C",NEPCISSSetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetSizes_C",NEPCISSGetSizes_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSSetThreshold_C",NEPCISSSetThreshold_CISS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPCISSGetThreshold_C",NEPCISSGetThreshold_CISS);CHKERRQ(ierr);
  /* set default values of parameters */
  ctx->center  = 0.0;
  ctx->radius  = 1.0;
  // ctx->vscale  = PETSC_DEFAULT;
  ctx->vscale  = 1.0;
  ctx->N       = 32;
  ctx->L       = 16;
  ctx->M       = ctx->N/4;
  ctx->delta   = 1e-12;
  ctx->npart   = 1;
  ctx->L_max   = 256;
  ctx->spurious_threshold = 1e-4;
  ctx->isreal  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

