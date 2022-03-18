/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Newton refinement for NEP, simple version
*/

#include <slepc/private/nepimpl.h>
#include <slepcblaslapack.h>

#define NREF_MAXIT 10

typedef struct {
  VecScatter    *scatter_id,nst;
  Mat           *A;
  Vec           nv,vg,v,w;
  FN            *fn;
} NEPSimpNRefctx;

typedef struct {
  Mat          M1;
  Vec          M2,M3;
  PetscScalar  M4,m3;
} NEP_REFINE_MATSHELL;

static PetscErrorCode MatMult_FS(Mat M ,Vec x,Vec y)
{
  NEP_REFINE_MATSHELL *ctx;
  PetscScalar         t;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(VecDot(x,ctx->M3,&t));
  t *= ctx->m3/ctx->M4;
  CHKERRQ(MatMult(ctx->M1,x,y));
  CHKERRQ(VecAXPY(y,-t,ctx->M2));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSimpleNRefSetUp(NEP nep,NEPSimpNRefctx **ctx_)
{
  PetscInt       i,si,j,n0,m0,nloc,*idx1,*idx2,ne;
  IS             is1,is2;
  NEPSimpNRefctx *ctx;
  Vec            v;
  PetscMPIInt    rank,size;
  MPI_Comm       child;

  PetscFunctionBegin;
  CHKERRQ(PetscCalloc1(1,ctx_));
  ctx = *ctx_;
  if (nep->npart==1) {
    ctx->A  = nep->A;
    ctx->fn = nep->f;
    nep->refinesubc = NULL;
    ctx->scatter_id = NULL;
  } else {
    CHKERRQ(PetscSubcommGetChild(nep->refinesubc,&child));
    CHKERRQ(PetscMalloc2(nep->nt,&ctx->A,nep->npart,&ctx->scatter_id));

    /* Duplicate matrices */
    for (i=0;i<nep->nt;i++) {
      CHKERRQ(MatCreateRedundantMatrix(nep->A[i],0,child,MAT_INITIAL_MATRIX,&ctx->A[i]));
      CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->A[i]));
    }
    CHKERRQ(MatCreateVecs(ctx->A[0],&ctx->v,NULL));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->v));

    /* Duplicate FNs */
    CHKERRQ(PetscMalloc1(nep->nt,&ctx->fn));
    for (i=0;i<nep->nt;i++) {
      CHKERRQ(FNDuplicate(nep->f[i],child,&ctx->fn[i]));
    }

    /* Create scatters for sending vectors to each subcommucator */
    CHKERRQ(BVGetColumn(nep->V,0,&v));
    CHKERRQ(VecGetOwnershipRange(v,&n0,&m0));
    CHKERRQ(BVRestoreColumn(nep->V,0,&v));
    CHKERRQ(VecGetLocalSize(ctx->v,&nloc));
    CHKERRQ(PetscMalloc2(m0-n0,&idx1,m0-n0,&idx2));
    CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)nep),nloc,PETSC_DECIDE,&ctx->vg));
    for (si=0;si<nep->npart;si++) {
      j = 0;
      for (i=n0;i<m0;i++) {
        idx1[j]   = i;
        idx2[j++] = i+nep->n*si;
      }
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)nep),(m0-n0),idx1,PETSC_COPY_VALUES,&is1));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)nep),(m0-n0),idx2,PETSC_COPY_VALUES,&is2));
      CHKERRQ(BVGetColumn(nep->V,0,&v));
      CHKERRQ(VecScatterCreate(v,is1,ctx->vg,is2,&ctx->scatter_id[si]));
      CHKERRQ(BVRestoreColumn(nep->V,0,&v));
      CHKERRQ(ISDestroy(&is1));
      CHKERRQ(ISDestroy(&is2));
    }
    CHKERRQ(PetscFree2(idx1,idx2));
  }
  if (nep->scheme==NEP_REFINE_SCHEME_EXPLICIT) {
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ctx->A[0]),&rank));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ctx->A[0]),&size));
    if (size>1) {
      if (nep->npart==1) {
        CHKERRQ(BVGetColumn(nep->V,0,&v));
      } else v = ctx->v;
      CHKERRQ(VecGetOwnershipRange(v,&n0,&m0));
      ne = (rank == size-1)?nep->n:0;
      CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)ctx->A[0]),ne,PETSC_DECIDE,&ctx->nv));
      CHKERRQ(PetscMalloc1(m0-n0,&idx1));
      for (i=n0;i<m0;i++) {
        idx1[i-n0] = i;
      }
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ctx->A[0]),(m0-n0),idx1,PETSC_COPY_VALUES,&is1));
      CHKERRQ(VecScatterCreate(v,is1,ctx->nv,is1,&ctx->nst));
      if (nep->npart==1) {
        CHKERRQ(BVRestoreColumn(nep->V,0,&v));
      }
      CHKERRQ(PetscFree(idx1));
      CHKERRQ(ISDestroy(&is1));
    }
  }  PetscFunctionReturn(0);
}

/*
  Gather Eigenpair idx from subcommunicator with color sc
*/
static PetscErrorCode NEPSimpleNRefGatherEigenpair(NEP nep,NEPSimpNRefctx *ctx,PetscInt sc,PetscInt idx,PetscInt *fail)
{
  PetscMPIInt    nproc,p;
  MPI_Comm       comm=((PetscObject)nep)->comm;
  Vec            v;
  PetscScalar    *array;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&nproc));
  p = (nproc/nep->npart)*(sc+1)+PetscMin(sc+1,nproc%nep->npart)-1;
  if (nep->npart>1) {
    /* Communicate convergence successful */
    CHKERRMPI(MPI_Bcast(fail,1,MPIU_INT,p,comm));
    if (!(*fail)) {
      /* Process 0 of subcommunicator sc broadcasts the eigenvalue */
      CHKERRMPI(MPI_Bcast(&nep->eigr[idx],1,MPIU_SCALAR,p,comm));
      /* Gather nep->V[idx] from the subcommuniator sc */
      CHKERRQ(BVGetColumn(nep->V,idx,&v));
      if (nep->refinesubc->color==sc) {
        CHKERRQ(VecGetArray(ctx->v,&array));
        CHKERRQ(VecPlaceArray(ctx->vg,array));
      }
      CHKERRQ(VecScatterBegin(ctx->scatter_id[sc],ctx->vg,v,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(ctx->scatter_id[sc],ctx->vg,v,INSERT_VALUES,SCATTER_REVERSE));
      if (nep->refinesubc->color==sc) {
        CHKERRQ(VecResetArray(ctx->vg));
        CHKERRQ(VecRestoreArray(ctx->v,&array));
      }
      CHKERRQ(BVRestoreColumn(nep->V,idx,&v));
    }
  } else {
    if (nep->scheme==NEP_REFINE_SCHEME_EXPLICIT && !(*fail)) {
      CHKERRMPI(MPI_Bcast(&nep->eigr[idx],1,MPIU_SCALAR,p,comm));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSimpleNRefScatterEigenvector(NEP nep,NEPSimpNRefctx *ctx,PetscInt sc,PetscInt idx)
{
  Vec            v;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (nep->npart>1) {
    CHKERRQ(BVGetColumn(nep->V,idx,&v));
    if (nep->refinesubc->color==sc) {
      CHKERRQ(VecGetArray(ctx->v,&array));
      CHKERRQ(VecPlaceArray(ctx->vg,array));
    }
    CHKERRQ(VecScatterBegin(ctx->scatter_id[sc],v,ctx->vg,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(ctx->scatter_id[sc],v,ctx->vg,INSERT_VALUES,SCATTER_FORWARD));
    if (nep->refinesubc->color==sc) {
      CHKERRQ(VecResetArray(ctx->vg));
      CHKERRQ(VecRestoreArray(ctx->v,&array));
    }
    CHKERRQ(BVRestoreColumn(nep->V,idx,&v));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPSimpleNRefSetUpSystem(NEP nep,NEPSimpNRefctx *ctx,Mat *A,PetscInt idx,Mat *Mt,Mat *T,Mat *P,PetscBool ini,Vec t,Vec v)
{
  PetscErrorCode      ierr;
  PetscInt            i,st,ml,m0,n0,m1,mg;
  PetscInt            *dnz,*onz,ncols,*cols2=NULL,*nnz,nt=nep->nt;
  PetscScalar         zero=0.0,*coeffs,*coeffs2;
  PetscMPIInt         rank,size;
  MPI_Comm            comm;
  const PetscInt      *cols;
  const PetscScalar   *vals,*array;
  NEP_REFINE_MATSHELL *fctx;
  Vec                 w=ctx->w;
  Mat                 M;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc2(nt,&coeffs,nt,&coeffs2));
  switch (nep->scheme) {
  case NEP_REFINE_SCHEME_SCHUR:
    if (ini) {
      CHKERRQ(PetscCalloc1(1,&fctx));
      CHKERRQ(MatGetSize(A[0],&m0,&n0));
      CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)A[0]),PETSC_DECIDE,PETSC_DECIDE,m0,n0,fctx,T));
      CHKERRQ(MatShellSetOperation(*T,MATOP_MULT,(void(*)(void))MatMult_FS));
    } else {
      CHKERRQ(MatShellGetContext(*T,&fctx));
    }
    M=fctx->M1;
    break;
  case NEP_REFINE_SCHEME_MBE:
    M=*T;
    break;
  case NEP_REFINE_SCHEME_EXPLICIT:
    M=*Mt;
    break;
  }
  if (ini) {
    CHKERRQ(MatDuplicate(A[0],MAT_COPY_VALUES,&M));
  } else {
    CHKERRQ(MatCopy(A[0],M,DIFFERENT_NONZERO_PATTERN));
  }
  for (i=0;i<nt;i++) {
    CHKERRQ(FNEvaluateFunction(ctx->fn[i],nep->eigr[idx],coeffs+i));
  }
  if (coeffs[0]!=1.0) {
    CHKERRQ(MatScale(M,coeffs[0]));
  }
  for (i=1;i<nt;i++) {
    CHKERRQ(MatAXPY(M,coeffs[i],A[i],(ini)?nep->mstr:SUBSET_NONZERO_PATTERN));
  }
  for (i=0;i<nt;i++) {
    CHKERRQ(FNEvaluateDerivative(ctx->fn[i],nep->eigr[idx],coeffs2+i));
  }
  st = 0;
  for (i=0;i<nt && PetscAbsScalar(coeffs2[i])==0.0;i++) st++;
  CHKERRQ(MatMult(A[st],v,w));
  if (coeffs2[st]!=1.0) {
    CHKERRQ(VecScale(w,coeffs2[st]));
  }
  for (i=st+1;i<nt;i++) {
    CHKERRQ(MatMult(A[i],v,t));
    CHKERRQ(VecAXPY(w,coeffs2[i],t));
  }

  switch (nep->scheme) {
  case NEP_REFINE_SCHEME_EXPLICIT:
    comm = PetscObjectComm((PetscObject)A[0]);
    CHKERRMPI(MPI_Comm_rank(comm,&rank));
    CHKERRMPI(MPI_Comm_size(comm,&size));
    CHKERRQ(MatGetSize(M,&mg,NULL));
    CHKERRQ(MatGetOwnershipRange(M,&m0,&m1));
    if (ini) {
      CHKERRQ(MatCreate(comm,T));
      CHKERRQ(MatGetLocalSize(M,&ml,NULL));
      if (rank==size-1) ml++;
      CHKERRQ(MatSetSizes(*T,ml,ml,mg+1,mg+1));
      CHKERRQ(MatSetFromOptions(*T));
      CHKERRQ(MatSetUp(*T));
      /* Preallocate M */
      if (size>1) {
        ierr = MatPreallocateInitialize(comm,ml,ml,dnz,onz);CHKERRQ(ierr);
        for (i=m0;i<m1;i++) {
          CHKERRQ(MatGetRow(M,i,&ncols,&cols,NULL));
          CHKERRQ(MatPreallocateSet(i,ncols,cols,dnz,onz));
          CHKERRQ(MatPreallocateSet(i,1,&mg,dnz,onz));
          CHKERRQ(MatRestoreRow(M,i,&ncols,&cols,NULL));
        }
        if (rank==size-1) {
          CHKERRQ(PetscCalloc1(mg+1,&cols2));
          for (i=0;i<mg+1;i++) cols2[i]=i;
          CHKERRQ(MatPreallocateSet(m1,mg+1,cols2,dnz,onz));
          CHKERRQ(PetscFree(cols2));
        }
        CHKERRQ(MatMPIAIJSetPreallocation(*T,0,dnz,0,onz));
        ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
      } else {
        CHKERRQ(PetscCalloc1(mg+1,&nnz));
        for (i=0;i<mg;i++) {
          CHKERRQ(MatGetRow(M,i,&ncols,NULL,NULL));
          nnz[i] = ncols+1;
          CHKERRQ(MatRestoreRow(M,i,&ncols,NULL,NULL));
        }
        nnz[mg] = mg+1;
        CHKERRQ(MatSeqAIJSetPreallocation(*T,0,nnz));
        CHKERRQ(PetscFree(nnz));
      }
      *Mt = M;
      *P  = *T;
    }

    /* Set values */
    CHKERRQ(VecGetArrayRead(w,&array));
    for (i=m0;i<m1;i++) {
      CHKERRQ(MatGetRow(M,i,&ncols,&cols,&vals));
      CHKERRQ(MatSetValues(*T,1,&i,ncols,cols,vals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(M,i,&ncols,&cols,&vals));
      CHKERRQ(MatSetValues(*T,1,&i,1,&mg,array+i-m0,INSERT_VALUES));
    }
    CHKERRQ(VecRestoreArrayRead(w,&array));
    CHKERRQ(VecConjugate(v));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A[0]),&size));
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A[0]),&rank));
    if (size>1) {
      if (rank==size-1) {
        CHKERRQ(PetscMalloc1(nep->n,&cols2));
        for (i=0;i<nep->n;i++) cols2[i]=i;
      }
      CHKERRQ(VecScatterBegin(ctx->nst,v,ctx->nv,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(ctx->nst,v,ctx->nv,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecGetArrayRead(ctx->nv,&array));
      if (rank==size-1) {
        CHKERRQ(MatSetValues(*T,1,&mg,nep->n,cols2,array,INSERT_VALUES));
        CHKERRQ(MatSetValues(*T,1,&mg,1,&mg,&zero,INSERT_VALUES));
      }
      CHKERRQ(VecRestoreArrayRead(ctx->nv,&array));
    } else {
      CHKERRQ(PetscMalloc1(m1-m0,&cols2));
      for (i=0;i<m1-m0;i++) cols2[i]=m0+i;
      CHKERRQ(VecGetArrayRead(v,&array));
      CHKERRQ(MatSetValues(*T,1,&mg,m1-m0,cols2,array,INSERT_VALUES));
      CHKERRQ(MatSetValues(*T,1,&mg,1,&mg,&zero,INSERT_VALUES));
      CHKERRQ(VecRestoreArrayRead(v,&array));
    }
    CHKERRQ(VecConjugate(v));
    CHKERRQ(MatAssemblyBegin(*T,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(*T,MAT_FINAL_ASSEMBLY));
    CHKERRQ(PetscFree(cols2));
    break;
  case NEP_REFINE_SCHEME_SCHUR:
    fctx->M2 = ctx->w;
    fctx->M3 = v;
    fctx->m3 = 1.0+PetscConj(nep->eigr[idx])*nep->eigr[idx];
    fctx->M4 = PetscConj(nep->eigr[idx]);
    fctx->M1 = M;
    if (ini) {
      CHKERRQ(MatDuplicate(M,MAT_COPY_VALUES,P));
    } else {
      CHKERRQ(MatCopy(M,*P,SAME_NONZERO_PATTERN));
    }
    if (fctx->M4!=0.0) {
      CHKERRQ(VecConjugate(v));
      CHKERRQ(VecPointwiseMult(t,v,w));
      CHKERRQ(VecConjugate(v));
      CHKERRQ(VecScale(t,-fctx->m3/fctx->M4));
      CHKERRQ(MatDiagonalSet(*P,t,ADD_VALUES));
    }
    break;
  case NEP_REFINE_SCHEME_MBE:
    *T = M;
    *P = M;
    break;
  }
  CHKERRQ(PetscFree2(coeffs,coeffs2));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPNewtonRefinementSimple(NEP nep,PetscInt *maxits,PetscReal tol,PetscInt k)
{
  PetscInt            i,n,its,idx=0,*idx_sc,*its_sc,color,*fail_sc;
  PetscMPIInt         rank,size;
  Mat                 Mt=NULL,T=NULL,P=NULL;
  MPI_Comm            comm;
  Vec                 r,v,dv,rr=NULL,dvv=NULL,t[2];
  const PetscScalar   *array;
  PetscScalar         *array2,deig=0.0,tt[2],ttt;
  PetscReal           norm,error;
  PetscBool           ini=PETSC_TRUE,sc_pend,solved=PETSC_FALSE;
  NEPSimpNRefctx      *ctx;
  NEP_REFINE_MATSHELL *fctx=NULL;
  KSPConvergedReason  reason;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(NEP_Refine,nep,0,0,0));
  CHKERRQ(NEPSimpleNRefSetUp(nep,&ctx));
  its = (maxits)?*maxits:NREF_MAXIT;
  if (!nep->refineksp) CHKERRQ(NEPRefineGetKSP(nep,&nep->refineksp));
  if (nep->npart==1) {
    CHKERRQ(BVGetColumn(nep->V,0,&v));
  } else v = ctx->v;
  CHKERRQ(VecDuplicate(v,&ctx->w));
  CHKERRQ(VecDuplicate(v,&r));
  CHKERRQ(VecDuplicate(v,&dv));
  CHKERRQ(VecDuplicate(v,&t[0]));
  CHKERRQ(VecDuplicate(v,&t[1]));
  if (nep->npart==1) {
    CHKERRQ(BVRestoreColumn(nep->V,0,&v));
    CHKERRQ(PetscObjectGetComm((PetscObject)nep,&comm));
  } else {
    CHKERRQ(PetscSubcommGetChild(nep->refinesubc,&comm));
  }
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(VecGetLocalSize(r,&n));
  CHKERRQ(PetscMalloc3(nep->npart,&idx_sc,nep->npart,&its_sc,nep->npart,&fail_sc));
  for (i=0;i<nep->npart;i++) fail_sc[i] = 0;
  for (i=0;i<nep->npart;i++) its_sc[i] = 0;
  color = (nep->npart==1)?0:nep->refinesubc->color;

  /* Loop performing iterative refinements */
  while (!solved) {
    for (i=0;i<nep->npart;i++) {
      sc_pend = PETSC_TRUE;
      if (its_sc[i]==0) {
        idx_sc[i] = idx++;
        if (idx_sc[i]>=k) {
          sc_pend = PETSC_FALSE;
        } else {
          CHKERRQ(NEPSimpleNRefScatterEigenvector(nep,ctx,i,idx_sc[i]));
        }
      }  else { /* Gather Eigenpair from subcommunicator i */
        CHKERRQ(NEPSimpleNRefGatherEigenpair(nep,ctx,i,idx_sc[i],&fail_sc[i]));
      }
      while (sc_pend) {
        if (!fail_sc[i]) {
          CHKERRQ(NEPComputeError(nep,idx_sc[i],NEP_ERROR_RELATIVE,&error));
        }
        if (error<=tol || its_sc[i]>=its || fail_sc[i]) {
          idx_sc[i] = idx++;
          its_sc[i] = 0;
          fail_sc[i] = 0;
          if (idx_sc[i]<k) CHKERRQ(NEPSimpleNRefScatterEigenvector(nep,ctx,i,idx_sc[i]));
        } else {
          sc_pend = PETSC_FALSE;
          its_sc[i]++;
        }
        if (idx_sc[i]>=k) sc_pend = PETSC_FALSE;
      }
    }
    solved = PETSC_TRUE;
    for (i=0;i<nep->npart&&solved;i++) solved = PetscNot(idx_sc[i]<k);
    if (idx_sc[color]<k) {
#if !defined(PETSC_USE_COMPLEX)
      PetscCheck(nep->eigi[idx_sc[color]]==0.0,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Simple Refinement not implemented in real scalar for complex eigenvalues");
#endif
      if (nep->npart==1) {
        CHKERRQ(BVGetColumn(nep->V,idx_sc[color],&v));
      } else v = ctx->v;
      CHKERRQ(NEPSimpleNRefSetUpSystem(nep,ctx,ctx->A,idx_sc[color],&Mt,&T,&P,ini,t[0],v));
      CHKERRQ(NEP_KSPSetOperators(nep->refineksp,T,P));
      if (ini) {
        CHKERRQ(KSPSetFromOptions(nep->refineksp));
        if (nep->scheme==NEP_REFINE_SCHEME_EXPLICIT) {
          CHKERRQ(MatCreateVecs(T,&dvv,NULL));
          CHKERRQ(VecDuplicate(dvv,&rr));
        }
        ini = PETSC_FALSE;
      }
      switch (nep->scheme) {
      case NEP_REFINE_SCHEME_EXPLICIT:
        CHKERRQ(MatMult(Mt,v,r));
        CHKERRQ(VecGetArrayRead(r,&array));
        if (rank==size-1) {
          CHKERRQ(VecGetArray(rr,&array2));
          CHKERRQ(PetscArraycpy(array2,array,n));
          array2[n] = 0.0;
          CHKERRQ(VecRestoreArray(rr,&array2));
        } else {
          CHKERRQ(VecPlaceArray(rr,array));
        }
        CHKERRQ(KSPSolve(nep->refineksp,rr,dvv));
        CHKERRQ(KSPGetConvergedReason(nep->refineksp,&reason));
        if (reason>0) {
          if (rank != size-1) {
            CHKERRQ(VecResetArray(rr));
          }
          CHKERRQ(VecRestoreArrayRead(r,&array));
          CHKERRQ(VecGetArrayRead(dvv,&array));
          CHKERRQ(VecPlaceArray(dv,array));
          CHKERRQ(VecAXPY(v,-1.0,dv));
          CHKERRQ(VecNorm(v,NORM_2,&norm));
          CHKERRQ(VecScale(v,1.0/norm));
          CHKERRQ(VecResetArray(dv));
          if (rank==size-1) nep->eigr[idx_sc[color]] -= array[n];
          CHKERRQ(VecRestoreArrayRead(dvv,&array));
        } else fail_sc[color] = 1;
        break;
      case NEP_REFINE_SCHEME_MBE:
        CHKERRQ(MatMult(T,v,r));
        /* Mixed block elimination */
        CHKERRQ(VecConjugate(v));
        CHKERRQ(KSPSolveTranspose(nep->refineksp,v,t[0]));
        CHKERRQ(KSPGetConvergedReason(nep->refineksp,&reason));
        if (reason>0) {
          CHKERRQ(VecConjugate(t[0]));
          CHKERRQ(VecDot(ctx->w,t[0],&tt[0]));
          CHKERRQ(KSPSolve(nep->refineksp,ctx->w,t[1]));
          CHKERRQ(KSPGetConvergedReason(nep->refineksp,&reason));
          if (reason>0) {
            CHKERRQ(VecDot(t[1],v,&tt[1]));
            CHKERRQ(VecDot(r,t[0],&ttt));
            tt[0] = ttt/tt[0];
            CHKERRQ(VecAXPY(r,-tt[0],ctx->w));
            CHKERRQ(KSPSolve(nep->refineksp,r,dv));
            CHKERRQ(KSPGetConvergedReason(nep->refineksp,&reason));
            if (reason>0) {
              CHKERRQ(VecDot(dv,v,&ttt));
              tt[1] = ttt/tt[1];
              CHKERRQ(VecAXPY(dv,-tt[1],t[1]));
              deig = tt[0]+tt[1];
            }
          }
          CHKERRQ(VecConjugate(v));
          CHKERRQ(VecAXPY(v,-1.0,dv));
          CHKERRQ(VecNorm(v,NORM_2,&norm));
          CHKERRQ(VecScale(v,1.0/norm));
          nep->eigr[idx_sc[color]] -= deig;
          fail_sc[color] = 0;
        } else {
          CHKERRQ(VecConjugate(v));
          fail_sc[color] = 1;
        }
        break;
      case NEP_REFINE_SCHEME_SCHUR:
        fail_sc[color] = 1;
        CHKERRQ(MatShellGetContext(T,&fctx));
        if (fctx->M4!=0.0) {
          CHKERRQ(MatMult(fctx->M1,v,r));
          CHKERRQ(KSPSolve(nep->refineksp,r,dv));
          CHKERRQ(KSPGetConvergedReason(nep->refineksp,&reason));
          if (reason>0) {
            CHKERRQ(VecDot(dv,v,&deig));
            deig *= -fctx->m3/fctx->M4;
            CHKERRQ(VecAXPY(v,-1.0,dv));
            CHKERRQ(VecNorm(v,NORM_2,&norm));
            CHKERRQ(VecScale(v,1.0/norm));
            nep->eigr[idx_sc[color]] -= deig;
            fail_sc[color] = 0;
          }
        }
        break;
      }
      if (nep->npart==1) CHKERRQ(BVRestoreColumn(nep->V,idx_sc[color],&v));
    }
  }
  CHKERRQ(VecDestroy(&t[0]));
  CHKERRQ(VecDestroy(&t[1]));
  CHKERRQ(VecDestroy(&dv));
  CHKERRQ(VecDestroy(&ctx->w));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(PetscFree3(idx_sc,its_sc,fail_sc));
  CHKERRQ(VecScatterDestroy(&ctx->nst));
  if (nep->npart>1) {
    CHKERRQ(VecDestroy(&ctx->vg));
    CHKERRQ(VecDestroy(&ctx->v));
    for (i=0;i<nep->nt;i++) {
      CHKERRQ(MatDestroy(&ctx->A[i]));
    }
    for (i=0;i<nep->npart;i++) {
      CHKERRQ(VecScatterDestroy(&ctx->scatter_id[i]));
    }
    CHKERRQ(PetscFree2(ctx->A,ctx->scatter_id));
  }
  if (fctx && nep->scheme==NEP_REFINE_SCHEME_SCHUR) {
    CHKERRQ(MatDestroy(&P));
    CHKERRQ(MatDestroy(&fctx->M1));
    CHKERRQ(PetscFree(fctx));
  }
  if (nep->scheme==NEP_REFINE_SCHEME_EXPLICIT) {
    CHKERRQ(MatDestroy(&Mt));
    CHKERRQ(VecDestroy(&dvv));
    CHKERRQ(VecDestroy(&rr));
    CHKERRQ(VecDestroy(&ctx->nv));
    if (nep->npart>1) {
      for (i=0;i<nep->nt;i++) CHKERRQ(FNDestroy(&ctx->fn[i]));
      CHKERRQ(PetscFree(ctx->fn));
    }
  }
  if (nep->scheme==NEP_REFINE_SCHEME_MBE) {
    if (nep->npart>1) {
      for (i=0;i<nep->nt;i++) CHKERRQ(FNDestroy(&ctx->fn[i]));
      CHKERRQ(PetscFree(ctx->fn));
    }
  }
  CHKERRQ(MatDestroy(&T));
  CHKERRQ(PetscFree(ctx));
  CHKERRQ(PetscLogEventEnd(NEP_Refine,nep,0,0,0));
  PetscFunctionReturn(0);
}
