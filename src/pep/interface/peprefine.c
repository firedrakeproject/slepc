/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Newton refinement for PEP, simple version
*/

#include <slepc/private/pepimpl.h>
#include <slepcblaslapack.h>

#define NREF_MAXIT 10

typedef struct {
  VecScatter *scatter_id,nst;
  Mat        *A;
  Vec        nv,vg,v,w;
} PEPSimpNRefctx;

typedef struct {
  Mat          M1;
  Vec          M2,M3;
  PetscScalar  M4,m3;
} PEP_REFINES_MATSHELL;

static PetscErrorCode MatMult_FS(Mat M ,Vec x,Vec y)
{
  PEP_REFINES_MATSHELL *ctx;
  PetscScalar          t;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(VecDot(x,ctx->M3,&t));
  t *= ctx->m3/ctx->M4;
  CHKERRQ(MatMult(ctx->M1,x,y));
  CHKERRQ(VecAXPY(y,-t,ctx->M2));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPSimpleNRefSetUp(PEP pep,PEPSimpNRefctx **ctx_)
{
  PetscInt       i,si,j,n0,m0,nloc,*idx1,*idx2,ne;
  IS             is1,is2;
  PEPSimpNRefctx *ctx;
  Vec            v;
  PetscMPIInt    rank,size;
  MPI_Comm       child;

  PetscFunctionBegin;
  CHKERRQ(PetscCalloc1(1,ctx_));
  ctx = *ctx_;
  if (pep->npart==1) {
    pep->refinesubc = NULL;
    ctx->scatter_id = NULL;
    ctx->A = pep->A;
  } else {
    CHKERRQ(PetscSubcommGetChild(pep->refinesubc,&child));
    CHKERRQ(PetscMalloc2(pep->nmat,&ctx->A,pep->npart,&ctx->scatter_id));

    /* Duplicate matrices */
    for (i=0;i<pep->nmat;i++) {
      CHKERRQ(MatCreateRedundantMatrix(pep->A[i],0,child,MAT_INITIAL_MATRIX,&ctx->A[i]));
      CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->A[i]));
    }
    CHKERRQ(MatCreateVecs(ctx->A[0],&ctx->v,NULL));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->v));

    /* Create scatters for sending vectors to each subcommucator */
    CHKERRQ(BVGetColumn(pep->V,0,&v));
    CHKERRQ(VecGetOwnershipRange(v,&n0,&m0));
    CHKERRQ(BVRestoreColumn(pep->V,0,&v));
    CHKERRQ(VecGetLocalSize(ctx->v,&nloc));
    CHKERRQ(PetscMalloc2(m0-n0,&idx1,m0-n0,&idx2));
    CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)pep),nloc,PETSC_DECIDE,&ctx->vg));
    for (si=0;si<pep->npart;si++) {
      j = 0;
      for (i=n0;i<m0;i++) {
        idx1[j]   = i;
        idx2[j++] = i+pep->n*si;
      }
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pep),(m0-n0),idx1,PETSC_COPY_VALUES,&is1));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pep),(m0-n0),idx2,PETSC_COPY_VALUES,&is2));
      CHKERRQ(BVGetColumn(pep->V,0,&v));
      CHKERRQ(VecScatterCreate(v,is1,ctx->vg,is2,&ctx->scatter_id[si]));
      CHKERRQ(BVRestoreColumn(pep->V,0,&v));
      CHKERRQ(ISDestroy(&is1));
      CHKERRQ(ISDestroy(&is2));
    }
    CHKERRQ(PetscFree2(idx1,idx2));
  }
  if (pep->scheme==PEP_REFINE_SCHEME_EXPLICIT) {
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ctx->A[0]),&rank));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ctx->A[0]),&size));
    if (size>1) {
      if (pep->npart==1) {
        CHKERRQ(BVGetColumn(pep->V,0,&v));
      } else v = ctx->v;
      CHKERRQ(VecGetOwnershipRange(v,&n0,&m0));
      ne = (rank == size-1)?pep->n:0;
      CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)ctx->A[0]),ne,PETSC_DECIDE,&ctx->nv));
      CHKERRQ(PetscMalloc1(m0-n0,&idx1));
      for (i=n0;i<m0;i++) idx1[i-n0] = i;
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)ctx->A[0]),(m0-n0),idx1,PETSC_COPY_VALUES,&is1));
      CHKERRQ(VecScatterCreate(v,is1,ctx->nv,is1,&ctx->nst));
      if (pep->npart==1) {
        CHKERRQ(BVRestoreColumn(pep->V,0,&v));
      }
      CHKERRQ(PetscFree(idx1));
      CHKERRQ(ISDestroy(&is1));
    }
  }
  PetscFunctionReturn(0);
}

/*
  Gather Eigenpair idx from subcommunicator with color sc
*/
static PetscErrorCode PEPSimpleNRefGatherEigenpair(PEP pep,PEPSimpNRefctx *ctx,PetscInt sc,PetscInt idx,PetscInt *fail)
{
  PetscMPIInt       nproc,p;
  MPI_Comm          comm=((PetscObject)pep)->comm;
  Vec               v;
  const PetscScalar *array;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&nproc));
  p = (nproc/pep->npart)*(sc+1)+PetscMin(nproc%pep->npart,sc+1)-1;
  if (pep->npart>1) {
    /* Communicate convergence successful */
    CHKERRMPI(MPI_Bcast(fail,1,MPIU_INT,p,comm));
    if (!(*fail)) {
      /* Process 0 of subcommunicator sc broadcasts the eigenvalue */
      CHKERRMPI(MPI_Bcast(&pep->eigr[idx],1,MPIU_SCALAR,p,comm));
      /* Gather pep->V[idx] from the subcommuniator sc */
      CHKERRQ(BVGetColumn(pep->V,idx,&v));
      if (pep->refinesubc->color==sc) {
        CHKERRQ(VecGetArrayRead(ctx->v,&array));
        CHKERRQ(VecPlaceArray(ctx->vg,array));
      }
      CHKERRQ(VecScatterBegin(ctx->scatter_id[sc],ctx->vg,v,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(ctx->scatter_id[sc],ctx->vg,v,INSERT_VALUES,SCATTER_REVERSE));
      if (pep->refinesubc->color==sc) {
        CHKERRQ(VecResetArray(ctx->vg));
        CHKERRQ(VecRestoreArrayRead(ctx->v,&array));
      }
      CHKERRQ(BVRestoreColumn(pep->V,idx,&v));
    }
  } else {
    if (pep->scheme==PEP_REFINE_SCHEME_EXPLICIT && !(*fail)) {
      CHKERRMPI(MPI_Bcast(&pep->eigr[idx],1,MPIU_SCALAR,p,comm));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPSimpleNRefScatterEigenvector(PEP pep,PEPSimpNRefctx *ctx,PetscInt sc,PetscInt idx)
{
  Vec               v;
  const PetscScalar *array;

  PetscFunctionBegin;
  if (pep->npart>1) {
    CHKERRQ(BVGetColumn(pep->V,idx,&v));
    if (pep->refinesubc->color==sc) {
      CHKERRQ(VecGetArrayRead(ctx->v,&array));
      CHKERRQ(VecPlaceArray(ctx->vg,array));
    }
    CHKERRQ(VecScatterBegin(ctx->scatter_id[sc],v,ctx->vg,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(ctx->scatter_id[sc],v,ctx->vg,INSERT_VALUES,SCATTER_FORWARD));
    if (pep->refinesubc->color==sc) {
      CHKERRQ(VecResetArray(ctx->vg));
      CHKERRQ(VecRestoreArrayRead(ctx->v,&array));
    }
    CHKERRQ(BVRestoreColumn(pep->V,idx,&v));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPEvaluateFunctionDerivatives(PEP pep,PetscScalar alpha,PetscScalar *vals)
{
  PetscInt    i,nmat=pep->nmat;
  PetscScalar a0,a1,a2;
  PetscReal   *a=pep->pbc,*b=a+nmat,*g=b+nmat;

  PetscFunctionBegin;
  a0 = 0.0;
  a1 = 1.0;
  vals[0] = 0.0;
  if (nmat>1) vals[1] = 1/a[0];
  for (i=2;i<nmat;i++) {
    a2 = ((alpha-b[i-2])*a1-g[i-2]*a0)/a[i-2];
    vals[i] = (a2+(alpha-b[i-1])*vals[i-1]-g[i-1]*vals[i-2])/a[i-1];
    a0 = a1; a1 = a2;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPSimpleNRefSetUpSystem(PEP pep,Mat *A,PEPSimpNRefctx *ctx,PetscInt idx,Mat *Mt,Mat *T,Mat *P,PetscBool ini,Vec t,Vec v)
{
  PetscErrorCode       ierr;
  PetscInt             i,nmat=pep->nmat,ml,m0,n0,m1,mg;
  PetscInt             *dnz,*onz,ncols,*cols2=NULL,*nnz;
  PetscScalar          zero=0.0,*coeffs,*coeffs2;
  PetscMPIInt          rank,size;
  MPI_Comm             comm;
  const PetscInt       *cols;
  const PetscScalar    *vals,*array;
  MatStructure         str;
  PEP_REFINES_MATSHELL *fctx;
  PEPRefineScheme      scheme=pep->scheme;
  Vec                  w=ctx->w;
  Mat                  M;

  PetscFunctionBegin;
  CHKERRQ(STGetMatStructure(pep->st,&str));
  CHKERRQ(PetscMalloc2(nmat,&coeffs,nmat,&coeffs2));
  switch (scheme) {
  case PEP_REFINE_SCHEME_SCHUR:
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
  case PEP_REFINE_SCHEME_MBE:
    M=*T;
    break;
  case PEP_REFINE_SCHEME_EXPLICIT:
    M=*Mt;
    break;
  }
  if (ini) {
    CHKERRQ(MatDuplicate(A[0],MAT_COPY_VALUES,&M));
  } else {
    CHKERRQ(MatCopy(A[0],M,DIFFERENT_NONZERO_PATTERN));
  }
  CHKERRQ(PEPEvaluateBasis(pep,pep->eigr[idx],0,coeffs,NULL));
  CHKERRQ(MatScale(M,coeffs[0]));
  for (i=1;i<nmat;i++) {
    CHKERRQ(MatAXPY(M,coeffs[i],A[i],(ini)?str:SUBSET_NONZERO_PATTERN));
  }
  CHKERRQ(PEPEvaluateFunctionDerivatives(pep,pep->eigr[idx],coeffs2));
  for (i=0;i<nmat && PetscAbsScalar(coeffs2[i])==0.0;i++);
  CHKERRQ(MatMult(A[i],v,w));
  if (coeffs2[i]!=1.0) {
    CHKERRQ(VecScale(w,coeffs2[i]));
  }
  for (i++;i<nmat;i++) {
    CHKERRQ(MatMult(A[i],v,t));
    CHKERRQ(VecAXPY(w,coeffs2[i],t));
  }
  switch (scheme) {
  case PEP_REFINE_SCHEME_EXPLICIT:
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
        CHKERRQ(PetscMalloc1(pep->n,&cols2));
        for (i=0;i<pep->n;i++) cols2[i]=i;
      }
      CHKERRQ(VecScatterBegin(ctx->nst,v,ctx->nv,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(ctx->nst,v,ctx->nv,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecGetArrayRead(ctx->nv,&array));
      if (rank==size-1) {
        CHKERRQ(MatSetValues(*T,1,&mg,pep->n,cols2,array,INSERT_VALUES));
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
  case PEP_REFINE_SCHEME_SCHUR:
    fctx->M2 = w;
    fctx->M3 = v;
    fctx->m3 = 0.0;
    for (i=1;i<nmat-1;i++) fctx->m3 += PetscConj(coeffs[i])*coeffs[i];
    fctx->M4 = 0.0;
    for (i=1;i<nmat-1;i++) fctx->M4 += PetscConj(coeffs[i])*coeffs2[i];
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
  case PEP_REFINE_SCHEME_MBE:
    *T = M;
    *P = M;
    break;
  }
  CHKERRQ(PetscFree2(coeffs,coeffs2));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPNewtonRefinementSimple(PEP pep,PetscInt *maxits,PetscReal tol,PetscInt k)
{
  PetscInt             i,n,its,idx=0,*idx_sc,*its_sc,color,*fail_sc;
  PetscMPIInt          rank,size;
  Mat                  Mt=NULL,T=NULL,P=NULL;
  MPI_Comm             comm;
  Vec                  r,v,dv,rr=NULL,dvv=NULL,t[2];
  PetscScalar          *array2,deig=0.0,tt[2],ttt;
  const PetscScalar    *array;
  PetscReal            norm,error;
  PetscBool            ini=PETSC_TRUE,sc_pend,solved=PETSC_FALSE;
  PEPSimpNRefctx       *ctx;
  PEP_REFINES_MATSHELL *fctx=NULL;
  KSPConvergedReason   reason;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(PEP_Refine,pep,0,0,0));
  CHKERRQ(PEPSimpleNRefSetUp(pep,&ctx));
  its = (maxits)?*maxits:NREF_MAXIT;
  if (!pep->refineksp) CHKERRQ(PEPRefineGetKSP(pep,&pep->refineksp));
  if (pep->npart==1) {
    CHKERRQ(BVGetColumn(pep->V,0,&v));
  } else v = ctx->v;
  CHKERRQ(VecDuplicate(v,&ctx->w));
  CHKERRQ(VecDuplicate(v,&r));
  CHKERRQ(VecDuplicate(v,&dv));
  CHKERRQ(VecDuplicate(v,&t[0]));
  CHKERRQ(VecDuplicate(v,&t[1]));
  if (pep->npart==1) {
    CHKERRQ(BVRestoreColumn(pep->V,0,&v));
    CHKERRQ(PetscObjectGetComm((PetscObject)pep,&comm));
  } else {
    CHKERRQ(PetscSubcommGetChild(pep->refinesubc,&comm));
  }
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(VecGetLocalSize(r,&n));
  CHKERRQ(PetscMalloc3(pep->npart,&idx_sc,pep->npart,&its_sc,pep->npart,&fail_sc));
  for (i=0;i<pep->npart;i++) fail_sc[i] = 0;
  for (i=0;i<pep->npart;i++) its_sc[i] = 0;
  color = (pep->npart==1)?0:pep->refinesubc->color;

  /* Loop performing iterative refinements */
  while (!solved) {
    for (i=0;i<pep->npart;i++) {
      sc_pend = PETSC_TRUE;
      if (its_sc[i]==0) {
        idx_sc[i] = idx++;
        if (idx_sc[i]>=k) {
          sc_pend = PETSC_FALSE;
        } else {
          CHKERRQ(PEPSimpleNRefScatterEigenvector(pep,ctx,i,idx_sc[i]));
        }
      }  else { /* Gather Eigenpair from subcommunicator i */
        CHKERRQ(PEPSimpleNRefGatherEigenpair(pep,ctx,i,idx_sc[i],&fail_sc[i]));
      }
      while (sc_pend) {
        if (!fail_sc[i]) {
          CHKERRQ(PEPComputeError(pep,idx_sc[i],PEP_ERROR_BACKWARD,&error));
        }
        if (error<=tol || its_sc[i]>=its || fail_sc[i]) {
          idx_sc[i] = idx++;
          its_sc[i] = 0;
          fail_sc[i] = 0;
          if (idx_sc[i]<k) CHKERRQ(PEPSimpleNRefScatterEigenvector(pep,ctx,i,idx_sc[i]));
        } else {
          sc_pend = PETSC_FALSE;
          its_sc[i]++;
        }
        if (idx_sc[i]>=k) sc_pend = PETSC_FALSE;
      }
    }
    solved = PETSC_TRUE;
    for (i=0;i<pep->npart&&solved;i++) solved = PetscNot(idx_sc[i]<k);
    if (idx_sc[color]<k) {
#if !defined(PETSC_USE_COMPLEX)
      PetscCheck(pep->eigi[idx_sc[color]]==0.0,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Simple Refinement not implemented in real scalars for complex eigenvalues");
#endif
      if (pep->npart==1) {
        CHKERRQ(BVGetColumn(pep->V,idx_sc[color],&v));
      } else v = ctx->v;
      CHKERRQ(PEPSimpleNRefSetUpSystem(pep,ctx->A,ctx,idx_sc[color],&Mt,&T,&P,ini,t[0],v));
      CHKERRQ(PEP_KSPSetOperators(pep->refineksp,T,P));
      if (ini) {
        CHKERRQ(KSPSetFromOptions(pep->refineksp));
        if (pep->scheme==PEP_REFINE_SCHEME_EXPLICIT) {
          CHKERRQ(MatCreateVecs(T,&dvv,NULL));
          CHKERRQ(VecDuplicate(dvv,&rr));
        }
        ini = PETSC_FALSE;
      }

      switch (pep->scheme) {
      case PEP_REFINE_SCHEME_EXPLICIT:
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
        CHKERRQ(KSPSolve(pep->refineksp,rr,dvv));
        CHKERRQ(KSPGetConvergedReason(pep->refineksp,&reason));
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
          if (rank==size-1) pep->eigr[idx_sc[color]] -= array[n];
          CHKERRQ(VecRestoreArrayRead(dvv,&array));
        } else fail_sc[color] = 1;
        break;
      case PEP_REFINE_SCHEME_MBE:
        CHKERRQ(MatMult(T,v,r));
        /* Mixed block elimination */
        CHKERRQ(VecConjugate(v));
        CHKERRQ(KSPSolveTranspose(pep->refineksp,v,t[0]));
        CHKERRQ(KSPGetConvergedReason(pep->refineksp,&reason));
        if (reason>0) {
          CHKERRQ(VecConjugate(t[0]));
          CHKERRQ(VecDot(ctx->w,t[0],&tt[0]));
          CHKERRQ(KSPSolve(pep->refineksp,ctx->w,t[1]));
          CHKERRQ(KSPGetConvergedReason(pep->refineksp,&reason));
          if (reason>0) {
            CHKERRQ(VecDot(t[1],v,&tt[1]));
            CHKERRQ(VecDot(r,t[0],&ttt));
            tt[0] = ttt/tt[0];
            CHKERRQ(VecAXPY(r,-tt[0],ctx->w));
            CHKERRQ(KSPSolve(pep->refineksp,r,dv));
            CHKERRQ(KSPGetConvergedReason(pep->refineksp,&reason));
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
          pep->eigr[idx_sc[color]] -= deig;
          fail_sc[color] = 0;
        } else {
          CHKERRQ(VecConjugate(v));
          fail_sc[color] = 1;
        }
        break;
      case PEP_REFINE_SCHEME_SCHUR:
        fail_sc[color] = 1;
        CHKERRQ(MatShellGetContext(T,&fctx));
        if (fctx->M4!=0.0) {
          CHKERRQ(MatMult(fctx->M1,v,r));
          CHKERRQ(KSPSolve(pep->refineksp,r,dv));
          CHKERRQ(KSPGetConvergedReason(pep->refineksp,&reason));
          if (reason>0) {
            CHKERRQ(VecDot(dv,v,&deig));
            deig *= -fctx->m3/fctx->M4;
            CHKERRQ(VecAXPY(v,-1.0,dv));
            CHKERRQ(VecNorm(v,NORM_2,&norm));
            CHKERRQ(VecScale(v,1.0/norm));
            pep->eigr[idx_sc[color]] -= deig;
            fail_sc[color] = 0;
          }
        }
        break;
      }
      if (pep->npart==1) CHKERRQ(BVRestoreColumn(pep->V,idx_sc[color],&v));
    }
  }
  CHKERRQ(VecDestroy(&t[0]));
  CHKERRQ(VecDestroy(&t[1]));
  CHKERRQ(VecDestroy(&dv));
  CHKERRQ(VecDestroy(&ctx->w));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(PetscFree3(idx_sc,its_sc,fail_sc));
  CHKERRQ(VecScatterDestroy(&ctx->nst));
  if (pep->npart>1) {
    CHKERRQ(VecDestroy(&ctx->vg));
    CHKERRQ(VecDestroy(&ctx->v));
    for (i=0;i<pep->nmat;i++) {
      CHKERRQ(MatDestroy(&ctx->A[i]));
    }
    for (i=0;i<pep->npart;i++) {
      CHKERRQ(VecScatterDestroy(&ctx->scatter_id[i]));
    }
    CHKERRQ(PetscFree2(ctx->A,ctx->scatter_id));
  }
  if (fctx && pep->scheme==PEP_REFINE_SCHEME_SCHUR) {
    CHKERRQ(MatDestroy(&P));
    CHKERRQ(MatDestroy(&fctx->M1));
    CHKERRQ(PetscFree(fctx));
  }
  if (pep->scheme==PEP_REFINE_SCHEME_EXPLICIT) {
    CHKERRQ(MatDestroy(&Mt));
    CHKERRQ(VecDestroy(&dvv));
    CHKERRQ(VecDestroy(&rr));
    CHKERRQ(VecDestroy(&ctx->nv));
  }
  CHKERRQ(MatDestroy(&T));
  CHKERRQ(PetscFree(ctx));
  CHKERRQ(PetscLogEventEnd(PEP_Refine,pep,0,0,0));
  PetscFunctionReturn(0);
}
