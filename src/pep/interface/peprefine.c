/*
   Newton refinement for PEP, simple version.

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

#include <slepc/private/pepimpl.h>
#include <slepcblaslapack.h>

#define NREF_MAXIT 10

typedef struct {
  VecScatter    *scatter_id;
  Mat           *A;
  Vec           vg,v;
} PEPSimpNRefctx;

#undef __FUNCT__
#define __FUNCT__ "PEPSimpleNRefSetUp"
static PetscErrorCode PEPSimpleNRefSetUp(PEP pep,PEPSimpNRefctx **ctx_)
{
  PetscErrorCode ierr;
  PetscInt       i,si,j,n0,m0,nloc,*idx1,*idx2;
  IS             is1,is2;
  PEPSimpNRefctx *ctx;
  Vec            v;

  PetscFunctionBegin;
  ierr = PetscMalloc1(1,ctx_);CHKERRQ(ierr);
  ctx = *ctx_;
  if (pep->npart==1) {
    pep->refinesubc = NULL;
    ctx->scatter_id = NULL;
    ctx->A = pep->A;
  } else {
    ierr = PetscMalloc2(pep->nmat,&ctx->A,pep->npart,&ctx->scatter_id);CHKERRQ(ierr);

    /* Duplicate matrices */
    for (i=0;i<pep->nmat;i++) {
      ierr = MatCreateRedundantMatrix(pep->A[i],0,PetscSubcommChild(pep->refinesubc),MAT_INITIAL_MATRIX,&ctx->A[i]);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(ctx->A[0],&ctx->v,NULL);CHKERRQ(ierr);

    /* Create scatters for sending vectors to each subcommucator */
    ierr = BVGetColumn(pep->V,0,&v);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(v,&n0,&m0);CHKERRQ(ierr);
    ierr = BVRestoreColumn(pep->V,0,&v);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ctx->v,&nloc);CHKERRQ(ierr);
    ierr = PetscMalloc2(m0-n0,&idx1,m0-n0,&idx2);CHKERRQ(ierr);
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)pep),nloc,PETSC_DECIDE,&ctx->vg);CHKERRQ(ierr);
    for (si=0;si<pep->npart;si++) {
      j = 0;
      for (i=n0;i<m0;i++) {
        idx1[j]   = i;
        idx2[j++] = i+pep->n*si;
      }
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pep),(m0-n0),idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pep),(m0-n0),idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
      ierr = BVGetColumn(pep->V,0,&v);CHKERRQ(ierr);
      ierr = VecScatterCreate(v,is1,ctx->vg,is2,&ctx->scatter_id[si]);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,0,&v);CHKERRQ(ierr);
      ierr = ISDestroy(&is1);CHKERRQ(ierr);
      ierr = ISDestroy(&is2);CHKERRQ(ierr);
    }
    ierr = PetscFree2(idx1,idx2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);  
}

/*
  Gather Eigenpair idx from subcommunicator with color sc
*/
#undef __FUNCT__
#define __FUNCT__ "PEPSimpleNRefGatherEigenpair"
static PetscErrorCode PEPSimpleNRefGatherEigenpair(PEP pep,PEPSimpNRefctx *ctx,PetscInt sc,PetscInt idx)
{
  PetscErrorCode    ierr;
  PetscMPIInt       nproc,p;
  MPI_Comm          comm=((PetscObject)pep)->comm;
  Vec               v;
  const PetscScalar *array;

  PetscFunctionBegin;
  /* The eigenvalue information is in the last process of the 
     subcommunicator sc. p is its mapping in the general comm */
  ierr = MPI_Comm_size(comm,&nproc);CHKERRQ(ierr);
  p = (nproc/pep->npart)*(sc+1)+PetscMin(nproc%pep->npart,sc+1)-1;
  ierr = MPI_Bcast(&pep->eigr[idx],1,MPIU_SCALAR,p,comm);CHKERRQ(ierr);

  if (pep->npart>1) {
    /* Gather pep->V[idx] from the subcommuniator sc */
    ierr = BVGetColumn(pep->V,idx,&v);CHKERRQ(ierr);
    if (pep->refinesubc->color==sc) {
      ierr = VecGetArrayRead(ctx->v,&array);CHKERRQ(ierr);
      ierr = VecPlaceArray(ctx->vg,array);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(ctx->scatter_id[sc],ctx->vg,v,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scatter_id[sc],ctx->vg,v,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    if (pep->refinesubc->color==sc) {
      ierr = VecResetArray(ctx->vg);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(ctx->v,&array);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(pep->V,idx,&v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSimpleNRefScatterEigenvector"
static PetscErrorCode PEPSimpleNRefScatterEigenvector(PEP pep,PEPSimpNRefctx *ctx,PetscInt sc,PetscInt idx)
{
  PetscErrorCode    ierr;
  Vec               v;
  const PetscScalar *array;
  
  PetscFunctionBegin;
  if (pep->npart>1) {
    ierr = BVGetColumn(pep->V,idx,&v);CHKERRQ(ierr);
    if (pep->refinesubc->color==sc) {
      ierr = VecGetArrayRead(ctx->v,&array);CHKERRQ(ierr);
      ierr = VecPlaceArray(ctx->vg,array);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(ctx->scatter_id[sc],v,ctx->vg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scatter_id[sc],v,ctx->vg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (pep->refinesubc->color==sc) {
      ierr = VecResetArray(ctx->vg);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(ctx->v,&array);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(pep->V,idx,&v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPEvaluateFunctionDerivatives"
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

#undef __FUNCT__
#define __FUNCT__ "PEPSimpleNRefSetUpSystem"
static PetscErrorCode PEPSimpleNRefSetUpSystem(PEP pep,Mat *A,PetscInt idx,Mat *M,Mat *T,PetscBool ini,Vec *t,Vec v)
{
  PetscErrorCode    ierr;
  PetscInt          i,nmat=pep->nmat,ml,m0,m1,mg;
  PetscInt          *dnz,*onz,ncols,*cols2,*nnz;
  PetscScalar       zero=0.0,*coeffs;
  PetscMPIInt       rank,size;
  MPI_Comm          comm;
  const PetscInt    *cols;
  const PetscScalar *vals,*array;
  MatStructure      str;
  Vec               w=t[1],q=t[0];

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)A[0]);
  ierr = STGetMatStructure(pep->st,&str);CHKERRQ(ierr);
  ierr = PetscMalloc1(nmat,&coeffs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (ini) {
    ierr = MatDuplicate(A[0],MAT_COPY_VALUES,T);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(A[0],*T,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = PEPEvaluateBasis(pep,pep->eigr[idx],0,coeffs,NULL);CHKERRQ(ierr);
  ierr = MatScale(*T,coeffs[0]);CHKERRQ(ierr);
  for (i=1;i<nmat;i++) {
    ierr = MatAXPY(*T,coeffs[i],A[i],(ini)?str:SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatGetSize(*T,&mg,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*T,&m0,&m1);CHKERRQ(ierr);
  if (ini) {
    ierr = MatCreate(comm,M);CHKERRQ(ierr);
    ierr = MatGetLocalSize(*T,&ml,NULL);CHKERRQ(ierr);
    if (rank==size-1) ml++;
    ierr = MatSetSizes(*M,ml,ml,mg+1,mg+1);CHKERRQ(ierr);
    ierr = MatSetFromOptions(*M);CHKERRQ(ierr);
    ierr = MatSetUp(*M);CHKERRQ(ierr);
    /* Preallocate M */
    if (size>1) {
      ierr = MatPreallocateInitialize(comm,ml,ml,dnz,onz);CHKERRQ(ierr);
      for (i=m0;i<m1;i++) {
        ierr = MatGetRow(*T,i,&ncols,&cols,NULL);CHKERRQ(ierr);
        ierr = MatPreallocateSet(i,ncols,cols,dnz,onz);CHKERRQ(ierr);
        ierr = MatPreallocateSet(i,1,&mg,dnz,onz);CHKERRQ(ierr);
        ierr = MatRestoreRow(*T,i,&ncols,&cols,NULL);CHKERRQ(ierr);
      }
      if (rank==size-1) {
        ierr = PetscCalloc1(mg+1,&cols2);CHKERRQ(ierr);
        for (i=0;i<mg+1;i++) cols2[i]=i;
        ierr = MatPreallocateSet(m1,mg+1,cols2,dnz,onz);CHKERRQ(ierr);
        ierr = PetscFree(cols2);CHKERRQ(ierr);
      }
      ierr = MatMPIAIJSetPreallocation(*M,0,dnz,0,onz);CHKERRQ(ierr);
      ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
    } else {
      ierr = PetscCalloc1(mg+1,&nnz);CHKERRQ(ierr);
      for (i=0;i<mg;i++) {
        ierr = MatGetRow(*T,i,&ncols,NULL,NULL);CHKERRQ(ierr);
        nnz[i] = ncols+1;
        ierr = MatRestoreRow(*T,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      }
      nnz[mg] = mg+1;
      ierr = MatSeqAIJSetPreallocation(*M,0,nnz);CHKERRQ(ierr);
      ierr = PetscFree(nnz);CHKERRQ(ierr);
    }
  }
  ierr = PEPEvaluateFunctionDerivatives(pep,pep->eigr[idx],coeffs);CHKERRQ(ierr);
  for (i=0;i<nmat && PetscAbsScalar(coeffs[i])==0.0;i++);
  ierr = MatMult(A[i],v,w);CHKERRQ(ierr);
  if (coeffs[i]!=1.0) {
    ierr = VecScale(w,coeffs[i]);CHKERRQ(ierr);
  }
  for (i++;i<nmat;i++) {
    ierr = MatMult(A[i],v,q);CHKERRQ(ierr);
    ierr = VecAXPY(w,coeffs[i],q);CHKERRQ(ierr);
  }
  
  /* Set values */
  ierr = PetscMalloc1(m1-m0,&cols2);CHKERRQ(ierr);
  for (i=0;i<m1-m0;i++) cols2[i]=m0+i;
  ierr = VecGetArrayRead(w,&array);CHKERRQ(ierr);
  for (i=m0;i<m1;i++) {
    ierr = MatGetRow(*T,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(*M,1,&i,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(*T,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(*M,1,&i,1,&mg,array+i-m0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(w,&array);CHKERRQ(ierr);
  ierr = VecConjugate(v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(v,&array);CHKERRQ(ierr);
  ierr = MatSetValues(*M,1,&mg,m1-m0,cols2,array,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*M,1,&mg,1,&mg,&zero,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v,&array);CHKERRQ(ierr);
  ierr = VecConjugate(v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = PetscFree(cols2);CHKERRQ(ierr);
  ierr = PetscFree(coeffs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPNewtonRefinementSimple"
PetscErrorCode PEPNewtonRefinementSimple(PEP pep,PetscInt *maxits,PetscReal *tol,PetscInt k)
{
  PetscErrorCode    ierr;
  PetscInt          i,n,its,idx=0,*idx_sc,*its_sc,color;
  PetscMPIInt       rank,size;
  KSP               ksp;
  Mat               M=NULL,T=NULL;
  MPI_Comm          comm;
  Vec               r,v,dv,rr=NULL,dvv=NULL,t[2];
  PetscScalar       *array2;
  const PetscScalar *array;
  PetscReal         norm,error;
  PetscBool         ini=PETSC_TRUE,sc_pend,solved=PETSC_FALSE;
  PEPSimpNRefctx    *ctx;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PEP_Refine,pep,0,0,0);CHKERRQ(ierr);
  ierr = PEPSimpleNRefSetUp(pep,&ctx);CHKERRQ(ierr);
  its = (maxits)?*maxits:NREF_MAXIT;
  comm = (pep->npart==1)?PetscObjectComm((PetscObject)pep):PetscSubcommChild(pep->refinesubc);
  ierr = PEPRefineGetKSP(pep,&ksp);CHKERRQ(ierr);
  if (pep->npart==1) {
    ierr = BVGetColumn(pep->V,0,&v);CHKERRQ(ierr);
  } else v = ctx->v;
  ierr = VecDuplicate(v,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&dv);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&t[0]);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&t[1]);CHKERRQ(ierr);
  if (pep->npart==1) { ierr = BVRestoreColumn(pep->V,0,&v);CHKERRQ(ierr); }
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = VecGetLocalSize(r,&n);CHKERRQ(ierr);
  ierr = PetscMalloc2(pep->npart,&idx_sc,pep->npart,&its_sc);CHKERRQ(ierr);
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
          ierr = PEPSimpleNRefScatterEigenvector(pep,ctx,i,idx_sc[i]);CHKERRQ(ierr);
        }
      }  else { /* Gather Eigenpair from subcommunicator i */
        ierr = PEPSimpleNRefGatherEigenpair(pep,ctx,i,idx_sc[i]);CHKERRQ(ierr);
      }
      while (sc_pend) {
        if (tol) {
          ierr = PEPComputeError(pep,idx_sc[i],PEP_ERROR_BACKWARD,&error);CHKERRQ(ierr);
        }
        if (error<=*tol || its_sc[i]>=its) {
          idx_sc[i] = idx++;
          its_sc[i] = 0;
          if (idx_sc[i]<k) { ierr = PEPSimpleNRefScatterEigenvector(pep,ctx,i,idx_sc[i]);CHKERRQ(ierr); }
        } else {
          sc_pend = PETSC_FALSE;
          its_sc[i]++;
        }
        if (idx_sc[i]>=k) sc_pend = PETSC_FALSE;
      }
    }
    solved = PETSC_TRUE;
    for (i=0;i<pep->npart&&solved;i++) solved = (idx_sc[i]<k)?PETSC_FALSE:PETSC_TRUE;
    if (idx_sc[color]<k) {
#if !defined(PETSC_USE_COMPLEX)
      if (pep->eigi[idx_sc[color]]!=0.0) SETERRQ(PetscObjectComm((PetscObject)pep),1,"Simple Refinement not implemented in real scalars for complex eigenvalues");
#endif
      if (pep->npart==1) {
        ierr = BVGetColumn(pep->V,idx_sc[color],&v);CHKERRQ(ierr);
      } else v = ctx->v; 
      ierr = PEPSimpleNRefSetUpSystem(pep,ctx->A,idx_sc[color],&M,&T,ini,t,v);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
      if (ini) {
        ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
        ierr = MatCreateVecs(M,&dvv,NULL);CHKERRQ(ierr);
        ierr = VecDuplicate(dvv,&rr);CHKERRQ(ierr);
        ini = PETSC_FALSE;
      }
      ierr = MatMult(T,v,r);CHKERRQ(ierr);
      ierr = VecGetArrayRead(r,&array);CHKERRQ(ierr);
      if (rank==size-1) {
        ierr = VecGetArray(rr,&array2);
        ierr = PetscMemcpy(array2,array,n*sizeof(PetscScalar));CHKERRQ(ierr);
        array2[n] = 0.0;
        ierr = VecRestoreArray(rr,&array2);
      } else {
        ierr = VecPlaceArray(rr,array);CHKERRQ(ierr);
      }
      ierr = KSPSolve(ksp,rr,dvv);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
      if (reason<0) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSP did not converge (reason=%s)",KSPConvergedReasons[reason]);
      if (rank != size-1) {
        ierr = VecResetArray(rr);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(r,&array);CHKERRQ(ierr);
      ierr = VecGetArrayRead(dvv,&array);CHKERRQ(ierr);
      ierr = VecPlaceArray(dv,array);CHKERRQ(ierr);
      ierr = VecAXPY(v,-1.0,dv);CHKERRQ(ierr);
      ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecScale(v,1.0/norm);CHKERRQ(ierr);
      ierr = VecResetArray(dv);CHKERRQ(ierr);
      if (rank==size-1) pep->eigr[idx_sc[color]] -= array[n];
      ierr = VecRestoreArrayRead(dvv,&array);CHKERRQ(ierr);
      if (pep->npart==1) { ierr = BVRestoreColumn(pep->V,idx_sc[color],&v);CHKERRQ(ierr); } 
    }
  }
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  ierr = VecDestroy(&t[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&t[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&dv);CHKERRQ(ierr);
  ierr = VecDestroy(&dvv);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&rr);CHKERRQ(ierr);
  ierr = PetscFree2(idx_sc,its_sc);CHKERRQ(ierr);
  if (pep->npart>1) {
    ierr = VecDestroy(&ctx->vg);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->v);CHKERRQ(ierr);
    for (i=0;i<pep->nmat;i++) {
      ierr = MatDestroy(&ctx->A[i]);CHKERRQ(ierr);
    }
    for (i=0;i<pep->npart;i++) {
      ierr = VecScatterDestroy(&ctx->scatter_id[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(ctx->A,ctx->scatter_id);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PEP_Refine,pep,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
