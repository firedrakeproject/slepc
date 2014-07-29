/*
   Newton refinement for NEP, simple version.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/nepimpl.h>
#include <slepcblaslapack.h>

#define NREF_MAXIT 100

#undef __FUNCT__
#define __FUNCT__ "NewtonSimpleRefSetUp"
static PetscErrorCode NewtonSimpleRefSetUp(NEP nep,PetscInt nmat,Mat *A,PetscInt idx,Mat *M,Mat *T,PetscBool ini,Vec *t)
{
  PetscErrorCode    ierr;
  PetscInt          i,st,ml,m0,m1,mg;
  PetscInt          *dnz,*onz,ncols,*cols2,*nnz;
  PetscScalar       *array,zero=0.0,*coeffs;
  PetscMPIInt       rank,size;
  MPI_Comm          comm;
  const PetscInt    *cols;
  const PetscScalar *vals;
  Vec               v,w=t[1],q=t[0];

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)A[0]);
  ierr = PetscMalloc1(nmat,&coeffs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (ini) {
    ierr = MatDuplicate(A[0],MAT_COPY_VALUES,T);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(A[0],*T,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  for (i=0;i<nmat;i++) {
    ierr = FNEvaluateFunction(nep->f[i],nep->eigr[idx],coeffs+i);CHKERRQ(ierr);
  }
  if (coeffs[0]!=1.0) {
    ierr = MatScale(*T,coeffs[0]);CHKERRQ(ierr);
  }
  for (i=1;i<nmat;i++) {
    ierr = MatAXPY(*T,coeffs[i],A[i],(ini)?DIFFERENT_NONZERO_PATTERN:SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
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
  for (i=0;i<nmat;i++) {
    ierr = FNEvaluateDerivative(nep->f[i],nep->eigr[idx],coeffs+i);CHKERRQ(ierr);
  }
  st = 0;
  for (i=0;i<nmat && PetscAbsScalar(coeffs[i])==0.0;i++) st++;
  ierr = BVGetColumn(nep->V,idx,&v);CHKERRQ(ierr);
  ierr = MatMult(A[st],v,w);CHKERRQ(ierr);
  if (coeffs[st]!=1.0) {
    ierr = VecScale(w,coeffs[st]);CHKERRQ(ierr);
  }
  for (i=st+1;i<nmat;i++) {
    ierr = MatMult(A[i],v,q);CHKERRQ(ierr);
    ierr = VecAXPY(w,coeffs[i],q);CHKERRQ(ierr);
  }
  ierr = BVRestoreColumn(nep->V,idx,&v);CHKERRQ(ierr);
  /* Set values */
  ierr = PetscMalloc1(m1-m0,&cols2);CHKERRQ(ierr);
  for (i=0;i<m1-m0;i++) cols2[i]=m0+i;
  ierr = VecGetArray(w,&array);CHKERRQ(ierr);
  for (i=m0;i<m1;i++) {
    ierr = MatGetRow(*T,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(*M,1,&i,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(*T,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(*M,1,&i,1,&mg,array+i-m0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(w,&array);CHKERRQ(ierr);
  ierr = BVGetColumn(nep->V,idx,&v);CHKERRQ(ierr);
  ierr = VecConjugate(v);CHKERRQ(ierr);
  ierr = VecGetArray(v,&array);CHKERRQ(ierr);
  ierr = MatSetValues(*M,1,&mg,m1-m0,cols2,array,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*M,1,&mg,1,&mg,&zero,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  ierr = VecConjugate(v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(nep->V,idx,&v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = PetscFree(cols2);CHKERRQ(ierr);
  ierr = PetscFree(coeffs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNewtonRefinementSimple"
PetscErrorCode NEPNewtonRefinementSimple(NEP nep,PetscInt *maxits,PetscReal *tol,PetscInt k)
{
  PetscErrorCode ierr;
  PetscInt       i,j,n,its;
  PetscMPIInt    rank,size;
  KSP            ksp;
  Mat            M,T;
  MPI_Comm       comm;
  Vec            r,v,dv,rr,dvv,t[2];
  PetscScalar    *array,*array2,dh;
  PetscReal      norm,error;
  PetscBool      ini=PETSC_TRUE;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(NEP_Refine,nep,0,0,0);CHKERRQ(ierr);
  its = (maxits)?*maxits:NREF_MAXIT;
  comm = PetscObjectComm((PetscObject)nep);
  ierr = KSPCreate(comm,&ksp);
  ierr = BVGetColumn(nep->V,0,&v);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&dv);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&t[0]);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&t[1]);CHKERRQ(ierr);
  ierr = BVRestoreColumn(nep->V,0,&v);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = VecGetLocalSize(r,&n);CHKERRQ(ierr);
  /* Loop performing iterative refinements */
  for (j=0;j<k;j++) {
#if !defined(PETSC_USE_COMPLEX)
      if (nep->eigi[j]!=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Simple Refinement not implemented in real scalar for complex eigenvalues");
#endif
    for (i=0;i<its;i++) {
      if (tol) {
        ierr = BVGetColumn(nep->V,j,&v);CHKERRQ(ierr);
        ierr = NEPComputeRelativeError_Private(nep,nep->eigr[j],v,&error);CHKERRQ(ierr);
        ierr = BVRestoreColumn(nep->V,j,&v);CHKERRQ(ierr);
        if (error<=*tol) break;
      }
      ierr = NewtonSimpleRefSetUp(nep,nep->nt,nep->A,j,&M,&T,ini,t);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
      if (ini) {
        ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
        ierr = MatGetVecs(M,&dvv,NULL);CHKERRQ(ierr);
        ierr = VecDuplicate(dvv,&rr);CHKERRQ(ierr);
        ini = PETSC_FALSE;
      }
      ierr = BVGetColumn(nep->V,j,&v);CHKERRQ(ierr);
      ierr = MatMult(T,v,r);CHKERRQ(ierr);
      ierr = BVRestoreColumn(nep->V,j,&v);CHKERRQ(ierr);
      ierr = VecGetArray(r,&array);CHKERRQ(ierr);
      if (rank==size-1) {
        ierr = VecGetArray(rr,&array2);
        ierr = PetscMemcpy(array2,array,n*sizeof(PetscScalar));CHKERRQ(ierr);
        array2[n] = 0.0;
        ierr = VecRestoreArray(rr,&array2);
      } else {
        ierr = VecPlaceArray(rr,array);CHKERRQ(ierr);
      }
      ierr = KSPSolve(ksp,rr,dvv);CHKERRQ(ierr);
      if (rank != size-1) {
        ierr = VecResetArray(rr);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(r,&array);CHKERRQ(ierr);
      ierr = VecGetArray(dvv,&array);CHKERRQ(ierr);
      ierr = VecPlaceArray(dv,array);CHKERRQ(ierr);
      ierr = BVGetColumn(nep->V,j,&v);CHKERRQ(ierr);
      ierr = VecAXPY(v,-1.0,dv);CHKERRQ(ierr);
      ierr = VecNorm(v,NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecScale(v,1.0/norm);CHKERRQ(ierr);
      ierr = BVRestoreColumn(nep->V,j,&v);CHKERRQ(ierr);
      ierr = VecResetArray(dv);CHKERRQ(ierr);
      if (rank==size-1) dh = array[n];
      ierr = VecRestoreArray(dvv,&array);CHKERRQ(ierr);
      ierr = MPI_Bcast(&dh,1,MPIU_SCALAR,size-1,comm);CHKERRQ(ierr);
      nep->eigr[j] -= dh;
    }
  }
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  ierr = VecDestroy(&t[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&t[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&dv);CHKERRQ(ierr);
  ierr = VecDestroy(&dvv);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&rr);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(NEP_Refine,nep,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

