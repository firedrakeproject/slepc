/*
   Subroutines related to special Vecs that share a common contiguous storage.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/vecimplslepc.h>            /*I "slepcvec.h" I*/
#include <petscblaslapack.h>

PetscLogEvent SLEPC_UpdateVectors = 0,SLEPC_VecMAXPBY = 0;

#undef __FUNCT__
#define __FUNCT__ "Vecs_ContiguousDestroy"
/*
  Frees the array of the contiguous vectors when all vectors have been destroyed.
*/
static PetscErrorCode Vecs_ContiguousDestroy(void *ctx)
{
  PetscErrorCode  ierr;
  Vecs_Contiguous *vc = (Vecs_Contiguous*)ctx;

  PetscFunctionBegin;
  ierr = PetscFree(vc->array);CHKERRQ(ierr);
  ierr = PetscFree(vc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcVecDuplicateVecs"
/*@
   SlepcVecDuplicateVecs - Creates several vectors of the same type as an existing vector,
   with contiguous storage.

   Collective on Vec

   Input Parameters:
+  v - a vector to mimic
-  m - the number of vectors to obtain

   Output Parameter:
.  V - location to put pointer to array of vectors

   Notes:
   The only difference with respect to PETSc's VecDuplicateVecs() is that storage is
   contiguous, that is, the array of values of V[1] immediately follows the array
   of V[0], and so on.

   Use SlepcVecDestroyVecs() to free the space.

   Level: developer

.seealso: SlepcVecDestroyVecs()
@*/
PetscErrorCode SlepcVecDuplicateVecs(Vec v,PetscInt m,Vec *V[])
{
  PetscErrorCode  ierr;
  PetscInt        i,nloc;
  PetscScalar     *pV;
  PetscContainer  container;
  Vecs_Contiguous *vc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(v,m,2);
  PetscValidPointer(V,3);
  PetscValidType(v,1);
  if (m <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %D",m);
  /* Allocate array */
  ierr = VecGetLocalSize(v,&nloc);CHKERRQ(ierr);
  ierr = PetscMalloc(m*nloc*sizeof(PetscScalar),&pV);CHKERRQ(ierr);
  /* Create container */
  ierr = PetscNew(Vecs_Contiguous,&vc);CHKERRQ(ierr);
  vc->nvecs = m;
  vc->array = pV;
  ierr = PetscContainerCreate(((PetscObject)v)->comm,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,vc);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,Vecs_ContiguousDestroy);CHKERRQ(ierr);
  /* Create vectors */
  ierr = PetscMalloc(m*sizeof(Vec),V);CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    ierr = VecCreateMPIWithArray(((PetscObject)v)->comm,nloc,PETSC_DECIDE,pV+i*nloc,*V+i);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)*(*V+i),"contiguous",(PetscObject)container);CHKERRQ(ierr);
  }
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcVecDestroyVecs"
/*@
   SlepcVecDestroyVecs - Frees a block of vectors obtained with SlepcVecDuplicateVecs().

   Collective on Vec

   Input Parameters:
+  m - the number of vectors previously obtained
-  V - pointer to array of vectors

   Level: developer

.seealso: SlepcVecDuplicateVecs()
@*/
PetscErrorCode SlepcVecDestroyVecs(PetscInt m,Vec *V[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(V,2);
  if (m <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %D",m);
  SlepcValidVecsContiguous(*V,m,2);
  for (i=0;i<m;i++) {
    ierr = VecDestroy(*V+i);CHKERRQ(ierr);
  }
  ierr = PetscFree(*V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcUpdateVectors"
/*@
   SlepcUpdateVectors - Update a set of vectors V as V(:,s:e-1) = V*Q(:,s:e-1).

   Not Collective

   Input parameters:
+  n      - number of vectors in V
.  s      - first column of V to be overwritten
.  e      - first column of V not to be overwritten
.  Q      - matrix containing the coefficients of the update
.  ldq    - leading dimension of Q
-  qtrans - flag indicating if Q is to be transposed

   Input/Output parameter:
.  V      - set of vectors

   Notes: 
   This function computes V(:,s:e-1) = V*Q(:,s:e-1), that is, given a set of
   vectors V, columns from s to e-1 are overwritten with columns from s to
   e-1 of the matrix-matrix product V*Q.

   Matrix V is represented as an array of Vec, whereas Q is represented as
   a column-major dense array of leading dimension ldq. Only columns s to e-1
   of Q are referenced.

   If qtrans=PETSC_TRUE, the operation is V*Q'.

   This routine is implemented with a call to BLAS, therefore V is an array 
   of Vec which have the data stored contiguously in memory as a Fortran matrix.
   PETSc does not create such arrays by default.

   Level: developer

@*/
PetscErrorCode SlepcUpdateVectors(PetscInt n_,Vec *V,PetscInt s,PetscInt e,const PetscScalar *Q,PetscInt ldq_,PetscBool qtrans)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SlepcUpdateStrideVectors(n_,V,s,1,e,Q,ldq_,qtrans);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcUpdateStrideVectors"
/*@
   SlepcUpdateStrideVectors - Update a set of vectors V as
   V(:,s:d:e-1) = V*Q(:,s:e-1).

   Not Collective

   Input parameters:
+  n      - number of vectors in V
.  s      - first column of V to be overwritten
.  d      - stride
.  e      - first column of V not to be overwritten
.  Q      - matrix containing the coefficients of the update
.  ldq    - leading dimension of Q
-  qtrans - flag indicating if Q is to be transposed

   Input/Output parameter:
.  V      - set of vectors

   Notes: 
   This function computes V(:,s:d:e-1) = V*Q(:,s:e-1), that is, given a set
   of vectors V, columns from s to e-1 are overwritten with columns from s to
   e-1 of the matrix-matrix product V*Q.

   Matrix V is represented as an array of Vec, whereas Q is represented as
   a column-major dense array of leading dimension ldq. Only columns s to e-1
   of Q are referenced.

   If qtrans=PETSC_TRUE, the operation is V*Q'.

   This routine is implemented with a call to BLAS, therefore V is an array 
   of Vec which have the data stored contiguously in memory as a Fortran matrix.
   PETSc does not create such arrays by default.

   Level: developer

@*/
PetscErrorCode SlepcUpdateStrideVectors(PetscInt n_,Vec *V,PetscInt s,PetscInt d,PetscInt e,const PetscScalar *Q,PetscInt ldq_,PetscBool qtrans)
{
  PetscErrorCode ierr;
  PetscInt       l;
  PetscBLASInt   i,j,k,bs=64,m,n,ldq,ls,ld;
  PetscScalar    *pv,*pw,*pq,*work,*pwork,one=1.0,zero=0.0;
  const char     *qt;

  PetscFunctionBegin;
  n = PetscBLASIntCast(n_/d);
  ldq = PetscBLASIntCast(ldq_);
  m = (e-s)/d;
  if (m==0) PetscFunctionReturn(0);
  PetscValidIntPointer(Q,5);
  if (m<0 || n<0 || s<0 || m>n) SETERRQ(((PetscObject)*V)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index argument out of range");
  ierr = PetscLogEventBegin(SLEPC_UpdateVectors,0,0,0,0);CHKERRQ(ierr);
  ierr = VecGetLocalSize(V[0],&l);CHKERRQ(ierr);
  ls = PetscBLASIntCast(l);
  ld = ls*PetscBLASIntCast(d);
  ierr = VecGetArray(V[0],&pv);CHKERRQ(ierr);
  if (qtrans) {
    pq = (PetscScalar*)Q+s;
    qt = "T";
  } else {
    pq = (PetscScalar*)Q+s*ldq;
    qt = "N";
  }
  ierr = PetscMalloc(sizeof(PetscScalar)*bs*m,&work);CHKERRQ(ierr);
  k = ls % bs;
  if (k) {
    BLASgemm_("N",qt,&k,&m,&n,&one,pv,&ld,pq,&ldq,&zero,work,&k);
    for (j=0;j<m;j++) {
      pw = pv+(s+j)*ld;
      pwork = work+j*k;
      for (i=0;i<k;i++) {
        *pw++ = *pwork++;
      }
    }        
  }
  for (;k<ls;k+=bs) {
    BLASgemm_("N",qt,&bs,&m,&n,&one,pv+k,&ld,pq,&ldq,&zero,work,&bs);
    for (j=0;j<m;j++) {
      pw = pv+(s+j)*ld+k;
      pwork = work+j*bs;
      for (i=0;i<bs;i++) {
        *pw++ = *pwork++;
      }
    }
  }
  ierr = VecRestoreArray(V[0],&pv);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscLogFlops(m*n*2.0*ls);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SLEPC_UpdateVectors,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcVecMAXPBY"
/*@
   SlepcVecMAXPBY - Computes y = beta*y + sum alpha*a[j]*x[j]

   Logically Collective on Vec

   Input parameters:
+  beta   - scalar beta
.  alpha  - scalar alpha
.  nv     - number of vectors in x and scalars in a
.  a      - array of scalars
-  x      - set of vectors

   Input/Output parameter:
.  y      - the vector to update

   Notes:
   This routine is implemented with a call to BLAS, therefore x is an array 
   of Vec which have the data stored contiguously in memory as a Fortran matrix.
   PETSc does not create such arrays by default.

   Level: developer

@*/
PetscErrorCode SlepcVecMAXPBY(Vec y,PetscScalar beta,PetscScalar alpha,PetscInt nv,PetscScalar a[],Vec x[])
{
  PetscErrorCode    ierr;
  PetscBLASInt      n,m,one=1;
  PetscScalar       *py;
  const PetscScalar *px;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(y,VEC_CLASSID,1);
  if (!nv || !(y)->map->n) PetscFunctionReturn(0);
  if (nv < 0) SETERRQ1(((PetscObject)y)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",nv);
  PetscValidLogicalCollectiveScalar(y,alpha,2);
  PetscValidLogicalCollectiveScalar(y,beta,3);
  PetscValidLogicalCollectiveInt(y,nv,4);
  PetscValidScalarPointer(a,5);
  PetscValidPointer(x,6);
  PetscValidHeaderSpecific(*x,VEC_CLASSID,6);
  PetscValidType(y,1);
  PetscValidType(*x,6);
  PetscCheckSameTypeAndComm(y,1,*x,6);
  if ((*x)->map->N != (y)->map->N) SETERRQ(((PetscObject)y)->comm,PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if ((*x)->map->n != (y)->map->n) SETERRQ(((PetscObject)y)->comm,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(SLEPC_VecMAXPBY,*x,y,0,0);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecGetArrayRead(*x,&px);CHKERRQ(ierr);
  n = PetscBLASIntCast(nv);
  m = PetscBLASIntCast((y)->map->n);
  BLASgemv_("N",&m,&n,&alpha,px,&m,a,&one,&beta,py,&one);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(*x,&px);CHKERRQ(ierr);
  ierr = PetscLogFlops(nv*2*(y)->map->n);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SLEPC_VecMAXPBY,*x,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

