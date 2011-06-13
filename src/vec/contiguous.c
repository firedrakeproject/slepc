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
#define __FUNCT__ "VecDuplicateVecs_Contiguous"
/*
  Version of VecDuplicateVecs that sets contiguous storage.
*/
static PetscErrorCode VecDuplicateVecs_Contiguous(Vec v,PetscInt m,Vec *V[])
{
  PetscErrorCode  ierr;
  PetscInt        i,nloc;
  PetscScalar     *pV;
  PetscContainer  container;
  Vecs_Contiguous *vc;

  PetscFunctionBegin;
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
#define __FUNCT__ "SlepcVecSetTemplate"
/*@
   SlepcVecSetTemplate - Sets a vector as a template for contiguous storage.

   Collective on Vec

   Input Parameters:
.  v - the vector

   Note:
   Once this function is called, subsequent calls to VecDuplicateVecs()
   with this vector will use a special version that generates vectors with
   contiguous storage, that is, the array of values of V[1] immediately
   follows the array of V[0], and so on.

   Level: developer
@*/
PetscErrorCode SlepcVecSetTemplate(Vec v)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  ierr = PetscTypeCompareAny((PetscObject)v,&flg,VECSEQ,VECMPI,"");CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)v)->comm,PETSC_ERR_SUP,"Only available for standard vectors (VECSEQ or VECMPI)");
  v->ops->duplicatevecs = VecDuplicateVecs_Contiguous;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcMatGetVecsTemplate"
/*@
   SlepcMatGetVecsTemplate - Get vectors compatible with a matrix,
   i.e. with the same parallel layout, and mark them as templates
   for contiguous storage.
   
   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  right - (optional) vector that the matrix can be multiplied against
-  left  - (optional) vector that the matrix vector product can be stored in

   Options Database Keys:
.  -slepc_non_contiguous - Disable contiguous vector storage

   Notes:
   Use -slepc_non_contiguous to disable contiguous storage throughout SLEPc.
   Contiguous storage is currently also disabled in AIJCUSP matrices.

   Level: developer

.seealso: SlepcVecSetTemplate()
@*/
PetscErrorCode SlepcMatGetVecsTemplate(Mat mat,Vec *right,Vec *left)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  Vec            v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  ierr = MatGetVecs(mat,right,left);CHKERRQ(ierr);
  v = right? *right: *left;
  ierr = PetscTypeCompareAny((PetscObject)v,&flg,VECSEQ,VECMPI,"");CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  ierr = PetscOptionsHasName(PETSC_NULL,"-slepc_non_contiguous",&flg);CHKERRQ(ierr);
  if (!flg) {
    if (right) { ierr = SlepcVecSetTemplate(*right);CHKERRQ(ierr); }
    if (left) { ierr = SlepcVecSetTemplate(*left);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcUpdateVectors_Noncontiguous_Inplace"
/*
   SlepcUpdateVectors_Noncontiguous_Inplace - V = V*Q for regular vectors
   (non-contiguous).
*/
static PetscErrorCode SlepcUpdateVectors_Noncontiguous_Inplace(PetscInt m,Vec *V,const PetscScalar *Q,PetscInt ldq,PetscBool qtrans)
{
  PetscInt       i,j,k,ls;
  PetscScalar    t,*pv,*work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(SLEPC_UpdateVectors,0,0,0,0);CHKERRQ(ierr);
  ierr = VecGetLocalSize(V[0],&ls);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*m,&work);CHKERRQ(ierr);
  for (i=0;i<ls;i++) {
    for (j=0;j<m;j++) {
      t = 0;
      for (k=0;k<m;k++) {
        ierr = VecGetArray(V[k],&pv);CHKERRQ(ierr);
        if (qtrans) t += pv[i]*Q[k*ldq+j];
        else        t += pv[i]*Q[j*ldq+k];
        ierr = VecRestoreArray(V[k],&pv);CHKERRQ(ierr);
      }
      work[j] = t;
    }
    for (j=0;j<m;j++) {
      ierr = VecGetArray(V[j],&pv);CHKERRQ(ierr);
      pv[i] = work[j];
      ierr = VecRestoreArray(V[j],&pv);CHKERRQ(ierr);    
    }
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscLogFlops(m*m*2.0*ls);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SLEPC_UpdateVectors,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcUpdateVectors_Noncontiguous"
/*
   SlepcUpdateVectors_Noncontiguous - V(:,s:e-1) = V*Q(:,s:e-1) for 
   regular vectors (non-contiguous).

   Writing V = [ V1 V2 V3 ] and Q = [ Q1 Q2 Q3 ], where the V2 and Q2
   correspond to the columns s:e-1, the computation is done as
                  V2 := V2*Q2 + V1*Q1 + V3*Q3
   (the first term is computed with SlepcUpdateVectors_Noncontiguous_Inplace).
*/
static PetscErrorCode SlepcUpdateVectors_Noncontiguous(PetscInt n,Vec *V,PetscInt s,PetscInt e,const PetscScalar *Q,PetscInt ldq,PetscBool qtrans)
{
  PetscInt       i,j,m,ln;
  PetscScalar    *pq,qt[100];
  PetscBool      allocated = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  m = e-s;
  if (qtrans) {
    ln = PetscMax(s,n-e);
    if (ln<=100) pq = qt;
    else {
      ierr = PetscMalloc(ln*sizeof(PetscScalar),&pq);CHKERRQ(ierr);
      allocated = PETSC_TRUE;
    }
  }
  /* V2 */
  pq = (PetscScalar*)Q+s*ldq+s;
  ierr = SlepcUpdateVectors_Noncontiguous_Inplace(m,V+s,pq,ldq,qtrans);CHKERRQ(ierr);
  /* V1 */
  if (s>0) {
    for (i=s;i<e;i++) {
      if (qtrans) {
        for (j=0;j<s;j++) pq[j] = Q[i+j*ldq];
      } else pq = (PetscScalar*)Q+i*ldq;
      ierr = VecMAXPY(V[i],s,pq,V);CHKERRQ(ierr);
    }
  }
  /* V3 */
  if (n>e) {
    for (i=s;i<e;i++) {
      if (qtrans) {
        for (j=0;j<n-e;j++) pq[j] = Q[i+(j+e)*ldq];
      } else pq = (PetscScalar*)Q+i*ldq+e;
      ierr = VecMAXPY(V[i],n-e,pq,V+e);CHKERRQ(ierr);
    }
  }
  if (allocated) { ierr = PetscFree(pq);CHKERRQ(ierr); }
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
PetscErrorCode SlepcUpdateVectors(PetscInt n,Vec *V,PetscInt s,PetscInt e,const PetscScalar *Q,PetscInt ldq,PetscBool qtrans)
{
  PetscContainer container;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",n);
  if (n==0 || s>=e) PetscFunctionReturn(0);
  PetscValidPointer(V,2);
  PetscValidHeaderSpecific(*V,VEC_CLASSID,2);
  PetscValidType(*V,2);
  PetscValidScalarPointer(Q,5);
  ierr = PetscObjectQuery((PetscObject)(V[0]),"contiguous",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    /* contiguous Vecs, use BLAS calls */
    ierr = SlepcUpdateStrideVectors(n,V,s,1,e,Q,ldq,qtrans);CHKERRQ(ierr);
  } else {
    /* use regular Vec operations */
    ierr = SlepcUpdateVectors_Noncontiguous(n,V,s,e,Q,ldq,qtrans);CHKERRQ(ierr);
  }
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
   If x are Vec's with contiguous storage, then the operation is done
   through a call to BLAS. Otherwise, VecMAXPY() is called.

   Level: developer

.seealso: SlepcVecSetTemplate()
@*/
PetscErrorCode SlepcVecMAXPBY(Vec y,PetscScalar beta,PetscScalar alpha,PetscInt nv,const PetscScalar a[],Vec x[])
{
  PetscErrorCode    ierr;
  PetscBLASInt      n,m,one=1;
  PetscScalar       *py;
  const PetscScalar *px;
  PetscContainer    container;
  Vec               z;

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

  ierr = PetscObjectQuery((PetscObject)(x[0]),"contiguous",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    /* assume x Vecs are contiguous, use BLAS calls */
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
  } else {
    /* use regular Vec operations */
    ierr = VecDuplicate(y,&z);CHKERRQ(ierr);
    ierr = VecCopy(y,z);CHKERRQ(ierr);
    ierr = VecMAXPY(y,nv,a,x);CHKERRQ(ierr);
    ierr = VecAXPBY(y,beta-alpha,alpha,z);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

