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

#include <slepcvec.h>            /*I "slepcvec.h" I*/

typedef struct {
  PetscScalar *array;    /* pointer to common storage */
  PetscInt    nvecs;     /* number of vectors that share this array */
} Vecs_Contiguous;

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
  PetscBool      contiguous;

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

