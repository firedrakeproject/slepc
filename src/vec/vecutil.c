/*
   Miscellaneous Vec-related functions.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/vecimplslepc.h>            /*I "slepcvec.h" I*/
#include <slepcsys.h>

#undef __FUNCT__  
#define __FUNCT__ "SlepcVecSetRandom"
/*@
   SlepcVecSetRandom - Sets all components of a vector to random numbers.

   Logically Collective on Vec

   Input/Output Parameter:
.  x  - the vector

   Input Parameter:
-  rctx - the random number context, formed by PetscRandomCreate(), or NULL and
          it will create one internally.

   Note:
   This operation is equivalent to VecSetRandom - the difference is that the
   vector generated by SlepcVecSetRandom is the same irrespective of the size
   of the communicator (if all processes pass a PetscRandom context initialized
   with the same seed).

   Level: developer
@*/
PetscErrorCode SlepcVecSetRandom(Vec x,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscRandom    randObj = NULL;
  PetscInt       i,n,low,high;
  PetscScalar    *px,t;
  MPI_Comm       comm;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_CLASSID,2);
  else {
    ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
    ierr = PetscRandomCreate(comm,&randObj);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(randObj,0x12345678);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randObj);CHKERRQ(ierr);
    rctx = randObj;
  }

  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscRandomGetValue(rctx,&t);CHKERRQ(ierr);
    if (i>=low && i<high) px[i-low] = t;
  }
  ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randObj);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcVecNormalize"
/*@C
   SlepcVecNormalize - Normalizes a possibly complex vector by the 2-norm.

   Collective on Vec

   Input parameters:
+  xr         - the real part of the vector (overwritten on output)
.  xi         - the imaginary part of the vector (not referenced if iscomplex is false)
-  iscomplex - a flag that indicating if the vector is complex

   Output parameter:
.  norm      - the vector norm before normalization (can be set to NULL)

   Level: developer

@*/
PetscErrorCode SlepcVecNormalize(Vec xr,Vec xi,PetscBool iscomplex,PetscReal *norm)
{
  PetscErrorCode ierr;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      normr,normi,alpha;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(xr,VEC_CLASSID,1);
#if !defined(PETSC_USE_COMPLEX)
  if (iscomplex) {
    PetscValidHeaderSpecific(xi,VEC_CLASSID,2);
    ierr = VecNormBegin(xr,NORM_2,&normr);CHKERRQ(ierr);
    ierr = VecNormBegin(xi,NORM_2,&normi);CHKERRQ(ierr);
    ierr = VecNormEnd(xr,NORM_2,&normr);CHKERRQ(ierr);
    ierr = VecNormEnd(xi,NORM_2,&normi);CHKERRQ(ierr);
    alpha = SlepcAbsEigenvalue(normr,normi);
    if (norm) *norm = alpha;
    alpha = 1.0 / alpha;
    ierr = VecScale(xr,alpha);CHKERRQ(ierr);
    ierr = VecScale(xi,alpha);CHKERRQ(ierr);
  } else
#endif
  {
    ierr = VecNormalize(xr,norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

