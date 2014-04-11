/*
   Basic BV routines.

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

#include <slepc-private/bvimpl.h>      /*I "slepcbv.h" I*/

PetscBool         BVRegisterAllCalled = PETSC_FALSE;
PetscFunctionList BVList = 0;

#undef __FUNCT__
#define __FUNCT__ "BVSetType"
/*@C
   BVSetType - Selects the type for the BV object.

   Logically Collective on BV

   Input Parameter:
+  bv   - the basis vectors context
-  type - a known type

   Options Database Key:
.  -bv_type <type> - Sets BV type

   Level: intermediate

.seealso: BVGetType()

@*/
PetscErrorCode BVSetType(BV bv,BVType type)
{
  PetscErrorCode ierr,(*r)(BV);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)bv,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(BVList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested BV type %s",type);

  if (bv->ops->destroy) { ierr = (*bv->ops->destroy)(bv);CHKERRQ(ierr); }
  ierr = PetscMemzero(bv->ops,sizeof(struct _BVOps));CHKERRQ(ierr);

  bv->setupcalled = 0;
  ierr = PetscObjectChangeTypeName((PetscObject)bv,type);CHKERRQ(ierr);
  if (bv->n < 0 && bv->N < 0) {
    bv->ops->create = r;
  } else {
    ierr = (*r)(bv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetType"
/*@C
   BVGetType - Gets the BV type name (as a string) from the BV context.

   Not Collective

   Input Parameter:
.  bv - the basis vectors context

   Output Parameter:
.  name - name of the type of basis vectors

   Level: intermediate

.seealso: BVSetType()

@*/
PetscErrorCode BVGetType(BV bv,BVType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)bv)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVSetSizes"
/*@
  BVSetSizes - Sets the local and global sizes, and the number of columns.

  Collective on BV

  Input Parameters:
+ bv - the basis vectors
. n  - the local size (or PETSC_DECIDE to have it set)
. N  - the global size (or PETSC_DECIDE)
- k  - the number of columns

  Notes:
  n and N cannot be both PETSC_DECIDE
  If one processor calls this with N of PETSC_DECIDE then all processors must,
  otherwise the program will hang.

  Level: basic

.seealso: BVGetSize()
@*/
PetscErrorCode BVSetSizes(BV bv,PetscInt n,PetscInt N,PetscInt k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  if (N >= 0) PetscValidLogicalCollectiveInt(bv,N,3);
  PetscValidLogicalCollectiveInt(bv,k,4);
  if (N >= 0 && n > N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size %D cannot be larger than global size %D",n,N);
  if (k <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of columns %D must be positive",k);
  if ((bv->n >= 0 || bv->N >= 0) && (bv->n != n || bv->N != N)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset vector sizes to %D local %D global after previously setting them to %D local %D global",n,N,bv->n,bv->N);
  if (bv->k > 0 && bv->k != k) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset the number of columns to %D after previously setting it to %D",k,bv->k);
  bv->n = n;
  bv->N = N;
  bv->k = k;
  if (bv->ops->create) {
    ierr = (*bv->ops->create)(bv);CHKERRQ(ierr);
    bv->ops->create = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetSizes"
/*@
  BVGetSizes - Returns the local and global sizes, and the number of columns.

  Not Collective

  Input Parameter:
. bv - the basis vectors

  Output Parameters:
+ n  - the local size
. N  - the global size
- k  - the number of columns

  Level: basic

.seealso: BVSetSize()
@*/
PetscErrorCode BVGetSizes(BV bv,PetscInt *n,PetscInt *N,PetscInt *k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  if (n) *n = bv->n;
  if (N) *N = bv->N;
  if (k) *k = bv->k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVSetFromOptions"
/*@
   BVSetFromOptions - Sets BV options from the options database.

   Collective on BV

   Input Parameter:
.  bv - the basis vectors context

   Level: beginner
@*/
PetscErrorCode BVSetFromOptions(BV bv)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  if (!BVRegisterAllCalled) { ierr = BVRegisterAll();CHKERRQ(ierr); }
  ierr = PetscObjectOptionsBegin((PetscObject)bv);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-bv_type","Basis Vectors type","BVSetType",BVList,(char*)(((PetscObject)bv)->type_name?((PetscObject)bv)->type_name:BVVECS),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = BVSetType(bv,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!((PetscObject)bv)->type_name) {
      ierr = BVSetType(bv,BVVECS);CHKERRQ(ierr);
    }

    if (bv->ops->setfromoptions) {
      ierr = (*bv->ops->setfromoptions)(bv);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)bv);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

