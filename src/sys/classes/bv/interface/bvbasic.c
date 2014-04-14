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

  Level: beginner

.seealso: BVSetSizesFromVec(), BVGetSizes()
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
  if (!bv->t) {  /* create template vector and get actual dimensions */
    ierr = VecCreate(PetscObjectComm((PetscObject)bv),&bv->t);CHKERRQ(ierr);
    ierr = VecSetSizes(bv->t,bv->n,bv->N);CHKERRQ(ierr);
    ierr = VecSetFromOptions(bv->t);CHKERRQ(ierr);
    ierr = VecGetSize(bv->t,&bv->N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(bv->t,&bv->n);CHKERRQ(ierr);
  }
  if (bv->ops->create) {
    ierr = (*bv->ops->create)(bv);CHKERRQ(ierr);
    bv->ops->create = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVSetSizesFromVec"
/*@
  BVSetSizesFromVec - Sets the local and global sizes, and the number of columns.
  Local and global sizes are specified indirectly by passing a template vector.

  Collective on BV

  Input Parameters:
+ bv - the basis vectors
. t  - the template vectors
- k  - the number of columns

  Level: beginner

.seealso: BVSetSizes(), BVGetSizes()
@*/
PetscErrorCode BVSetSizesFromVec(BV bv,Vec t,PetscInt k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidHeaderSpecific(t,VEC_CLASSID,2);
  PetscCheckSameComm(bv,1,t,2);
  PetscValidLogicalCollectiveInt(bv,k,3);
  if (k <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of columns %D must be positive",k);
  if (bv->t) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Template vector was already set by a previous call to BVSetSizes/FromVec");
  ierr = VecGetSize(t,&bv->N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(t,&bv->n);CHKERRQ(ierr);
  bv->k = k;
  bv->t = t;
  ierr = PetscObjectReference((PetscObject)t);CHKERRQ(ierr);
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

  Level: beginner

.seealso: BVSetSizes(), BVSetSizesFromVec()
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

#undef __FUNCT__
#define __FUNCT__ "BVGetColumn"
/*@
   BVGetColumn - Returns a Vec object that contains the entries of the
   requested column of the basis vectors object.

   Collective on BV

   Input Parameters:
+  bv - the basis vectors context
-  j  - the index of the requested column

   Output Parameter:
.  v  - vector containing the jth column

   Notes:
   The returned Vec must be seen as a reference (not a copy) of the BV
   column, that is, modifying the Vec will change the BV entries as well.

   The returned Vec must not be destroyed. BVRestoreColumn() must be
   called when it is no longer needed. At most, two columns can be fetched,
   that is, this function can only be called twice before the corresponding
   BVRestoreColumn() is invoked.

   Level: beginner

.seealso: BVRestoreColumn()
@*/
PetscErrorCode BVGetColumn(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode ierr;
  PetscInt       l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidType(bv,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  if (j<0) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Column index must be non-negative");
  if (j>=bv->k) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"You requested column %D but only %D are available",j,bv->k);
  if (j==bv->ci[0] || j==bv->ci[1]) SETERRQ1(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Column %D already fetched in a previous call to BVGetColumn",j);
  l = BVAvailableVec;
  if (l==-1) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Too many requested columns; you must call BVReleaseColumn for one of the previously fetched columns");
  if (!bv->ops->getcolumn) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_POINTER,"BVGetColumn operation not defined");
  ierr = (*bv->ops->getcolumn)(bv,j,v);CHKERRQ(ierr);
  bv->ci[l] = j;
  ierr = PetscObjectStateGet((PetscObject)bv->cv[l],&bv->st[l]);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)bv->cv[l],&bv->id[l]);CHKERRQ(ierr);
  *v = bv->cv[l];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVRestoreColumn"
/*@
   BVRestoreColumn - Restore a column obtained with BVGetColumn().

   Logically Collective on BV

   Input Parameters:
+  bv - the basis vectors context
.  j  - the index of the column
-  v  - vector obtained with BVGetColumn()

   Note:
   The arguments must match the corresponding call to BVGetColumn().

   Level: beginner

.seealso: BVGetColumn()
@*/
PetscErrorCode BVRestoreColumn(BV bv,PetscInt j,Vec *v)
{
  PetscErrorCode   ierr;
  PetscObjectId    id;
  PetscObjectState st;
  PetscInt         l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidType(bv,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  PetscValidPointer(v,3);
  PetscValidHeaderSpecific(*v,VEC_CLASSID,3);
  if (j<0) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Column index must be non-negative");
  if (j>=bv->k) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"You requested column %D but only %D are available",j,bv->k);
  if (j!=bv->ci[0] && j!=bv->ci[1]) SETERRQ1(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_WRONG,"Column %D has not been fetched with a call to BVGetColumn",j);
  l = (j==bv->ci[0])? 0: 1;
  ierr = PetscObjectGetId((PetscObject)*v,&id);CHKERRQ(ierr);
  if (id!=bv->id[l]) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_WRONG,"Argument 3 is not the same Vec that was obtained with BVGetColumn");
  ierr = PetscObjectStateGet((PetscObject)*v,&st);CHKERRQ(ierr);
  if (st!=bv->st[l]) {
    ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  }
  if (bv->ops->restorecolumn) {
    ierr = (*bv->ops->restorecolumn)(bv,j,v);CHKERRQ(ierr);
  } else bv->cv[l] = NULL;
  bv->ci[l] = -1;
  bv->st[l] = -1;
  bv->id[l] = 0;
  *v = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetVec"
/*@
   BVGetVec - Creates a new Vec object with the same type and dimensions
   as the columns of the basis vectors object.

   Collective on BV

   Input Parameters:
.  bv - the basis vectors context

   Output Parameter:
.  v  - the new vector

   Note:
   The user is responsible of destroying the returned vector.

   Level: beginner
@*/
PetscErrorCode BVGetVec(BV bv,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidType(bv,1);
  PetscValidPointer(v,2);
  if (!bv->k) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_WRONGSTATE,"Must call BVSetSizes (or BVSetSizesFromVec) first");
  ierr = VecDuplicate(bv->t,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

