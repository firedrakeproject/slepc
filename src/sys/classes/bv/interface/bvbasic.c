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
  ierr = (*r)(bv);CHKERRQ(ierr);
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

