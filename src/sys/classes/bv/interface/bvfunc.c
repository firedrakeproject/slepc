/*
   BV (basis vectors) interface routines, callable by users.

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

#include <slepc-private/bvimpl.h>            /*I "slepcbv.h" I*/

PetscClassId     BV_CLASSID = 0;
PetscLogEvent    BV_Mult = 0,BV_MultVec = 0,BV_Dot = 0;
static PetscBool BVPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "BVFinalizePackage"
/*@C
   BVFinalizePackage - This function destroys everything in the Slepc interface
   to the BV package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode BVFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&BVList);CHKERRQ(ierr);
  BVPackageInitialized = PETSC_FALSE;
  BVRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVInitializePackage"
/*@C
   BVInitializePackage - This function initializes everything in the BV package.
   It is called from PetscDLLibraryRegister() when using dynamic libraries, and
   on the first call to BVCreate() when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode BVInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (BVPackageInitialized) PetscFunctionReturn(0);
  BVPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Basis Vectors",&BV_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = BVRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("BVMult",BV_CLASSID,&BV_Mult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVMultVec",BV_CLASSID,&BV_MultVec);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVDot",BV_CLASSID,&BV_Dot);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"bv",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(BV_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"bv",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(BV_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(BVFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDestroy"
/*@C
   BVDestroy - Destroys BV context that was created with BVCreate().

   Collective on BV

   Input Parameter:
.  bv - the basis vectors context

   Level: beginner

.seealso: BVCreate(), BVSetUp()
@*/
PetscErrorCode BVDestroy(BV *bv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*bv) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*bv,BV_CLASSID,1);
  if (--((PetscObject)(*bv))->refct > 0) { *bv = 0; PetscFunctionReturn(0); }
  if ((*bv)->ops->destroy) { ierr = (*(*bv)->ops->destroy)(*bv);CHKERRQ(ierr); }
  ierr = VecDestroy(&(*bv)->t);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCreate"
/*@C
   BVCreate - Creates a basis vectors context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  bv - location to put the basis vectors context

   Level: beginner

.seealso: BVSetUp(), BVDestroy(), BV
@*/
PetscErrorCode BVCreate(MPI_Comm comm,BV *newbv)
{
  PetscErrorCode ierr;
  BV             bv;

  PetscFunctionBegin;
  PetscValidPointer(newbv,2);
  *newbv = 0;
  ierr = BVInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(bv,_p_BV,struct _BVOps,BV_CLASSID,"BV","Basis Vectors","BV",comm,BVDestroy,BVView);CHKERRQ(ierr);

  bv->t            = NULL;
  bv->n            = -1;
  bv->N            = -1;
  bv->k            = 0;

  bv->cv[0]        = NULL;
  bv->cv[1]        = NULL;
  bv->ci[0]        = -1;
  bv->ci[1]        = -1;
  bv->st[0]        = -1;
  bv->st[1]        = -1;
  bv->id[0]        = 0;
  bv->id[1]        = 0;
  bv->data         = 0;

  *newbv = bv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVSetOptionsPrefix"
/*@C
   BVSetOptionsPrefix - Sets the prefix used for searching for all
   BV options in the database.

   Logically Collective on BV

   Input Parameters:
+  bv     - the basis vectors context
-  prefix - the prefix string to prepend to all BV option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: BVAppendOptionsPrefix(), BVGetOptionsPrefix()
@*/
PetscErrorCode BVSetOptionsPrefix(BV bv,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)bv,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVAppendOptionsPrefix"
/*@C
   BVAppendOptionsPrefix - Appends to the prefix used for searching for all
   BV options in the database.

   Logically Collective on BV

   Input Parameters:
+  bv     - the basis vectors context
-  prefix - the prefix string to prepend to all BV option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: BVSetOptionsPrefix(), BVGetOptionsPrefix()
@*/
PetscErrorCode BVAppendOptionsPrefix(BV bv,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)bv,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetOptionsPrefix"
/*@C
   BVGetOptionsPrefix - Gets the prefix used for searching for all
   BV options in the database.

   Not Collective

   Input Parameters:
.  bv - the basis vectors context

   Output Parameters:
.  prefix - pointer to the prefix string used, is returned

   Notes: On the Fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: BVSetOptionsPrefix(), BVAppendOptionsPrefix()
@*/
PetscErrorCode BVGetOptionsPrefix(BV bv,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)bv,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVView"
/*@C
   BVView - Prints the BV data structure.

   Collective on BV

   Input Parameters:
+  bv     - the BV context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization contexts with
   PetscViewerASCIIOpen() (output to a specified file).

   Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode BVView(BV bv,PetscViewer viewer)
{
  PetscErrorCode ierr;
  BVType         cstr;
  PetscBool      isascii,isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)bv));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(bv,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)bv,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  %D columns of global length %D\n",bv->k,bv->N);CHKERRQ(ierr);
    if (bv->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*bv->ops->view)(bv,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = BVGetType(bv,&cstr);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-7.7s",cstr);CHKERRQ(ierr);
    if (bv->ops->view) { ierr = (*bv->ops->view)(bv,viewer);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVRegister"
/*@C
   BVRegister - Adds a new storage format to de BV package.

   Not collective

   Input Parameters:
+  name     - name of a new user-defined BV
-  function - routine to create context

   Notes:
   BVRegister() may be called multiple times to add several user-defined
   basis vectors.

   Level: advanced

.seealso: BVRegisterAll()
@*/
PetscErrorCode BVRegister(const char *name,PetscErrorCode (*function)(BV))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&BVList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

