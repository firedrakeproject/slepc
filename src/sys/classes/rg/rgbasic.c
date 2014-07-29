/*
   Basic routines

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

#include <slepc-private/rgimpl.h>      /*I "slepcrg.h" I*/

PetscFunctionList RGList = 0;
PetscBool         RGRegisterAllCalled = PETSC_FALSE;
PetscClassId      RG_CLASSID = 0;
static PetscBool  RGPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "RGFinalizePackage"
/*@C
   RGFinalizePackage - This function destroys everything in the Slepc interface
   to the RG package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode RGFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&RGList);CHKERRQ(ierr);
  RGPackageInitialized = PETSC_FALSE;
  RGRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGInitializePackage"
/*@C
  RGInitializePackage - This function initializes everything in the RG package.
  It is called from PetscDLLibraryRegister() when using dynamic libraries, and
  on the first call to RGCreate() when using static libraries.

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode RGInitializePackage(void)
{
  char             logList[256];
  char             *className;
  PetscBool        opt;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (RGPackageInitialized) PetscFunctionReturn(0);
  RGPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Region",&RG_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = RGRegisterAll();CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"rg",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(RG_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"rg",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(RG_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(RGFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCreate"
/*@C
   RGCreate - Creates an RG context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newrg - location to put the RG context

   Level: beginner

.seealso: RGDestroy(), RG
@*/
PetscErrorCode RGCreate(MPI_Comm comm,RG *newrg)
{
  RG             rg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newrg,2);
  *newrg = 0;
  ierr = RGInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(rg,_p_RG,struct _RGOps,RG_CLASSID,"RG","Region","RG",comm,RGDestroy,RGView);CHKERRQ(ierr);
  rg->complement = PETSC_FALSE;
  rg->data       = NULL;

  *newrg = rg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetOptionsPrefix"
/*@C
   RGSetOptionsPrefix - Sets the prefix used for searching for all
   RG options in the database.

   Logically Collective on RG

   Input Parameters:
+  rg     - the region context
-  prefix - the prefix string to prepend to all RG option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: RGAppendOptionsPrefix()
@*/
PetscErrorCode RGSetOptionsPrefix(RG rg,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)rg,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGAppendOptionsPrefix"
/*@C
   RGAppendOptionsPrefix - Appends to the prefix used for searching for all
   RG options in the database.

   Logically Collective on RG

   Input Parameters:
+  rg     - the region context
-  prefix - the prefix string to prepend to all RG option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: RGSetOptionsPrefix()
@*/
PetscErrorCode RGAppendOptionsPrefix(RG rg,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)rg,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGGetOptionsPrefix"
/*@C
   RGGetOptionsPrefix - Gets the prefix used for searching for all
   RG options in the database.

   Not Collective

   Input Parameters:
.  rg - the region context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: RGSetOptionsPrefix(), RGAppendOptionsPrefix()
@*/
PetscErrorCode RGGetOptionsPrefix(RG rg,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)rg,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetType"
/*@C
   RGSetType - Selects the type for the RG object.

   Logically Collective on RG

   Input Parameter:
+  rg   - the region context
-  type - a known type

   Level: intermediate

.seealso: RGGetType()
@*/
PetscErrorCode RGSetType(RG rg,RGType type)
{
  PetscErrorCode ierr,(*r)(RG);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)rg,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(RGList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested RG type %s",type);

  if (rg->ops->destroy) { ierr = (*rg->ops->destroy)(rg);CHKERRQ(ierr); }
  ierr = PetscMemzero(rg->ops,sizeof(struct _RGOps));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)rg,type);CHKERRQ(ierr);
  ierr = (*r)(rg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGGetType"
/*@C
   RGGetType - Gets the RG type name (as a string) from the RG context.

   Not Collective

   Input Parameter:
.  rg - the region context

   Output Parameter:
.  name - name of the region

   Level: intermediate

.seealso: RGSetType()
@*/
PetscErrorCode RGGetType(RG rg,RGType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)rg)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetFromOptions"
/*@
   RGSetFromOptions - Sets RG options from the options database.

   Collective on RG

   Input Parameters:
.  rg - the region context

   Notes:
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode RGSetFromOptions(RG rg)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  if (!RGRegisterAllCalled) { ierr = RGRegisterAll();CHKERRQ(ierr); }
  ierr = PetscObjectOptionsBegin((PetscObject)rg);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-rg_type","Region type","RGSetType",RGList,(char*)(((PetscObject)rg)->type_name?((PetscObject)rg)->type_name:RGINTERVAL),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = RGSetType(rg,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!((PetscObject)rg)->type_name) {
      ierr = RGSetType(rg,RGINTERVAL);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBool("-rg_complement","Whether region is complemented or not","RGSetComplement",rg->complement,&rg->complement,&flg);CHKERRQ(ierr);

    if (rg->ops->setfromoptions) {
      ierr = (*rg->ops->setfromoptions)(rg);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)rg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGView"
/*@C
   RGView - Prints the RG data structure.

   Collective on RG

   Input Parameters:
+  rg - the region context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

   Level: beginner
@*/
PetscErrorCode RGView(RG rg,PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)rg));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(rg,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)rg,viewer);CHKERRQ(ierr);
    if (rg->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*rg->ops->view)(rg,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (rg->complement) {
      ierr = PetscViewerASCIIPrintf(viewer,"  selected region is the complement of the specified one\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIsTrivial"
/*@
   RGIsTrivial - Whether it is the trivial region (whole complex plane).

   Not Collective

   Input Parameter:
.  rg - the region context

   Output Parameter:
.  trivial - true if the region is equal to the whole complex plane, e.g.,
             an interval region with all four endpoints unbounded or an
             ellipse with infinite radius.

   Level: basic
@*/
PetscErrorCode RGIsTrivial(RG rg,PetscBool *trivial)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidType(rg,1);
  PetscValidPointer(trivial,2);
  if (*rg->ops->istrivial) {
    ierr = (*rg->ops->istrivial)(rg,trivial);CHKERRQ(ierr);
  } else *trivial = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCheckInside"
/*@
   RGCheckInside - Determines if a set of given points are inside the region or not.

   Not Collective

   Input Parameters:
+  rg - the region context
.  n  - number of points to check
.  ar - array of real parts
-  ai - array of imaginary parts

   Output Parameter:
.  inside - array of results (1=inside, 0=on the contour, -1=outside)

   Note:
   The point a is expressed as a couple of PetscScalar variables ar,ai.
   If built with complex scalars, the point is supposed to be stored in ar,
   otherwise ar,ai contain the real and imaginary parts, respectively.

   Level: intermediate
@*/
PetscErrorCode RGCheckInside(RG rg,PetscInt n,PetscScalar *ar,PetscScalar *ai,PetscInt *inside)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidType(rg,1);
  PetscValidPointer(ar,3);
#if defined(PETSC_USE_COMPLEX)
  PetscValidPointer(ai,4);
#endif
  PetscValidPointer(inside,5);
  ierr = (*rg->ops->checkinside)(rg,n,ar,ai,inside);CHKERRQ(ierr);
  if (rg->complement) {
    for (i=0;i<n;i++) inside[i] = -inside[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGComputeContour"
/*@
   RGComputeContour - Computes the coordinates of several points lying in the
   contour of the region.

   Not Collective

   Input Parameters:
+  rg - the region context
-  n  - number of points to compute

   Output Parameter:
+  cr - location to store real parts
-  ci - location to store imaginary parts

   Level: intermediate
@*/
PetscErrorCode RGComputeContour(RG rg,PetscInt n,PetscScalar *cr,PetscScalar *ci)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidType(rg,1);
  PetscValidPointer(cr,3);
#if defined(PETSC_USE_COMPLEX)
  PetscValidPointer(ci,4);
#endif
  ierr = (*rg->ops->computecontour)(rg,n,cr,ci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetComplement"
/*@
   RGSetComplement - Sets a flag to indicate that the region is the complement
   of the specified one.

   Logically Collective on RG

   Input Parameters:
+  rg  - the region context
-  flg - the boolean flag

   Options Database Key:
.  -rg_complement <bool> - Activate/deactivate the complementation of the region.

   Level: intermediate

.seealso: RGGetComplement()
@*/
PetscErrorCode RGSetComplement(RG rg,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidLogicalCollectiveBool(rg,flg,2);
  rg->complement = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGGetComplement"
/*@
   RGGetComplement - Gets a flag that that indicates whether the region
   is complemented or not.

   Not Collective

   Input Parameter:
.  rg - the region context

   Output Parameter:
.  flg - the flag

   Level: intermediate

.seealso: RGSetComplement()
@*/
PetscErrorCode RGGetComplement(RG rg,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidPointer(flg,2);
  *flg = rg->complement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGDestroy"
/*@C
   RGDestroy - Destroys RG context that was created with RGCreate().

   Collective on RG

   Input Parameter:
.  rg - the region context

   Level: beginner

.seealso: RGCreate()
@*/
PetscErrorCode RGDestroy(RG *rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*rg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*rg,RG_CLASSID,1);
  if (--((PetscObject)(*rg))->refct > 0) { *rg = 0; PetscFunctionReturn(0); }
  if ((*rg)->ops->destroy) { ierr = (*(*rg)->ops->destroy)(*rg);CHKERRQ(ierr); }
  ierr = PetscHeaderDestroy(rg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGRegister"
/*@C
   RGRegister - See Adds a mathematical function to the RG package.

   Not collective

   Input Parameters:
+  name - name of a new user-defined RG
-  function - routine to create context

   Notes:
   RGRegister() may be called multiple times to add several user-defined inner products.

   Level: advanced

.seealso: RGRegisterAll()
@*/
PetscErrorCode RGRegister(const char *name,PetscErrorCode (*function)(RG))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&RGList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode RGCreate_Interval(RG);
PETSC_EXTERN PetscErrorCode RGCreate_Ellipse(RG);

#undef __FUNCT__
#define __FUNCT__ "RGRegisterAll"
/*@C
   RGRegisterAll - Registers all of the regions in the RG package.

   Not Collective

   Level: advanced
@*/
PetscErrorCode RGRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  RGRegisterAllCalled = PETSC_TRUE;
  ierr = RGRegister(RGINTERVAL,RGCreate_Interval);CHKERRQ(ierr);
  ierr = RGRegister(RGELLIPSE,RGCreate_Ellipse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

