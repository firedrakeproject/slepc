/*
   Basic PS routines

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/psimpl.h>      /*I "slepcps.h" I*/

PetscFList       PSList = 0;
PetscBool        PSRegisterAllCalled = PETSC_FALSE;
PetscClassId     PS_CLASSID = 0;
PetscLogEvent    PS_Solve = 0,PS_Sort = 0;
static PetscBool PSPackageInitialized = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PSFinalizePackage"
/*@C
   PSFinalizePackage - This function destroys everything in the Slepc interface 
   to the PS package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode PSFinalizePackage(void) 
{
  PetscFunctionBegin;
  PSPackageInitialized = PETSC_FALSE;
  PSList               = 0;
  PSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSInitializePackage"
/*@C
  PSInitializePackage - This function initializes everything in the PS package.
  It is called from PetscDLLibraryRegister() when using dynamic libraries, and
  on the first call to PSCreate() when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode PSInitializePackage(const char *path) 
{
  char             logList[256];
  char             *className;
  PetscBool        opt;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (PSPackageInitialized) PetscFunctionReturn(0);
  PSPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Projected system",&PS_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PSRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("PSSolve",PS_CLASSID,&PS_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PSSort",PS_CLASSID,&PS_Sort);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"ps",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(PS_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"ps",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(PS_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(PSFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSCreate"
/*@C
   PSCreate - Creates a PS context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newps - location to put the PS context

   Level: beginner

   Note: 
   PS objects are not intended for normal users but only for
   advanced user that for instance implement their own solvers.

.seealso: PSDestroy(), PS
@*/
PetscErrorCode PSCreate(MPI_Comm comm,PS *newps)
{
  PS             ps;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newps,2);
  ierr = PetscHeaderCreate(ps,_p_PS,struct _PSOps,PS_CLASSID,-1,"PS","Projected System","PS",comm,PSDestroy,PSView);CHKERRQ(ierr);
  *newps    = ps;
  ps->state = PS_STATE_RAW;
  ps->ld    = 0;
  ps->l     = 0;
  ps->n     = 0;
  ps->k     = 0;
  for (i=0;i<PS_NUM_MAT;i++) {
    ps->mat[i]  = PETSC_NULL;
    ps->rmat[i] = PETSC_NULL;
  }
  ps->work  = PETSC_NULL;
  ps->rwork = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSetOptionsPrefix"
/*@C
   PSSetOptionsPrefix - Sets the prefix used for searching for all 
   PS options in the database.

   Logically Collective on PS

   Input Parameters:
+  ps - the projected system context
-  prefix - the prefix string to prepend to all PS option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: PSAppendOptionsPrefix()
@*/
PetscErrorCode PSSetOptionsPrefix(PS ps,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ps,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSAppendOptionsPrefix"
/*@C
   PSAppendOptionsPrefix - Appends to the prefix used for searching for all 
   PS options in the database.

   Logically Collective on PS

   Input Parameters:
+  ps - the projected system context
-  prefix - the prefix string to prepend to all PS option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: PSSetOptionsPrefix()
@*/
PetscErrorCode PSAppendOptionsPrefix(PS ps,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ps,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PSGetOptionsPrefix"
/*@C
   PSGetOptionsPrefix - Gets the prefix used for searching for all 
   PS options in the database.

   Not Collective

   Input Parameters:
.  ps - the projected system context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: PSSetOptionsPrefix(), PSAppendOptionsPrefix()
@*/
PetscErrorCode PSGetOptionsPrefix(PS ps,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ps,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSetType"
/*@C
   PSSetType - Selects the type for the PS object.

   Logically Collective on PS

   Input Parameter:
+  ps   - the projected system context.
-  type - a known type

   Level: advanced

.seealso: PSGetType()

@*/
PetscErrorCode PSSetType(PS ps,const PSType type)
{
  PetscErrorCode ierr,(*r)(PS);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)ps,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFListFind(PSList,((PetscObject)ps)->comm,type,PETSC_TRUE,(void (**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(((PetscObject)ps)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PS type %s",type);

  ierr = PetscMemzero(ps->ops,sizeof(struct _PSOps));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)ps,type);CHKERRQ(ierr);
  ierr = (*r)(ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSGetType"
/*@C
   PSGetType - Gets the PS type name (as a string) from the PS context.

   Not Collective

   Input Parameter:
.  ps - the projected system context

   Output Parameter:
.  name - name of the projected system

   Level: advanced

.seealso: PSSetType()

@*/
PetscErrorCode PSGetType(PS ps,const PSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)ps)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSetFromOptions"
/*@
   PSSetFromOptions - Sets PS options from the options database.

   Collective on PS

   Input Parameters:
.  ps - the projected system context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode PSSetFromOptions(PS ps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (!PSRegisterAllCalled) { ierr = PSRegisterAll(PETSC_NULL);CHKERRQ(ierr); }
  /* Set default type (we do not allow changing it with -ps_type) */
  if (!((PetscObject)ps)->type_name) {
    ierr = PSSetType(ps,PSNHEP);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBegin(((PetscObject)ps)->comm,((PetscObject)ps)->prefix,"Projecte System (PS) Options","PS");CHKERRQ(ierr);
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)ps);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSView"
/*@C
   PSView - Prints the PS data structure.

   Collective on PS

   Input Parameters:
+  ps - the projected system context
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

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode PSView(PS ps,PetscViewer viewer)
{
  PetscBool      isascii;
  const char     *state;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)ps)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ps,1,viewer,2);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)ps,viewer,"PS Object");CHKERRQ(ierr);
     switch (ps->state) {
       case PS_STATE_RAW:          state = "raw"; break;
       case PS_STATE_INTERMEDIATE: state = "intermediate"; break;
       case PS_STATE_CONDENSED:    state = "condensed"; break;
       case PS_STATE_SORTED:       state = "sorted"; break;
       default: SETERRQ(((PetscObject)ps)->comm,1,"Wrong value of ps->state");
     }
    ierr = PetscViewerASCIIPrintf(viewer,"  current state: %s\n",state);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  dimensions: ld=%d, n=%d, l=%d, k=%d\n",ps->ld,ps->n,ps->l,ps->k);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)ps)->comm,1,"Viewer type %s not supported for PS",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSAllocate"
/*@
   PSAllocate - Allocates memory for internal storage or matrices in PS.

   Collective on PS

   Input Parameters:
+  ps - the projected system context
-  ld - leading dimension (maximum allowed dimension for the matrices)

   Level: advanced

.seealso: PSGetLeadingDimension(), PSSetDimensions()
@*/
PetscErrorCode PSAllocate(PS ps,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ps,ld,2);
  if (ld<1) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Leading dimension should be at least one");
  ierr = PSReset(ps);CHKERRQ(ierr);
  ps->ld = ld;
  ps->n  = ld;
  ierr = (*ps->ops->allocate)(ps,ld);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSAllocateMat_Private"
PetscErrorCode PSAllocateMat_Private(PS ps,PSMatType m)
{
  PetscInt sz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sz = ps->ld*ps->ld*sizeof(PetscScalar);
  ierr = PetscMalloc(sz,&ps->mat[m]);CHKERRQ(ierr); 
  ierr = PetscMemzero(ps->mat[m],sz);CHKERRQ(ierr); 
  ierr = PetscLogObjectMemory(ps,sz);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSGetLeadingDimension"
/*@
   PSGetLeadingDimension - Returns the leading dimension of the allocated
   matrices.

   Not Collective

   Input Parameter:
.  ps - the projected system context

   Output Parameter:
.  ld - leading dimension (maximum allowed dimension for the matrices)

   Level: advanced

.seealso: PSAllocate(), PSSetDimensions()
@*/
PetscErrorCode PSGetLeadingDimension(PS ps,PetscInt *ld)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (ld) *ld = ps->ld;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSetState"
/*@
   PSSetState - Change the state of the PS object.

   Collective on PS

   Input Parameters:
+  ps    - the projected system context
-  state - the new state

   Notes:
   The state indicates that the projected system is in an initial state (raw),
   in an intermediate state (such as tridiagonal, Hessenberg or 
   Hessenberg-triangular), in a condensed state (such as diagonal, Schur or
   generalized Schur), or in a sorted condensed state (according to a given
   sorting criterion).

   This function is normally used to return to the raw state when the
   condensed structure is destroyed.

   Level: advanced

.seealso: PSGetState()
@*/
PetscErrorCode PSSetState(PS ps,PSStateType state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ps,state,2);
  switch (state) {
    case PS_STATE_RAW:
    case PS_STATE_INTERMEDIATE:
    case PS_STATE_CONDENSED:
    case PS_STATE_SORTED:
      if (ps->state<state) { ierr = PetscInfo(ps,"PS state has been increased\n");CHKERRQ(ierr); }
      ps->state = state;
    default:
      SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_WRONG,"Wrong state");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSGetState"
/*@
   PSGetState - Returns the current state.

   Not Collective

   Input Parameter:
.  ps - the projected system context

   Output Parameter:
.  state - current state

   Level: advanced

.seealso: PSSetState()
@*/
PetscErrorCode PSGetState(PS ps,PSStateType *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (state) *state = ps->state;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSSetDimensions"
/*@
   PSSetDimensions - Resize the matrices in the PS object.

   Collective on PS

   Input Parameters:
+  ps - the projected system context
.  n  - the new size
.  l  - number of locked (inactive) leading columns
-  k  - intermediate dimension (e.g., position of arrow)

   Note:
   The internal arrays are not reallocated.

   Level: advanced

.seealso: PSGetDimensions(), PSAllocate()
@*/
PetscErrorCode PSSetDimensions(PS ps,PetscInt n,PetscInt l,PetscInt k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ps,n,2);
  PetscValidLogicalCollectiveInt(ps,l,3);
  PetscValidLogicalCollectiveInt(ps,k,4);
  if (!ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ORDER,"Must call PSAllocate() first");
  if (n!=PETSC_IGNORE) {
    if (n==PETSC_DECIDE || n==PETSC_DEFAULT) {
      ps->n = ps->ld;
    } else {
      if (n<1 || n>ps->ld) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of n. Must be between 1 and ld");
      ps->n = n;
    }
  }
  if (l!=PETSC_IGNORE) {
    if (l==PETSC_DECIDE || l==PETSC_DEFAULT) {
      ps->l = 0;
    } else {
      if (l<0 || l>ps->n) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of l. Must be between 0 and n");
      ps->l = l;
    }
  }
  if (k!=PETSC_IGNORE) {
    if (k==PETSC_DECIDE || k==PETSC_DEFAULT) {
      ps->k = ps->n/2;
    } else {
      if (k<0 || k>ps->n) SETERRQ(((PetscObject)ps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of k. Must be between 0 and n");
      ps->k = k;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSGetDimensions"
/*@
   PSGetDimensions - Returns the current dimensions.

   Not Collective

   Input Parameter:
.  ps - the projected system context

   Output Parameter:
.  state - current dimensions

   Level: advanced

.seealso: PSSetDimensions()
@*/
PetscErrorCode PSGetDimensions(PS ps,PetscInt *n,PetscInt *l,PetscInt *k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  if (n) *n = ps->n;
  if (l) *l = ps->l;
  if (k) *k = ps->k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSReset"
/*@
   PSReset - Resets the PS context to the initial state.

   Collective on PS

   Input Parameter:
.  ps - the projected system context

   Level: advanced

.seealso: PSDestroy()
@*/
PetscErrorCode PSReset(PS ps)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ps,PS_CLASSID,1);
  ps->state = PS_STATE_RAW;
  ps->ld    = 0;
  ps->l     = 0;
  ps->n     = 0;
  ps->k     = 0;
  for (i=0;i<PS_NUM_MAT;i++) {
    ierr = PetscFree(ps->mat[i]);CHKERRQ(ierr);
    ierr = PetscFree(ps->rmat[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ps->work);CHKERRQ(ierr);
  ierr = PetscFree(ps->rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSDestroy"
/*@C
   PSDestroy - Destroys PS context that was created with PSCreate().

   Collective on PS

   Input Parameter:
.  ps - the projected system context

   Level: beginner

.seealso: PSCreate()
@*/
PetscErrorCode PSDestroy(PS *ps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ps) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*ps,PS_CLASSID,1);
  if (--((PetscObject)(*ps))->refct > 0) { *ps = 0; PetscFunctionReturn(0); }
  ierr = PSReset(*ps);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSRegister"
/*@C
   PSRegister - See PSRegisterDynamic()

   Level: advanced
@*/
PetscErrorCode PSRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(PS))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&PSList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PSRegisterDestroy"
/*@
   PSRegisterDestroy - Frees the list of PS methods that were
   registered by PSRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: PSRegisterDynamic(), PSRegisterAll()
@*/
PetscErrorCode PSRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&PSList);CHKERRQ(ierr);
  PSRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode PSCreate_NHEP(PS);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PSRegisterAll"
/*@C
   PSRegisterAll - Registers all of the projected systems in the PS package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced
@*/
PetscErrorCode PSRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PSRegisterAllCalled = PETSC_TRUE;
  ierr = PSRegisterDynamic(PSNHEP,path,"PSCreate_NHEP",PSCreate_NHEP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

