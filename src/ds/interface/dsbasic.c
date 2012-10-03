/*
   Basic DS routines

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

#include <slepc-private/dsimpl.h>      /*I "slepcds.h" I*/

PetscFList       DSList = 0;
PetscBool        DSRegisterAllCalled = PETSC_FALSE;
PetscClassId     DS_CLASSID = 0;
PetscLogEvent    DS_Solve = 0,DS_Vectors = 0,DS_Other = 0;
static PetscBool DSPackageInitialized = PETSC_FALSE;
const char       *DSMatName[DS_NUM_MAT] = {"A","B","C","T","D","Q","Z","X","Y","U","VT","W"};

#undef __FUNCT__  
#define __FUNCT__ "DSFinalizePackage"
/*@C
   DSFinalizePackage - This function destroys everything in the SLEPc interface 
   to the DS package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode DSFinalizePackage(void) 
{
  PetscFunctionBegin;
  DSPackageInitialized = PETSC_FALSE;
  DSList               = 0;
  DSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSInitializePackage"
/*@C
  DSInitializePackage - This function initializes everything in the DS package.
  It is called from PetscDLLibraryRegister() when using dynamic libraries, and
  on the first call to DSCreate() when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode DSInitializePackage(const char *path) 
{
  char             logList[256];
  char             *className;
  PetscBool        opt;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (DSPackageInitialized) PetscFunctionReturn(0);
  DSPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Direct solver",&DS_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = DSRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("DSSolve",DS_CLASSID,&DS_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DSVectors",DS_CLASSID,&DS_Vectors);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DSOther",DS_CLASSID,&DS_Other);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"ds",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(DS_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"ds",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(DS_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(DSFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSCreate"
/*@C
   DSCreate - Creates a DS context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newds - location to put the DS context

   Level: beginner

   Note: 
   DS objects are not intended for normal users but only for
   advanced user that for instance implement their own solvers.

.seealso: DSDestroy(), DS
@*/
PetscErrorCode DSCreate(MPI_Comm comm,DS *newds)
{
  DS             ds;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newds,2);
  ierr = SlepcHeaderCreate(ds,_p_DS,struct _DSOps,DS_CLASSID,-1,"DS","Direct Solver (or Dense System)","DS",comm,DSDestroy,DSView);CHKERRQ(ierr);
  *newds       = ds;
  ds->state    = DS_STATE_RAW;
  ds->method   = 0;
  ds->compact  = PETSC_FALSE;
  ds->refined  = PETSC_FALSE;
  ds->extrarow = PETSC_FALSE;
  ds->ld       = 0;
  ds->l        = 0;
  ds->n        = 0;
  ds->m        = 0;
  ds->k        = 0;
  ds->t        = 0;
  for (i=0;i<DS_NUM_MAT;i++) {
    ds->mat[i]  = PETSC_NULL;
    ds->rmat[i] = PETSC_NULL;
  }
  ds->perm     = PETSC_NULL;
  ds->work     = PETSC_NULL;
  ds->rwork    = PETSC_NULL;
  ds->iwork    = PETSC_NULL;
  ds->lwork    = 0;
  ds->lrwork   = 0;
  ds->liwork   = 0;
  ds->comp_fun = PETSC_NULL;
  ds->comp_ctx = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetOptionsPrefix"
/*@C
   DSSetOptionsPrefix - Sets the prefix used for searching for all 
   DS options in the database.

   Logically Collective on DS

   Input Parameters:
+  ds - the direct solver context
-  prefix - the prefix string to prepend to all DS option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: DSAppendOptionsPrefix()
@*/
PetscErrorCode DSSetOptionsPrefix(DS ds,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ds,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSAppendOptionsPrefix"
/*@C
   DSAppendOptionsPrefix - Appends to the prefix used for searching for all 
   DS options in the database.

   Logically Collective on DS

   Input Parameters:
+  ds - the direct solver context
-  prefix - the prefix string to prepend to all DS option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: DSSetOptionsPrefix()
@*/
PetscErrorCode DSAppendOptionsPrefix(DS ds,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ds,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSGetOptionsPrefix"
/*@C
   DSGetOptionsPrefix - Gets the prefix used for searching for all 
   DS options in the database.

   Not Collective

   Input Parameters:
.  ds - the direct solver context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: DSSetOptionsPrefix(), DSAppendOptionsPrefix()
@*/
PetscErrorCode DSGetOptionsPrefix(DS ds,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ds,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetType"
/*@C
   DSSetType - Selects the type for the DS object.

   Logically Collective on DS

   Input Parameter:
+  ds   - the direct solver context
-  type - a known type

   Level: intermediate

.seealso: DSGetType()
@*/
PetscErrorCode DSSetType(DS ds,DSType type)
{
  PetscErrorCode ierr,(*r)(DS);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)ds,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFListFind(DSList,((PetscObject)ds)->comm,type,PETSC_TRUE,(void (**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(((PetscObject)ds)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested DS type %s",type);

  ierr = PetscMemzero(ds->ops,sizeof(struct _DSOps));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)ds,type);CHKERRQ(ierr);
  ierr = (*r)(ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetType"
/*@C
   DSGetType - Gets the DS type name (as a string) from the DS context.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameter:
.  name - name of the direct solver

   Level: intermediate

.seealso: DSSetType()
@*/
PetscErrorCode DSGetType(DS ds,DSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)ds)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetMethod"
/*@
   DSSetMethod - Selects the method to be used to solve the problem.

   Logically Collective on DS

   Input Parameter:
+  ds   - the direct solver context
-  meth - an index indentifying the method

   Level: intermediate

.seealso: DSGetMethod()
@*/
PetscErrorCode DSSetMethod(DS ds,PetscInt meth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,meth,2);
  if (meth<0) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"The method must be a non-negative integer");
  if (meth>DS_MAX_SOLVE) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Too large value for the method");
  ds->method = meth;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetMethod"
/*@
   DSGetMethod - Gets the method currently used in the DS.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameter:
.  meth - identifier of the method

   Level: intermediate

.seealso: DSSetMethod()
@*/
PetscErrorCode DSGetMethod(DS ds,PetscInt *meth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(meth,2);
  *meth = ds->method;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetCompact"
/*@
   DSSetCompact - Switch to compact storage of matrices.

   Logically Collective on DS

   Input Parameter:
+  ds   - the direct solver context
-  comp - a boolean flag

   Notes:
   Compact storage is used in some DS types such as DSHEP when the matrix
   is tridiagonal. This flag can be used to indicate whether the user
   provides the matrix entries via the compact form (the tridiagonal DS_MAT_T)
   or the non-compact one (DS_MAT_A).

   The default is PETSC_FALSE.

   Level: advanced

.seealso: DSGetCompact()
@*/
PetscErrorCode DSSetCompact(DS ds,PetscBool comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveBool(ds,comp,2);
  ds->compact = comp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetCompact"
/*@
   DSGetCompact - Gets the compact storage flag.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameter:
.  comp - the flag

   Level: advanced

.seealso: DSSetCompact()
@*/
PetscErrorCode DSGetCompact(DS ds,PetscBool *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(comp,2);
  *comp = ds->compact;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetExtraRow"
/*@
   DSSetExtraRow - Sets a flag to indicate that the matrix has one extra
   row.

   Logically Collective on DS

   Input Parameter:
+  ds  - the direct solver context
-  ext - a boolean flag

   Notes:
   In Krylov methods it is useful that the matrix representing the direct solver
   has one extra row, i.e., has dimension (n+1) x n. If this flag is activated, all
   transformations applied to the right of the matrix also affect this additional
   row. In that case, (n+1) must be less or equal than the leading dimension.

   The default is PETSC_FALSE.

   Level: advanced

.seealso: DSSolve(), DSAllocate(), DSGetExtraRow()
@*/
PetscErrorCode DSSetExtraRow(DS ds,PetscBool ext)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveBool(ds,ext,2);
  if (ds->n>0 && ds->n==ds->ld) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ORDER,"Cannot set extra row after setting n=ld");
  ds->extrarow = ext;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetExtraRow"
/*@
   DSGetExtraRow - Gets the extra row flag.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameter:
.  ext - the flag

   Level: advanced

.seealso: DSSetExtraRow()
@*/
PetscErrorCode DSGetExtraRow(DS ds,PetscBool *ext)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(ext,2);
  *ext = ds->extrarow;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetRefined"
/*@
   DSSetRefined - Sets a flag to indicate that refined vectors must be
   computed.

   Logically Collective on DS

   Input Parameter:
+  ds  - the direct solver context
-  ref - a boolean flag

   Notes:
   Normally the vectors returned in DS_MAT_X are eigenvectors of the
   projected matrix. With this flag activated, DSVectors() will return
   the right singular vector of the smallest singular value of matrix
   \tilde{A}-theta*I, where \tilde{A} is the extended (n+1)xn matrix
   and theta is the Ritz value. This is used in the refined Ritz
   approximation.

   The default is PETSC_FALSE.

   Level: advanced

.seealso: DSVectors(), DSGetRefined()
@*/
PetscErrorCode DSSetRefined(DS ds,PetscBool ref)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveBool(ds,ref,2);
  ds->refined = ref;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetRefined"
/*@
   DSGetRefined - Gets the refined vectors flag.

   Not Collective

   Input Parameter:
.  ds - the direct solver context

   Output Parameter:
.  ref - the flag

   Level: advanced

.seealso: DSSetRefined()
@*/
PetscErrorCode DSGetRefined(DS ds,PetscBool *ref)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidPointer(ref,2);
  *ref = ds->refined;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetEigenvalueComparison"
/*@C
   DSSetEigenvalueComparison - Specifies the eigenvalue comparison function
   to be used for sorting.

   Logically Collective on DS

   Input Parameters:
+  ds  - the direct solver context
.  fun - a pointer to the comparison function
-  ctx - a context pointer (the last parameter to the comparison function)

   Calling Sequence of fun:
$  func(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *res,void *ctx)

+   ar     - real part of the 1st eigenvalue
.   ai     - imaginary part of the 1st eigenvalue
.   br     - real part of the 2nd eigenvalue
.   bi     - imaginary part of the 2nd eigenvalue
.   res    - result of comparison
-   ctx    - optional context, as set by DSSetEigenvalueComparison()

   Note:
   The returning parameter 'res' can be:
+  negative - if the 1st eigenvalue is preferred to the 2st one
.  zero     - if both eigenvalues are equally preferred
-  positive - if the 2st eigenvalue is preferred to the 1st one

   Level: developer

.seealso: DSSort()
@*/
PetscErrorCode DSSetEigenvalueComparison(DS ds,PetscErrorCode (*fun)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void* ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  ds->comp_fun = fun;
  ds->comp_ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSGetEigenvalueComparison"
/*@C
   DSGetEigenvalueComparison - Gets the eigenvalue comparison function
   used for sorting.

   Not Collective

   Input Parameter:
.  ds  - the direct solver context

   Output Parameters:
+  fun - a pointer to the comparison function
-  ctx - a context pointer (the last parameter to the comparison function)

   Calling Sequence of fun:
$  func(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *res,void *ctx)

+   ar     - real part of the 1st eigenvalue
.   ai     - imaginary part of the 1st eigenvalue
.   br     - real part of the 2nd eigenvalue
.   bi     - imaginary part of the 2nd eigenvalue
.   res    - result of comparison
-   ctx    - optional context, as set by DSSetEigenvalueComparison()

   Note:
   The returning parameter 'res' can be:
+  negative - if the 1st eigenvalue is preferred to the 2st one
.  zero     - if both eigenvalues are equally preferred
-  positive - if the 2st eigenvalue is preferred to the 1st one

   Level: developer

.seealso: DSSort(), DSSetEigenvalueComparison()
@*/
PetscErrorCode DSGetEigenvalueComparison(DS ds,PetscErrorCode (**fun)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void** ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (fun) *fun = ds->comp_fun;
  if (ctx) *ctx = ds->comp_ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSetFromOptions"
/*@
   DSSetFromOptions - Sets DS options from the options database.

   Collective on DS

   Input Parameters:
.  ds - the direct solver context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode DSSetFromOptions(DS ds)
{
  PetscErrorCode ierr;
  PetscInt       meth;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (!DSRegisterAllCalled) { ierr = DSRegisterAll(PETSC_NULL);CHKERRQ(ierr); }
  /* Set default type (we do not allow changing it with -ds_type) */
  if (!((PetscObject)ds)->type_name) {
    ierr = DSSetType(ds,DSNHEP);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBegin(((PetscObject)ds)->comm,((PetscObject)ds)->prefix,"Direct Solver (DS) Options","DS");CHKERRQ(ierr);
    meth = 0;
    ierr = PetscOptionsInt("-ds_method","Method to be used for the dense system","DSSetMethod",ds->method,&meth,PETSC_NULL);CHKERRQ(ierr);
    ierr = DSSetMethod(ds,meth);CHKERRQ(ierr);
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)ds);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSView"
/*@C
   DSView - Prints the DS data structure.

   Collective on DS

   Input Parameters:
+  ds - the direct solver context
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
PetscErrorCode DSView(DS ds,PetscViewer viewer)
{
  PetscBool         isascii,issvd;
  const char        *state;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)ds)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ds,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)ds,viewer,"DS Object");CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      switch (ds->state) {
        case DS_STATE_RAW:          state = "raw"; break;
        case DS_STATE_INTERMEDIATE: state = "intermediate"; break;
        case DS_STATE_CONDENSED:    state = "condensed"; break;
        case DS_STATE_TRUNCATED:    state = "truncated"; break;
        default: SETERRQ(((PetscObject)ds)->comm,1,"Wrong value of ds->state");
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  current state: %s\n",state);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)ds,DSSVD,&issvd);CHKERRQ(ierr);
      if (issvd) {
        ierr = PetscViewerASCIIPrintf(viewer,"  dimensions: ld=%d, n=%d, m=%d, l=%d, k=%d",ds->ld,ds->n,ds->m,ds->l,ds->k);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  dimensions: ld=%d, n=%d, l=%d, k=%d",ds->ld,ds->n,ds->l,ds->k);CHKERRQ(ierr);
      }
      if (ds->state==DS_STATE_TRUNCATED) {
        ierr = PetscViewerASCIIPrintf(viewer,", t=%d\n",ds->t);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  flags:%s%s%s\n",ds->compact?" compact":"",ds->extrarow?" extrarow":"",ds->refined?" refined":"");CHKERRQ(ierr);
    }
    if (ds->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ds->ops->view)(ds,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else SETERRQ1(((PetscObject)ds)->comm,1,"Viewer type %s not supported for DS",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSAllocate"
/*@
   DSAllocate - Allocates memory for internal storage or matrices in DS.

   Logically Collective on DS

   Input Parameters:
+  ds - the direct solver context
-  ld - leading dimension (maximum allowed dimension for the matrices, including
        the extra row if present)

   Level: intermediate

.seealso: DSGetLeadingDimension(), DSSetDimensions(), DSSetExtraRow()
@*/
PetscErrorCode DSAllocate(DS ds,PetscInt ld)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ds,ld,2);
  if (ld<1) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Leading dimension should be at least one");
  ds->ld = ld;
  ierr = (*ds->ops->allocate)(ds,ld);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSReset"
/*@
   DSReset - Resets the DS context to the initial state.

   Collective on DS

   Input Parameter:
.  ds - the direct solver context

   Level: advanced

.seealso: DSDestroy()
@*/
PetscErrorCode DSReset(DS ds)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  ds->state    = DS_STATE_RAW;
  ds->compact  = PETSC_FALSE;
  ds->refined  = PETSC_FALSE;
  ds->extrarow = PETSC_FALSE;
  ds->ld       = 0;
  ds->l        = 0;
  ds->n        = 0;
  ds->m        = 0;
  ds->k        = 0;
  for (i=0;i<DS_NUM_MAT;i++) {
    ierr = PetscFree(ds->mat[i]);CHKERRQ(ierr);
    ierr = PetscFree(ds->rmat[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ds->perm);CHKERRQ(ierr);
  ierr = PetscFree(ds->work);CHKERRQ(ierr);
  ierr = PetscFree(ds->rwork);CHKERRQ(ierr);
  ierr = PetscFree(ds->iwork);CHKERRQ(ierr);
  ds->lwork    = 0;
  ds->lrwork   = 0;
  ds->liwork   = 0;
  ds->comp_fun = PETSC_NULL;
  ds->comp_ctx = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSDestroy"
/*@C
   DSDestroy - Destroys DS context that was created with DSCreate().

   Collective on DS

   Input Parameter:
.  ds - the direct solver context

   Level: beginner

.seealso: DSCreate()
@*/
PetscErrorCode DSDestroy(DS *ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ds) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*ds,DS_CLASSID,1);
  if (--((PetscObject)(*ds))->refct > 0) { *ds = 0; PetscFunctionReturn(0); }
  ierr = DSReset(*ds);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSRegister"
/*@C
   DSRegister - See DSRegisterDynamic()

   Level: advanced
@*/
PetscErrorCode DSRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(DS))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&DSList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSRegisterDestroy"
/*@
   DSRegisterDestroy - Frees the list of DS methods that were
   registered by DSRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: DSRegisterDynamic(), DSRegisterAll()
@*/
PetscErrorCode DSRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&DSList);CHKERRQ(ierr);
  DSRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode DSCreate_HEP(DS);
extern PetscErrorCode DSCreate_NHEP(DS);
extern PetscErrorCode DSCreate_GHEP(DS);
extern PetscErrorCode DSCreate_GHIEP(DS);
extern PetscErrorCode DSCreate_GNHEP(DS);
extern PetscErrorCode DSCreate_SVD(DS);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "DSRegisterAll"
/*@C
   DSRegisterAll - Registers all of the direct solvers in the DS package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced
@*/
PetscErrorCode DSRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  DSRegisterAllCalled = PETSC_TRUE;
  ierr = DSRegisterDynamic(DSHEP,path,"DSCreate_HEP",DSCreate_HEP);CHKERRQ(ierr);
  ierr = DSRegisterDynamic(DSNHEP,path,"DSCreate_NHEP",DSCreate_NHEP);CHKERRQ(ierr);
  ierr = DSRegisterDynamic(DSGHEP,path,"DSCreate_GHEP",DSCreate_GHEP);CHKERRQ(ierr);
  ierr = DSRegisterDynamic(DSGHIEP,path,"DSCreate_GHIEP",DSCreate_GHIEP);CHKERRQ(ierr);
  ierr = DSRegisterDynamic(DSGNHEP,path,"DSCreate_GNHEP",DSCreate_GNHEP);CHKERRQ(ierr);
  ierr = DSRegisterDynamic(DSSVD,path,"DSCreate_SVD",DSCreate_SVD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

