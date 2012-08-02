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
#include <slepcblaslapack.h>

PetscFList       DSList = 0;
PetscBool        DSRegisterAllCalled = PETSC_FALSE;
PetscClassId     DS_CLASSID = 0;
PetscLogEvent    DS_Solve = 0,DS_Vectors = 0,DS_Other = 0;
static PetscBool DSPackageInitialized = PETSC_FALSE;
const char       *DSMatName[DS_NUM_MAT] = {"A","B","C","T","D","Q","Z","X","Y","U","VT","W"};

PetscErrorCode SlepcDenseMatProd(PetscScalar *C, PetscInt _ldC, PetscScalar b, PetscScalar a, const PetscScalar *A, PetscInt _ldA, PetscInt rA, PetscInt cA, PetscBool At, const PetscScalar *B, PetscInt _ldB, PetscInt rB, PetscInt cB, PetscBool Bt);

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
PetscErrorCode DSSetType(DS ds,const DSType type)
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
PetscErrorCode DSGetType(DS ds,const DSType *type)
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
      ierr = PetscViewerASCIIPrintf(viewer,"  flags: %s %s %s\n",ds->compact?"compact":"",ds->extrarow?"extrarow":"",ds->refined?"refined":"");CHKERRQ(ierr);
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
#define __FUNCT__ "DSAllocateMat_Private"
PetscErrorCode DSAllocateMat_Private(DS ds,DSMatType m)
{
  PetscInt       sz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (m==DS_MAT_T) sz = 3*ds->ld*sizeof(PetscScalar);
  else if (m==DS_MAT_D) sz = ds->ld*sizeof(PetscScalar);
  else sz = ds->ld*ds->ld*sizeof(PetscScalar);
  if (ds->mat[m]) { ierr = PetscFree(ds->mat[m]);CHKERRQ(ierr); }
  else { ierr = PetscLogObjectMemory(ds,sz);CHKERRQ(ierr); }
  ierr = PetscMalloc(sz,&ds->mat[m]);CHKERRQ(ierr); 
  ierr = PetscMemzero(ds->mat[m],sz);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSAllocateMatReal_Private"
PetscErrorCode DSAllocateMatReal_Private(DS ds,DSMatType m)
{
  PetscInt       sz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (m==DS_MAT_T) sz = 3*ds->ld*sizeof(PetscReal);
  else if (m==DS_MAT_D) sz = ds->ld*sizeof(PetscReal);
  else sz = ds->ld*ds->ld*sizeof(PetscReal);
  if (!ds->rmat[m]) {
    ierr = PetscLogObjectMemory(ds,sz);CHKERRQ(ierr);
    ierr = PetscMalloc(sz,&ds->rmat[m]);CHKERRQ(ierr); 
  }
  ierr = PetscMemzero(ds->rmat[m],sz);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSAllocateWork_Private"
PetscErrorCode DSAllocateWork_Private(DS ds,PetscInt s,PetscInt r,PetscInt i)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s>ds->lwork) {
    ierr = PetscFree(ds->work);CHKERRQ(ierr);
    ierr = PetscMalloc(s*sizeof(PetscScalar),&ds->work);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ds,(s-ds->lwork)*sizeof(PetscScalar));CHKERRQ(ierr); 
    ds->lwork = s;
  }
  if (r>ds->lrwork) {
    ierr = PetscFree(ds->rwork);CHKERRQ(ierr);
    ierr = PetscMalloc(r*sizeof(PetscReal),&ds->rwork);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ds,(r-ds->lrwork)*sizeof(PetscReal));CHKERRQ(ierr); 
    ds->lrwork = r;
  }
  if (i>ds->liwork) {
    ierr = PetscFree(ds->iwork);CHKERRQ(ierr);
    ierr = PetscMalloc(i*sizeof(PetscBLASInt),&ds->iwork);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ds,(i-ds->liwork)*sizeof(PetscBLASInt));CHKERRQ(ierr); 
    ds->liwork = i;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSViewMat_Private"
PetscErrorCode DSViewMat_Private(DS ds,PetscViewer viewer,DSMatType m)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,rows,cols;
  PetscScalar       *v;
  PetscViewerFormat format;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         allreal = PETSC_TRUE;
#endif

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  if (ds->state==DS_STATE_TRUNCATED && m>=DS_MAT_Q) rows = ds->t;
  else rows = (m==DS_MAT_A && ds->extrarow)? ds->n+1: ds->n;
  cols = (ds->m!=0)? ds->m: ds->n;
#if defined(PETSC_USE_COMPLEX)
  /* determine if matrix has all real values */
  v = ds->mat[m];
  for (i=0;i<rows;i++)
    for (j=0;j<cols;j++)
      if (PetscImaginaryPart(v[i+j*ds->ld])) { allreal = PETSC_FALSE; break; }
#endif
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D\n",rows,cols);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s = [\n",DSMatName[m]);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"Matrix %s =\n",DSMatName[m]);CHKERRQ(ierr);
  }

  for (i=0;i<rows;i++) {
    v = ds->mat[m]+i;
    for (j=0;j<cols;j++) {
#if defined(PETSC_USE_COMPLEX)
      if (allreal) {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e ",PetscRealPart(*v));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei ",PetscRealPart(*v),PetscImaginaryPart(*v));CHKERRQ(ierr);
      }
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e ",*v);CHKERRQ(ierr);
#endif
      v += ds->ld;
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }

  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSortEigenvalues_Private"
PetscErrorCode DSSortEigenvalues_Private(DS ds,PetscScalar *wr,PetscScalar *wi,PetscInt *perm,PetscBool isghiep)
{
  PetscErrorCode ierr;
  PetscScalar    re,im,wi0;
  PetscInt       i,j,result,tmp1,tmp2=0,d=1;

  PetscFunctionBegin;
  for (i=0;i<ds->n;i++) perm[i] = i;
  /* insertion sort */
  i=ds->l+1;
#if !defined(PETSC_USE_COMPLEX)
  if (wi && wi[perm[i-1]]!=0.0) i++; /* initial value is complex */
#else
  if (isghiep && PetscImaginaryPart(wr[perm[i-1]])!=0.0) i++;
#endif
  for (;i<ds->n;i+=d) {
    re = wr[perm[i]];
    if (wi) im = wi[perm[i]];
    else im = 0.0;
    tmp1 = perm[i];
#if !defined(PETSC_USE_COMPLEX)
    if (im!=0.0) { d = 2; tmp2 = perm[i+1]; }
    else d = 1;
#else
    if (isghiep && PetscImaginaryPart(re)!=0.0) { d = 2; tmp2 = perm[i+1]; }
    else d = 1;
#endif
    j = i-1;
    if (wi) wi0 = wi[perm[j]];
    else wi0 = 0.0;
    ierr = (*ds->comp_fun)(re,im,wr[perm[j]],wi0,&result,ds->comp_ctx);CHKERRQ(ierr);
    while (result<0 && j>=ds->l) {
      perm[j+d] = perm[j];
      j--;
#if !defined(PETSC_USE_COMPLEX)
      if (wi && wi[perm[j+1]]!=0)
#else
      if (isghiep && PetscImaginaryPart(wr[perm[j+1]])!=0)
#endif
        { perm[j+d] = perm[j]; j--; }

     if (j>=ds->l) {
       if (wi) wi0 = wi[perm[j]];
       else wi0 = 0.0;
       ierr = (*ds->comp_fun)(re,im,wr[perm[j]],wi0,&result,ds->comp_ctx);CHKERRQ(ierr);
     }
    }
    perm[j+1] = tmp1;
    if(d==2) perm[j+2] = tmp2;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSSortEigenvaluesReal_Private"
PetscErrorCode DSSortEigenvaluesReal_Private(DS ds,PetscReal *eig,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscScalar    re;
  PetscInt       i,j,result,tmp,l,n;

  PetscFunctionBegin;
  n = ds->n;
  l = ds->l;
  for (i=0;i<n;i++) perm[i] = i;
  /* insertion sort */
  for (i=l+1;i<n;i++) {
    re = eig[perm[i]];
    j = i-1;
    ierr = (*ds->comp_fun)(re,0.0,eig[perm[j]],0.0,&result,ds->comp_ctx);CHKERRQ(ierr);
    while (result<0 && j>=l) {
      tmp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp; j--;
      if (j>=l) {
        ierr = (*ds->comp_fun)(re,0.0,eig[perm[j]],0.0,&result,ds->comp_ctx);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSCopyMatrix_Private"
/*
  DSCopyMatrix_Private - Copies the trailing block of a matrix (from
  rows/columns l to n).
*/
PetscErrorCode DSCopyMatrix_Private(DS ds,DSMatType dst,DSMatType src)
{
  PetscErrorCode ierr;
  PetscInt    j,m,off,ld;
  PetscScalar *S,*D;

  PetscFunctionBegin;
  ld  = ds->ld;
  m   = ds->n-ds->l;
  off = ds->l+ds->l*ld;
  S   = ds->mat[src];
  D   = ds->mat[dst];
  for (j=0;j<m;j++) {
    ierr = PetscMemcpy(D+off+j*ld,S+off+j*ld,m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSPermuteColumns_Private"
PetscErrorCode DSPermuteColumns_Private(DS ds,PetscInt l,PetscInt n,DSMatType mat,PetscInt *perm)
{
  PetscInt    i,j,k,p,ld;
  PetscScalar *Q,rtmp;

  PetscFunctionBegin;
  ld = ds->ld;
  Q  = ds->mat[mat];
  for (i=l;i<n;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap columns i and j */
      for (k=0;k<n;k++) {
        rtmp = Q[k+p*ld]; Q[k+p*ld] = Q[k+i*ld]; Q[k+i*ld] = rtmp;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSPermuteRows_Private"
PetscErrorCode DSPermuteRows_Private(DS ds,PetscInt l,PetscInt n,DSMatType mat,PetscInt *perm)
{
  PetscInt    i,j,m=ds->m,k,p,ld;
  PetscScalar *Q,rtmp;

  PetscFunctionBegin;
  if (m==0) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"m was not set");
  ld = ds->ld;
  Q  = ds->mat[mat];
  for (i=l;i<n;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap rows i and j */
      for (k=0;k<m;k++) {
        rtmp = Q[p+k*ld]; Q[p+k*ld] = Q[i+k*ld]; Q[i+k*ld] = rtmp;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSPermuteBoth_Private"
PetscErrorCode DSPermuteBoth_Private(DS ds,PetscInt l,PetscInt n,DSMatType mat1,DSMatType mat2,PetscInt *perm)
{
  PetscInt    i,j,m=ds->m,k,p,ld;
  PetscScalar *U,*VT,rtmp;

  PetscFunctionBegin;
  if (m==0) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"m was not set");
  ld = ds->ld;
  U  = ds->mat[mat1];
  VT = ds->mat[mat2];
  for (i=l;i<n;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap columns i and j of U */
      for (k=0;k<n;k++) {
        rtmp = U[k+p*ld]; U[k+p*ld] = U[k+i*ld]; U[k+i*ld] = rtmp;
      }
      /* swap rows i and j of VT */
      for (k=0;k<m;k++) {
        rtmp = VT[p+k*ld]; VT[p+k*ld] = VT[i+k*ld]; VT[i+k*ld] = rtmp;
      }
    }
  }
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

#undef __FUNCT__  
#define __FUNCT__ "DSSetIdentity"
/*@
   DSSetIdentity - Copy the identity (a diagonal matrix with ones) on the
   active part of a matrix.

   Logically Collective on DS

   Input Parameters:
+  ds  - the direct solver context
-  mat - a matrix

   Level: advanced
@*/
PetscErrorCode DSSetIdentity(DS ds,DSMatType mat)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt       i,ld,n,l;

  PetscFunctionBegin;
  ierr = DSGetDimensions(ds,&n,PETSC_NULL,&l,PETSC_NULL);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(ds,&ld);CHKERRQ(ierr);
  ierr = DSGetArray(ds,mat,&x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscMemzero(&x[ld*l],ld*(n-l)*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=l;i<n;i++) {
    x[ld*i+i] = 1.0;
  }
  ierr = DSRestoreArray(ds,mat,&x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DSOrthogonalize"
/*@
   DSOrthogonalize - Orthogonalize the columns of a matrix.

   Logically Collective on DS

   Input Parameters:
+  ds   - the direct solver context
.  mat  - a matrix
-  cols - number of columns to orthogonalize (starting from the column zero)

   Output Parameter:
.  lindcols - number of linearly independent columns of the matrix (can be PETSC_NULL) 

   Level: advanced
@*/
PetscErrorCode DSOrthogonalize(DS ds,DSMatType mat,PetscInt cols,PetscInt *lindcols)
{
#if defined(PETSC_MISSING_LAPACK_GEQRF) || defined(SLEPC_MISSING_LAPACK_ORGQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEQRF/ORGQR - Lapack routine is unavailable");
#else
  PetscErrorCode  ierr;
  PetscInt        n,l,ld;
  PetscBLASInt    ld_,rA,cA,info,ltau,lw;
  PetscScalar     *A,*tau,*w,saux;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ds,mat,2);
  PetscValidLogicalCollectiveInt(ds,cols,3);
  ierr = DSGetDimensions(ds,&n,PETSC_NULL,&l,PETSC_NULL);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(ds,&ld);CHKERRQ(ierr);
  n = n - l;
  if (cols > n) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"Invalid number of columns");
  if (n == 0 || cols == 0) { PetscFunctionReturn(0); }
  ierr = DSGetArray(ds,mat,&A);CHKERRQ(ierr);
  ltau = PetscBLASIntCast(PetscMin(cols,n));
  ld_ = PetscBLASIntCast(ld);
  rA = PetscBLASIntCast(n);
  cA = PetscBLASIntCast(cols);
  lw = -1;
  LAPACKgeqrf_(&rA,&cA,A,&ld_,PETSC_NULL,&saux,&lw,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  lw = (PetscBLASInt)PetscRealPart(saux);
  ierr = DSAllocateWork_Private(ds,lw+ltau,0,0);CHKERRQ(ierr);
  tau = ds->work;
  w = &tau[ltau];
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgeqrf_(&rA,&cA,&A[ld*l+l],&ld_,tau,w,&lw,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  LAPACKorgqr_(&rA,&ltau,&ltau,&A[ld*l+l],&ld_,tau,w,&lw,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = DSRestoreArray(ds,mat,&A);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  if (lindcols) *lindcols = ltau;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "DSPseudoOrthogonalize"
/*@
   DSPseudoOrthogonalize - Orthogonalize the columns of a matrix with Modified
   Gram-Schmidt in an indefinite inner product space defined by a signature.

   Logically Collective on DS

   Input Parameters:
+  ds   - the direct solver context
.  mat  - the matrix
.  cols - number of columns to orthogonalize (starting from the column zero)
-  s    - the signature that defines the inner product

   Output Parameter:
+  lindcols - linear independent columns of the matrix (can be PETSC_NULL) 
-  ns - the new norm of the vectors (can be PETSC_NULL)

   Level: advanced
@*/
PetscErrorCode DSPseudoOrthogonalize(DS ds,DSMatType mat,PetscInt cols,PetscReal *s,PetscInt *lindcols,PetscReal *ns)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,k,l,n,ld;
  PetscBLASInt    one=1,rA_;
  PetscScalar     alpha,*A,*A_,*m,*h,nr0;
  PetscReal       nr_o,nr,*ns_;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ds,DS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ds,mat,2);
  PetscValidLogicalCollectiveInt(ds,cols,3);
  PetscValidScalarPointer(s,4);
  if (ns) PetscValidPointer(ns,6);
  ierr = DSGetDimensions(ds,&n,PETSC_NULL,&l,PETSC_NULL);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(ds,&ld);CHKERRQ(ierr);
  n = n - l;
  if (cols > n) SETERRQ(((PetscObject)ds)->comm,PETSC_ERR_ARG_WRONG,"Invalid number of columns");
  if (n == 0 || cols == 0) { PetscFunctionReturn(0); }
  rA_ = PetscBLASIntCast(n);
  ierr = DSGetArray(ds,mat,&A_);CHKERRQ(ierr);
  A = &A_[ld*l+l];
  ierr = DSAllocateWork_Private(ds,n+cols,ns?0:cols,0);CHKERRQ(ierr);
  m = ds->work;
  h = &m[n];
  ns_ = ns ? ns : ds->rwork;
  ierr = PetscLogEventBegin(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  for (i=0; i<cols; i++) {
    /* m <- diag(s)*A[i] */
    for (k=0; k<n; k++) m[k] = s[k]*A[k+i*ld];
    /* nr_o <- mynorm(A[i]'*m), mynorm(x) = sign(x)*sqrt(|x|) */
    ierr = SlepcDenseMatProd(&nr0,1,0.0,1.0,&A[ld*i],ld,n,1,PETSC_TRUE,m,n,n,1,PETSC_FALSE);CHKERRQ(ierr);
    nr = nr_o = PetscSign(PetscRealPart(nr0))*PetscSqrtReal(PetscAbsScalar(nr0));
    for (j=0; j<3 && i>0; j++) {
      /* h <- A[0:i-1]'*m */
      ierr = SlepcDenseMatProd(h,i,0.0,1.0,A,ld,n,i,PETSC_TRUE,m,n,n,1,PETSC_FALSE);CHKERRQ(ierr);
      /* h <- diag(ns)*h */
      for (k=0; k<i; k++) h[k] *= ns_[k];
      /* A[i] <- A[i] - A[0:i-1]*h */
      ierr = SlepcDenseMatProd(&A[ld*i],ld,1.0,-1.0,A,ld,n,i,PETSC_FALSE,h,i,i,1,PETSC_FALSE);CHKERRQ(ierr);
      /* m <- diag(s)*A[i] */
      for (k=0; k<n; k++) m[k] = s[k]*A[k+i*ld];
      /* nr_o <- mynorm(A[i]'*m) */
      ierr = SlepcDenseMatProd(&nr0,1,0.0,1.0,&A[ld*i],ld,n,1,PETSC_TRUE,m,n,n,1,PETSC_FALSE);CHKERRQ(ierr);
      nr = PetscSign(PetscRealPart(nr0))*PetscSqrtReal(PetscAbsScalar(nr0));
      if (PetscAbs(nr) < PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_SELF,1, "Linear dependency detected");
      if (PetscAbs(nr) > 0.7*PetscAbs(nr_o)) break;
      nr_o = nr;
    }
    ns_[i] = PetscSign(nr);
    /* A[i] <- A[i]/|nr| */
    alpha = 1.0/PetscAbs(nr);
    BLASscal_(&rA_,&alpha,&A[i*ld],&one);
  }
  ierr = PetscLogEventEnd(DS_Other,ds,0,0,0);CHKERRQ(ierr);
  ierr = DSRestoreArray(ds,mat,&A_);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)ds);CHKERRQ(ierr);
  if (lindcols) *lindcols = cols;
  PetscFunctionReturn(0);
}
