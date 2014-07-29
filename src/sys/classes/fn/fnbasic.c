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

#include <slepc-private/fnimpl.h>      /*I "slepcfn.h" I*/

PetscFunctionList FNList = 0;
PetscBool         FNRegisterAllCalled = PETSC_FALSE;
PetscClassId      FN_CLASSID = 0;
static PetscBool  FNPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "FNFinalizePackage"
/*@C
   FNFinalizePackage - This function destroys everything in the Slepc interface
   to the FN package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode FNFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&FNList);CHKERRQ(ierr);
  FNPackageInitialized = PETSC_FALSE;
  FNRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNInitializePackage"
/*@C
  FNInitializePackage - This function initializes everything in the FN package.
  It is called from PetscDLLibraryRegister() when using dynamic libraries, and
  on the first call to FNCreate() when using static libraries.

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode FNInitializePackage(void)
{
  char             logList[256];
  char             *className;
  PetscBool        opt;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (FNPackageInitialized) PetscFunctionReturn(0);
  FNPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Math function",&FN_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = FNRegisterAll();CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"fn",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(FN_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"fn",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(FN_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(FNFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNCreate"
/*@C
   FNCreate - Creates an FN context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newfn - location to put the FN context

   Level: beginner

.seealso: FNDestroy(), FN
@*/
PetscErrorCode FNCreate(MPI_Comm comm,FN *newfn)
{
  FN             fn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newfn,2);
  *newfn = 0;
  ierr = FNInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(fn,_p_FN,struct _FNOps,FN_CLASSID,"FN","Math Function","FN",comm,FNDestroy,FNView);CHKERRQ(ierr);
  fn->na       = 0;
  fn->alpha    = NULL;
  fn->nb       = 0;
  fn->beta     = NULL;

  *newfn = fn;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNSetOptionsPrefix"
/*@C
   FNSetOptionsPrefix - Sets the prefix used for searching for all
   FN options in the database.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  prefix - the prefix string to prepend to all FN option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: FNAppendOptionsPrefix()
@*/
PetscErrorCode FNSetOptionsPrefix(FN fn,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNAppendOptionsPrefix"
/*@C
   FNAppendOptionsPrefix - Appends to the prefix used for searching for all
   FN options in the database.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  prefix - the prefix string to prepend to all FN option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: FNSetOptionsPrefix()
@*/
PetscErrorCode FNAppendOptionsPrefix(FN fn,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNGetOptionsPrefix"
/*@C
   FNGetOptionsPrefix - Gets the prefix used for searching for all
   FN options in the database.

   Not Collective

   Input Parameters:
.  fn - the math function context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: FNSetOptionsPrefix(), FNAppendOptionsPrefix()
@*/
PetscErrorCode FNGetOptionsPrefix(FN fn,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNSetType"
/*@C
   FNSetType - Selects the type for the FN object.

   Logically Collective on FN

   Input Parameter:
+  fn   - the math function context
-  type - a known type

   Notes:
   The default is FNRATIONAL, which includes polynomials as a particular
   case as well as simple functions such as f(x)=x and f(x)=constant.

   Level: intermediate

.seealso: FNGetType()
@*/
PetscErrorCode FNSetType(FN fn,FNType type)
{
  PetscErrorCode ierr,(*r)(FN);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)fn,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(FNList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested FN type %s",type);

  ierr = PetscMemzero(fn->ops,sizeof(struct _FNOps));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)fn,type);CHKERRQ(ierr);
  ierr = (*r)(fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNGetType"
/*@C
   FNGetType - Gets the FN type name (as a string) from the FN context.

   Not Collective

   Input Parameter:
.  fn - the math function context

   Output Parameter:
.  name - name of the math function

   Level: intermediate

.seealso: FNSetType()
@*/
PetscErrorCode FNGetType(FN fn,FNType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)fn)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNSetParameters"
/*@
   FNSetParameters - Sets the parameters that define the matematical function.

   Logically Collective on FN

   Input Parameters:
+  fn    - the math function context
.  na    - number of parameters in the first group
.  alpha - first group of parameters (array of scalar values)
.  nb    - number of parameters in the second group
-  beta  - second group of parameters (array of scalar values)

   Notes:
   In a rational function r(x) = p(x)/q(x), where p(x) and q(x) are polynomials,
   the parameters alpha and beta represent the coefficients of p(x) and q(x),
   respectively. Hence, p(x) is of degree na-1 and q(x) of degree nb-1.
   If nb is zero, then the function is assumed to be polynomial, r(x) = p(x).

   In other functions the parameters have other meanings.

   In polynomials, high order coefficients are stored in the first positions
   of the array, e.g. to represent x^2-3 use {1,0,-3}.

   Level: intermediate

.seealso: FNGetParameters()
@*/
PetscErrorCode FNSetParameters(FN fn,PetscInt na,PetscScalar *alpha,PetscInt nb,PetscScalar *beta)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveInt(fn,na,2);
  if (na<0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Argument na cannot be negative");
  if (na) PetscValidPointer(alpha,3);
  PetscValidLogicalCollectiveInt(fn,nb,4);
  if (nb<0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Argument nb cannot be negative");
  if (nb) PetscValidPointer(beta,5);
  fn->na = na;
  ierr = PetscFree(fn->alpha);CHKERRQ(ierr);
  if (na) {
    ierr = PetscMalloc1(na,&fn->alpha);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)fn,na*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<na;i++) fn->alpha[i] = alpha[i];
  }
  fn->nb = nb;
  ierr = PetscFree(fn->beta);CHKERRQ(ierr);
  if (nb) {
    ierr = PetscMalloc1(nb,&fn->beta);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)fn,nb*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<nb;i++) fn->beta[i] = beta[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNGetParameters"
/*@
   FNGetParameters - Returns the parameters that define the matematical function.

   Not Collective

   Input Parameter:
.  fn    - the math function context

   Output Parameters:
+  na    - number of parameters in the first group
.  alpha - first group of parameters (array of scalar values)
.  nb    - number of parameters in the second group
-  beta  - second group of parameters (array of scalar values)

   Level: intermediate

.seealso: FNSetParameters()
@*/
PetscErrorCode FNGetParameters(FN fn,PetscInt *na,PetscScalar *alpha[],PetscInt *nb,PetscScalar *beta[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  if (na)    *na = fn->na;
  if (alpha) *alpha = fn->alpha;
  if (nb)    *nb = fn->nb;
  if (beta)  *beta = fn->beta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunction"
/*@
   FNEvaluateFunction - Computes the value of the function f(x) for a given x.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  x  - the value where the function must be evaluated

   Output Parameter:
.  y  - the result of f(x)

   Level: intermediate

.seealso: FNEvaluateDerivative()
@*/
PetscErrorCode FNEvaluateFunction(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveScalar(fn,x,2);
  PetscValidPointer(y,3);
  if (!((PetscObject)fn)->type_name) {
    ierr = FNSetType(fn,FNRATIONAL);CHKERRQ(ierr);
  }
  ierr = (*fn->ops->evaluatefunction)(fn,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative"
/*@
   FNEvaluateDerivative - Computes the value of the derivative f'(x) for a given x.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  x  - the value where the derivative must be evaluated

   Output Parameter:
.  y  - the result of f'(x)

   Level: intermediate

.seealso: FNEvaluateFunction()
@*/
PetscErrorCode FNEvaluateDerivative(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveScalar(fn,x,2);
  PetscValidPointer(y,3);
  if (!((PetscObject)fn)->type_name) {
    ierr = FNSetType(fn,FNRATIONAL);CHKERRQ(ierr);
  }
  ierr = (*fn->ops->evaluatederivative)(fn,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNSetFromOptions"
/*@
   FNSetFromOptions - Sets FN options from the options database.

   Collective on FN

   Input Parameters:
.  fn - the math function context

   Notes:
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode FNSetFromOptions(FN fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  if (!FNRegisterAllCalled) { ierr = FNRegisterAll();CHKERRQ(ierr); }
  /* Set default type (we do not allow changing it with -fn_type) */
  if (!((PetscObject)fn)->type_name) {
    ierr = FNSetType(fn,FNRATIONAL);CHKERRQ(ierr);
  }
  ierr = PetscObjectOptionsBegin((PetscObject)fn);CHKERRQ(ierr);
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)fn);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNView"
/*@C
   FNView - Prints the FN data structure.

   Collective on FN

   Input Parameters:
+  fn - the math function context
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
PetscErrorCode FNView(FN fn,PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)fn));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(fn,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)fn,viewer);CHKERRQ(ierr);
    if (fn->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*fn->ops->view)(fn,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNDestroy"
/*@C
   FNDestroy - Destroys FN context that was created with FNCreate().

   Collective on FN

   Input Parameter:
.  fn - the math function context

   Level: beginner

.seealso: FNCreate()
@*/
PetscErrorCode FNDestroy(FN *fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*fn) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*fn,FN_CLASSID,1);
  if (--((PetscObject)(*fn))->refct > 0) { *fn = 0; PetscFunctionReturn(0); }
  ierr = PetscFree((*fn)->alpha);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->beta);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNRegister"
/*@C
   FNRegister - See Adds a mathematical function to the FN package.

   Not collective

   Input Parameters:
+  name - name of a new user-defined FN
-  function - routine to create context

   Notes:
   FNRegister() may be called multiple times to add several user-defined inner products.

   Level: advanced

.seealso: FNRegisterAll()
@*/
PetscErrorCode FNRegister(const char *name,PetscErrorCode (*function)(FN))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&FNList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode FNCreate_Rational(FN);
PETSC_EXTERN PetscErrorCode FNCreate_Exp(FN);

#undef __FUNCT__
#define __FUNCT__ "FNRegisterAll"
/*@C
   FNRegisterAll - Registers all of the math functions in the FN package.

   Not Collective

   Level: advanced
@*/
PetscErrorCode FNRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  FNRegisterAllCalled = PETSC_TRUE;
  ierr = FNRegister(FNRATIONAL,FNCreate_Rational);CHKERRQ(ierr);
  ierr = FNRegister(FNEXP,FNCreate_Exp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

