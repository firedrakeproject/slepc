/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   MFN routines related to options that can be set via the command-line
   or procedurally
*/

#include <slepc/private/mfnimpl.h>   /*I "slepcmfn.h" I*/
#include <petscdraw.h>

/*@C
   MFNMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type
   indicated by the user.

   Collective on mfn

   Input Parameters:
+  mfn  - the matrix function context
.  opt  - the command line option for this monitor
.  name - the monitor type one is seeking
-  ctx  - an optional user context for the monitor, or NULL

   Level: developer

.seealso: MFNMonitorSet()
@*/
PetscErrorCode MFNMonitorSetFromOptions(MFN mfn,const char opt[],const char name[],void *ctx)
{
  PetscErrorCode       (*mfunc)(MFN,PetscInt,PetscReal,void*);
  PetscErrorCode       (*cfunc)(PetscViewer,PetscViewerFormat,void*,PetscViewerAndFormat**);
  PetscErrorCode       (*dfunc)(PetscViewerAndFormat**);
  PetscViewerAndFormat *vf;
  PetscViewer          viewer;
  PetscViewerFormat    format;
  PetscViewerType      vtype;
  char                 key[PETSC_MAX_PATH_LEN];
  PetscBool            flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)mfn),((PetscObject)mfn)->options,((PetscObject)mfn)->prefix,opt,&viewer,&format,&flg));
  if (!flg) PetscFunctionReturn(0);

  CHKERRQ(PetscViewerGetType(viewer,&vtype));
  CHKERRQ(SlepcMonitorMakeKey_Internal(name,vtype,format,key));
  CHKERRQ(PetscFunctionListFind(MFNMonitorList,key,&mfunc));
  PetscCheck(mfunc,PetscObjectComm((PetscObject)mfn),PETSC_ERR_SUP,"Specified viewer and format not supported");
  CHKERRQ(PetscFunctionListFind(MFNMonitorCreateList,key,&cfunc));
  CHKERRQ(PetscFunctionListFind(MFNMonitorDestroyList,key,&dfunc));
  if (!cfunc) cfunc = PetscViewerAndFormatCreate_Internal;
  if (!dfunc) dfunc = PetscViewerAndFormatDestroy;

  CHKERRQ((*cfunc)(viewer,format,ctx,&vf));
  CHKERRQ(PetscObjectDereference((PetscObject)viewer));
  CHKERRQ(MFNMonitorSet(mfn,mfunc,vf,(PetscErrorCode(*)(void **))dfunc));
  PetscFunctionReturn(0);
}

/*@
   MFNSetFromOptions - Sets MFN options from the options database.
   This routine must be called before MFNSetUp() if the user is to be
   allowed to set the solver type.

   Collective on mfn

   Input Parameters:
.  mfn - the matrix function context

   Notes:
   To see all options, run your program with the -help option.

   Level: beginner

.seealso: MFNSetOptionsPrefix()
@*/
PetscErrorCode MFNSetFromOptions(MFN mfn)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscBool      set,flg,flg1,flg2;
  PetscReal      r;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  CHKERRQ(MFNRegisterAll());
  ierr = PetscObjectOptionsBegin((PetscObject)mfn);CHKERRQ(ierr);
    CHKERRQ(PetscOptionsFList("-mfn_type","Matrix Function method","MFNSetType",MFNList,(char*)(((PetscObject)mfn)->type_name?((PetscObject)mfn)->type_name:MFNKRYLOV),type,sizeof(type),&flg));
    if (flg) {
      CHKERRQ(MFNSetType(mfn,type));
    } else if (!((PetscObject)mfn)->type_name) {
      CHKERRQ(MFNSetType(mfn,MFNKRYLOV));
    }

    i = mfn->max_it;
    CHKERRQ(PetscOptionsInt("-mfn_max_it","Maximum number of iterations","MFNSetTolerances",mfn->max_it,&i,&flg1));
    if (!flg1) i = PETSC_DEFAULT;
    r = mfn->tol;
    CHKERRQ(PetscOptionsReal("-mfn_tol","Tolerance","MFNSetTolerances",SlepcDefaultTol(mfn->tol),&r,&flg2));
    if (flg1 || flg2) CHKERRQ(MFNSetTolerances(mfn,r,i));

    CHKERRQ(PetscOptionsInt("-mfn_ncv","Number of basis vectors","MFNSetDimensions",mfn->ncv,&i,&flg));
    if (flg) CHKERRQ(MFNSetDimensions(mfn,i));

    CHKERRQ(PetscOptionsBool("-mfn_error_if_not_converged","Generate error if solver does not converge","MFNSetErrorIfNotConverged",mfn->errorifnotconverged,&mfn->errorifnotconverged,NULL));

    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to MFNSetFromOptions()
    */
    CHKERRQ(PetscOptionsBool("-mfn_monitor_cancel","Remove any hardwired monitor routines","MFNMonitorCancel",PETSC_FALSE,&flg,&set));
    if (set && flg) CHKERRQ(MFNMonitorCancel(mfn));
    CHKERRQ(MFNMonitorSetFromOptions(mfn,"-mfn_monitor","error_estimate",NULL));

    /* -----------------------------------------------------------------------*/
    CHKERRQ(PetscOptionsName("-mfn_view","Print detailed information on solver used","MFNView",NULL));

    if (mfn->ops->setfromoptions) {
      CHKERRQ((*mfn->ops->setfromoptions)(PetscOptionsObject,mfn));
    }
    CHKERRQ(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)mfn));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!mfn->V) CHKERRQ(MFNGetBV(mfn,&mfn->V));
  CHKERRQ(BVSetFromOptions(mfn->V));
  if (!mfn->fn) CHKERRQ(MFNGetFN(mfn,&mfn->fn));
  CHKERRQ(FNSetFromOptions(mfn->fn));
  PetscFunctionReturn(0);
}

/*@C
   MFNGetTolerances - Gets the tolerance and maximum iteration count used
   by the MFN convergence tests.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: MFNSetTolerances()
@*/
PetscErrorCode MFNGetTolerances(MFN mfn,PetscReal *tol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (tol)    *tol    = mfn->tol;
  if (maxits) *maxits = mfn->max_it;
  PetscFunctionReturn(0);
}

/*@
   MFNSetTolerances - Sets the tolerance and maximum iteration count used
   by the MFN convergence tests.

   Logically Collective on mfn

   Input Parameters:
+  mfn - the matrix function context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -mfn_tol <tol> - Sets the convergence tolerance
-  -mfn_max_it <maxits> - Sets the maximum number of iterations allowed

   Notes:
   Use PETSC_DEFAULT for either argument to assign a reasonably good value.

   Level: intermediate

.seealso: MFNGetTolerances()
@*/
PetscErrorCode MFNSetTolerances(MFN mfn,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidLogicalCollectiveReal(mfn,tol,2);
  PetscValidLogicalCollectiveInt(mfn,maxits,3);
  if (tol == PETSC_DEFAULT) {
    mfn->tol = PETSC_DEFAULT;
    mfn->setupcalled = 0;
  } else {
    PetscCheck(tol>0.0,PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
    mfn->tol = tol;
  }
  if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
    mfn->max_it = PETSC_DEFAULT;
    mfn->setupcalled = 0;
  } else {
    PetscCheck(maxits>0,PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
    mfn->max_it = maxits;
  }
  PetscFunctionReturn(0);
}

/*@
   MFNGetDimensions - Gets the dimension of the subspace used by the solver.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameter:
.  ncv - the maximum dimension of the subspace to be used by the solver

   Level: intermediate

.seealso: MFNSetDimensions()
@*/
PetscErrorCode MFNGetDimensions(MFN mfn,PetscInt *ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidIntPointer(ncv,2);
  *ncv = mfn->ncv;
  PetscFunctionReturn(0);
}

/*@
   MFNSetDimensions - Sets the dimension of the subspace to be used by the solver.

   Logically Collective on mfn

   Input Parameters:
+  mfn - the matrix function context
-  ncv - the maximum dimension of the subspace to be used by the solver

   Options Database Keys:
.  -mfn_ncv <ncv> - Sets the dimension of the subspace

   Notes:
   Use PETSC_DEFAULT for ncv to assign a reasonably good value, which is
   dependent on the solution method.

   Level: intermediate

.seealso: MFNGetDimensions()
@*/
PetscErrorCode MFNSetDimensions(MFN mfn,PetscInt ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidLogicalCollectiveInt(mfn,ncv,2);
  if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
    mfn->ncv = PETSC_DEFAULT;
  } else {
    PetscCheck(ncv>0,PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
    mfn->ncv = ncv;
  }
  mfn->setupcalled = 0;
  PetscFunctionReturn(0);
}

/*@
   MFNSetErrorIfNotConverged - Causes MFNSolve() to generate an error if the
   solver has not converged.

   Logically Collective on mfn

   Input Parameters:
+  mfn - the matrix function context
-  flg - PETSC_TRUE indicates you want the error generated

   Options Database Keys:
.  -mfn_error_if_not_converged - this takes an optional truth value (0/1/no/yes/true/false)

   Level: intermediate

   Note:
   Normally SLEPc continues if the solver fails to converge, you can call
   MFNGetConvergedReason() after a MFNSolve() to determine if it has converged.

.seealso: MFNGetErrorIfNotConverged()
@*/
PetscErrorCode MFNSetErrorIfNotConverged(MFN mfn,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidLogicalCollectiveBool(mfn,flg,2);
  mfn->errorifnotconverged = flg;
  PetscFunctionReturn(0);
}

/*@
   MFNGetErrorIfNotConverged - Return a flag indicating whether MFNSolve() will
   generate an error if the solver does not converge.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameter:
.  flag - PETSC_TRUE if it will generate an error, else PETSC_FALSE

   Level: intermediate

.seealso: MFNSetErrorIfNotConverged()
@*/
PetscErrorCode MFNGetErrorIfNotConverged(MFN mfn,PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidBoolPointer(flag,2);
  *flag = mfn->errorifnotconverged;
  PetscFunctionReturn(0);
}

/*@C
   MFNSetOptionsPrefix - Sets the prefix used for searching for all
   MFN options in the database.

   Logically Collective on mfn

   Input Parameters:
+  mfn - the matrix function context
-  prefix - the prefix string to prepend to all MFN option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different MFN contexts, one could call
.vb
      MFNSetOptionsPrefix(mfn1,"fun1_")
      MFNSetOptionsPrefix(mfn2,"fun2_")
.ve

   Level: advanced

.seealso: MFNAppendOptionsPrefix(), MFNGetOptionsPrefix()
@*/
PetscErrorCode MFNSetOptionsPrefix(MFN mfn,const char *prefix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (!mfn->V) CHKERRQ(MFNGetBV(mfn,&mfn->V));
  CHKERRQ(BVSetOptionsPrefix(mfn->V,prefix));
  if (!mfn->fn) CHKERRQ(MFNGetFN(mfn,&mfn->fn));
  CHKERRQ(FNSetOptionsPrefix(mfn->fn,prefix));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)mfn,prefix));
  PetscFunctionReturn(0);
}

/*@C
   MFNAppendOptionsPrefix - Appends to the prefix used for searching for all
   MFN options in the database.

   Logically Collective on mfn

   Input Parameters:
+  mfn - the matrix function context
-  prefix - the prefix string to prepend to all MFN option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: MFNSetOptionsPrefix(), MFNGetOptionsPrefix()
@*/
PetscErrorCode MFNAppendOptionsPrefix(MFN mfn,const char *prefix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (!mfn->V) CHKERRQ(MFNGetBV(mfn,&mfn->V));
  CHKERRQ(BVAppendOptionsPrefix(mfn->V,prefix));
  if (!mfn->fn) CHKERRQ(MFNGetFN(mfn,&mfn->fn));
  CHKERRQ(FNAppendOptionsPrefix(mfn->fn,prefix));
  CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)mfn,prefix));
  PetscFunctionReturn(0);
}

/*@C
   MFNGetOptionsPrefix - Gets the prefix used for searching for all
   MFN options in the database.

   Not Collective

   Input Parameters:
.  mfn - the matrix function context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Note:
   On the Fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: MFNSetOptionsPrefix(), MFNAppendOptionsPrefix()
@*/
PetscErrorCode MFNGetOptionsPrefix(MFN mfn,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(prefix,2);
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)mfn,prefix));
  PetscFunctionReturn(0);
}
