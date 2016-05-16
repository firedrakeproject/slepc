/*
      MFN routines related to options that can be set via the command-line
      or procedurally.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/mfnimpl.h>   /*I "slepcmfn.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "MFNMonitorSetFromOptions"
/*@C
   MFNMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type
   indicated by the user.

   Collective on MFN

   Input Parameters:
+  mfn      - the eigensolver context
.  name     - the monitor option name
.  help     - message indicating what monitoring is done
.  manual   - manual page for the monitor
.  monitor  - the monitor function, whose context is a PetscViewerAndFormat
-  trackall - whether this monitor tracks all eigenvalues or not

   Level: developer

.seealso: MFNMonitorSet(), MFNSetTrackAll(), MFNConvMonitorSetFromOptions()
@*/
PetscErrorCode MFNMonitorSetFromOptions(MFN mfn,const char name[],const char help[],const char manual[],PetscErrorCode (*monitor)(MFN,PetscInt,PetscReal,PetscViewerAndFormat*))
{
  PetscErrorCode       ierr;
  PetscBool            flg;
  PetscViewer          viewer;
  PetscViewerFormat    format;
  PetscViewerAndFormat *vf;

  PetscFunctionBegin;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)mfn),((PetscObject)mfn)->prefix,name,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerAndFormatCreate(viewer,format,&vf);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    ierr = MFNMonitorSet(mfn,(PetscErrorCode (*)(MFN,PetscInt,PetscReal,void*))monitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetFromOptions"
/*@
   MFNSetFromOptions - Sets MFN options from the options database.
   This routine must be called before MFNSetUp() if the user is to be
   allowed to set the solver type.

   Collective on MFN

   Input Parameters:
.  mfn - the matrix function context

   Notes:
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode MFNSetFromOptions(MFN mfn)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscBool      set,flg,flg1,flg2;
  PetscReal      r;
  PetscInt       i;
  PetscDrawLG    lg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  ierr = MFNRegisterAll();CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)mfn);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-mfn_type","Matrix Function method","MFNSetType",MFNList,(char*)(((PetscObject)mfn)->type_name?((PetscObject)mfn)->type_name:MFNKRYLOV),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MFNSetType(mfn,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!((PetscObject)mfn)->type_name) {
      ierr = MFNSetType(mfn,MFNKRYLOV);CHKERRQ(ierr);
    }

    i = mfn->max_it;
    ierr = PetscOptionsInt("-mfn_max_it","Maximum number of iterations","MFNSetTolerances",mfn->max_it,&i,&flg1);CHKERRQ(ierr);
    if (!flg1) i = PETSC_DEFAULT;
    r = mfn->tol;
    ierr = PetscOptionsReal("-mfn_tol","Tolerance","MFNSetTolerances",mfn->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:mfn->tol,&r,&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) {
      ierr = MFNSetTolerances(mfn,r,i);CHKERRQ(ierr);
    }

    ierr = PetscOptionsInt("-mfn_ncv","Number of basis vectors","MFNSetDimensions",mfn->ncv,&i,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MFNSetDimensions(mfn,i);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBool("-mfn_error_if_not_converged","Generate error if solver does not converge","MFNSetErrorIfNotConverged",mfn->errorifnotconverged,&mfn->errorifnotconverged,NULL);CHKERRQ(ierr);

    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to MFNSetFromOptions()
    */
    ierr = PetscOptionsBool("-mfn_monitor_cancel","Remove any hardwired monitor routines","MFNMonitorCancel",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
    if (set && flg) {
      ierr = MFNMonitorCancel(mfn);CHKERRQ(ierr);
    }
    /*
      Text monitors
    */
    ierr = MFNMonitorSetFromOptions(mfn,"-mfn_monitor","Monitor error estimate","MFNMonitorDefault",MFNMonitorDefault);CHKERRQ(ierr);
    /*
      Line graph monitors
    */
    ierr = PetscOptionsBool("-mfn_monitor_lg","Monitor error estimate graphically","MFNMonitorSet",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
    if (set && flg) {
      ierr = MFNMonitorLGCreate(PetscObjectComm((PetscObject)mfn),NULL,"Error estimate",PETSC_DECIDE,PETSC_DECIDE,300,300,&lg);CHKERRQ(ierr);
      ierr = MFNMonitorSet(mfn,MFNMonitorLG,lg,(PetscErrorCode (*)(void**))PetscDrawLGDestroy);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/

    ierr = PetscOptionsName("-mfn_view","Print detailed information on solver used","MFNView",NULL);CHKERRQ(ierr);

    if (mfn->ops->setfromoptions) {
      ierr = (*mfn->ops->setfromoptions)(PetscOptionsObject,mfn);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)mfn);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!mfn->V) { ierr = MFNGetBV(mfn,&mfn->V);CHKERRQ(ierr); }
  ierr = BVSetFromOptions(mfn->V);CHKERRQ(ierr);
  if (!mfn->fn) { ierr = MFNGetFN(mfn,&mfn->fn);CHKERRQ(ierr); }
  ierr = FNSetFromOptions(mfn->fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetTolerances"
/*@
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

#undef __FUNCT__
#define __FUNCT__ "MFNSetTolerances"
/*@
   MFNSetTolerances - Sets the tolerance and maximum iteration count used
   by the MFN convergence tests.

   Logically Collective on MFN

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
    if (tol <= 0.0) SETERRQ(PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
    mfn->tol = tol;
  }
  if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
    mfn->max_it = 0;
    mfn->setupcalled = 0;
  } else {
    if (maxits <= 0) SETERRQ(PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
    mfn->max_it = maxits;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetDimensions"
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
  PetscValidPointer(ncv,2);
  *ncv = mfn->ncv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetDimensions"
/*@
   MFNSetDimensions - Sets the dimension of the subspace to be used by the solver.

   Logically Collective on MFN

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
    mfn->ncv = 0;
  } else {
    if (ncv<1) SETERRQ(PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
    mfn->ncv = ncv;
  }
  mfn->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetErrorIfNotConverged"
/*@
   MFNSetErrorIfNotConverged - Causes MFNSolve() to generate an error if the
   solver has not converged.

   Logically Collective on MFN

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

#undef __FUNCT__
#define __FUNCT__ "MFNGetErrorIfNotConverged"
/*@
   MFNGetErrorIfNotConverged - Return a flag indicating whether MFNSolve() will
   generate an error if the solver does not converge.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameter:
.  flag - PETSC_TRUE if it will generate an error, else PETSC_FALSE

   Level: intermediate

.seealso:  MFNSetErrorIfNotConverged()
@*/
PetscErrorCode MFNGetErrorIfNotConverged(MFN mfn,PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = mfn->errorifnotconverged;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetOptionsPrefix"
/*@C
   MFNSetOptionsPrefix - Sets the prefix used for searching for all
   MFN options in the database.

   Logically Collective on MFN

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (!mfn->V) { ierr = MFNGetBV(mfn,&mfn->V);CHKERRQ(ierr); }
  ierr = BVSetOptionsPrefix(mfn->V,prefix);CHKERRQ(ierr);
  if (!mfn->fn) { ierr = MFNGetFN(mfn,&mfn->fn);CHKERRQ(ierr); }
  ierr = FNSetOptionsPrefix(mfn->fn,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)mfn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNAppendOptionsPrefix"
/*@C
   MFNAppendOptionsPrefix - Appends to the prefix used for searching for all
   MFN options in the database.

   Logically Collective on MFN

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (!mfn->V) { ierr = MFNGetBV(mfn,&mfn->V);CHKERRQ(ierr); }
  ierr = BVSetOptionsPrefix(mfn->V,prefix);CHKERRQ(ierr);
  if (!mfn->fn) { ierr = MFNGetFN(mfn,&mfn->fn);CHKERRQ(ierr); }
  ierr = FNSetOptionsPrefix(mfn->fn,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)mfn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetOptionsPrefix"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)mfn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
