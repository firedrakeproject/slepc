/*
      NEP routines related to options that can be set via the command-line
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

#include <slepc/private/nepimpl.h>       /*I "slepcnep.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "NEPMonitorSetFromOptions"
/*@C
   NEPMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type
   indicated by the user.

   Collective on NEP

   Input Parameters:
+  nep      - the nonlinear eigensolver context
.  name     - the monitor option name
.  help     - message indicating what monitoring is done
.  manual   - manual page for the monitor
.  monitor  - the monitor function, whose context is a PetscViewerAndFormat
-  trackall - whether this monitor tracks all eigenvalues or not

   Level: developer

.seealso: NEPMonitorSet(), NEPSetTrackAll(), NEPConvMonitorSetFromOptions()
@*/
PetscErrorCode NEPMonitorSetFromOptions(NEP nep,const char name[],const char help[],const char manual[],PetscErrorCode (*monitor)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*),PetscBool trackall)
{
  PetscErrorCode       ierr;
  PetscBool            flg;
  PetscViewer          viewer;
  PetscViewerFormat    format;
  PetscViewerAndFormat *vf;

  PetscFunctionBegin;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->prefix,name,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerAndFormatCreate(viewer,format,&vf);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    ierr = NEPMonitorSet(nep,(PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))monitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);
    if (trackall) {
      ierr = NEPSetTrackAll(nep,PETSC_TRUE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPConvMonitorSetFromOptions"
/*@C
   NEPConvMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type
   indicated by the user (for monitors that only show iteration numbers of convergence).

   Collective on NEP

   Input Parameters:
+  nep      - the nonlinear eigensolver context
.  name     - the monitor option name
.  help     - message indicating what monitoring is done
.  manual   - manual page for the monitor
-  monitor  - the monitor function, whose context is a SlepcConvMonitor

   Level: developer

.seealso: NEPMonitorSet(), NEPMonitorSetFromOptions()
@*/
PetscErrorCode NEPConvMonitorSetFromOptions(NEP nep,const char name[],const char help[],const char manual[],PetscErrorCode (*monitor)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,SlepcConvMonitor))
{
  PetscErrorCode    ierr;
  PetscBool         flg;
  PetscViewer       viewer;
  PetscViewerFormat format;
  SlepcConvMonitor  ctx;

  PetscFunctionBegin;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->prefix,name,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SlepcConvMonitorCreate(viewer,format,&ctx);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    ierr = NEPMonitorSet(nep,(PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))monitor,ctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions"
/*@
   NEPSetFromOptions - Sets NEP options from the options database.
   This routine must be called before NEPSetUp() if the user is to be
   allowed to set the solver type.

   Collective on NEP

   Input Parameters:
.  nep - the nonlinear eigensolver context

   Notes:
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode NEPSetFromOptions(NEP nep)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscBool      set,flg,flg1,flg2,flg3;
  PetscReal      r;
  PetscScalar    s;
  PetscInt       i,j,k;
  PetscDrawLG    lg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = NEPRegisterAll();CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)nep);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-nep_type","Nonlinear Eigenvalue Problem method","NEPSetType",NEPList,(char*)(((PetscObject)nep)->type_name?((PetscObject)nep)->type_name:NEPRII),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = NEPSetType(nep,type);CHKERRQ(ierr);
    } else if (!((PetscObject)nep)->type_name) {
      ierr = NEPSetType(nep,NEPRII);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEnum("-nep_refine","Iterative refinement method","NEPSetRefine",NEPRefineTypes,(PetscEnum)nep->refine,(PetscEnum*)&nep->refine,NULL);CHKERRQ(ierr);

    i = nep->npart;
    ierr = PetscOptionsInt("-nep_refine_partitions","Number of partitions of the communicator for iterative refinement","NEPSetRefine",nep->npart,&i,&flg1);CHKERRQ(ierr);
    r = nep->rtol;
    ierr = PetscOptionsReal("-nep_refine_tol","Tolerance for iterative refinement","NEPSetRefine",nep->rtol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL/1000:nep->rtol,&r,&flg2);CHKERRQ(ierr);
    j = nep->rits;
    ierr = PetscOptionsInt("-nep_refine_its","Maximum number of iterations for iterative refinement","NEPSetRefine",nep->rits,&j,&flg3);CHKERRQ(ierr);
    if (flg1 || flg2 || flg3) {
      ierr = NEPSetRefine(nep,nep->refine,i,r,j,nep->scheme);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEnum("-nep_refine_scheme","Scheme used for linear systems within iterative refinement","NEPSetRefine",NEPRefineSchemes,(PetscEnum)nep->scheme,(PetscEnum*)&nep->scheme,NULL);CHKERRQ(ierr);

    i = nep->max_it? nep->max_it: PETSC_DEFAULT;
    ierr = PetscOptionsInt("-nep_max_it","Maximum number of iterations","NEPSetTolerances",nep->max_it,&i,&flg1);CHKERRQ(ierr);
    r = nep->tol;
    ierr = PetscOptionsReal("-nep_tol","Tolerance","NEPSetTolerances",nep->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:nep->tol,&r,&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) {
      ierr = NEPSetTolerances(nep,r,i);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBoolGroupBegin("-nep_conv_rel","Relative error convergence test","NEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetConvergenceTest(nep,NEP_CONV_REL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_conv_norm","Convergence test relative to the matrix norms","NEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetConvergenceTest(nep,NEP_CONV_NORM);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_conv_abs","Absolute error convergence test","NEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetConvergenceTest(nep,NEP_CONV_ABS);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-nep_conv_user","User-defined convergence test","NEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetConvergenceTest(nep,NEP_CONV_USER);CHKERRQ(ierr); }

    ierr = PetscOptionsBoolGroupBegin("-nep_stop_basic","Stop iteration if all eigenvalues converged or max_it reached","NEPSetStoppingTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetStoppingTest(nep,NEP_STOP_BASIC);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-nep_stop_user","User-defined stopping test","NEPSetStoppingTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetStoppingTest(nep,NEP_STOP_USER);CHKERRQ(ierr); }

    i = nep->nev;
    ierr = PetscOptionsInt("-nep_nev","Number of eigenvalues to compute","NEPSetDimensions",nep->nev,&i,&flg1);CHKERRQ(ierr);
    j = nep->ncv? nep->ncv: PETSC_DEFAULT;
    ierr = PetscOptionsInt("-nep_ncv","Number of basis vectors","NEPSetDimensions",nep->ncv,&j,&flg2);CHKERRQ(ierr);
    k = nep->mpd? nep->mpd: PETSC_DEFAULT;
    ierr = PetscOptionsInt("-nep_mpd","Maximum dimension of projected problem","NEPSetDimensions",nep->mpd,&k,&flg3);CHKERRQ(ierr);
    if (flg1 || flg2 || flg3) {
      ierr = NEPSetDimensions(nep,i,j,k);CHKERRQ(ierr);
    }

    ierr = PetscOptionsScalar("-nep_target","Value of the target","NEPSetTarget",nep->target,&s,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = NEPSetWhichEigenpairs(nep,NEP_TARGET_MAGNITUDE);CHKERRQ(ierr);
      ierr = NEPSetTarget(nep,s);CHKERRQ(ierr);
    }

    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to NEPSetFromOptions()
    */
    ierr = PetscOptionsBool("-nep_monitor_cancel","Remove any hardwired monitor routines","NEPMonitorCancel",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
    if (set && flg) {
      ierr = NEPMonitorCancel(nep);CHKERRQ(ierr);
    }
    /*
      Text monitors
    */
    ierr = NEPMonitorSetFromOptions(nep,"-nep_monitor","Monitor first unconverged approximate eigenvalue and error estimate","NEPMonitorFirst",NEPMonitorFirst,PETSC_FALSE);CHKERRQ(ierr);
    ierr = NEPConvMonitorSetFromOptions(nep,"-nep_monitor_conv","Monitor approximate eigenvalues and error estimates as they converge","NEPMonitorConverged",NEPMonitorConverged);CHKERRQ(ierr);
    ierr = NEPMonitorSetFromOptions(nep,"-nep_monitor_all","Monitor approximate eigenvalues and error estimates","NEPMonitorAll",NEPMonitorAll,PETSC_TRUE);CHKERRQ(ierr);
    /*
      Line graph monitors
    */
    ierr = PetscOptionsBool("-nep_monitor_lg","Monitor first unconverged approximate error estimate graphically","NEPMonitorSet",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
    if (set && flg) {
      ierr = NEPMonitorLGCreate(PetscObjectComm((PetscObject)nep),NULL,"Error estimates",PETSC_DECIDE,PETSC_DECIDE,300,300,&lg);CHKERRQ(ierr);
      ierr = NEPMonitorSet(nep,NEPMonitorLG,lg,(PetscErrorCode (*)(void**))PetscDrawLGDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsBool("-nep_monitor_lg_all","Monitor error estimates graphically","NEPMonitorSet",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
    if (set && flg) {
      ierr = NEPMonitorLGCreate(PetscObjectComm((PetscObject)nep),NULL,"Error estimates",PETSC_DECIDE,PETSC_DECIDE,300,300,&lg);CHKERRQ(ierr);
      ierr = NEPMonitorSet(nep,NEPMonitorLGAll,lg,(PetscErrorCode (*)(void**))PetscDrawLGDestroy);CHKERRQ(ierr);
      ierr = NEPSetTrackAll(nep,PETSC_TRUE);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/

    ierr = PetscOptionsBoolGroupBegin("-nep_largest_magnitude","compute largest eigenvalues in magnitude","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_LARGEST_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_smallest_magnitude","compute smallest eigenvalues in magnitude","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_SMALLEST_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_largest_real","compute largest real parts","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_LARGEST_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_smallest_real","compute smallest real parts","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_SMALLEST_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_largest_imaginary","compute largest imaginary parts","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_LARGEST_IMAGINARY);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_smallest_imaginary","compute smallest imaginary parts","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_SMALLEST_IMAGINARY);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_target_magnitude","compute nearest eigenvalues to target","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_TARGET_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_target_real","compute eigenvalues with real parts close to target","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_TARGET_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-nep_target_imaginary","compute eigenvalues with imaginary parts close to target","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_TARGET_IMAGINARY);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-nep_all","compute all eigenvalues in a region","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_ALL);CHKERRQ(ierr); }

    ierr = PetscOptionsName("-nep_view","Print detailed information on solver used","NEPView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-nep_view_vectors","View computed eigenvectors","NEPVectorsView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-nep_view_values","View computed eigenvalues","NEPValuesView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-nep_converged_reason","Print reason for convergence, and number of iterations","NEPReasonView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-nep_error_absolute","Print absolute errors of each eigenpair","NEPErrorView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-nep_error_relative","Print relative errors of each eigenpair","NEPErrorView",NULL);CHKERRQ(ierr);

    if (nep->ops->setfromoptions) {
      ierr = (*nep->ops->setfromoptions)(PetscOptionsObject,nep);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)nep);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!nep->V) { ierr = NEPGetBV(nep,&nep->V);CHKERRQ(ierr); }
  ierr = BVSetFromOptions(nep->V);CHKERRQ(ierr);
  if (!nep->rg) { ierr = NEPGetRG(nep,&nep->rg);CHKERRQ(ierr); }
  ierr = RGSetFromOptions(nep->rg);CHKERRQ(ierr);
  if (!nep->ds) { ierr = NEPGetDS(nep,&nep->ds);CHKERRQ(ierr); }
  ierr = DSSetFromOptions(nep->ds);CHKERRQ(ierr);
  if (!nep->refineksp) { ierr = NEPRefineGetKSP(nep,&nep->refineksp);CHKERRQ(ierr); }
  ierr = KSPSetFromOptions(nep->refineksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetTolerances"
/*@
   NEPGetTolerances - Gets the tolerance and maximum iteration count used
   by the NEP convergence tests.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: NEPSetTolerances()
@*/
PetscErrorCode NEPGetTolerances(NEP nep,PetscReal *tol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (tol)    *tol    = nep->tol;
  if (maxits) *maxits = nep->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetTolerances"
/*@
   NEPSetTolerances - Sets the tolerance and maximum iteration count used
   by the NEP convergence tests.

   Logically Collective on NEP

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  tol    - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -nep_tol <tol> - Sets the convergence tolerance
-  -nep_max_it <maxits> - Sets the maximum number of iterations allowed

   Notes:
   Use PETSC_DEFAULT for either argument to assign a reasonably good value.

   Level: intermediate

.seealso: NEPGetTolerances()
@*/
PetscErrorCode NEPSetTolerances(NEP nep,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(nep,tol,2);
  PetscValidLogicalCollectiveInt(nep,maxits,3);
  if (tol == PETSC_DEFAULT) {
    nep->tol   = PETSC_DEFAULT;
    nep->state = NEP_STATE_INITIAL;
  } else {
    if (tol <= 0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
    nep->tol = tol;
  }
  if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
    nep->max_it = 0;
    nep->state  = NEP_STATE_INITIAL;
  } else {
    if (maxits <= 0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
    nep->max_it = maxits;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetDimensions"
/*@
   NEPGetDimensions - Gets the number of eigenvalues to compute
   and the dimension of the subspace.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: NEPSetDimensions()
@*/
PetscErrorCode NEPGetDimensions(NEP nep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (nev) *nev = nep->nev;
  if (ncv) *ncv = nep->ncv;
  if (mpd) *mpd = nep->mpd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetDimensions"
/*@
   NEPSetDimensions - Sets the number of eigenvalues to compute
   and the dimension of the subspace.

   Logically Collective on NEP

   Input Parameters:
+  nep - the nonlinear eigensolver context
.  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Options Database Keys:
+  -nep_nev <nev> - Sets the number of eigenvalues
.  -nep_ncv <ncv> - Sets the dimension of the subspace
-  -nep_mpd <mpd> - Sets the maximum projected dimension

   Notes:
   Use PETSC_DEFAULT for ncv and mpd to assign a reasonably good value, which is
   dependent on the solution method.

   The parameters ncv and mpd are intimately related, so that the user is advised
   to set one of them at most. Normal usage is that
   (a) in cases where nev is small, the user sets ncv (a reasonable default is 2*nev); and
   (b) in cases where nev is large, the user sets mpd.

   The value of ncv should always be between nev and (nev+mpd), typically
   ncv=nev+mpd. If nev is not too large, mpd=nev is a reasonable choice, otherwise
   a smaller value should be used.

   Level: intermediate

.seealso: NEPGetDimensions()
@*/
PetscErrorCode NEPSetDimensions(NEP nep,PetscInt nev,PetscInt ncv,PetscInt mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,nev,2);
  PetscValidLogicalCollectiveInt(nep,ncv,3);
  PetscValidLogicalCollectiveInt(nep,mpd,4);
  if (nev<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nev. Must be > 0");
  nep->nev = nev;
  if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
    nep->ncv = 0;
  } else {
    if (ncv<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
    nep->ncv = ncv;
  }
  if (mpd == PETSC_DECIDE || mpd == PETSC_DEFAULT) {
    nep->mpd = 0;
  } else {
    if (mpd<1) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of mpd. Must be > 0");
    nep->mpd = mpd;
  }
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetWhichEigenpairs"
/*@
    NEPSetWhichEigenpairs - Specifies which portion of the spectrum is
    to be sought.

    Logically Collective on NEP

    Input Parameters:
+   nep   - eigensolver context obtained from NEPCreate()
-   which - the portion of the spectrum to be sought

    Possible values:
    The parameter 'which' can have one of these values

+     NEP_LARGEST_MAGNITUDE - largest eigenvalues in magnitude (default)
.     NEP_SMALLEST_MAGNITUDE - smallest eigenvalues in magnitude
.     NEP_LARGEST_REAL - largest real parts
.     NEP_SMALLEST_REAL - smallest real parts
.     NEP_LARGEST_IMAGINARY - largest imaginary parts
.     NEP_SMALLEST_IMAGINARY - smallest imaginary parts
.     NEP_TARGET_MAGNITUDE - eigenvalues closest to the target (in magnitude)
.     NEP_TARGET_REAL - eigenvalues with real part closest to target
.     NEP_TARGET_IMAGINARY - eigenvalues with imaginary part closest to target
.     NEP_ALL - all eigenvalues contained in a given region
-     NEP_WHICH_USER - user defined ordering set with NEPSetEigenvalueComparison()

    Options Database Keys:
+   -nep_largest_magnitude - Sets largest eigenvalues in magnitude
.   -nep_smallest_magnitude - Sets smallest eigenvalues in magnitude
.   -nep_largest_real - Sets largest real parts
.   -nep_smallest_real - Sets smallest real parts
.   -nep_largest_imaginary - Sets largest imaginary parts
.   -nep_smallest_imaginary - Sets smallest imaginary parts
.   -nep_target_magnitude - Sets eigenvalues closest to target
.   -nep_target_real - Sets real parts closest to target
.   -nep_target_imaginary - Sets imaginary parts closest to target
-   -nep_all - Sets all eigenvalues in a region

    Notes:
    Not all eigensolvers implemented in NEP account for all the possible values
    stated above. If SLEPc is compiled for real numbers NEP_LARGEST_IMAGINARY
    and NEP_SMALLEST_IMAGINARY use the absolute value of the imaginary part
    for eigenvalue selection.

    The target is a scalar value provided with NEPSetTarget().

    NEP_ALL is intended for use in the context of the CISS solver for
    computing all eigenvalues in a region.

    Level: intermediate

.seealso: NEPGetWhichEigenpairs(), NEPSetTarget(), NEPSetEigenvalueComparison(), NEPWhich
@*/
PetscErrorCode NEPSetWhichEigenpairs(NEP nep,NEPWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(nep,which,2);
  switch (which) {
    case NEP_LARGEST_MAGNITUDE:
    case NEP_SMALLEST_MAGNITUDE:
    case NEP_LARGEST_REAL:
    case NEP_SMALLEST_REAL:
    case NEP_LARGEST_IMAGINARY:
    case NEP_SMALLEST_IMAGINARY:
    case NEP_TARGET_MAGNITUDE:
    case NEP_TARGET_REAL:
#if defined(PETSC_USE_COMPLEX)
    case NEP_TARGET_IMAGINARY:
#endif
    case EPS_ALL:
    case NEP_WHICH_USER:
      if (nep->which != which) {
        nep->state = NEP_STATE_INITIAL;
        nep->which = which;
      }
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'which' value");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetWhichEigenpairs"
/*@
    NEPGetWhichEigenpairs - Returns which portion of the spectrum is to be
    sought.

    Not Collective

    Input Parameter:
.   nep - eigensolver context obtained from NEPCreate()

    Output Parameter:
.   which - the portion of the spectrum to be sought

    Notes:
    See NEPSetWhichEigenpairs() for possible values of 'which'.

    Level: intermediate

.seealso: NEPSetWhichEigenpairs(), NEPWhich
@*/
PetscErrorCode NEPGetWhichEigenpairs(NEP nep,NEPWhich *which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(which,2);
  *which = nep->which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetEigenvalueComparison"
/*@C
   NEPSetEigenvalueComparison - Specifies the eigenvalue comparison function
   when NEPSetWhichEigenpairs() is set to NEP_WHICH_USER.

   Logically Collective on NEP

   Input Parameters:
+  pep  - eigensolver context obtained from NEPCreate()
.  func - a pointer to the comparison function
-  ctx  - a context pointer (the last parameter to the comparison function)

   Calling Sequence of func:
$   func(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *res,void *ctx)

+   ar     - real part of the 1st eigenvalue
.   ai     - imaginary part of the 1st eigenvalue
.   br     - real part of the 2nd eigenvalue
.   bi     - imaginary part of the 2nd eigenvalue
.   res    - result of comparison
-   ctx    - optional context, as set by NEPSetEigenvalueComparison()

   Note:
   The returning parameter 'res' can be
+  negative - if the 1st eigenvalue is preferred to the 2st one
.  zero     - if both eigenvalues are equally preferred
-  positive - if the 2st eigenvalue is preferred to the 1st one

   Level: advanced

.seealso: NEPSetWhichEigenpairs(), NEPWhich
@*/
PetscErrorCode NEPSetEigenvalueComparison(NEP pep,PetscErrorCode (*func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void* ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,NEP_CLASSID,1);
  pep->sc->comparison    = func;
  pep->sc->comparisonctx = ctx;
  pep->which             = NEP_WHICH_USER;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetConvergenceTestFunction"
/*@C
   NEPSetConvergenceTestFunction - Sets a function to compute the error estimate
   used in the convergence test.

   Logically Collective on NEP

   Input Parameters:
+  nep     - nonlinear eigensolver context obtained from NEPCreate()
.  func    - a pointer to the convergence test function
.  ctx     - context for private data for the convergence routine (may be null)
-  destroy - a routine for destroying the context (may be null)

   Calling Sequence of func:
$   func(NEP nep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)

+   nep    - nonlinear eigensolver context obtained from NEPCreate()
.   eigr   - real part of the eigenvalue
.   eigi   - imaginary part of the eigenvalue
.   res    - residual norm associated to the eigenpair
.   errest - (output) computed error estimate
-   ctx    - optional context, as set by NEPSetConvergenceTestFunction()

   Note:
   If the error estimate returned by the convergence test function is less than
   the tolerance, then the eigenvalue is accepted as converged.

   Level: advanced

.seealso: NEPSetConvergenceTest(), NEPSetTolerances()
@*/
PetscErrorCode NEPSetConvergenceTestFunction(NEP nep,PetscErrorCode (*func)(NEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*),void* ctx,PetscErrorCode (*destroy)(void*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (nep->convergeddestroy) {
    ierr = (*nep->convergeddestroy)(nep->convergedctx);CHKERRQ(ierr);
  }
  nep->converged        = func;
  nep->convergeddestroy = destroy;
  nep->convergedctx     = ctx;
  if (func == NEPConvergedRelative) nep->conv = NEP_CONV_REL;
  else if (func == NEPConvergedNorm) nep->conv = NEP_CONV_NORM;
  else if (func == NEPConvergedAbsolute) nep->conv = NEP_CONV_ABS;
  else nep->conv = NEP_CONV_USER;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetConvergenceTest"
/*@
   NEPSetConvergenceTest - Specifies how to compute the error estimate
   used in the convergence test.

   Logically Collective on NEP

   Input Parameters:
+  nep  - nonlinear eigensolver context obtained from NEPCreate()
-  conv - the type of convergence test

   Options Database Keys:
+  -nep_conv_abs  - Sets the absolute convergence test
.  -nep_conv_rel  - Sets the convergence test relative to the eigenvalue
-  -nep_conv_user - Selects the user-defined convergence test

   Note:
   The parameter 'conv' can have one of these values
+     NEP_CONV_ABS  - absolute error ||r||
.     NEP_CONV_REL  - error relative to the eigenvalue l, ||r||/|l|
.     NEP_CONV_NORM - error relative matrix norms, ||r||/sum_i(|f_i(l)|*||A_i||)
-     NEP_CONV_USER - function set by NEPSetConvergenceTestFunction()

   Level: intermediate

.seealso: NEPGetConvergenceTest(), NEPSetConvergenceTestFunction(), NEPSetStoppingTest(), NEPConv
@*/
PetscErrorCode NEPSetConvergenceTest(NEP nep,NEPConv conv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(nep,conv,2);
  switch (conv) {
    case NEP_CONV_ABS:  nep->converged = NEPConvergedAbsolute; break;
    case NEP_CONV_REL:  nep->converged = NEPConvergedRelative; break;
    case NEP_CONV_NORM: nep->converged = NEPConvergedNorm; break;
    case NEP_CONV_USER: break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'conv' value");
  }
  nep->conv = conv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetConvergenceTest"
/*@
   NEPGetConvergenceTest - Gets the method used to compute the error estimate
   used in the convergence test.

   Not Collective

   Input Parameters:
.  nep   - nonlinear eigensolver context obtained from NEPCreate()

   Output Parameters:
.  conv  - the type of convergence test

   Level: intermediate

.seealso: NEPSetConvergenceTest(), NEPConv
@*/
PetscErrorCode NEPGetConvergenceTest(NEP nep,NEPConv *conv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(conv,2);
  *conv = nep->conv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetStoppingTestFunction"
/*@C
   NEPSetStoppingTestFunction - Sets a function to decide when to stop the outer
   iteration of the eigensolver.

   Logically Collective on NEP

   Input Parameters:
+  nep     - nonlinear eigensolver context obtained from NEPCreate()
.  func    - pointer to the stopping test function
.  ctx     - context for private data for the stopping routine (may be null)
-  destroy - a routine for destroying the context (may be null)

   Calling Sequence of func:
$   func(NEP nep,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,NEPConvergedReason *reason,void *ctx)

+   nep    - nonlinear eigensolver context obtained from NEPCreate()
.   its    - current number of iterations
.   max_it - maximum number of iterations
.   nconv  - number of currently converged eigenpairs
.   nev    - number of requested eigenpairs
.   reason - (output) result of the stopping test
-   ctx    - optional context, as set by NEPSetStoppingTestFunction()

   Note:
   Normal usage is to first call the default routine NEPStoppingBasic() and then
   set reason to NEP_CONVERGED_USER if some user-defined conditions have been
   met. To let the eigensolver continue iterating, the result must be left as
   NEP_CONVERGED_ITERATING.

   Level: advanced

.seealso: NEPSetStoppingTest(), NEPStoppingBasic()
@*/
PetscErrorCode NEPSetStoppingTestFunction(NEP nep,PetscErrorCode (*func)(NEP,PetscInt,PetscInt,PetscInt,PetscInt,NEPConvergedReason*,void*),void* ctx,PetscErrorCode (*destroy)(void*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (nep->stoppingdestroy) {
    ierr = (*nep->stoppingdestroy)(nep->stoppingctx);CHKERRQ(ierr);
  }
  nep->stopping        = func;
  nep->stoppingdestroy = destroy;
  nep->stoppingctx     = ctx;
  if (func == NEPStoppingBasic) nep->stop = NEP_STOP_BASIC;
  else nep->stop = NEP_STOP_USER;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetStoppingTest"
/*@
   NEPSetStoppingTest - Specifies how to decide the termination of the outer
   loop of the eigensolver.

   Logically Collective on NEP

   Input Parameters:
+  nep  - nonlinear eigensolver context obtained from NEPCreate()
-  stop - the type of stopping test

   Options Database Keys:
+  -nep_stop_basic - Sets the default stopping test
-  -nep_stop_user  - Selects the user-defined stopping test

   Note:
   The parameter 'stop' can have one of these values
+     NEP_STOP_BASIC - default stopping test
-     NEP_STOP_USER  - function set by NEPSetStoppingTestFunction()

   Level: advanced

.seealso: NEPGetStoppingTest(), NEPSetStoppingTestFunction(), NEPSetConvergenceTest(), NEPStop
@*/
PetscErrorCode NEPSetStoppingTest(NEP nep,NEPStop stop)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(nep,stop,2);
  switch (stop) {
    case NEP_STOP_BASIC: nep->stopping = NEPStoppingBasic; break;
    case NEP_STOP_USER:  break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'stop' value");
  }
  nep->stop = stop;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetStoppingTest"
/*@
   NEPGetStoppingTest - Gets the method used to decide the termination of the outer
   loop of the eigensolver.

   Not Collective

   Input Parameters:
.  nep   - nonlinear eigensolver context obtained from NEPCreate()

   Output Parameters:
.  stop  - the type of stopping test

   Level: advanced

.seealso: NEPSetStoppingTest(), NEPStop
@*/
PetscErrorCode NEPGetStoppingTest(NEP nep,NEPStop *stop)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(stop,2);
  *stop = nep->stop;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetTrackAll"
/*@
   NEPSetTrackAll - Specifies if the solver must compute the residual of all
   approximate eigenpairs or not.

   Logically Collective on NEP

   Input Parameters:
+  nep      - the eigensolver context
-  trackall - whether compute all residuals or not

   Notes:
   If the user sets trackall=PETSC_TRUE then the solver explicitly computes
   the residual for each eigenpair approximation. Computing the residual is
   usually an expensive operation and solvers commonly compute the associated
   residual to the first unconverged eigenpair.
   The options '-nep_monitor_all' and '-nep_monitor_lg_all' automatically
   activate this option.

   Level: developer

.seealso: NEPGetTrackAll()
@*/
PetscErrorCode NEPSetTrackAll(NEP nep,PetscBool trackall)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(nep,trackall,2);
  nep->trackall = trackall;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetTrackAll"
/*@
   NEPGetTrackAll - Returns the flag indicating whether all residual norms must
   be computed or not.

   Not Collective

   Input Parameter:
.  nep - the eigensolver context

   Output Parameter:
.  trackall - the returned flag

   Level: developer

.seealso: NEPSetTrackAll()
@*/
PetscErrorCode NEPGetTrackAll(NEP nep,PetscBool *trackall)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(trackall,2);
  *trackall = nep->trackall;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetRefine"
/*@
   NEPSetRefine - Specifies the refinement type (and options) to be used
   after the solve.

   Logically Collective on NEP

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  refine - refinement type
.  npart  - number of partitions of the communicator
.  tol    - the convergence tolerance
.  its    - maximum number of refinement iterations
-  scheme - which scheme to be used for solving the involved linear systems

   Options Database Keys:
+  -nep_refine <type> - refinement type, one of <none,simple,multiple>
.  -nep_refine_partitions <n> - the number of partitions
.  -nep_refine_tol <tol> - the tolerance
.  -nep_refine_its <its> - number of iterations
-  -nep_refine_scheme - to set the scheme for the linear solves

   Notes:
   By default, iterative refinement is disabled, since it may be very
   costly. There are two possible refinement strategies: simple and multiple.
   The simple approach performs iterative refinement on each of the
   converged eigenpairs individually, whereas the multiple strategy works
   with the invariant pair as a whole, refining all eigenpairs simultaneously.
   The latter may be required for the case of multiple eigenvalues.

   In some cases, especially when using direct solvers within the
   iterative refinement method, it may be helpful for improved scalability
   to split the communicator in several partitions. The npart parameter
   indicates how many partitions to use (defaults to 1).

   The tol and its parameters specify the stopping criterion. In the simple
   method, refinement continues until the residual of each eigenpair is
   below the tolerance (tol defaults to the NEP tol, but may be set to a
   different value). In contrast, the multiple method simply performs its
   refinement iterations (just one by default).

   The scheme argument is used to change the way in which linear systems are
   solved. Possible choices are: explicit, mixed block elimination (MBE), 
   and Schur complement.

   Level: intermediate

.seealso: NEPGetRefine()
@*/
PetscErrorCode NEPSetRefine(NEP nep,NEPRefine refine,PetscInt npart,PetscReal tol,PetscInt its,NEPRefineScheme scheme)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(nep,refine,2);
  PetscValidLogicalCollectiveInt(nep,npart,3);
  PetscValidLogicalCollectiveReal(nep,tol,4);
  PetscValidLogicalCollectiveInt(nep,its,5);
  PetscValidLogicalCollectiveEnum(nep,scheme,6);
  nep->refine = refine;
  if (refine) {  /* process parameters only if not REFINE_NONE */
    if (npart == PETSC_DEFAULT || npart == PETSC_DECIDE) {
      nep->npart = 1;
    } else {
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)nep),&size);CHKERRQ(ierr);
      if (npart<1 || npart>size) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of npart");
      nep->npart = npart;
    }
    if (tol == PETSC_DEFAULT || tol == PETSC_DECIDE) {
      nep->rtol = PetscMax(nep->tol/1000,PETSC_MACHINE_EPSILON);
    } else {
      if (tol<=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
      nep->rtol = tol;
    }
    if (its==PETSC_DECIDE || its==PETSC_DEFAULT) {
      nep->rits = PETSC_DEFAULT;
    } else {
      if (its<0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of its. Must be >= 0");
      nep->rits = its;
    }
    nep->scheme = scheme;
  }
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetRefine"
/*@
   NEPGetRefine - Gets the refinement strategy used by the NEP object, and the
   associated parameters.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  refine - refinement type
.  npart  - number of partitions of the communicator
.  tol    - the convergence tolerance
-  its    - maximum number of refinement iterations
-  scheme - the scheme used for solving linear systems

   Level: intermediate

   Note:
   The user can specify NULL for any parameter that is not needed.

.seealso: NEPSetRefine()
@*/
PetscErrorCode NEPGetRefine(NEP nep,NEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,NEPRefineScheme *scheme)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (refine) *refine = nep->refine;
  if (npart)  *npart  = nep->npart;
  if (tol)    *tol    = nep->rtol;
  if (its)    *its    = nep->rits;
  if (scheme) *scheme = nep->scheme;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetOptionsPrefix"
/*@C
   NEPSetOptionsPrefix - Sets the prefix used for searching for all
   NEP options in the database.

   Logically Collective on NEP

   Input Parameters:
+  nep - the nonlinear eigensolver context
-  prefix - the prefix string to prepend to all NEP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different NEP contexts, one could call
.vb
      NEPSetOptionsPrefix(nep1,"neig1_")
      NEPSetOptionsPrefix(nep2,"neig2_")
.ve

   Level: advanced

.seealso: NEPAppendOptionsPrefix(), NEPGetOptionsPrefix()
@*/
PetscErrorCode NEPSetOptionsPrefix(NEP nep,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!nep->V) { ierr = NEPGetBV(nep,&nep->V);CHKERRQ(ierr); }
  ierr = BVSetOptionsPrefix(nep->V,prefix);CHKERRQ(ierr);
  if (!nep->ds) { ierr = NEPGetDS(nep,&nep->ds);CHKERRQ(ierr); }
  ierr = DSSetOptionsPrefix(nep->ds,prefix);CHKERRQ(ierr);
  if (!nep->rg) { ierr = NEPGetRG(nep,&nep->rg);CHKERRQ(ierr); }
  ierr = RGSetOptionsPrefix(nep->rg,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)nep,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPAppendOptionsPrefix"
/*@C
   NEPAppendOptionsPrefix - Appends to the prefix used for searching for all
   NEP options in the database.

   Logically Collective on NEP

   Input Parameters:
+  nep - the nonlinear eigensolver context
-  prefix - the prefix string to prepend to all NEP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: NEPSetOptionsPrefix(), NEPGetOptionsPrefix()
@*/
PetscErrorCode NEPAppendOptionsPrefix(NEP nep,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!nep->V) { ierr = NEPGetBV(nep,&nep->V);CHKERRQ(ierr); }
  ierr = BVSetOptionsPrefix(nep->V,prefix);CHKERRQ(ierr);
  if (!nep->ds) { ierr = NEPGetDS(nep,&nep->ds);CHKERRQ(ierr); }
  ierr = DSSetOptionsPrefix(nep->ds,prefix);CHKERRQ(ierr);
  if (!nep->rg) { ierr = NEPGetRG(nep,&nep->rg);CHKERRQ(ierr); }
  ierr = RGSetOptionsPrefix(nep->rg,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)nep,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetOptionsPrefix"
/*@C
   NEPGetOptionsPrefix - Gets the prefix used for searching for all
   NEP options in the database.

   Not Collective

   Input Parameters:
.  nep - the nonlinear eigensolver context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Note:
   On the Fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: NEPSetOptionsPrefix(), NEPAppendOptionsPrefix()
@*/
PetscErrorCode NEPGetOptionsPrefix(NEP nep,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)nep,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

