/*
      NEP routines related to options that can be set via the command-line
      or procedurally.

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

#include <slepc-private/nepimpl.h>       /*I "slepcnep.h" I*/

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
  PetscErrorCode   ierr;
  char             type[256],monfilename[PETSC_MAX_PATH_LEN];
  PetscBool        flg,flg1,flg2,flg3,flg4,flg5;
  PetscReal        r1,r2,r3;
  PetscScalar      s;
  PetscInt         i,j,k;
  PetscViewer      monviewer;
  SlepcConvMonitor ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!NEPRegisterAllCalled) { ierr = NEPRegisterAll();CHKERRQ(ierr); }
  ierr = PetscObjectOptionsBegin((PetscObject)nep);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-nep_type","Nonlinear Eigenvalue Problem method","NEPSetType",NEPList,(char*)(((PetscObject)nep)->type_name?((PetscObject)nep)->type_name:NEPRII),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = NEPSetType(nep,type);CHKERRQ(ierr);
    } else if (!((PetscObject)nep)->type_name) {
      ierr = NEPSetType(nep,NEPRII);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEnum("-nep_refine","Iterative refinement method","NEPSetRefine",NEPRefineTypes,(PetscEnum)nep->refine,(PetscEnum*)&nep->refine,NULL);CHKERRQ(ierr);

    r1 = nep->reftol;
    ierr = PetscOptionsReal("-nep_refine_tol","Tolerance for iterative refinement","NEPSetRefine",nep->reftol,&r1,&flg1);CHKERRQ(ierr);
    j = nep->rits;
    ierr = PetscOptionsInt("-nep_refine_its","Maximum number of iterations for iterative refinement","NEPSetRefine",nep->rits,&j,&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) {
      ierr = NEPSetRefine(nep,nep->refine,r1,j);CHKERRQ(ierr);
    }

    i = nep->max_it? nep->max_it: PETSC_DEFAULT;
    ierr = PetscOptionsInt("-nep_max_it","Maximum number of iterations","NEPSetTolerances",nep->max_it,&i,&flg1);CHKERRQ(ierr);
    j = nep->max_funcs? nep->max_funcs: PETSC_DEFAULT;
    ierr = PetscOptionsInt("-nep_max_funcs","Maximum number of function evaluations","NEPSetTolerances",nep->max_funcs,&j,&flg2);CHKERRQ(ierr);
    r1 = nep->abstol;
    ierr = PetscOptionsReal("-nep_atol","Absolute tolerance for residual norm","NEPSetTolerances",nep->abstol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:nep->abstol,&r1,&flg3);CHKERRQ(ierr);
    r2 = nep->rtol;
    ierr = PetscOptionsReal("-nep_rtol","Relative tolerance for residual norm","NEPSetTolerances",nep->rtol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:nep->rtol,&r2,&flg4);CHKERRQ(ierr);
    r3 = nep->stol;
    ierr = PetscOptionsReal("-nep_stol","Relative tolerance for step length","NEPSetTolerances",nep->stol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:nep->stol,&r3,&flg5);CHKERRQ(ierr);
    if (flg1 || flg2 || flg3 || flg4 || flg5) {
      ierr = NEPSetTolerances(nep,r1,r2,r3,i,j);CHKERRQ(ierr);
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-nep_convergence_default","Default (relative error) convergence test","NEPSetConvergenceTest",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = NEPSetConvergenceTest(nep,NEPConvergedDefault,NULL,NULL);CHKERRQ(ierr);
    }

    i = nep->nev;
    ierr = PetscOptionsInt("-nep_nev","Number of eigenvalues to compute","NEPSetDimensions",nep->nev,&i,&flg1);CHKERRQ(ierr);
    j = nep->ncv? nep->ncv: PETSC_DEFAULT;
    ierr = PetscOptionsInt("-nep_ncv","Number of basis vectors","NEPSetDimensions",nep->ncv,&j,&flg2);CHKERRQ(ierr);
    k = nep->mpd? nep->mpd: PETSC_DEFAULT;
    ierr = PetscOptionsInt("-nep_mpd","Maximum dimension of projected problem","NEPSetDimensions",nep->mpd,&k,&flg3);CHKERRQ(ierr);
    if (flg1 || flg2 || flg3) {
      ierr = NEPSetDimensions(nep,i,j,k);CHKERRQ(ierr);
    }

    i = 0;
    ierr = PetscOptionsInt("-nep_lag_preconditioner","Interval to rebuild preconditioner","NEPSetLagPreconditioner",nep->lag,&i,&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetLagPreconditioner(nep,i);CHKERRQ(ierr); }

    ierr = PetscOptionsBool("-nep_const_correction_tol","Constant correction tolerance for the linear solver","NEPSetConstCorrectionTol",nep->cctol,&nep->cctol,NULL);CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-nep_target","Value of the target","NEPSetTarget",nep->target,&s,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = NEPSetWhichEigenpairs(nep,NEP_TARGET_MAGNITUDE);CHKERRQ(ierr);
      ierr = NEPSetTarget(nep,s);CHKERRQ(ierr);
    }

    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to NEPSetFromOptions()
    */
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-nep_monitor_cancel","Remove any hardwired monitor routines","NEPMonitorCancel",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = NEPMonitorCancel(nep);CHKERRQ(ierr);
    }
    /*
      Prints approximate eigenvalues and error estimates at each iteration
    */
    ierr = PetscOptionsString("-nep_monitor","Monitor first unconverged approximate eigenvalue and error estimate","NEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)nep),monfilename,&monviewer);CHKERRQ(ierr);
      ierr = NEPMonitorSet(nep,NEPMonitorFirst,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-nep_monitor_conv","Monitor approximate eigenvalues and error estimates as they converge","NEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscNew(&ctx);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)nep),monfilename,&ctx->viewer);CHKERRQ(ierr);
      ierr = NEPMonitorSet(nep,NEPMonitorConverged,ctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-nep_monitor_all","Monitor approximate eigenvalues and error estimates","NEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)nep),monfilename,&monviewer);CHKERRQ(ierr);
      ierr = NEPMonitorSet(nep,NEPMonitorAll,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
      ierr = NEPSetTrackAll(nep,PETSC_TRUE);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-nep_monitor_lg","Monitor first unconverged approximate error estimate graphically","NEPMonitorSet",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = NEPMonitorSet(nep,NEPMonitorLG,NULL,NULL);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-nep_monitor_lg_all","Monitor error estimates graphically","NEPMonitorSet",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = NEPMonitorSet(nep,NEPMonitorLGAll,NULL,NULL);CHKERRQ(ierr);
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
    ierr = PetscOptionsBoolGroupEnd("-nep_target_imaginary","compute eigenvalues with imaginary parts close to target","NEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = NEPSetWhichEigenpairs(nep,NEP_TARGET_IMAGINARY);CHKERRQ(ierr); }

    ierr = PetscOptionsName("-nep_view","Print detailed information on solver used","NEPView",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-nep_plot_eigs","Make a plot of the computed eigenvalues","NEPSolve",0);CHKERRQ(ierr);

    if (nep->ops->setfromoptions) {
      ierr = (*nep->ops->setfromoptions)(nep);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)nep);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!nep->V) { ierr = NEPGetBV(nep,&nep->V);CHKERRQ(ierr); }
  ierr = BVSetFromOptions(nep->V);CHKERRQ(ierr);
  if (!nep->rg) { ierr = NEPGetRG(nep,&nep->rg);CHKERRQ(ierr); }
  ierr = RGSetFromOptions(nep->rg);CHKERRQ(ierr);
  if (!nep->ds) { ierr = NEPGetDS(nep,&nep->ds);CHKERRQ(ierr); }
  ierr = DSSetFromOptions(nep->ds);CHKERRQ(ierr);
  if (!nep->ksp) { ierr = NEPGetKSP(nep,&nep->ksp);CHKERRQ(ierr); }
  ierr = KSPSetOperators(nep->ksp,nep->function,nep->function_pre);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(nep->ksp);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(nep->rand);CHKERRQ(ierr);
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
+  abstol - absolute convergence tolerance
.  rtol   - relative convergence tolerance
.  stol   - convergence tolerance in terms of the norm of the change in the
           solution between steps, || delta x || < stol*|| x ||
.  maxit  - maximum number of iterations
-  maxf   - maximum number of function evaluations

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: NEPSetTolerances()
@*/
PetscErrorCode NEPGetTolerances(NEP nep,PetscReal *abstol,PetscReal *rtol,PetscReal *stol,PetscInt *maxit,PetscInt *maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (abstol) *abstol = nep->abstol;
  if (rtol)   *rtol   = nep->rtol;
  if (stol)   *stol   = nep->stol;
  if (maxit)  *maxit  = nep->max_it;
  if (maxf)   *maxf   = nep->max_funcs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetTolerances"
/*@
   NEPSetTolerances - Sets various parameters used in convergence tests.

   Logically Collective on NEP

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  abstol - absolute convergence tolerance
.  rtol   - relative convergence tolerance
.  stol   - convergence tolerance in terms of the norm of the change in the
            solution between steps, || delta x || < stol*|| x ||
.  maxit  - maximum number of iterations
-  maxf   - maximum number of function evaluations

   Options Database Keys:
+    -nep_atol <abstol> - Sets abstol
.    -nep_rtol <rtol> - Sets rtol
.    -nep_stol <stol> - Sets stol
.    -nep_max_it <maxit> - Sets maxit
-    -nep_max_funcs <maxf> - Sets maxf

   Notes:
   Use PETSC_DEFAULT for either argument to assign a reasonably good value.

   Level: intermediate

.seealso: NEPGetTolerances()
@*/
PetscErrorCode NEPSetTolerances(NEP nep,PetscReal abstol,PetscReal rtol,PetscReal stol,PetscInt maxit,PetscInt maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(nep,abstol,2);
  PetscValidLogicalCollectiveReal(nep,rtol,3);
  PetscValidLogicalCollectiveReal(nep,stol,4);
  PetscValidLogicalCollectiveInt(nep,maxit,5);
  PetscValidLogicalCollectiveInt(nep,maxf,6);
  if (abstol == PETSC_DEFAULT) {
    nep->abstol = PETSC_DEFAULT;
  } else {
    if (abstol < 0.0) SETERRQ1(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %g must be non-negative",(double)abstol);
    nep->abstol = abstol;
  }
  if (rtol == PETSC_DEFAULT) {
    nep->rtol = PETSC_DEFAULT;
  } else {
    if (rtol < 0.0 || 1.0 <= rtol) SETERRQ1(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %g must be non-negative and less than 1.0",(double)rtol);
    nep->rtol = rtol;
  }
  if (stol == PETSC_DEFAULT) {
    nep->stol = PETSC_DEFAULT;
  } else {
    if (stol < 0.0) SETERRQ1(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Step tolerance %g must be non-negative",(double)stol);
    nep->stol = stol;
  }
  if (maxit == PETSC_DEFAULT || maxit == PETSC_DECIDE) {
    nep->max_it = 0;
    nep->state = NEP_STATE_INITIAL;
  } else {
    if (maxit < 0) SETERRQ1(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %D must be non-negative",maxit);
    nep->max_it = maxit;
  }
  if (maxf == PETSC_DEFAULT || maxf == PETSC_DECIDE) {
    nep->max_it = 0;
    nep->state = NEP_STATE_INITIAL;
  } else {
    if (maxf < 0) SETERRQ1(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of function evaluations %D must be non-negative",maxf);
    nep->max_funcs = maxf;
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
-     NEP_TARGET_IMAGINARY - eigenvalues with imaginary part closest to target

    Options Database Keys:
+   -nep_largest_magnitude - Sets largest eigenvalues in magnitude
.   -nep_smallest_magnitude - Sets smallest eigenvalues in magnitude
.   -nep_largest_real - Sets largest real parts
.   -nep_smallest_real - Sets smallest real parts
.   -nep_largest_imaginary - Sets largest imaginary parts
.   -nep_smallest_imaginary - Sets smallest imaginary parts
.   -nep_target_magnitude - Sets eigenvalues closest to target
.   -nep_target_real - Sets real parts closest to target
-   -nep_target_imaginary - Sets imaginary parts closest to target

    Notes:
    Not all eigensolvers implemented in NEP account for all the possible values
    stated above. If SLEPc is compiled for real numbers NEP_LARGEST_IMAGINARY
    and NEP_SMALLEST_IMAGINARY use the absolute value of the imaginary part
    for eigenvalue selection.

    Level: intermediate

.seealso: NEPGetWhichEigenpairs(), NEPWhich
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
/*@C
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
#define __FUNCT__ "NEPSetLagPreconditioner"
/*@
    NEPSetLagPreconditioner - Determines when the preconditioner is rebuilt in the
    nonlinear solve.

    Logically Collective on NEP

    Input Parameters:
+   nep - the NEP context
-   lag - 0 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is
          computed within the nonlinear iteration, 2 means every second time
          the Jacobian is built, etc.

    Options Database Keys:
.   -nep_lag_preconditioner <lag>

    Notes:
    The default is 1.
    The preconditioner is ALWAYS built in the first iteration of a nonlinear solve.

    Level: intermediate

.seealso: NEPGetLagPreconditioner()
@*/
PetscErrorCode NEPSetLagPreconditioner(NEP nep,PetscInt lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,lag,2);
  if (lag<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lag must be non-negative");
  nep->lag = lag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetLagPreconditioner"
/*@
    NEPGetLagPreconditioner - Indicates how often the preconditioner is rebuilt.

    Not Collective

    Input Parameter:
.   nep - the NEP context

    Output Parameter:
.   lag - the lag parameter

    Level: intermediate

.seealso: NEPSetLagPreconditioner()
@*/
PetscErrorCode NEPGetLagPreconditioner(NEP nep,PetscInt *lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(lag,2);
  *lag = nep->lag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetConstCorrectionTol"
/*@
    NEPSetConstCorrectionTol - Sets a flag to keep the tolerance used
    in the linear solver constant.

    Logically Collective on NEP

    Input Parameters:
+   nep - the NEP context
-   cct - a boolean value

    Options Database Keys:
.   -nep_const_correction_tol <cct>

    Notes:
    By default, an exponentially decreasing tolerance is set in the KSP used
    within the nonlinear iteration, so that each Newton iteration requests
    better accuracy than the previous one. The constant correction tolerance
    flag stops this behaviour.

    Level: intermediate

.seealso: NEPGetConstCorrectionTol()
@*/
PetscErrorCode NEPSetConstCorrectionTol(NEP nep,PetscBool cct)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(nep,cct,2);
  nep->cctol = cct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetConstCorrectionTol"
/*@
    NEPGetConstCorrectionTol - Returns the constant tolerance flag.

    Not Collective

    Input Parameter:
.   nep - the NEP context

    Output Parameter:
.   cct - the value of the constant tolerance flag

    Level: intermediate

.seealso: NEPSetConstCorrectionTol()
@*/
PetscErrorCode NEPGetConstCorrectionTol(NEP nep,PetscBool *cct)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(cct,2);
  *cct = nep->cctol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetConvergenceTest"
/*@C
    NEPSetConvergenceTest - Sets the function to be used to test convergence
    of the nonlinear iterative solution.

    Logically Collective on NEP

    Input Parameters:
+   nep     - the NEP context
.   func    - a pointer to the convergence test function
.   ctx     - [optional] context for private data for the convergence routine
              (may be NULL)
-   destroy - [optional] destructor for the context (may be NULL;
              PETSC_NULL_FUNCTION in Fortran)

    Calling Sequence of func:
$   func(NEP nep,PetscInt it,PetscReal xnorm,PetscReal snorm,PetscReal fnorm,NEPConvergedReason reason*,void *fctx)

+   nep    - the NEP context
.   it     - iteration number
.   xnorm  - norm of the current solution
.   snorm  - norm of the step (difference between two consecutive solutions)
.   fnorm  - norm of the function (residual)
.   reason - (output) result of the convergence test
-   fctx   - optional context, as set by NEPSetConvergenceTest()

    Level: advanced

.seealso: NEPSetTolerances()
@*/
PetscErrorCode NEPSetConvergenceTest(NEP nep,PetscErrorCode (*func)(NEP,PetscInt,PetscReal,PetscReal,PetscReal,NEPConvergedReason*,void*),void* ctx,PetscErrorCode (*destroy)(void*))
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

   Level: intermediate

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

   Level: intermediate

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
.  tol    - the convergence tolerance
-  its    - maximum number of refinement iterations

   Options Database Keys:
+  -nep_refine <type> - refinement type, one of <none,simple,multiple>
.  -nep_refine_tol <tol> - the tolerance
-  -nep_refine_its <its> - number of iterations

   Notes:
   By default, iterative refinement is disabled, since it may be very
   costly. There are two possible refinement strategies: simple and multiple.
   The simple approach performs iterative refinement on each of the
   converged eigenpairs individually, whereas the multiple strategy works
   with the invariant pair as a whole, refining all eigenpairs simultaneously.
   The latter may be required for the case of multiple eigenvalues.

   The tol and its parameters specify the stopping criterion. In the simple
   method, refinement continues until the residual of each eigenpair is
   below the tolerance (tol defaults to the NEP tol, but may be set to a
   different value). In contrast, the multiple method simply performs its
   refinement iterations (just one by default).

   Level: intermediate

.seealso: NEPGetRefine()
@*/
PetscErrorCode NEPSetRefine(NEP nep,NEPRefine refine,PetscReal tol,PetscInt its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(nep,refine,2);
  PetscValidLogicalCollectiveReal(nep,tol,3);
  PetscValidLogicalCollectiveInt(nep,its,4);
  nep->refine = refine;
  if (refine) {  /* process parameters only if not REFINE_NONE */
    if (tol == PETSC_DEFAULT || tol == PETSC_DECIDE) {
      nep->reftol = nep->rtol;
    } else {
      if (tol<=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
      nep->reftol = tol;
    }
    if (its==PETSC_DECIDE || its==PETSC_DEFAULT) {
      nep->rits = PETSC_DEFAULT;
    } else {
      if (its<=0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of its. Must be > 0");
      nep->rits = its;
    }
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
.  tol    - the convergence tolerance
-  its    - maximum number of refinement iterations

   Level: intermediate

   Note:
   The user can specify NULL for any parameter that is not needed.

.seealso: NEPSetRefine()
@*/
PetscErrorCode NEPGetRefine(NEP nep,NEPRefine *refine,PetscReal *tol,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (refine) *refine = nep->refine;
  if (tol)    *tol    = nep->reftol;
  if (its)    *its    = nep->rits;
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
  if (!nep->ksp) { ierr = NEPGetKSP(nep,&nep->ksp);CHKERRQ(ierr); }
  ierr = KSPSetOptionsPrefix(nep->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(nep->ksp,"nep_");CHKERRQ(ierr);
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
  if (!nep->ksp) { ierr = NEPGetKSP(nep,&nep->ksp);CHKERRQ(ierr); }
  ierr = KSPSetOptionsPrefix(nep->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(nep->ksp,"nep_");CHKERRQ(ierr);
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

   Notes: On the fortran side, the user should pass in a string 'prefix' of
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

