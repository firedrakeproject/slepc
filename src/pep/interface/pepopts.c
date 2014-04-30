/*
      PEP routines related to options that can be set via the command-line
      or procedurally.

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

#include <slepc-private/pepimpl.h>       /*I "slepcpep.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions"
/*@
   PEPSetFromOptions - Sets PEP options from the options database.
   This routine must be called before PEPSetUp() if the user is to be
   allowed to set the solver type.

   Collective on PEP

   Input Parameters:
.  pep - the polynomial eigensolver context

   Notes:
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode PEPSetFromOptions(PEP pep)
{
  PetscErrorCode   ierr;
  char             type[256],monfilename[PETSC_MAX_PATH_LEN];
  PetscBool        flg,val;
  PetscReal        r;
  PetscScalar      s;
  PetscInt         i,j,k;
  PetscViewer      monviewer;
  SlepcConvMonitor ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!PEPRegisterAllCalled) { ierr = PEPRegisterAll();CHKERRQ(ierr); }
  ierr = PetscObjectOptionsBegin((PetscObject)pep);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-pep_type","Polynomial Eigenvalue Problem method","PEPSetType",PEPList,(char*)(((PetscObject)pep)->type_name?((PetscObject)pep)->type_name:PEPTOAR),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PEPSetType(pep,type);CHKERRQ(ierr);
    } else if (!((PetscObject)pep)->type_name) {
      ierr = PEPSetType(pep,PEPTOAR);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBoolGroupBegin("-pep_general","general polynomial eigenvalue problem","PEPSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetProblemType(pep,PEP_GENERAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_hermitian","hermitian polynomial eigenvalue problem","PEPSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetProblemType(pep,PEP_HERMITIAN);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-pep_gyroscopic","gyroscopic polynomial eigenvalue problem","PEPSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetProblemType(pep,PEP_GYROSCOPIC);CHKERRQ(ierr); }

    r = 0;
    ierr = PetscOptionsReal("-pep_scale","Scale factor","PEPSetScaleFactor",pep->sfactor,&r,NULL);CHKERRQ(ierr);
    ierr = PEPSetScaleFactor(pep,r);CHKERRQ(ierr);

    ierr = PetscOptionsBool("-pep_balance","Enable diagonal scaling (balancing)","PEPSetBalance",pep->balance,&pep->balance,NULL);CHKERRQ(ierr);
    r = j = 0;
    ierr = PetscOptionsInt("-pep_balance_its","Number of iterations in balancing","PEPSetBalance",pep->balance_its,&j,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pep_balance_w","Estimate of eigenvalue (modulus) for balancing","PEPSetBalance",pep->balance_w,&r,NULL);CHKERRQ(ierr);
    ierr = PEPSetBalance(pep,pep->balance,j,r);CHKERRQ(ierr);

    r = i = 0;
    ierr = PetscOptionsInt("-pep_max_it","Maximum number of iterations","PEPSetTolerances",pep->max_it,&i,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pep_tol","Tolerance","PEPSetTolerances",pep->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:pep->tol,&r,NULL);CHKERRQ(ierr);
    ierr = PEPSetTolerances(pep,r,i);CHKERRQ(ierr);
    ierr = PetscOptionsBoolGroupBegin("-pep_conv_eig","Relative error convergence test","PEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetConvergenceTest(pep,PEP_CONV_EIG);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_conv_norm","Convergence test relative to the eigenvalue and the matrix norms","PEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetConvergenceTest(pep,PEP_CONV_NORM);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-pep_conv_abs","Absolute error convergence test","PEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetConvergenceTest(pep,PEP_CONV_ABS);CHKERRQ(ierr); }

    i = j = k = 0;
    ierr = PetscOptionsInt("-pep_nev","Number of eigenvalues to compute","PEPSetDimensions",pep->nev,&i,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_ncv","Number of basis vectors","PEPSetDimensions",pep->ncv,&j,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pep_mpd","Maximum dimension of projected problem","PEPSetDimensions",pep->mpd,&k,NULL);CHKERRQ(ierr);
    ierr = PEPSetDimensions(pep,i,j,k);CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-pep_target","Value of the target","PEPSetTarget",pep->target,&s,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PEPSetWhichEigenpairs(pep,PEP_TARGET_MAGNITUDE);CHKERRQ(ierr);
      ierr = PEPSetTarget(pep,s);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEnum("-pep_basis","Polynomial basis","PEPSetBasis",PEPBasisTypes,(PetscEnum)pep->basis,(PetscEnum*)&pep->basis,NULL);CHKERRQ(ierr);

    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to PEPSetFromOptions()
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-pep_monitor_cancel","Remove any hardwired monitor routines","PEPMonitorCancel",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PEPMonitorCancel(pep);CHKERRQ(ierr);
    }
    /*
      Prints approximate eigenvalues and error estimates at each iteration
    */
    ierr = PetscOptionsString("-pep_monitor","Monitor first unconverged approximate eigenvalue and error estimate","PEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pep),monfilename,&monviewer);CHKERRQ(ierr);
      ierr = PEPMonitorSet(pep,PEPMonitorFirst,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-pep_monitor_conv","Monitor approximate eigenvalues and error estimates as they converge","PEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscNew(&ctx);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pep),monfilename,&ctx->viewer);CHKERRQ(ierr);
      ierr = PEPMonitorSet(pep,PEPMonitorConverged,ctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-pep_monitor_all","Monitor approximate eigenvalues and error estimates","PEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pep),monfilename,&monviewer);CHKERRQ(ierr);
      ierr = PEPMonitorSet(pep,PEPMonitorAll,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
      ierr = PEPSetTrackAll(pep,PETSC_TRUE);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-pep_monitor_lg","Monitor first unconverged approximate error estimate graphically","PEPMonitorSet",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PEPMonitorSet(pep,PEPMonitorLG,NULL,NULL);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-pep_monitor_lg_all","Monitor error estimates graphically","PEPMonitorSet",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PEPMonitorSet(pep,PEPMonitorLGAll,NULL,NULL);CHKERRQ(ierr);
      ierr = PEPSetTrackAll(pep,PETSC_TRUE);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/

    ierr = PetscOptionsBoolGroupBegin("-pep_largest_magnitude","compute largest eigenvalues in magnitude","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_LARGEST_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_smallest_magnitude","compute smallest eigenvalues in magnitude","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_SMALLEST_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_largest_real","compute largest real parts","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_LARGEST_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_smallest_real","compute smallest real parts","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_SMALLEST_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_largest_imaginary","compute largest imaginary parts","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_LARGEST_IMAGINARY);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_smallest_imaginary","compute smallest imaginary parts","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_SMALLEST_IMAGINARY);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_target_magnitude","compute nearest eigenvalues to target","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_TARGET_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-pep_target_real","compute eigenvalues with real parts close to target","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_TARGET_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-pep_target_imaginary","compute eigenvalues with imaginary parts close to target","PEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = PEPSetWhichEigenpairs(pep,PEP_TARGET_IMAGINARY);CHKERRQ(ierr); }

    ierr = PetscOptionsBool("-pep_left_vectors","Compute left eigenvectors also","PEPSetLeftVectorsWanted",pep->leftvecs,&val,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PEPSetLeftVectorsWanted(pep,val);CHKERRQ(ierr);
    }

    ierr = PetscOptionsName("-pep_view","Print detailed information on solver used","PEPView",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pep_plot_eigs","Make a plot of the computed eigenvalues","PEPSolve",0);CHKERRQ(ierr);

    if (pep->ops->setfromoptions) {
      ierr = (*pep->ops->setfromoptions)(pep);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)pep);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!pep->ip) { ierr = PEPGetIP(pep,&pep->ip);CHKERRQ(ierr); }
  ierr = IPSetFromOptions(pep->ip);CHKERRQ(ierr);
  if (!pep->ds) { ierr = PEPGetDS(pep,&pep->ds);CHKERRQ(ierr); }
  ierr = DSSetFromOptions(pep->ds);CHKERRQ(ierr);
  if (!pep->st) { ierr = PEPGetST(pep,&pep->st);CHKERRQ(ierr); }
  ierr = STSetFromOptions(pep->st);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(pep->rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetTolerances"
/*@
   PEPGetTolerances - Gets the tolerance and maximum iteration count used
   by the PEP convergence tests.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: PEPSetTolerances()
@*/
PetscErrorCode PEPGetTolerances(PEP pep,PetscReal *tol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (tol)    *tol    = pep->tol;
  if (maxits) *maxits = pep->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetTolerances"
/*@
   PEPSetTolerances - Sets the tolerance and maximum iteration count used
   by the PEP convergence tests.

   Logically Collective on PEP

   Input Parameters:
+  pep - the polynomial eigensolver context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -pep_tol <tol> - Sets the convergence tolerance
-  -pep_max_it <maxits> - Sets the maximum number of iterations allowed

   Notes:
   Pass 0 for an argument that need not be changed.

   Use PETSC_DECIDE for maxits to assign a reasonably good value, which is
   dependent on the solution method.

   Level: intermediate

.seealso: PEPGetTolerances()
@*/
PetscErrorCode PEPSetTolerances(PEP pep,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,tol,2);
  PetscValidLogicalCollectiveInt(pep,maxits,3);
  if (tol) {
    if (tol == PETSC_DEFAULT) {
      pep->tol = PETSC_DEFAULT;
    } else {
      if (tol < 0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
      pep->tol = tol;
    }
  }
  if (maxits) {
    if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
      pep->max_it = 0;
      pep->setupcalled = 0;
    } else {
      if (maxits < 0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
      pep->max_it = maxits;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetDimensions"
/*@
   PEPGetDimensions - Gets the number of eigenvalues to compute
   and the dimension of the subspace.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameters:
+  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: PEPSetDimensions()
@*/
PetscErrorCode PEPGetDimensions(PEP pep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (nev) *nev = pep->nev;
  if (ncv) *ncv = pep->ncv;
  if (mpd) *mpd = pep->mpd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetDimensions"
/*@
   PEPSetDimensions - Sets the number of eigenvalues to compute
   and the dimension of the subspace.

   Logically Collective on PEP

   Input Parameters:
+  pep - the polynomial eigensolver context
.  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Options Database Keys:
+  -pep_nev <nev> - Sets the number of eigenvalues
.  -pep_ncv <ncv> - Sets the dimension of the subspace
-  -pep_mpd <mpd> - Sets the maximum projected dimension

   Notes:
   Pass 0 to retain the previous value of any parameter.

   Use PETSC_DECIDE for ncv and mpd to assign a reasonably good value, which is
   dependent on the solution method.

   The parameters ncv and mpd are intimately related, so that the user is advised
   to set one of them at most. Normal usage is that
   (a) in cases where nev is small, the user sets ncv (a reasonable default is 2*nev); and
   (b) in cases where nev is large, the user sets mpd.

   The value of ncv should always be between nev and (nev+mpd), typically
   ncv=nev+mpd. If nev is not too large, mpd=nev is a reasonable choice, otherwise
   a smaller value should be used.

   Level: intermediate

.seealso: PEPGetDimensions()
@*/
PetscErrorCode PEPSetDimensions(PEP pep,PetscInt nev,PetscInt ncv,PetscInt mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,nev,2);
  PetscValidLogicalCollectiveInt(pep,ncv,3);
  PetscValidLogicalCollectiveInt(pep,mpd,4);
  if (nev) {
    if (nev<1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nev. Must be > 0");
    pep->nev = nev;
    pep->setupcalled = 0;
  }
  if (ncv) {
    if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
      pep->ncv = 0;
    } else {
      if (ncv<1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
      pep->ncv = ncv;
    }
    pep->setupcalled = 0;
  }
  if (mpd) {
    if (mpd == PETSC_DECIDE || mpd == PETSC_DEFAULT) {
      pep->mpd = 0;
    } else {
      if (mpd<1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of mpd. Must be > 0");
      pep->mpd = mpd;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetWhichEigenpairs"
/*@
    PEPSetWhichEigenpairs - Specifies which portion of the spectrum is
    to be sought.

    Logically Collective on PEP

    Input Parameters:
+   pep   - eigensolver context obtained from PEPCreate()
-   which - the portion of the spectrum to be sought

    Possible values:
    The parameter 'which' can have one of these values

+     PEP_LARGEST_MAGNITUDE - largest eigenvalues in magnitude (default)
.     PEP_SMALLEST_MAGNITUDE - smallest eigenvalues in magnitude
.     PEP_LARGEST_REAL - largest real parts
.     PEP_SMALLEST_REAL - smallest real parts
.     PEP_LARGEST_IMAGINARY - largest imaginary parts
.     PEP_SMALLEST_IMAGINARY - smallest imaginary parts
.     PEP_TARGET_MAGNITUDE - eigenvalues closest to the target (in magnitude)
.     PEP_TARGET_REAL - eigenvalues with real part closest to target
-     PEP_TARGET_IMAGINARY - eigenvalues with imaginary part closest to target

    Options Database Keys:
+   -pep_largest_magnitude - Sets largest eigenvalues in magnitude
.   -pep_smallest_magnitude - Sets smallest eigenvalues in magnitude
.   -pep_largest_real - Sets largest real parts
.   -pep_smallest_real - Sets smallest real parts
.   -pep_largest_imaginary - Sets largest imaginary parts
.   -pep_smallest_imaginary - Sets smallest imaginary parts
.   -pep_target_magnitude - Sets eigenvalues closest to target
.   -pep_target_real - Sets real parts closest to target
-   -pep_target_imaginary - Sets imaginary parts closest to target

    Notes:
    Not all eigensolvers implemented in PEP account for all the possible values
    stated above. If SLEPc is compiled for real numbers PEP_LARGEST_IMAGINARY
    and PEP_SMALLEST_IMAGINARY use the absolute value of the imaginary part
    for eigenvalue selection.

    Level: intermediate

.seealso: PEPGetWhichEigenpairs(), PEPWhich
@*/
PetscErrorCode PEPSetWhichEigenpairs(PEP pep,PEPWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pep,which,2);
  if (which) {
    if (which==PETSC_DECIDE || which==PETSC_DEFAULT) pep->which = (PEPWhich)0;
    else switch (which) {
      case PEP_LARGEST_MAGNITUDE:
      case PEP_SMALLEST_MAGNITUDE:
      case PEP_LARGEST_REAL:
      case PEP_SMALLEST_REAL:
      case PEP_LARGEST_IMAGINARY:
      case PEP_SMALLEST_IMAGINARY:
      case PEP_TARGET_MAGNITUDE:
      case PEP_TARGET_REAL:
#if defined(PETSC_USE_COMPLEX)
      case PEP_TARGET_IMAGINARY:
#endif
        if (pep->which != which) {
          pep->setupcalled = 0;
          pep->which = which;
        }
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'which' value");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetWhichEigenpairs"
/*@C
    PEPGetWhichEigenpairs - Returns which portion of the spectrum is to be
    sought.

    Not Collective

    Input Parameter:
.   pep - eigensolver context obtained from PEPCreate()

    Output Parameter:
.   which - the portion of the spectrum to be sought

    Notes:
    See PEPSetWhichEigenpairs() for possible values of 'which'.

    Level: intermediate

.seealso: PEPSetWhichEigenpairs(), PEPWhich
@*/
PetscErrorCode PEPGetWhichEigenpairs(PEP pep,PEPWhich *which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(which,2);
  *which = pep->which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetLeftVectorsWanted"
/*@
    PEPSetLeftVectorsWanted - Specifies which eigenvectors are required.

    Logically Collective on PEP

    Input Parameters:
+   pep      - the polynomial eigensolver context
-   leftvecs - whether left eigenvectors are required or not

    Options Database Keys:
.   -pep_left_vectors <boolean> - Sets/resets the boolean flag 'leftvecs'

    Notes:
    If the user sets leftvecs=PETSC_TRUE then the solver uses a variant of
    the algorithm that computes both right and left eigenvectors. This is
    usually much more costly. This option is not available in all solvers.

    Level: intermediate

.seealso: PEPGetLeftVectorsWanted()
@*/
PetscErrorCode PEPSetLeftVectorsWanted(PEP pep,PetscBool leftvecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,leftvecs,2);
  if (pep->leftvecs != leftvecs) {
    pep->leftvecs = leftvecs;
    pep->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetLeftVectorsWanted"
/*@C
    PEPGetLeftVectorsWanted - Returns the flag indicating whether left
    eigenvectors are required or not.

    Not Collective

    Input Parameter:
.   pep - the eigensolver context

    Output Parameter:
.   leftvecs - the returned flag

    Level: intermediate

.seealso: PEPSetLeftVectorsWanted()
@*/
PetscErrorCode PEPGetLeftVectorsWanted(PEP pep,PetscBool *leftvecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(leftvecs,2);
  *leftvecs = pep->leftvecs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetScaleFactor"
/*@
   PEPGetScaleFactor - Gets the factor used for scaling the polynomial eigenproblem.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameters:
.  alpha - the scaling factor

   Notes:
   If the user did not specify a scaling factor, then after PEPSolve() the
   default value is returned.

   Level: intermediate

.seealso: PEPSetScaleFactor(), PEPSolve()
@*/
PetscErrorCode PEPGetScaleFactor(PEP pep,PetscReal *alpha)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(alpha,2);
  *alpha = pep->sfactor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetScaleFactor"
/*@
   PEPSetScaleFactor - Sets the scaling factor to be used for scaling the
   polynomial problem before attempting to solve.

   Logically Collective on PEP

   Input Parameters:
+  pep   - the polynomial eigensolver context
-  alpha - the scaling factor

   Options Database Keys:
.  -pep_scale <alpha> - Sets the scaling factor

   Notes:
   For the problem (l^2*M + l*C + K)*x = 0, the effect of scaling is to work
   with matrices (alpha^2*M, alpha*C, K), then scale the computed eigenvalue.

   The default is to scale with alpha = norm(K)/norm(M).

   Level: intermediate

.seealso: PEPGetScaleFactor()
@*/
PetscErrorCode PEPSetScaleFactor(PEP pep,PetscReal alpha)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,alpha,2);
  if (alpha) {
    if (alpha == PETSC_DEFAULT || alpha == PETSC_DECIDE) {
      pep->sfactor = 0.0;
      pep->sfactor_set = PETSC_FALSE;
    } else {
      if (alpha < 0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of alpha. Must be > 0");
      pep->sfactor = alpha;
      pep->sfactor_set = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetProblemType"
/*@
   PEPSetProblemType - Specifies the type of the polynomial eigenvalue problem.

   Logically Collective on PEP

   Input Parameters:
+  pep      - the polynomial eigensolver context
-  type     - a known type of polynomial eigenvalue problem

   Options Database Keys:
+  -pep_general - general problem with no particular structure
.  -pep_hermitian - problem whose coefficient matrices are Hermitian
-  -pep_gyroscopic - problem with Hamiltonian structure

   Notes:
   Allowed values for the problem type are: general (PEP_GENERAL), Hermitian
   (PEP_HERMITIAN), and gyroscopic (PEP_GYROSCOPIC).

   This function is used to instruct SLEPc to exploit certain structure in
   the polynomial eigenproblem. By default, no particular structure is assumed.

   If the problem matrices are Hermitian (symmetric in the real case) or
   Hermitian/skew-Hermitian then the solver can exploit this fact to perform
   less operations or provide better stability.

   Level: intermediate

.seealso: PEPSetOperators(), PEPSetType(), PEPGetProblemType(), PEPProblemType
@*/
PetscErrorCode PEPSetProblemType(PEP pep,PEPProblemType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pep,type,2);
  if (type!=PEP_GENERAL && type!=PEP_HERMITIAN && type!=PEP_GYROSCOPIC)
    SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"Unknown eigenvalue problem type");
  pep->problem_type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetProblemType"
/*@C
   PEPGetProblemType - Gets the problem type from the PEP object.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameter:
.  type - name of PEP problem type

   Level: intermediate

.seealso: PEPSetProblemType(), PEPProblemType
@*/
PetscErrorCode PEPGetProblemType(PEP pep,PEPProblemType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = pep->problem_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetBasis"
/*@
   PEPSetBasis - Specifies the type of polynomial basis used to describe the
   polynomial eigenvalue problem.

   Logically Collective on PEP

   Input Parameters:
+  pep   - the polynomial eigensolver context
-  basis - the type of polynomial basis

   Options Database Key:
.  -pep_basis <basis> - Select the basis type

   Notes:
   By default, the coefficient matrices passed via PEPSetOperators() are
   expressed in the monomial basis, i.e. 
   P(lambda) = A_0 + lambda*A_1 + lambda^2*A_2 + ... + lambda^d*A_d.
   Other polynomial bases may have better numerical behaviour, but the user
   must then pass the coefficient matrices accordingly.

   Level: intermediate

.seealso: PEPSetOperators(), PEPGetBasis(), PEPBasis
@*/
PetscErrorCode PEPSetBasis(PEP pep,PEPBasis basis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pep,basis,2);
  pep->basis = basis;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetBasis"
/*@C
   PEPGetBasis - Gets the type of polynomial basis from the PEP object.

   Not Collective

   Input Parameter:
.  pep - the polynomial eigensolver context

   Output Parameter:
.  basis - the polynomial basis

   Level: intermediate

.seealso: PEPSetBasis(), PEPBasis
@*/
PetscErrorCode PEPGetBasis(PEP pep,PEPBasis *basis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(basis,2);
  *basis = pep->basis;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetTrackAll"
/*@
   PEPSetTrackAll - Specifies if the solver must compute the residual of all
   approximate eigenpairs or not.

   Logically Collective on PEP

   Input Parameters:
+  pep      - the eigensolver context
-  trackall - whether compute all residuals or not

   Notes:
   If the user sets trackall=PETSC_TRUE then the solver explicitly computes
   the residual for each eigenpair approximation. Computing the residual is
   usually an expensive operation and solvers commonly compute the associated
   residual to the first unconverged eigenpair.
   The options '-pep_monitor_all' and '-pep_monitor_lg_all' automatically
   activate this option.

   Level: intermediate

.seealso: PEPGetTrackAll()
@*/
PetscErrorCode PEPSetTrackAll(PEP pep,PetscBool trackall)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,trackall,2);
  pep->trackall = trackall;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetTrackAll"
/*@
   PEPGetTrackAll - Returns the flag indicating whether all residual norms must
   be computed or not.

   Not Collective

   Input Parameter:
.  pep - the eigensolver context

   Output Parameter:
.  trackall - the returned flag

   Level: intermediate

.seealso: PEPSetTrackAll()
@*/
PetscErrorCode PEPGetTrackAll(PEP pep,PetscBool *trackall)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(trackall,2);
  *trackall = pep->trackall;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetConvergenceTest"
/*@
   PEPSetConvergenceTest - Specifies how to compute the error estimate
   used in the convergence test.

   Logically Collective on PEP

   Input Parameters:
+  pep   - eigensolver context obtained from PEPCreate()
-  conv  - the type of convergence test

   Options Database Keys:
+  -pep_conv_abs - Sets the absolute convergence test
-  -pep_conv_eig - Sets the convergence test relative to the eigenvalue

   Note:
   The parameter 'conv' can have one of these values
+     PEP_CONV_ABS - absolute error ||r||
-     PEP_CONV_EIG - error relative to the eigenvalue l, ||r||/|l|

   Level: intermediate

.seealso: PEPGetConvergenceTest(), PEPConv
@*/
PetscErrorCode PEPSetConvergenceTest(PEP pep,PEPConv conv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pep,conv,2);
  switch (conv) {
    case PEP_CONV_EIG:   pep->converged = PEPConvergedEigRelative; break;
    case PEP_CONV_ABS:   pep->converged = PEPConvergedAbsolute; break;
    case PEP_CONV_NORM:  pep->converged = PEPConvergedNormRelative; break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'conv' value");
  }
  pep->conv = conv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetConvergenceTest"
/*@
   PEPGetConvergenceTest - Gets the method used to compute the error estimate
   used in the convergence test.

   Not Collective

   Input Parameters:
.  pep   - eigensolver context obtained from PEPCreate()

   Output Parameters:
.  conv  - the type of convergence test

   Level: intermediate

.seealso: PEPSetConvergenceTest(), PEPConv
@*/
PetscErrorCode PEPGetConvergenceTest(PEP pep,PEPConv *conv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(conv,2);
  *conv = pep->conv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetBalance"
/*@
   PEPSetBalance - Specifies if eigensolver performs balancing.

   Logically Collective on PEP

   Input Parameters:
+  pep    - the eigensolver context
.  bal    - flag indicating if balancing is required
.  its    - number of iterations of the balancing algorithm
-  w      - approximation to wanted eigenvalues (norm)

   Options Database Keys:
+  -pep_balance  - flag 
.  -pep_balance_its <its> - number of iterations
-  -pep_balance_lambda <w> - approximation to eigenvalues

   Notes:
   When balancing is enabled, the solver works implicitly with matrix Dr*A*Dl,
   where Dr and Dl are appropriate diagonal matrices. This improves the accuracy
   of the computed results in some cases.

   By default, balancing is disabled and it requires MATAIJ matrices.

   The parameter 'its' is the number of iterations performed by the method.
   Parameter 'w' must be positive. Use PETSC_DECIDE or set w = 1.0 if no
   information about eigenvalues is available.

   Level: intermediate

.seealso: PEPGetBalance()
@*/
PetscErrorCode PEPSetBalance(PEP pep,PetscBool bal,PetscInt its,PetscReal w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,bal,2);
  PetscValidLogicalCollectiveInt(pep,its,3);
  PetscValidLogicalCollectiveReal(pep,w,4);
  pep->balance = bal;
  if (its) {
    if (its==PETSC_DECIDE || its==PETSC_DEFAULT) pep->balance_its = 5;
    else pep->balance_its = its;
  }
  if (w) {
    if (w==PETSC_DECIDE || w==PETSC_DEFAULT) pep->balance_w = 1.0;
    else if (w<0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"w must be positive");
    else pep->balance_w = w;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetBalance"
/*@
   PEPGetBalance - Gets the balancing type used by the PEP object, and the associated
   parameters.

   Not Collective

   Input Parameter:
.  pep - the eigensolver context

   Output Parameters:
+  bal    - flag indicating whether balancing is required or not
.  its    - number of iterations of the balancing algorithm
-  w      - magnitude of wanted eigenvalue

   Level: intermediate

   Note:
   The user can specify NULL for any parameter that is not needed.

.seealso: PEPSetBalance()
@*/
PetscErrorCode PEPGetBalance(PEP pep,PetscBool *bal,PetscInt *its,PetscReal *w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (bal) *bal = pep->balance;
  if (its) *its = pep->balance_its;
  if (w)   *w   = pep->balance_w;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetOptionsPrefix"
/*@C
   PEPSetOptionsPrefix - Sets the prefix used for searching for all
   PEP options in the database.

   Logically Collective on PEP

   Input Parameters:
+  pep - the polynomial eigensolver context
-  prefix - the prefix string to prepend to all PEP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different PEP contexts, one could call
.vb
      PEPSetOptionsPrefix(pep1,"qeig1_")
      PEPSetOptionsPrefix(pep2,"qeig2_")
.ve

   Level: advanced

.seealso: PEPAppendOptionsPrefix(), PEPGetOptionsPrefix()
@*/
PetscErrorCode PEPSetOptionsPrefix(PEP pep,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!pep->ip) { ierr = PEPGetIP(pep,&pep->ip);CHKERRQ(ierr); }
  ierr = IPSetOptionsPrefix(pep->ip,prefix);CHKERRQ(ierr);
  if (!pep->ds) { ierr = PEPGetDS(pep,&pep->ds);CHKERRQ(ierr); }
  ierr = DSSetOptionsPrefix(pep->ds,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)pep,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPAppendOptionsPrefix"
/*@C
   PEPAppendOptionsPrefix - Appends to the prefix used for searching for all
   PEP options in the database.

   Logically Collective on PEP

   Input Parameters:
+  pep - the polynomial eigensolver context
-  prefix - the prefix string to prepend to all PEP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: PEPSetOptionsPrefix(), PEPGetOptionsPrefix()
@*/
PetscErrorCode PEPAppendOptionsPrefix(PEP pep,const char *prefix)
{
  PetscErrorCode ierr;
  /*PetscBool      flg;
  EPS            eps;*/

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!pep->ip) { ierr = PEPGetIP(pep,&pep->ip);CHKERRQ(ierr); }
  ierr = IPSetOptionsPrefix(pep->ip,prefix);CHKERRQ(ierr);
  if (!pep->ds) { ierr = PEPGetDS(pep,&pep->ds);CHKERRQ(ierr); }
  ierr = DSSetOptionsPrefix(pep->ds,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)pep,prefix);CHKERRQ(ierr);
  /*ierr = PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PEPLinearGetEPS(pep,&eps);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(eps,((PetscObject)pep)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(eps,"pep_");CHKERRQ(ierr);
  }*/
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetOptionsPrefix"
/*@C
   PEPGetOptionsPrefix - Gets the prefix used for searching for all
   PEP options in the database.

   Not Collective

   Input Parameters:
.  pep - the polynomial eigensolver context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: PEPSetOptionsPrefix(), PEPAppendOptionsPrefix()
@*/
PetscErrorCode PEPGetOptionsPrefix(PEP pep,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)pep,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

