/*
      QEP routines related to options that can be set via the command-line
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

#include <slepc-private/qepimpl.h>       /*I "slepcqep.h" I*/

#undef __FUNCT__
#define __FUNCT__ "QEPSetFromOptions"
/*@
   QEPSetFromOptions - Sets QEP options from the options database.
   This routine must be called before QEPSetUp() if the user is to be
   allowed to set the solver type.

   Collective on QEP

   Input Parameters:
.  qep - the quadratic eigensolver context

   Notes:
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode QEPSetFromOptions(QEP qep)
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
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (!QEPRegisterAllCalled) { ierr = QEPRegisterAll();CHKERRQ(ierr); }
  ierr = PetscObjectOptionsBegin((PetscObject)qep);CHKERRQ(ierr);
    ierr = PetscOptionsList("-qep_type","Quadratic Eigenvalue Problem method","QEPSetType",QEPList,(char*)(((PetscObject)qep)->type_name?((PetscObject)qep)->type_name:QEPLINEAR),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = QEPSetType(qep,type);CHKERRQ(ierr);
    } else if (!((PetscObject)qep)->type_name) {
      ierr = QEPSetType(qep,QEPLINEAR);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBoolGroupBegin("-qep_general","general quadratic eigenvalue problem","QEPSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetProblemType(qep,QEP_GENERAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-qep_hermitian","hermitian quadratic eigenvalue problem","QEPSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetProblemType(qep,QEP_HERMITIAN);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-qep_gyroscopic","gyroscopic quadratic eigenvalue problem","QEPSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetProblemType(qep,QEP_GYROSCOPIC);CHKERRQ(ierr); }

    r = 0;
    ierr = PetscOptionsReal("-qep_scale","Scale factor","QEPSetScaleFactor",qep->sfactor,&r,NULL);CHKERRQ(ierr);
    ierr = QEPSetScaleFactor(qep,r);CHKERRQ(ierr);

    r = i = 0;
    ierr = PetscOptionsInt("-qep_max_it","Maximum number of iterations","QEPSetTolerances",qep->max_it,&i,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-qep_tol","Tolerance","QEPSetTolerances",qep->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:qep->tol,&r,NULL);CHKERRQ(ierr);
    ierr = QEPSetTolerances(qep,r,i);CHKERRQ(ierr);
    ierr = PetscOptionsBoolGroupBegin("-qep_convergence_default","Default (relative error) convergence test","QEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetConvergenceTest(qep,QEPConvergedDefault,NULL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-qep_convergence_absolute","Absolute error convergence test","QEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetConvergenceTest(qep,QEPConvergedAbsolute,NULL);CHKERRQ(ierr); }

    i = j = k = 0;
    ierr = PetscOptionsInt("-qep_nev","Number of eigenvalues to compute","QEPSetDimensions",qep->nev,&i,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-qep_ncv","Number of basis vectors","QEPSetDimensions",qep->ncv,&j,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-qep_mpd","Maximum dimension of projected problem","QEPSetDimensions",qep->mpd,&k,NULL);CHKERRQ(ierr);
    ierr = QEPSetDimensions(qep,i,j,k);CHKERRQ(ierr);

    ierr = PetscOptionsScalar("-qep_target","Value of the target","QEPSetTarget",qep->target,&s,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = QEPSetWhichEigenpairs(qep,QEP_TARGET_MAGNITUDE);CHKERRQ(ierr);
      ierr = QEPSetTarget(qep,s);CHKERRQ(ierr);
    }

    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to QEPSetFromOptions()
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-qep_monitor_cancel","Remove any hardwired monitor routines","QEPMonitorCancel",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = QEPMonitorCancel(qep);CHKERRQ(ierr);
    }
    /*
      Prints approximate eigenvalues and error estimates at each iteration
    */
    ierr = PetscOptionsString("-qep_monitor","Monitor first unconverged approximate eigenvalue and error estimate","QEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)qep),monfilename,&monviewer);CHKERRQ(ierr);
      ierr = QEPMonitorSet(qep,QEPMonitorFirst,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-qep_monitor_conv","Monitor approximate eigenvalues and error estimates as they converge","QEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscNew(struct _n_SlepcConvMonitor,&ctx);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)qep),monfilename,&ctx->viewer);CHKERRQ(ierr);
      ierr = QEPMonitorSet(qep,QEPMonitorConverged,ctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-qep_monitor_all","Monitor approximate eigenvalues and error estimates","QEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)qep),monfilename,&monviewer);CHKERRQ(ierr);
      ierr = QEPMonitorSet(qep,QEPMonitorAll,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
      ierr = QEPSetTrackAll(qep,PETSC_TRUE);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-qep_monitor_lg","Monitor first unconverged approximate error estimate graphically","QEPMonitorSet",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = QEPMonitorSet(qep,QEPMonitorLG,NULL,NULL);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-qep_monitor_lg_all","Monitor error estimates graphically","QEPMonitorSet",flg,&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = QEPMonitorSet(qep,QEPMonitorLGAll,NULL,NULL);CHKERRQ(ierr);
      ierr = QEPSetTrackAll(qep,PETSC_TRUE);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/

    ierr = PetscOptionsBoolGroupBegin("-qep_largest_magnitude","compute largest eigenvalues in magnitude","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_LARGEST_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-qep_smallest_magnitude","compute smallest eigenvalues in magnitude","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_SMALLEST_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-qep_largest_real","compute largest real parts","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_LARGEST_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-qep_smallest_real","compute smallest real parts","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_SMALLEST_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-qep_largest_imaginary","compute largest imaginary parts","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_LARGEST_IMAGINARY);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-qep_smallest_imaginary","compute smallest imaginary parts","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_SMALLEST_IMAGINARY);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-qep_target_magnitude","compute nearest eigenvalues to target","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_TARGET_MAGNITUDE);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroup("-qep_target_real","compute eigenvalues with real parts close to target","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_TARGET_REAL);CHKERRQ(ierr); }
    ierr = PetscOptionsBoolGroupEnd("-qep_target_imaginary","compute eigenvalues with imaginary parts close to target","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) { ierr = QEPSetWhichEigenpairs(qep,QEP_TARGET_IMAGINARY);CHKERRQ(ierr); }

    ierr = PetscOptionsBool("-qep_left_vectors","Compute left eigenvectors also","QEPSetLeftVectorsWanted",qep->leftvecs,&val,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = QEPSetLeftVectorsWanted(qep,val);CHKERRQ(ierr);
    }

    ierr = PetscOptionsName("-qep_view","Print detailed information on solver used","QEPView",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-qep_plot_eigs","Make a plot of the computed eigenvalues","QEPSolve",0);CHKERRQ(ierr);

    if (qep->ops->setfromoptions) {
      ierr = (*qep->ops->setfromoptions)(qep);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)qep);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!qep->ip) { ierr = QEPGetIP(qep,&qep->ip);CHKERRQ(ierr); }
  ierr = IPSetFromOptions(qep->ip);CHKERRQ(ierr);
  if (!qep->ds) { ierr = QEPGetDS(qep,&qep->ds);CHKERRQ(ierr); }
  ierr = DSSetFromOptions(qep->ds);CHKERRQ(ierr);
  if (!qep->st) { ierr = QEPGetST(qep,&qep->st);CHKERRQ(ierr); }
  ierr = STSetFromOptions(qep->st);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(qep->rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetTolerances"
/*@
   QEPGetTolerances - Gets the tolerance and maximum iteration count used
   by the QEP convergence tests.

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context

   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: QEPSetTolerances()
@*/
PetscErrorCode QEPGetTolerances(QEP qep,PetscReal *tol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (tol)    *tol    = qep->tol;
  if (maxits) *maxits = qep->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetTolerances"
/*@
   QEPSetTolerances - Sets the tolerance and maximum iteration count used
   by the QEP convergence tests.

   Logically Collective on QEP

   Input Parameters:
+  qep - the quadratic eigensolver context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -qep_tol <tol> - Sets the convergence tolerance
-  -qep_max_it <maxits> - Sets the maximum number of iterations allowed

   Notes:
   Pass 0 for an argument that need not be changed.

   Use PETSC_DECIDE for maxits to assign a reasonably good value, which is
   dependent on the solution method.

   Level: intermediate

.seealso: QEPGetTolerances()
@*/
PetscErrorCode QEPSetTolerances(QEP qep,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(qep,tol,2);
  PetscValidLogicalCollectiveInt(qep,maxits,3);
  if (tol) {
    if (tol == PETSC_DEFAULT) {
      qep->tol = PETSC_DEFAULT;
    } else {
      if (tol < 0.0) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
      qep->tol = tol;
    }
  }
  if (maxits) {
    if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
      qep->max_it = 0;
      qep->setupcalled = 0;
    } else {
      if (maxits < 0) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
      qep->max_it = maxits;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetDimensions"
/*@
   QEPGetDimensions - Gets the number of eigenvalues to compute
   and the dimension of the subspace.

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context

   Output Parameters:
+  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.seealso: QEPSetDimensions()
@*/
PetscErrorCode QEPGetDimensions(QEP qep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (nev) *nev = qep->nev;
  if (ncv) *ncv = qep->ncv;
  if (mpd) *mpd = qep->mpd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetDimensions"
/*@
   QEPSetDimensions - Sets the number of eigenvalues to compute
   and the dimension of the subspace.

   Logically Collective on QEP

   Input Parameters:
+  qep - the quadratic eigensolver context
.  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Options Database Keys:
+  -qep_nev <nev> - Sets the number of eigenvalues
.  -qep_ncv <ncv> - Sets the dimension of the subspace
-  -qep_mpd <mpd> - Sets the maximum projected dimension

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

.seealso: QEPGetDimensions()
@*/
PetscErrorCode QEPSetDimensions(QEP qep,PetscInt nev,PetscInt ncv,PetscInt mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(qep,nev,2);
  PetscValidLogicalCollectiveInt(qep,ncv,3);
  PetscValidLogicalCollectiveInt(qep,mpd,4);
  if (nev) {
    if (nev<1) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nev. Must be > 0");
    qep->nev = nev;
    qep->setupcalled = 0;
  }
  if (ncv) {
    if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
      qep->ncv = 0;
    } else {
      if (ncv<1) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
      qep->ncv = ncv;
    }
    qep->setupcalled = 0;
  }
  if (mpd) {
    if (mpd == PETSC_DECIDE || mpd == PETSC_DEFAULT) {
      qep->mpd = 0;
    } else {
      if (mpd<1) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of mpd. Must be > 0");
      qep->mpd = mpd;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetWhichEigenpairs"
/*@
    QEPSetWhichEigenpairs - Specifies which portion of the spectrum is
    to be sought.

    Logically Collective on QEP

    Input Parameters:
+   qep   - eigensolver context obtained from QEPCreate()
-   which - the portion of the spectrum to be sought

    Possible values:
    The parameter 'which' can have one of these values

+     QEP_LARGEST_MAGNITUDE - largest eigenvalues in magnitude (default)
.     QEP_SMALLEST_MAGNITUDE - smallest eigenvalues in magnitude
.     QEP_LARGEST_REAL - largest real parts
.     QEP_SMALLEST_REAL - smallest real parts
.     QEP_LARGEST_IMAGINARY - largest imaginary parts
.     QEP_SMALLEST_IMAGINARY - smallest imaginary parts
.     QEP_TARGET_MAGNITUDE - eigenvalues closest to the target (in magnitude)
.     QEP_TARGET_REAL - eigenvalues with real part closest to target
-     QEP_TARGET_IMAGINARY - eigenvalues with imaginary part closest to target

    Options Database Keys:
+   -qep_largest_magnitude - Sets largest eigenvalues in magnitude
.   -qep_smallest_magnitude - Sets smallest eigenvalues in magnitude
.   -qep_largest_real - Sets largest real parts
.   -qep_smallest_real - Sets smallest real parts
.   -qep_largest_imaginary - Sets largest imaginary parts
.   -qep_smallest_imaginary - Sets smallest imaginary parts
.   -qep_target_magnitude - Sets eigenvalues closest to target
.   -qep_target_real - Sets real parts closest to target
-   -qep_target_imaginary - Sets imaginary parts closest to target

    Notes:
    Not all eigensolvers implemented in QEP account for all the possible values
    stated above. If SLEPc is compiled for real numbers QEP_LARGEST_IMAGINARY
    and QEP_SMALLEST_IMAGINARY use the absolute value of the imaginary part
    for eigenvalue selection.

    Level: intermediate

.seealso: QEPGetWhichEigenpairs(), QEPWhich
@*/
PetscErrorCode QEPSetWhichEigenpairs(QEP qep,QEPWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(qep,which,2);
  if (which) {
    if (which==PETSC_DECIDE || which==PETSC_DEFAULT) qep->which = (QEPWhich)0;
    else switch (which) {
      case QEP_LARGEST_MAGNITUDE:
      case QEP_SMALLEST_MAGNITUDE:
      case QEP_LARGEST_REAL:
      case QEP_SMALLEST_REAL:
      case QEP_LARGEST_IMAGINARY:
      case QEP_SMALLEST_IMAGINARY:
      case QEP_TARGET_MAGNITUDE:
      case QEP_TARGET_REAL:
#if defined(PETSC_USE_COMPLEX)
      case QEP_TARGET_IMAGINARY:
#endif
        if (qep->which != which) {
          qep->setupcalled = 0;
          qep->which = which;
        }
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'which' value");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetWhichEigenpairs"
/*@C
    QEPGetWhichEigenpairs - Returns which portion of the spectrum is to be
    sought.

    Not Collective

    Input Parameter:
.   qep - eigensolver context obtained from QEPCreate()

    Output Parameter:
.   which - the portion of the spectrum to be sought

    Notes:
    See QEPSetWhichEigenpairs() for possible values of 'which'.

    Level: intermediate

.seealso: QEPSetWhichEigenpairs(), QEPWhich
@*/
PetscErrorCode QEPGetWhichEigenpairs(QEP qep,QEPWhich *which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(which,2);
  *which = qep->which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetLeftVectorsWanted"
/*@
    QEPSetLeftVectorsWanted - Specifies which eigenvectors are required.

    Logically Collective on QEP

    Input Parameters:
+   qep      - the quadratic eigensolver context
-   leftvecs - whether left eigenvectors are required or not

    Options Database Keys:
.   -qep_left_vectors <boolean> - Sets/resets the boolean flag 'leftvecs'

    Notes:
    If the user sets leftvecs=PETSC_TRUE then the solver uses a variant of
    the algorithm that computes both right and left eigenvectors. This is
    usually much more costly. This option is not available in all solvers.

    Level: intermediate

.seealso: QEPGetLeftVectorsWanted()
@*/
PetscErrorCode QEPSetLeftVectorsWanted(QEP qep,PetscBool leftvecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(qep,leftvecs,2);
  if (qep->leftvecs != leftvecs) {
    qep->leftvecs = leftvecs;
    qep->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetLeftVectorsWanted"
/*@C
    QEPGetLeftVectorsWanted - Returns the flag indicating whether left
    eigenvectors are required or not.

    Not Collective

    Input Parameter:
.   qep - the eigensolver context

    Output Parameter:
.   leftvecs - the returned flag

    Level: intermediate

.seealso: QEPSetLeftVectorsWanted()
@*/
PetscErrorCode QEPGetLeftVectorsWanted(QEP qep,PetscBool *leftvecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(leftvecs,2);
  *leftvecs = qep->leftvecs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetScaleFactor"
/*@
   QEPGetScaleFactor - Gets the factor used for scaling the quadratic eigenproblem.

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context

   Output Parameters:
.  alpha - the scaling factor

   Notes:
   If the user did not specify a scaling factor, then after QEPSolve() the
   default value is returned.

   Level: intermediate

.seealso: QEPSetScaleFactor(), QEPSolve()
@*/
PetscErrorCode QEPGetScaleFactor(QEP qep,PetscReal *alpha)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(alpha,2);
  *alpha = qep->sfactor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetScaleFactor"
/*@
   QEPSetScaleFactor - Sets the scaling factor to be used for scaling the
   quadratic problem before attempting to solve.

   Logically Collective on QEP

   Input Parameters:
+  qep   - the quadratic eigensolver context
-  alpha - the scaling factor

   Options Database Keys:
.  -qep_scale <alpha> - Sets the scaling factor

   Notes:
   For the problem (l^2*M + l*C + K)*x = 0, the effect of scaling is to work
   with matrices (alpha^2*M, alpha*C, K), then scale the computed eigenvalue.

   The default is to scale with alpha = norm(K)/norm(M).

   Level: intermediate

.seealso: QEPGetScaleFactor()
@*/
PetscErrorCode QEPSetScaleFactor(QEP qep,PetscReal alpha)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(qep,alpha,2);
  if (alpha) {
    if (alpha == PETSC_DEFAULT || alpha == PETSC_DECIDE) {
      qep->sfactor = 0.0;
      qep->sfactor_set = PETSC_FALSE;
    } else {
      if (alpha < 0.0) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of alpha. Must be > 0");
      qep->sfactor = alpha;
      qep->sfactor_set = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetProblemType"
/*@
   QEPSetProblemType - Specifies the type of the quadratic eigenvalue problem.

   Logically Collective on QEP

   Input Parameters:
+  qep      - the quadratic eigensolver context
-  type     - a known type of quadratic eigenvalue problem

   Options Database Keys:
+  -qep_general - general problem with no particular structure
.  -qep_hermitian - problem whose coefficient matrices are Hermitian
-  -qep_gyroscopic - problem with Hamiltonian structure

   Notes:
   Allowed values for the problem type are: general (QEP_GENERAL), Hermitian
   (QEP_HERMITIAN), and gyroscopic (QEP_GYROSCOPIC).

   This function is used to instruct SLEPc to exploit certain structure in
   the quadratic eigenproblem. By default, no particular structure is assumed.

   If the problem matrices are Hermitian (symmetric in the real case) or
   Hermitian/skew-Hermitian then the solver can exploit this fact to perform
   less operations or provide better stability.

   Level: intermediate

.seealso: QEPSetOperators(), QEPSetType(), QEPGetProblemType(), QEPProblemType
@*/
PetscErrorCode QEPSetProblemType(QEP qep,QEPProblemType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(qep,type,2);
  if (type!=QEP_GENERAL && type!=QEP_HERMITIAN && type!=QEP_GYROSCOPIC)
    SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_ARG_WRONG,"Unknown eigenvalue problem type");
  qep->problem_type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetProblemType"
/*@C
   QEPGetProblemType - Gets the problem type from the QEP object.

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context

   Output Parameter:
.  type - name of QEP problem type

   Level: intermediate

.seealso: QEPSetProblemType(), QEPProblemType
@*/
PetscErrorCode QEPGetProblemType(QEP qep,QEPProblemType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = qep->problem_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetConvergenceTest"
/*@C
    QEPSetConvergenceTest - Sets a function to compute the error estimate used in
    the convergence test.

    Logically Collective on QEP

    Input Parameters:
+   qep  - eigensolver context obtained from QEPCreate()
.   func - a pointer to the convergence test function
-   ctx  - a context pointer (the last parameter to the convergence test function)

    Calling Sequence of func:
$   func(QEP qep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal* errest,void *ctx)

+   qep    - eigensolver context obtained from QEPCreate()
.   eigr   - real part of the eigenvalue
.   eigi   - imaginary part of the eigenvalue
.   res    - residual norm associated to the eigenpair
.   errest - (output) computed error estimate
-   ctx    - optional context, as set by QEPSetConvergenceTest()

    Note:
    If the error estimate returned by the convergence test function is less than
    the tolerance, then the eigenvalue is accepted as converged.

    Level: advanced

.seealso: QEPSetTolerances()
@*/
PetscErrorCode QEPSetConvergenceTest(QEP qep,PetscErrorCode (*func)(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*),void* ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  qep->converged    = func;
  qep->convergedctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetTrackAll"
/*@
   QEPSetTrackAll - Specifies if the solver must compute the residual of all
   approximate eigenpairs or not.

   Logically Collective on QEP

   Input Parameters:
+  qep      - the eigensolver context
-  trackall - whether compute all residuals or not

   Notes:
   If the user sets trackall=PETSC_TRUE then the solver explicitly computes
   the residual for each eigenpair approximation. Computing the residual is
   usually an expensive operation and solvers commonly compute the associated
   residual to the first unconverged eigenpair.
   The options '-qep_monitor_all' and '-qep_monitor_lg_all' automatically
   activate this option.

   Level: intermediate

.seealso: QEPGetTrackAll()
@*/
PetscErrorCode QEPSetTrackAll(QEP qep,PetscBool trackall)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(qep,trackall,2);
  qep->trackall = trackall;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetTrackAll"
/*@
   QEPGetTrackAll - Returns the flag indicating whether all residual norms must
   be computed or not.

   Not Collective

   Input Parameter:
.  qep - the eigensolver context

   Output Parameter:
.  trackall - the returned flag

   Level: intermediate

.seealso: QEPSetTrackAll()
@*/
PetscErrorCode QEPGetTrackAll(QEP qep,PetscBool *trackall)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(trackall,2);
  *trackall = qep->trackall;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetOptionsPrefix"
/*@C
   QEPSetOptionsPrefix - Sets the prefix used for searching for all
   QEP options in the database.

   Logically Collective on QEP

   Input Parameters:
+  qep - the quadratic eigensolver context
-  prefix - the prefix string to prepend to all QEP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different QEP contexts, one could call
.vb
      QEPSetOptionsPrefix(qep1,"qeig1_")
      QEPSetOptionsPrefix(qep2,"qeig2_")
.ve

   Level: advanced

.seealso: QEPAppendOptionsPrefix(), QEPGetOptionsPrefix()
@*/
PetscErrorCode QEPSetOptionsPrefix(QEP qep,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (!qep->ip) { ierr = QEPGetIP(qep,&qep->ip);CHKERRQ(ierr); }
  ierr = IPSetOptionsPrefix(qep->ip,prefix);CHKERRQ(ierr);
  if (!qep->ds) { ierr = QEPGetDS(qep,&qep->ds);CHKERRQ(ierr); }
  ierr = DSSetOptionsPrefix(qep->ds,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)qep,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPAppendOptionsPrefix"
/*@C
   QEPAppendOptionsPrefix - Appends to the prefix used for searching for all
   QEP options in the database.

   Logically Collective on QEP

   Input Parameters:
+  qep - the quadratic eigensolver context
-  prefix - the prefix string to prepend to all QEP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: QEPSetOptionsPrefix(), QEPGetOptionsPrefix()
@*/
PetscErrorCode QEPAppendOptionsPrefix(QEP qep,const char *prefix)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  EPS            eps;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (!qep->ip) { ierr = QEPGetIP(qep,&qep->ip);CHKERRQ(ierr); }
  ierr = IPSetOptionsPrefix(qep->ip,prefix);CHKERRQ(ierr);
  if (!qep->ds) { ierr = QEPGetDS(qep,&qep->ds);CHKERRQ(ierr); }
  ierr = DSSetOptionsPrefix(qep->ds,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)qep,prefix);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)qep,QEPLINEAR,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = QEPLinearGetEPS(qep,&eps);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(eps,((PetscObject)qep)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(eps,"qep_");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetOptionsPrefix"
/*@C
   QEPGetOptionsPrefix - Gets the prefix used for searching for all
   QEP options in the database.

   Not Collective

   Input Parameters:
.  qep - the quadratic eigensolver context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: QEPSetOptionsPrefix(), QEPAppendOptionsPrefix()
@*/
PetscErrorCode QEPGetOptionsPrefix(QEP qep,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)qep,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
