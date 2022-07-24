/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   EPS routines related to various viewers
*/

#include <slepc/private/epsimpl.h>      /*I "slepceps.h" I*/
#include <slepc/private/bvimpl.h>
#include <petscdraw.h>

/*@C
   EPSView - Prints the EPS data structure.

   Collective on eps

   Input Parameters:
+  eps - the eigenproblem solver context
-  viewer - optional visualization context

   Options Database Key:
.  -eps_view -  Calls EPSView() at end of EPSSolve()

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

.seealso: STView()
@*/
PetscErrorCode EPSView(EPS eps,PetscViewer viewer)
{
  const char     *type=NULL,*extr=NULL,*bal=NULL;
  char           str[50];
  PetscBool      isascii,isexternal,istrivial;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)eps),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(eps,1,viewer,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)eps,viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(eps,view,viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    if (eps->problem_type) {
      switch (eps->problem_type) {
        case EPS_HEP:    type = SLEPC_STRING_HERMITIAN " eigenvalue problem"; break;
        case EPS_GHEP:   type = "generalized " SLEPC_STRING_HERMITIAN " eigenvalue problem"; break;
        case EPS_NHEP:   type = "non-" SLEPC_STRING_HERMITIAN " eigenvalue problem"; break;
        case EPS_GNHEP:  type = "generalized non-" SLEPC_STRING_HERMITIAN " eigenvalue problem"; break;
        case EPS_PGNHEP: type = "generalized non-" SLEPC_STRING_HERMITIAN " eigenvalue problem with " SLEPC_STRING_HERMITIAN " positive definite B"; break;
        case EPS_GHIEP:  type = "generalized " SLEPC_STRING_HERMITIAN "-indefinite eigenvalue problem"; break;
      }
    } else type = "not yet set";
    PetscCall(PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type));
    if (eps->extraction) {
      switch (eps->extraction) {
        case EPS_RITZ:              extr = "Rayleigh-Ritz"; break;
        case EPS_HARMONIC:          extr = "harmonic Ritz"; break;
        case EPS_HARMONIC_RELATIVE: extr = "relative harmonic Ritz"; break;
        case EPS_HARMONIC_RIGHT:    extr = "right harmonic Ritz"; break;
        case EPS_HARMONIC_LARGEST:  extr = "largest harmonic Ritz"; break;
        case EPS_REFINED:           extr = "refined Ritz"; break;
        case EPS_REFINED_HARMONIC:  extr = "refined harmonic Ritz"; break;
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"  extraction type: %s\n",extr));
    }
    if (!eps->ishermitian && eps->balance!=EPS_BALANCE_NONE) {
      switch (eps->balance) {
        case EPS_BALANCE_NONE:    break;
        case EPS_BALANCE_ONESIDE: bal = "one-sided Krylov"; break;
        case EPS_BALANCE_TWOSIDE: bal = "two-sided Krylov"; break;
        case EPS_BALANCE_USER:    bal = "user-defined matrix"; break;
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"  balancing enabled: %s",bal));
      PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      if (eps->balance==EPS_BALANCE_ONESIDE || eps->balance==EPS_BALANCE_TWOSIDE) PetscCall(PetscViewerASCIIPrintf(viewer,", with its=%" PetscInt_FMT,eps->balance_its));
      if (eps->balance==EPS_BALANCE_TWOSIDE && eps->balance_cutoff!=0.0) PetscCall(PetscViewerASCIIPrintf(viewer," and cutoff=%g",(double)eps->balance_cutoff));
      PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: "));
    PetscCall(SlepcSNPrintfScalar(str,sizeof(str),eps->target,PETSC_FALSE));
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    if (!eps->which) PetscCall(PetscViewerASCIIPrintf(viewer,"not yet set\n"));
    else switch (eps->which) {
      case EPS_WHICH_USER:
        PetscCall(PetscViewerASCIIPrintf(viewer,"user defined\n"));
        break;
      case EPS_TARGET_MAGNITUDE:
        PetscCall(PetscViewerASCIIPrintf(viewer,"closest to target: %s (in magnitude)\n",str));
        break;
      case EPS_TARGET_REAL:
        PetscCall(PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the real axis)\n",str));
        break;
      case EPS_TARGET_IMAGINARY:
        PetscCall(PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the imaginary axis)\n",str));
        break;
      case EPS_LARGEST_MAGNITUDE:
        PetscCall(PetscViewerASCIIPrintf(viewer,"largest eigenvalues in magnitude\n"));
        break;
      case EPS_SMALLEST_MAGNITUDE:
        PetscCall(PetscViewerASCIIPrintf(viewer,"smallest eigenvalues in magnitude\n"));
        break;
      case EPS_LARGEST_REAL:
        PetscCall(PetscViewerASCIIPrintf(viewer,"largest real parts\n"));
        break;
      case EPS_SMALLEST_REAL:
        PetscCall(PetscViewerASCIIPrintf(viewer,"smallest real parts\n"));
        break;
      case EPS_LARGEST_IMAGINARY:
        PetscCall(PetscViewerASCIIPrintf(viewer,"largest imaginary parts\n"));
        break;
      case EPS_SMALLEST_IMAGINARY:
        PetscCall(PetscViewerASCIIPrintf(viewer,"smallest imaginary parts\n"));
        break;
      case EPS_ALL:
        if (eps->inta || eps->intb) PetscCall(PetscViewerASCIIPrintf(viewer,"all eigenvalues in interval [%g,%g]\n",(double)eps->inta,(double)eps->intb));
        else PetscCall(PetscViewerASCIIPrintf(viewer,"all eigenvalues in the region\n"));
        break;
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    if (eps->twosided && eps->problem_type!=EPS_HEP && eps->problem_type!=EPS_GHEP) PetscCall(PetscViewerASCIIPrintf(viewer,"  using two-sided variant (for left eigenvectors)\n"));
    if (eps->purify) PetscCall(PetscViewerASCIIPrintf(viewer,"  postprocessing eigenvectors with purification\n"));
    if (eps->trueres) PetscCall(PetscViewerASCIIPrintf(viewer,"  computing true residuals explicitly\n"));
    if (eps->trackall) PetscCall(PetscViewerASCIIPrintf(viewer,"  computing all residuals (for tracking convergence)\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %" PetscInt_FMT "\n",eps->nev));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %" PetscInt_FMT "\n",eps->ncv));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %" PetscInt_FMT "\n",eps->mpd));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %" PetscInt_FMT "\n",eps->max_it));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)eps->tol));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  convergence test: "));
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    switch (eps->conv) {
    case EPS_CONV_ABS:
      PetscCall(PetscViewerASCIIPrintf(viewer,"absolute\n"));break;
    case EPS_CONV_REL:
      PetscCall(PetscViewerASCIIPrintf(viewer,"relative to the eigenvalue\n"));break;
    case EPS_CONV_NORM:
      PetscCall(PetscViewerASCIIPrintf(viewer,"relative to the eigenvalue and matrix norms\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  computed matrix norms: norm(A)=%g",(double)eps->nrma));
      if (eps->isgeneralized) PetscCall(PetscViewerASCIIPrintf(viewer,", norm(B)=%g",(double)eps->nrmb));
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      break;
    case EPS_CONV_USER:
      PetscCall(PetscViewerASCIIPrintf(viewer,"user-defined\n"));break;
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    if (eps->nini) PetscCall(PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %" PetscInt_FMT "\n",PetscAbs(eps->nini)));
    if (eps->ninil) PetscCall(PetscViewerASCIIPrintf(viewer,"  dimension of user-provided left initial space: %" PetscInt_FMT "\n",PetscAbs(eps->ninil)));
    if (eps->nds) PetscCall(PetscViewerASCIIPrintf(viewer,"  dimension of user-provided deflation space: %" PetscInt_FMT "\n",PetscAbs(eps->nds)));
  } else PetscTryTypeMethod(eps,view,viewer);
  PetscCall(PetscObjectTypeCompareAny((PetscObject)eps,&isexternal,EPSARPACK,EPSBLOPEX,EPSELEMENTAL,EPSFEAST,EPSPRIMME,EPSSCALAPACK,EPSELPA,EPSEVSL,EPSTRLAN,""));
  if (!isexternal) {
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
    if (!eps->V) PetscCall(EPSGetBV(eps,&eps->V));
    PetscCall(BVView(eps->V,viewer));
    if (eps->rg) {
      PetscCall(RGIsTrivial(eps->rg,&istrivial));
      if (!istrivial) PetscCall(RGView(eps->rg,viewer));
    }
    if (eps->useds) {
      if (!eps->ds) PetscCall(EPSGetDS(eps,&eps->ds));
      PetscCall(DSView(eps->ds,viewer));
    }
    PetscCall(PetscViewerPopFormat(viewer));
  }
  if (!eps->st) PetscCall(EPSGetST(eps,&eps->st));
  PetscCall(STView(eps->st,viewer));
  PetscFunctionReturn(0);
}

/*@C
   EPSViewFromOptions - View from options

   Collective on EPS

   Input Parameters:
+  eps  - the eigensolver context
.  obj  - optional object
-  name - command line option

   Level: intermediate

.seealso: EPSView(), EPSCreate()
@*/
PetscErrorCode EPSViewFromOptions(EPS eps,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)eps,obj,name));
  PetscFunctionReturn(0);
}

/*@C
   EPSConvergedReasonView - Displays the reason an EPS solve converged or diverged.

   Collective on eps

   Input Parameters:
+  eps - the eigensolver context
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -eps_converged_reason - print reason for convergence, and number of iterations

   Note:
   To change the format of the output call PetscViewerPushFormat(viewer,format) before
   this call. Use PETSC_VIEWER_DEFAULT for the default, use PETSC_VIEWER_FAILED to only
   display a reason if it fails. The latter can be set in the command line with
   -eps_converged_reason ::failed

   Level: intermediate

.seealso: EPSSetConvergenceTest(), EPSSetTolerances(), EPSGetIterationNumber(), EPSConvergedReasonViewFromOptions()
@*/
PetscErrorCode EPSConvergedReasonView(EPS eps,PetscViewer viewer)
{
  PetscBool         isAscii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)eps));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii));
  if (isAscii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    PetscCall(PetscViewerASCIIAddTab(viewer,((PetscObject)eps)->tablevel));
    if (eps->reason > 0 && format != PETSC_VIEWER_FAILED) PetscCall(PetscViewerASCIIPrintf(viewer,"%s Linear eigensolve converged (%" PetscInt_FMT " eigenpair%s) due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)eps)->prefix?((PetscObject)eps)->prefix:"",eps->nconv,(eps->nconv>1)?"s":"",EPSConvergedReasons[eps->reason],eps->its));
    else if (eps->reason <= 0) PetscCall(PetscViewerASCIIPrintf(viewer,"%s Linear eigensolve did not converge due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)eps)->prefix?((PetscObject)eps)->prefix:"",EPSConvergedReasons[eps->reason],eps->its));
    PetscCall(PetscViewerASCIISubtractTab(viewer,((PetscObject)eps)->tablevel));
  }
  PetscFunctionReturn(0);
}

/*@
   EPSConvergedReasonViewFromOptions - Processes command line options to determine if/how
   the EPS converged reason is to be viewed.

   Collective on eps

   Input Parameter:
.  eps - the eigensolver context

   Level: developer

.seealso: EPSConvergedReasonView()
@*/
PetscErrorCode EPSConvergedReasonViewFromOptions(EPS eps)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)eps),((PetscObject)eps)->options,((PetscObject)eps)->prefix,"-eps_converged_reason",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(EPSConvergedReasonView(eps,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSErrorView_ASCII(EPS eps,EPSErrorType etype,PetscViewer viewer)
{
  PetscReal      error;
  PetscInt       i,j,k,nvals;

  PetscFunctionBegin;
  nvals = (eps->which==EPS_ALL)? eps->nconv: eps->nev;
  if (eps->which!=EPS_ALL && eps->nconv<nvals) {
    PetscCall(PetscViewerASCIIPrintf(viewer," Problem: less than %" PetscInt_FMT " eigenvalues converged\n\n",eps->nev));
    PetscFunctionReturn(0);
  }
  if (eps->which==EPS_ALL && !nvals) {
    PetscCall(PetscViewerASCIIPrintf(viewer," No eigenvalues have been found\n\n"));
    PetscFunctionReturn(0);
  }
  for (i=0;i<nvals;i++) {
    PetscCall(EPSComputeError(eps,i,etype,&error));
    if (error>=5.0*eps->tol) {
      PetscCall(PetscViewerASCIIPrintf(viewer," Problem: some of the first %" PetscInt_FMT " relative errors are higher than the tolerance\n\n",nvals));
      PetscFunctionReturn(0);
    }
  }
  if (eps->which==EPS_ALL) PetscCall(PetscViewerASCIIPrintf(viewer," Found %" PetscInt_FMT " eigenvalues, all of them computed up to the required tolerance:",nvals));
  else PetscCall(PetscViewerASCIIPrintf(viewer," All requested eigenvalues computed up to the required tolerance:"));
  for (i=0;i<=(nvals-1)/8;i++) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"\n     "));
    for (j=0;j<PetscMin(8,nvals-8*i);j++) {
      k = eps->perm[8*i+j];
      PetscCall(SlepcPrintEigenvalueASCII(viewer,eps->eigr[k],eps->eigi[k]));
      if (8*i+j+1<nvals) PetscCall(PetscViewerASCIIPrintf(viewer,", "));
    }
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,"\n\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSErrorView_DETAIL(EPS eps,EPSErrorType etype,PetscViewer viewer)
{
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i;
  char           ex[30],sep[]=" ---------------------- --------------------\n";

  PetscFunctionBegin;
  if (!eps->nconv) PetscFunctionReturn(0);
  switch (etype) {
    case EPS_ERROR_ABSOLUTE:
      PetscCall(PetscSNPrintf(ex,sizeof(ex),"   ||Ax-k%sx||",eps->isgeneralized?"B":""));
      break;
    case EPS_ERROR_RELATIVE:
      PetscCall(PetscSNPrintf(ex,sizeof(ex),"||Ax-k%sx||/||kx||",eps->isgeneralized?"B":""));
      break;
    case EPS_ERROR_BACKWARD:
      PetscCall(PetscSNPrintf(ex,sizeof(ex),"    eta(x,k)"));
      break;
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,"%s            k             %s\n%s",sep,ex,sep));
  for (i=0;i<eps->nconv;i++) {
    PetscCall(EPSGetEigenpair(eps,i,&kr,&ki,NULL,NULL));
    PetscCall(EPSComputeError(eps,i,etype,&error));
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(kr);
    im = PetscImaginaryPart(kr);
#else
    re = kr;
    im = ki;
#endif
    if (im!=0.0) PetscCall(PetscViewerASCIIPrintf(viewer,"  % 9f%+9fi      %12g\n",(double)re,(double)im,(double)error));
    else PetscCall(PetscViewerASCIIPrintf(viewer,"    % 12f           %12g\n",(double)re,(double)error));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,"%s",sep));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSErrorView_MATLAB(EPS eps,EPSErrorType etype,PetscViewer viewer)
{
  PetscReal      error;
  PetscInt       i;
  const char     *name;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)eps,&name));
  PetscCall(PetscViewerASCIIPrintf(viewer,"Error_%s = [\n",name));
  for (i=0;i<eps->nconv;i++) {
    PetscCall(EPSComputeError(eps,i,etype,&error));
    PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)error));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,"];\n"));
  PetscFunctionReturn(0);
}

/*@C
   EPSErrorView - Displays the errors associated with the computed solution
   (as well as the eigenvalues).

   Collective on eps

   Input Parameters:
+  eps    - the eigensolver context
.  etype  - error type
-  viewer - optional visualization context

   Options Database Keys:
+  -eps_error_absolute - print absolute errors of each eigenpair
.  -eps_error_relative - print relative errors of each eigenpair
-  -eps_error_backward - print backward errors of each eigenpair

   Notes:
   By default, this function checks the error of all eigenpairs and prints
   the eigenvalues if all of them are below the requested tolerance.
   If the viewer has format=PETSC_VIEWER_ASCII_INFO_DETAIL then a table with
   eigenvalues and corresponding errors is printed.

   Level: intermediate

.seealso: EPSSolve(), EPSValuesView(), EPSVectorsView()
@*/
PetscErrorCode EPSErrorView(EPS eps,EPSErrorType etype,PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)eps),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(eps,1,viewer,3);
  EPSCheckSolved(eps,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (!isascii) PetscFunctionReturn(0);

  PetscCall(PetscViewerGetFormat(viewer,&format));
  switch (format) {
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_ASCII_INFO:
      PetscCall(EPSErrorView_ASCII(eps,etype,viewer));
      break;
    case PETSC_VIEWER_ASCII_INFO_DETAIL:
      PetscCall(EPSErrorView_DETAIL(eps,etype,viewer));
      break;
    case PETSC_VIEWER_ASCII_MATLAB:
      PetscCall(EPSErrorView_MATLAB(eps,etype,viewer));
      break;
    default:
      PetscCall(PetscInfo(eps,"Unsupported viewer format %s\n",PetscViewerFormats[format]));
  }
  PetscFunctionReturn(0);
}

/*@
   EPSErrorViewFromOptions - Processes command line options to determine if/how
   the errors of the computed solution are to be viewed.

   Collective on eps

   Input Parameter:
.  eps - the eigensolver context

   Level: developer

.seealso: EPSErrorView()
@*/
PetscErrorCode EPSErrorViewFromOptions(EPS eps)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)eps),((PetscObject)eps)->options,((PetscObject)eps)->prefix,"-eps_error_absolute",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(EPSErrorView(eps,EPS_ERROR_ABSOLUTE,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)eps),((PetscObject)eps)->options,((PetscObject)eps)->prefix,"-eps_error_relative",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)eps),((PetscObject)eps)->options,((PetscObject)eps)->prefix,"-eps_error_backward",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(EPSErrorView(eps,EPS_ERROR_BACKWARD,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSValuesView_DRAW(EPS eps,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscDrawSP    drawsp;
  PetscReal      re,im;
  PetscInt       i,k;

  PetscFunctionBegin;
  if (!eps->nconv) PetscFunctionReturn(0);
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawSetTitle(draw,"Computed Eigenvalues"));
  PetscCall(PetscDrawSPCreate(draw,1,&drawsp));
  for (i=0;i<eps->nconv;i++) {
    k = eps->perm[i];
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(eps->eigr[k]);
    im = PetscImaginaryPart(eps->eigr[k]);
#else
    re = eps->eigr[k];
    im = eps->eigi[k];
#endif
    PetscCall(PetscDrawSPAddPoint(drawsp,&re,&im));
  }
  PetscCall(PetscDrawSPDraw(drawsp,PETSC_TRUE));
  PetscCall(PetscDrawSPSave(drawsp));
  PetscCall(PetscDrawSPDestroy(&drawsp));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSValuesView_BINARY(EPS eps,PetscViewer viewer)
{
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt       i,k;
  PetscComplex   *ev;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_COMPLEX)
  PetscCall(PetscMalloc1(eps->nconv,&ev));
  for (i=0;i<eps->nconv;i++) {
    k = eps->perm[i];
#if defined(PETSC_USE_COMPLEX)
    ev[i] = eps->eigr[k];
#else
    ev[i] = PetscCMPLX(eps->eigr[k],eps->eigi[k]);
#endif
  }
  PetscCall(PetscViewerBinaryWrite(viewer,ev,eps->nconv,PETSC_COMPLEX));
  PetscCall(PetscFree(ev));
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode EPSValuesView_HDF5(EPS eps,PetscViewer viewer)
{
  PetscInt       i,k,n,N;
  PetscMPIInt    rank;
  Vec            v;
  char           vname[30];
  const char     *ename;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)eps),&rank));
  N = eps->nconv;
  n = rank? 0: N;
  /* create a vector containing the eigenvalues */
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)eps),n,N,&v));
  PetscCall(PetscObjectGetName((PetscObject)eps,&ename));
  PetscCall(PetscSNPrintf(vname,sizeof(vname),"eigr_%s",ename));
  PetscCall(PetscObjectSetName((PetscObject)v,vname));
  if (!rank) {
    for (i=0;i<eps->nconv;i++) {
      k = eps->perm[i];
      PetscCall(VecSetValue(v,i,eps->eigr[k],INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscCall(VecView(v,viewer));
#if !defined(PETSC_USE_COMPLEX)
  /* in real scalars write the imaginary part as a separate vector */
  PetscCall(PetscSNPrintf(vname,sizeof(vname),"eigi_%s",ename));
  PetscCall(PetscObjectSetName((PetscObject)v,vname));
  if (!rank) {
    for (i=0;i<eps->nconv;i++) {
      k = eps->perm[i];
      PetscCall(VecSetValue(v,i,eps->eigi[k],INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscCall(VecView(v,viewer));
#endif
  PetscCall(VecDestroy(&v));
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode EPSValuesView_ASCII(EPS eps,PetscViewer viewer)
{
  PetscInt       i,k;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer,"Eigenvalues = \n"));
  for (i=0;i<eps->nconv;i++) {
    k = eps->perm[i];
    PetscCall(PetscViewerASCIIPrintf(viewer,"   "));
    PetscCall(SlepcPrintEigenvalueASCII(viewer,eps->eigr[k],eps->eigi[k]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSValuesView_MATLAB(EPS eps,PetscViewer viewer)
{
  PetscInt       i,k;
  PetscReal      re,im;
  const char     *name;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)eps,&name));
  PetscCall(PetscViewerASCIIPrintf(viewer,"Lambda_%s = [\n",name));
  for (i=0;i<eps->nconv;i++) {
    k = eps->perm[i];
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(eps->eigr[k]);
    im = PetscImaginaryPart(eps->eigr[k]);
#else
    re = eps->eigr[k];
    im = eps->eigi[k];
#endif
    if (im!=0.0) PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e%+18.16ei\n",(double)re,(double)im));
    else PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)re));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,"];\n"));
  PetscFunctionReturn(0);
}

/*@C
   EPSValuesView - Displays the computed eigenvalues in a viewer.

   Collective on eps

   Input Parameters:
+  eps    - the eigensolver context
-  viewer - the viewer

   Options Database Key:
.  -eps_view_values - print computed eigenvalues

   Level: intermediate

.seealso: EPSSolve(), EPSVectorsView(), EPSErrorView()
@*/
PetscErrorCode EPSValuesView(EPS eps,PetscViewer viewer)
{
  PetscBool         isascii,isdraw,isbinary;
  PetscViewerFormat format;
#if defined(PETSC_HAVE_HDF5)
  PetscBool         ishdf5;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)eps),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(eps,1,viewer,2);
  EPSCheckSolved(eps,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isdraw) PetscCall(EPSValuesView_DRAW(eps,viewer));
  else if (isbinary) PetscCall(EPSValuesView_BINARY(eps,viewer));
#if defined(PETSC_HAVE_HDF5)
  else if (ishdf5) PetscCall(EPSValuesView_HDF5(eps,viewer));
#endif
  else if (isascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    switch (format) {
      case PETSC_VIEWER_DEFAULT:
      case PETSC_VIEWER_ASCII_INFO:
      case PETSC_VIEWER_ASCII_INFO_DETAIL:
        PetscCall(EPSValuesView_ASCII(eps,viewer));
        break;
      case PETSC_VIEWER_ASCII_MATLAB:
        PetscCall(EPSValuesView_MATLAB(eps,viewer));
        break;
      default:
        PetscCall(PetscInfo(eps,"Unsupported viewer format %s\n",PetscViewerFormats[format]));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   EPSValuesViewFromOptions - Processes command line options to determine if/how
   the computed eigenvalues are to be viewed.

   Collective on eps

   Input Parameters:
.  eps - the eigensolver context

   Level: developer

.seealso: EPSValuesView()
@*/
PetscErrorCode EPSValuesViewFromOptions(EPS eps)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)eps),((PetscObject)eps)->options,((PetscObject)eps)->prefix,"-eps_view_values",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(EPSValuesView(eps,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   EPSVectorsView - Outputs computed eigenvectors to a viewer.

   Collective on eps

   Input Parameters:
+  eps    - the eigensolver context
-  viewer - the viewer

   Options Database Key:
.  -eps_view_vectors - output eigenvectors.

   Notes:
   If PETSc was configured with real scalars, complex conjugate eigenvectors
   will be viewed as two separate real vectors, one containing the real part
   and another one containing the imaginary part.

   If left eigenvectors were computed with a two-sided eigensolver, the right
   and left eigenvectors are interleaved, that is, the vectors are output in
   the following order X0, Y0, X1, Y1, X2, Y2, ...

   Level: intermediate

.seealso: EPSSolve(), EPSValuesView(), EPSErrorView()
@*/
PetscErrorCode EPSVectorsView(EPS eps,PetscViewer viewer)
{
  PetscInt       i,k;
  Vec            xr,xi=NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)eps),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(eps,1,viewer,2);
  EPSCheckSolved(eps,1);
  if (eps->nconv) {
    PetscCall(EPSComputeVectors(eps));
    PetscCall(BVCreateVec(eps->V,&xr));
#if !defined(PETSC_USE_COMPLEX)
    PetscCall(BVCreateVec(eps->V,&xi));
#endif
    for (i=0;i<eps->nconv;i++) {
      k = eps->perm[i];
      PetscCall(BV_GetEigenvector(eps->V,k,eps->eigi[k],xr,xi));
      PetscCall(SlepcViewEigenvector(viewer,xr,xi,"X",i,(PetscObject)eps));
      if (eps->twosided) {
        PetscCall(BV_GetEigenvector(eps->W,k,eps->eigi[k],xr,xi));
        PetscCall(SlepcViewEigenvector(viewer,xr,xi,"Y",i,(PetscObject)eps));
      }
    }
    PetscCall(VecDestroy(&xr));
#if !defined(PETSC_USE_COMPLEX)
    PetscCall(VecDestroy(&xi));
#endif
  }
  PetscFunctionReturn(0);
}

/*@
   EPSVectorsViewFromOptions - Processes command line options to determine if/how
   the computed eigenvectors are to be viewed.

   Collective on eps

   Input Parameter:
.  eps - the eigensolver context

   Level: developer

.seealso: EPSVectorsView()
@*/
PetscErrorCode EPSVectorsViewFromOptions(EPS eps)
{
  PetscViewer       viewer;
  PetscBool         flg = PETSC_FALSE;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)eps),((PetscObject)eps)->options,((PetscObject)eps)->prefix,"-eps_view_vectors",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(EPSVectorsView(eps,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}
