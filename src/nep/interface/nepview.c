/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   NEP routines related to various viewers
*/

#include <slepc/private/nepimpl.h>      /*I "slepcnep.h" I*/
#include <slepc/private/bvimpl.h>
#include <petscdraw.h>

/*@C
   NEPView - Prints the NEP data structure.

   Collective on nep

   Input Parameters:
+  nep - the nonlinear eigenproblem solver context
-  viewer - optional visualization context

   Options Database Key:
.  -nep_view -  Calls NEPView() at end of NEPSolve()

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

.seealso: FNView()
@*/
PetscErrorCode NEPView(NEP nep,PetscViewer viewer)
{
  const char     *type=NULL;
  char           str[50];
  PetscInt       i;
  PetscBool      isascii,istrivial;
  PetscViewer    sviewer;
  MPI_Comm       child;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)nep),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(nep,1,viewer,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)nep,viewer));
    if (nep->ops->view) {
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ((*nep->ops->view)(nep,viewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
    if (nep->problem_type) {
      switch (nep->problem_type) {
        case NEP_GENERAL:  type = "general nonlinear eigenvalue problem"; break;
        case NEP_RATIONAL: type = "rational eigenvalue problem"; break;
      }
    } else type = "not yet set";
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type));
    if (nep->fui) {
      switch (nep->fui) {
      case NEP_USER_INTERFACE_CALLBACK:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  nonlinear operator from user callbacks\n"));
        break;
      case NEP_USER_INTERFACE_SPLIT:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  nonlinear operator in split form\n"));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"    number of terms: %" PetscInt_FMT "\n",nep->nt));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"    nonzero pattern of the matrices: %s\n",MatStructures[nep->mstr]));
        break;
      }
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  nonlinear operator not specified yet\n"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: "));
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),nep->target,PETSC_FALSE));
    if (!nep->which) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"not yet set\n"));
    } else switch (nep->which) {
      case NEP_WHICH_USER:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"user defined\n"));
        break;
      case NEP_TARGET_MAGNITUDE:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"closest to target: %s (in magnitude)\n",str));
        break;
      case NEP_TARGET_REAL:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the real axis)\n",str));
        break;
      case NEP_TARGET_IMAGINARY:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the imaginary axis)\n",str));
        break;
      case NEP_LARGEST_MAGNITUDE:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"largest eigenvalues in magnitude\n"));
        break;
      case NEP_SMALLEST_MAGNITUDE:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"smallest eigenvalues in magnitude\n"));
        break;
      case NEP_LARGEST_REAL:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"largest real parts\n"));
        break;
      case NEP_SMALLEST_REAL:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"smallest real parts\n"));
        break;
      case NEP_LARGEST_IMAGINARY:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"largest imaginary parts\n"));
        break;
      case NEP_SMALLEST_IMAGINARY:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"smallest imaginary parts\n"));
        break;
      case NEP_ALL:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"all eigenvalues in the region\n"));
        break;
    }
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    if (nep->twosided) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using two-sided variant (for left eigenvectors)\n"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %" PetscInt_FMT "\n",nep->nev));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %" PetscInt_FMT "\n",nep->ncv));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %" PetscInt_FMT "\n",nep->mpd));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %" PetscInt_FMT "\n",nep->max_it));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)nep->tol));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  convergence test: "));
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    switch (nep->conv) {
    case NEP_CONV_ABS:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"absolute\n"));break;
    case NEP_CONV_REL:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"relative to the eigenvalue\n"));break;
    case NEP_CONV_NORM:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"relative to the matrix norms\n"));
      if (nep->nrma) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  computed matrix norms: %g",(double)nep->nrma[0]));
        for (i=1;i<nep->nt;i++) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,", %g",(double)nep->nrma[i]));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      break;
    case NEP_CONV_USER:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"user-defined\n"));break;
    }
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    if (nep->refine) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  iterative refinement: %s, with %s scheme\n",NEPRefineTypes[nep->refine],NEPRefineSchemes[nep->scheme]));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  refinement stopping criterion: tol=%g, its=%" PetscInt_FMT "\n",(double)nep->rtol,nep->rits));
      if (nep->npart>1) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  splitting communicator in %" PetscInt_FMT " partitions for refinement\n",nep->npart));
      }
    }
    if (nep->nini) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %" PetscInt_FMT "\n",PetscAbs(nep->nini)));
    }
  } else {
    if (nep->ops->view) {
      CHKERRQ((*nep->ops->view)(nep,viewer));
    }
  }
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
  if (!nep->V) CHKERRQ(NEPGetBV(nep,&nep->V));
  CHKERRQ(BVView(nep->V,viewer));
  if (!nep->rg) CHKERRQ(NEPGetRG(nep,&nep->rg));
  CHKERRQ(RGIsTrivial(nep->rg,&istrivial));
  if (!istrivial) CHKERRQ(RGView(nep->rg,viewer));
  if (nep->useds) {
    if (!nep->ds) CHKERRQ(NEPGetDS(nep,&nep->ds));
    CHKERRQ(DSView(nep->ds,viewer));
  }
  CHKERRQ(PetscViewerPopFormat(viewer));
  if (nep->refine!=NEP_REFINE_NONE) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    if (nep->npart>1) {
      if (nep->refinesubc->color==0) {
        CHKERRQ(PetscSubcommGetChild(nep->refinesubc,&child));
        CHKERRQ(PetscViewerASCIIGetStdout(child,&sviewer));
        CHKERRQ(KSPView(nep->refineksp,sviewer));
      }
    } else {
      CHKERRQ(KSPView(nep->refineksp,viewer));
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
   NEPViewFromOptions - View from options

   Collective on NEP

   Input Parameters:
+  nep  - the nonlinear eigensolver context
.  obj  - optional object
-  name - command line option

   Level: intermediate

.seealso: NEPView(), NEPCreate()
@*/
PetscErrorCode NEPViewFromOptions(NEP nep,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)nep,obj,name));
  PetscFunctionReturn(0);
}

/*@C
   NEPConvergedReasonView - Displays the reason a NEP solve converged or diverged.

   Collective on nep

   Input Parameters:
+  nep - the nonlinear eigensolver context
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -nep_converged_reason - print reason for convergence, and number of iterations

   Note:
   To change the format of the output call PetscViewerPushFormat(viewer,format) before
   this call. Use PETSC_VIEWER_DEFAULT for the default, use PETSC_VIEWER_FAILED to only
   display a reason if it fails. The latter can be set in the command line with
   -nep_converged_reason ::failed

   Level: intermediate

.seealso: NEPSetConvergenceTest(), NEPSetTolerances(), NEPGetIterationNumber(), NEPConvergedReasonViewFromOptions(
@*/
PetscErrorCode NEPConvergedReasonView(NEP nep,PetscViewer viewer)
{
  PetscBool         isAscii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii));
  if (isAscii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)nep)->tablevel));
    if (nep->reason > 0 && format != PETSC_VIEWER_FAILED) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s Nonlinear eigensolve converged (%" PetscInt_FMT " eigenpair%s) due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)nep)->prefix?((PetscObject)nep)->prefix:"",nep->nconv,(nep->nconv>1)?"s":"",NEPConvergedReasons[nep->reason],nep->its));
    } else if (nep->reason <= 0) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s Nonlinear eigensolve did not converge due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)nep)->prefix?((PetscObject)nep)->prefix:"",NEPConvergedReasons[nep->reason],nep->its));
    }
    CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)nep)->tablevel));
  }
  PetscFunctionReturn(0);
}

/*@
   NEPConvergedReasonViewFromOptions - Processes command line options to determine if/how
   the NEP converged reason is to be viewed.

   Collective on nep

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Level: developer

.seealso: NEPConvergedReasonView()
@*/
PetscErrorCode NEPConvergedReasonViewFromOptions(NEP nep)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->options,((PetscObject)nep)->prefix,"-nep_converged_reason",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(NEPConvergedReasonView(nep,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPErrorView_ASCII(NEP nep,NEPErrorType etype,PetscViewer viewer)
{
  PetscReal      error;
  PetscInt       i,j,k,nvals;

  PetscFunctionBegin;
  nvals = (nep->which==NEP_ALL)? nep->nconv: nep->nev;
  if (nep->which!=NEP_ALL && nep->nconv<nvals) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Problem: less than %" PetscInt_FMT " eigenvalues converged\n\n",nep->nev));
    PetscFunctionReturn(0);
  }
  if (nep->which==NEP_ALL && !nvals) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," No eigenvalues have been found\n\n"));
    PetscFunctionReturn(0);
  }
  for (i=0;i<nvals;i++) {
    CHKERRQ(NEPComputeError(nep,i,etype,&error));
    if (error>=5.0*nep->tol) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer," Problem: some of the first %" PetscInt_FMT " relative errors are higher than the tolerance\n\n",nvals));
      PetscFunctionReturn(0);
    }
  }
  if (nep->which==NEP_ALL) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Found %" PetscInt_FMT " eigenvalues, all of them computed up to the required tolerance:",nvals));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," All requested eigenvalues computed up to the required tolerance:"));
  }
  for (i=0;i<=(nvals-1)/8;i++) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n     "));
    for (j=0;j<PetscMin(8,nvals-8*i);j++) {
      k = nep->perm[8*i+j];
      CHKERRQ(SlepcPrintEigenvalueASCII(viewer,nep->eigr[k],nep->eigi[k]));
      if (8*i+j+1<nvals) CHKERRQ(PetscViewerASCIIPrintf(viewer,", "));
    }
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPErrorView_DETAIL(NEP nep,NEPErrorType etype,PetscViewer viewer)
{
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i;
  char           ex[30],sep[]=" ---------------------- --------------------\n";

  PetscFunctionBegin;
  if (!nep->nconv) PetscFunctionReturn(0);
  switch (etype) {
    case NEP_ERROR_ABSOLUTE:
      CHKERRQ(PetscSNPrintf(ex,sizeof(ex),"    ||T(k)x||"));
      break;
    case NEP_ERROR_RELATIVE:
      CHKERRQ(PetscSNPrintf(ex,sizeof(ex)," ||T(k)x||/||kx||"));
      break;
    case NEP_ERROR_BACKWARD:
      CHKERRQ(PetscSNPrintf(ex,sizeof(ex),"    eta(x,k)"));
      break;
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s            k             %s\n%s",sep,ex,sep));
  for (i=0;i<nep->nconv;i++) {
    CHKERRQ(NEPGetEigenpair(nep,i,&kr,&ki,NULL,NULL));
    CHKERRQ(NEPComputeError(nep,i,etype,&error));
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(kr);
    im = PetscImaginaryPart(kr);
#else
    re = kr;
    im = ki;
#endif
    if (im!=0.0) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  % 9f%+9fi      %12g\n",(double)re,(double)im,(double)error));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"    % 12f           %12g\n",(double)re,(double)error));
    }
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s",sep));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPErrorView_MATLAB(NEP nep,NEPErrorType etype,PetscViewer viewer)
{
  PetscReal      error;
  PetscInt       i;
  const char     *name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)nep,&name));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Error_%s = [\n",name));
  for (i=0;i<nep->nconv;i++) {
    CHKERRQ(NEPComputeError(nep,i,etype,&error));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)error));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"];\n"));
  PetscFunctionReturn(0);
}

/*@C
   NEPErrorView - Displays the errors associated with the computed solution
   (as well as the eigenvalues).

   Collective on nep

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  etype  - error type
-  viewer - optional visualization context

   Options Database Key:
+  -nep_error_absolute - print absolute errors of each eigenpair
.  -nep_error_relative - print relative errors of each eigenpair
-  -nep_error_backward - print backward errors of each eigenpair

   Notes:
   By default, this function checks the error of all eigenpairs and prints
   the eigenvalues if all of them are below the requested tolerance.
   If the viewer has format=PETSC_VIEWER_ASCII_INFO_DETAIL then a table with
   eigenvalues and corresponding errors is printed.

   Level: intermediate

.seealso: NEPSolve(), NEPValuesView(), NEPVectorsView()
@*/
PetscErrorCode NEPErrorView(NEP nep,NEPErrorType etype,PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)nep),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(nep,1,viewer,3);
  NEPCheckSolved(nep,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (!isascii) PetscFunctionReturn(0);

  CHKERRQ(PetscViewerGetFormat(viewer,&format));
  switch (format) {
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_ASCII_INFO:
      CHKERRQ(NEPErrorView_ASCII(nep,etype,viewer));
      break;
    case PETSC_VIEWER_ASCII_INFO_DETAIL:
      CHKERRQ(NEPErrorView_DETAIL(nep,etype,viewer));
      break;
    case PETSC_VIEWER_ASCII_MATLAB:
      CHKERRQ(NEPErrorView_MATLAB(nep,etype,viewer));
      break;
    default:
      CHKERRQ(PetscInfo(nep,"Unsupported viewer format %s\n",PetscViewerFormats[format]));
  }
  PetscFunctionReturn(0);
}

/*@
   NEPErrorViewFromOptions - Processes command line options to determine if/how
   the errors of the computed solution are to be viewed.

   Collective on nep

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Level: developer

.seealso: NEPErrorView()
@*/
PetscErrorCode NEPErrorViewFromOptions(NEP nep)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->options,((PetscObject)nep)->prefix,"-nep_error_absolute",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_ABSOLUTE,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->options,((PetscObject)nep)->prefix,"-nep_error_relative",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->options,((PetscObject)nep)->prefix,"-nep_error_backward",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_BACKWARD,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPValuesView_DRAW(NEP nep,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscDrawSP    drawsp;
  PetscReal      re,im;
  PetscInt       i,k;

  PetscFunctionBegin;
  if (!nep->nconv) PetscFunctionReturn(0);
  CHKERRQ(PetscViewerDrawGetDraw(viewer,0,&draw));
  CHKERRQ(PetscDrawSetTitle(draw,"Computed Eigenvalues"));
  CHKERRQ(PetscDrawSPCreate(draw,1,&drawsp));
  for (i=0;i<nep->nconv;i++) {
    k = nep->perm[i];
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(nep->eigr[k]);
    im = PetscImaginaryPart(nep->eigr[k]);
#else
    re = nep->eigr[k];
    im = nep->eigi[k];
#endif
    CHKERRQ(PetscDrawSPAddPoint(drawsp,&re,&im));
  }
  CHKERRQ(PetscDrawSPDraw(drawsp,PETSC_TRUE));
  CHKERRQ(PetscDrawSPSave(drawsp));
  CHKERRQ(PetscDrawSPDestroy(&drawsp));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPValuesView_BINARY(NEP nep,PetscViewer viewer)
{
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt       i,k;
  PetscComplex   *ev;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_COMPLEX)
  CHKERRQ(PetscMalloc1(nep->nconv,&ev));
  for (i=0;i<nep->nconv;i++) {
    k = nep->perm[i];
#if defined(PETSC_USE_COMPLEX)
    ev[i] = nep->eigr[k];
#else
    ev[i] = PetscCMPLX(nep->eigr[k],nep->eigi[k]);
#endif
  }
  CHKERRQ(PetscViewerBinaryWrite(viewer,ev,nep->nconv,PETSC_COMPLEX));
  CHKERRQ(PetscFree(ev));
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode NEPValuesView_HDF5(NEP nep,PetscViewer viewer)
{
  PetscInt       i,k,n,N;
  PetscMPIInt    rank;
  Vec            v;
  char           vname[30];
  const char     *ename;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)nep),&rank));
  N = nep->nconv;
  n = rank? 0: N;
  /* create a vector containing the eigenvalues */
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)nep),n,N,&v));
  CHKERRQ(PetscObjectGetName((PetscObject)nep,&ename));
  CHKERRQ(PetscSNPrintf(vname,sizeof(vname),"eigr_%s",ename));
  CHKERRQ(PetscObjectSetName((PetscObject)v,vname));
  if (!rank) {
    for (i=0;i<nep->nconv;i++) {
      k = nep->perm[i];
      CHKERRQ(VecSetValue(v,i,nep->eigr[k],INSERT_VALUES));
    }
  }
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  CHKERRQ(VecView(v,viewer));
#if !defined(PETSC_USE_COMPLEX)
  /* in real scalars write the imaginary part as a separate vector */
  CHKERRQ(PetscSNPrintf(vname,sizeof(vname),"eigi_%s",ename));
  CHKERRQ(PetscObjectSetName((PetscObject)v,vname));
  if (!rank) {
    for (i=0;i<nep->nconv;i++) {
      k = nep->perm[i];
      CHKERRQ(VecSetValue(v,i,nep->eigi[k],INSERT_VALUES));
    }
  }
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  CHKERRQ(VecView(v,viewer));
#endif
  CHKERRQ(VecDestroy(&v));
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode NEPValuesView_ASCII(NEP nep,PetscViewer viewer)
{
  PetscInt       i,k;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Eigenvalues = \n"));
  for (i=0;i<nep->nconv;i++) {
    k = nep->perm[i];
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"   "));
    CHKERRQ(SlepcPrintEigenvalueASCII(viewer,nep->eigr[k],nep->eigi[k]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode NEPValuesView_MATLAB(NEP nep,PetscViewer viewer)
{
  PetscInt       i,k;
  PetscReal      re,im;
  const char     *name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)nep,&name));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Lambda_%s = [\n",name));
  for (i=0;i<nep->nconv;i++) {
    k = nep->perm[i];
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(nep->eigr[k]);
    im = PetscImaginaryPart(nep->eigr[k]);
#else
    re = nep->eigr[k];
    im = nep->eigi[k];
#endif
    if (im!=0.0) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e%+18.16ei\n",(double)re,(double)im));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)re));
    }
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"];\n"));
  PetscFunctionReturn(0);
}

/*@C
   NEPValuesView - Displays the computed eigenvalues in a viewer.

   Collective on nep

   Input Parameters:
+  nep    - the nonlinear eigensolver context
-  viewer - the viewer

   Options Database Key:
.  -nep_view_values - print computed eigenvalues

   Level: intermediate

.seealso: NEPSolve(), NEPVectorsView(), NEPErrorView()
@*/
PetscErrorCode NEPValuesView(NEP nep,PetscViewer viewer)
{
  PetscBool         isascii,isdraw,isbinary;
  PetscViewerFormat format;
#if defined(PETSC_HAVE_HDF5)
  PetscBool         ishdf5;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)nep),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(nep,1,viewer,2);
  NEPCheckSolved(nep,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
#if defined(PETSC_HAVE_HDF5)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isdraw) {
    CHKERRQ(NEPValuesView_DRAW(nep,viewer));
  } else if (isbinary) {
    CHKERRQ(NEPValuesView_BINARY(nep,viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    CHKERRQ(NEPValuesView_HDF5(nep,viewer));
#endif
  } else if (isascii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    switch (format) {
      case PETSC_VIEWER_DEFAULT:
      case PETSC_VIEWER_ASCII_INFO:
      case PETSC_VIEWER_ASCII_INFO_DETAIL:
        CHKERRQ(NEPValuesView_ASCII(nep,viewer));
        break;
      case PETSC_VIEWER_ASCII_MATLAB:
        CHKERRQ(NEPValuesView_MATLAB(nep,viewer));
        break;
      default:
        CHKERRQ(PetscInfo(nep,"Unsupported viewer format %s\n",PetscViewerFormats[format]));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   NEPValuesViewFromOptions - Processes command line options to determine if/how
   the computed eigenvalues are to be viewed.

   Collective on nep

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Level: developer

.seealso: NEPValuesView()
@*/
PetscErrorCode NEPValuesViewFromOptions(NEP nep)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->options,((PetscObject)nep)->prefix,"-nep_view_values",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(NEPValuesView(nep,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   NEPVectorsView - Outputs computed eigenvectors to a viewer.

   Collective on nep

   Input Parameters:
+  nep    - the nonlinear eigensolver context
-  viewer - the viewer

   Options Database Keys:
.  -nep_view_vectors - output eigenvectors.

   Notes:
   If PETSc was configured with real scalars, complex conjugate eigenvectors
   will be viewed as two separate real vectors, one containing the real part
   and another one containing the imaginary part.

   If left eigenvectors were computed with a two-sided eigensolver, the right
   and left eigenvectors are interleaved, that is, the vectors are output in
   the following order X0, Y0, X1, Y1, X2, Y2, ...

   Level: intermediate

.seealso: NEPSolve(), NEPValuesView(), NEPErrorView()
@*/
PetscErrorCode NEPVectorsView(NEP nep,PetscViewer viewer)
{
  PetscInt       i,k;
  Vec            xr,xi=NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)nep),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(nep,1,viewer,2);
  NEPCheckSolved(nep,1);
  if (nep->nconv) {
    CHKERRQ(NEPComputeVectors(nep));
    CHKERRQ(BVCreateVec(nep->V,&xr));
#if !defined(PETSC_USE_COMPLEX)
    CHKERRQ(BVCreateVec(nep->V,&xi));
#endif
    for (i=0;i<nep->nconv;i++) {
      k = nep->perm[i];
      CHKERRQ(BV_GetEigenvector(nep->V,k,nep->eigi[k],xr,xi));
      CHKERRQ(SlepcViewEigenvector(viewer,xr,xi,"X",i,(PetscObject)nep));
      if (nep->twosided) {
        CHKERRQ(BV_GetEigenvector(nep->W,k,nep->eigi[k],xr,xi));
        CHKERRQ(SlepcViewEigenvector(viewer,xr,xi,"Y",i,(PetscObject)nep));
      }
    }
    CHKERRQ(VecDestroy(&xr));
#if !defined(PETSC_USE_COMPLEX)
    CHKERRQ(VecDestroy(&xi));
#endif
  }
  PetscFunctionReturn(0);
}

/*@
   NEPVectorsViewFromOptions - Processes command line options to determine if/how
   the computed eigenvectors are to be viewed.

   Collective on nep

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Level: developer

.seealso: NEPVectorsView()
@*/
PetscErrorCode NEPVectorsViewFromOptions(NEP nep)
{
  PetscViewer       viewer;
  PetscBool         flg = PETSC_FALSE;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->options,((PetscObject)nep)->prefix,"-nep_view_vectors",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(NEPVectorsView(nep,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}
