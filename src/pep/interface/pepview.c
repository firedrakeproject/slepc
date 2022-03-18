/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   The PEP routines related to various viewers
*/

#include <slepc/private/pepimpl.h>      /*I "slepcpep.h" I*/
#include <slepc/private/bvimpl.h>
#include <petscdraw.h>

/*@C
   PEPView - Prints the PEP data structure.

   Collective on pep

   Input Parameters:
+  pep - the polynomial eigenproblem solver context
-  viewer - optional visualization context

   Options Database Key:
.  -pep_view -  Calls PEPView() at end of PEPSolve()

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
PetscErrorCode PEPView(PEP pep,PetscViewer viewer)
{
  const char     *type=NULL;
  char           str[50];
  PetscBool      isascii,islinear,istrivial;
  PetscInt       i;
  PetscViewer    sviewer;
  MPI_Comm       child;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pep),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(pep,1,viewer,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)pep,viewer));
    if (pep->ops->view) {
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ((*pep->ops->view)(pep,viewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
    if (pep->problem_type) {
      switch (pep->problem_type) {
        case PEP_GENERAL:    type = "general polynomial eigenvalue problem"; break;
        case PEP_HERMITIAN:  type = SLEPC_STRING_HERMITIAN " polynomial eigenvalue problem"; break;
        case PEP_HYPERBOLIC: type = "hyperbolic polynomial eigenvalue problem"; break;
        case PEP_GYROSCOPIC: type = "gyroscopic polynomial eigenvalue problem"; break;
      }
    } else type = "not yet set";
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  polynomial represented in %s basis\n",PEPBasisTypes[pep->basis]));
    switch (pep->scale) {
      case PEP_SCALE_NONE:
        break;
      case PEP_SCALE_SCALAR:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  parameter scaling enabled, with scaling factor=%g\n",(double)pep->sfactor));
        break;
      case PEP_SCALE_DIAGONAL:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  diagonal balancing enabled, with its=%" PetscInt_FMT " and lambda=%g\n",pep->sits,(double)pep->slambda));
        break;
      case PEP_SCALE_BOTH:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  parameter scaling & diagonal balancing enabled, with scaling factor=%g, its=%" PetscInt_FMT " and lambda=%g\n",(double)pep->sfactor,pep->sits,(double)pep->slambda));
        break;
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: "));
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),pep->target,PETSC_FALSE));
    if (!pep->which) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"not yet set\n"));
    } else switch (pep->which) {
      case PEP_WHICH_USER:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"user defined\n"));
        break;
      case PEP_TARGET_MAGNITUDE:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"closest to target: %s (in magnitude)\n",str));
        break;
      case PEP_TARGET_REAL:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the real axis)\n",str));
        break;
      case PEP_TARGET_IMAGINARY:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the imaginary axis)\n",str));
        break;
      case PEP_LARGEST_MAGNITUDE:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"largest eigenvalues in magnitude\n"));
        break;
      case PEP_SMALLEST_MAGNITUDE:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"smallest eigenvalues in magnitude\n"));
        break;
      case PEP_LARGEST_REAL:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"largest real parts\n"));
        break;
      case PEP_SMALLEST_REAL:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"smallest real parts\n"));
        break;
      case PEP_LARGEST_IMAGINARY:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"largest imaginary parts\n"));
        break;
      case PEP_SMALLEST_IMAGINARY:
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"smallest imaginary parts\n"));
        break;
      case PEP_ALL:
        if (pep->inta || pep->intb) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"all eigenvalues in interval [%g,%g]\n",(double)pep->inta,(double)pep->intb));
        } else {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"all eigenvalues in the region\n"));
        }
        break;
    }
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %" PetscInt_FMT "\n",pep->nev));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %" PetscInt_FMT "\n",pep->ncv));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %" PetscInt_FMT "\n",pep->mpd));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %" PetscInt_FMT "\n",pep->max_it));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)pep->tol));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  convergence test: "));
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    switch (pep->conv) {
    case PEP_CONV_ABS:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"absolute\n"));break;
    case PEP_CONV_REL:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"relative to the eigenvalue\n"));break;
    case PEP_CONV_NORM:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"relative to the matrix norms\n"));
      if (pep->nrma) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  computed matrix norms: %g",(double)pep->nrma[0]));
        for (i=1;i<pep->nmat;i++) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,", %g",(double)pep->nrma[i]));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
      }
      break;
    case PEP_CONV_USER:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"user-defined\n"));break;
    }
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  extraction type: %s\n",PEPExtractTypes[pep->extract]));
    if (pep->refine) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  iterative refinement: %s, with %s scheme\n",PEPRefineTypes[pep->refine],PEPRefineSchemes[pep->scheme]));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  refinement stopping criterion: tol=%g, its=%" PetscInt_FMT "\n",(double)pep->rtol,pep->rits));
      if (pep->npart>1) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  splitting communicator in %" PetscInt_FMT " partitions for refinement\n",pep->npart));
      }
    }
    if (pep->nini) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %" PetscInt_FMT "\n",PetscAbs(pep->nini)));
    }
  } else {
    if (pep->ops->view) {
      CHKERRQ((*pep->ops->view)(pep,viewer));
    }
  }
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
  if (!pep->V) CHKERRQ(PEPGetBV(pep,&pep->V));
  CHKERRQ(BVView(pep->V,viewer));
  if (!pep->rg) CHKERRQ(PEPGetRG(pep,&pep->rg));
  CHKERRQ(RGIsTrivial(pep->rg,&istrivial));
  if (!istrivial) CHKERRQ(RGView(pep->rg,viewer));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&islinear));
  if (!islinear) {
    if (!pep->ds) CHKERRQ(PEPGetDS(pep,&pep->ds));
    CHKERRQ(DSView(pep->ds,viewer));
  }
  CHKERRQ(PetscViewerPopFormat(viewer));
  if (!pep->st) CHKERRQ(PEPGetST(pep,&pep->st));
  CHKERRQ(STView(pep->st,viewer));
  if (pep->refine!=PEP_REFINE_NONE) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    if (pep->npart>1) {
      if (pep->refinesubc->color==0) {
        CHKERRQ(PetscSubcommGetChild(pep->refinesubc,&child));
        CHKERRQ(PetscViewerASCIIGetStdout(child,&sviewer));
        CHKERRQ(KSPView(pep->refineksp,sviewer));
      }
    } else {
      CHKERRQ(KSPView(pep->refineksp,viewer));
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
   PEPViewFromOptions - View from options

   Collective on PEP

   Input Parameters:
+  pep  - the eigensolver context
.  obj  - optional object
-  name - command line option

   Level: intermediate

.seealso: PEPView(), PEPCreate()
@*/
PetscErrorCode PEPViewFromOptions(PEP pep,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)pep,obj,name));
  PetscFunctionReturn(0);
}

/*@C
   PEPConvergedReasonView - Displays the reason a PEP solve converged or diverged.

   Collective on pep

   Input Parameters:
+  pep - the eigensolver context
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -pep_converged_reason - print reason for convergence, and number of iterations

   Note:
   To change the format of the output call PetscViewerPushFormat(viewer,format) before
   this call. Use PETSC_VIEWER_DEFAULT for the default, use PETSC_VIEWER_FAILED to only
   display a reason if it fails. The latter can be set in the command line with
   -pep_converged_reason ::failed

   Level: intermediate

.seealso: PEPSetConvergenceTest(), PEPSetTolerances(), PEPGetIterationNumber(), PEPConvergedReasonViewFromOptions()
@*/
PetscErrorCode PEPConvergedReasonView(PEP pep,PetscViewer viewer)
{
  PetscBool         isAscii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pep));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii));
  if (isAscii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)pep)->tablevel));
    if (pep->reason > 0 && format != PETSC_VIEWER_FAILED) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s Polynomial eigensolve converged (%" PetscInt_FMT " eigenpair%s) due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)pep)->prefix?((PetscObject)pep)->prefix:"",pep->nconv,(pep->nconv>1)?"s":"",PEPConvergedReasons[pep->reason],pep->its));
    } else if (pep->reason <= 0) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s Polynomial eigensolve did not converge due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)pep)->prefix?((PetscObject)pep)->prefix:"",PEPConvergedReasons[pep->reason],pep->its));
    }
    CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)pep)->tablevel));
  }
  PetscFunctionReturn(0);
}

/*@
   PEPConvergedReasonViewFromOptions - Processes command line options to determine if/how
   the PEP converged reason is to be viewed.

   Collective on pep

   Input Parameter:
.  pep - the eigensolver context

   Level: developer

.seealso: PEPConvergedReasonView()
@*/
PetscErrorCode PEPConvergedReasonViewFromOptions(PEP pep)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)pep),((PetscObject)pep)->options,((PetscObject)pep)->prefix,"-pep_converged_reason",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PEPConvergedReasonView(pep,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPErrorView_ASCII(PEP pep,PEPErrorType etype,PetscViewer viewer)
{
  PetscReal      error;
  PetscInt       i,j,k,nvals;

  PetscFunctionBegin;
  nvals = (pep->which==PEP_ALL)? pep->nconv: pep->nev;
  if (pep->which!=PEP_ALL && pep->nconv<pep->nev) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Problem: less than %" PetscInt_FMT " eigenvalues converged\n\n",pep->nev));
    PetscFunctionReturn(0);
  }
  if (pep->which==PEP_ALL && !nvals) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," No eigenvalues have been found\n\n"));
    PetscFunctionReturn(0);
  }
  for (i=0;i<nvals;i++) {
    CHKERRQ(PEPComputeError(pep,i,etype,&error));
    if (error>=5.0*pep->tol) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer," Problem: some of the first %" PetscInt_FMT " relative errors are higher than the tolerance\n\n",nvals));
      PetscFunctionReturn(0);
    }
  }
  if (pep->which==PEP_ALL) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Found %" PetscInt_FMT " eigenvalues, all of them computed up to the required tolerance:",nvals));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," All requested eigenvalues computed up to the required tolerance:"));
  }
  for (i=0;i<=(nvals-1)/8;i++) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n     "));
    for (j=0;j<PetscMin(8,nvals-8*i);j++) {
      k = pep->perm[8*i+j];
      CHKERRQ(SlepcPrintEigenvalueASCII(viewer,pep->eigr[k],pep->eigi[k]));
      if (8*i+j+1<nvals) CHKERRQ(PetscViewerASCIIPrintf(viewer,", "));
    }
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPErrorView_DETAIL(PEP pep,PEPErrorType etype,PetscViewer viewer)
{
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i;
  char           ex[30],sep[]=" ---------------------- --------------------\n";

  PetscFunctionBegin;
  if (!pep->nconv) PetscFunctionReturn(0);
  switch (etype) {
    case PEP_ERROR_ABSOLUTE:
      CHKERRQ(PetscSNPrintf(ex,sizeof(ex),"   ||P(k)x||"));
      break;
    case PEP_ERROR_RELATIVE:
      CHKERRQ(PetscSNPrintf(ex,sizeof(ex),"||P(k)x||/||kx||"));
      break;
    case PEP_ERROR_BACKWARD:
      CHKERRQ(PetscSNPrintf(ex,sizeof(ex),"    eta(x,k)"));
      break;
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s            k             %s\n%s",sep,ex,sep));
  for (i=0;i<pep->nconv;i++) {
    CHKERRQ(PEPGetEigenpair(pep,i,&kr,&ki,NULL,NULL));
    CHKERRQ(PEPComputeError(pep,i,etype,&error));
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

static PetscErrorCode PEPErrorView_MATLAB(PEP pep,PEPErrorType etype,PetscViewer viewer)
{
  PetscReal      error;
  PetscInt       i;
  const char     *name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)pep,&name));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Error_%s = [\n",name));
  for (i=0;i<pep->nconv;i++) {
    CHKERRQ(PEPComputeError(pep,i,etype,&error));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)error));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"];\n"));
  PetscFunctionReturn(0);
}

/*@C
   PEPErrorView - Displays the errors associated with the computed solution
   (as well as the eigenvalues).

   Collective on pep

   Input Parameters:
+  pep    - the eigensolver context
.  etype  - error type
-  viewer - optional visualization context

   Options Database Key:
+  -pep_error_absolute - print absolute errors of each eigenpair
.  -pep_error_relative - print relative errors of each eigenpair
-  -pep_error_backward - print backward errors of each eigenpair

   Notes:
   By default, this function checks the error of all eigenpairs and prints
   the eigenvalues if all of them are below the requested tolerance.
   If the viewer has format=PETSC_VIEWER_ASCII_INFO_DETAIL then a table with
   eigenvalues and corresponding errors is printed.

   Level: intermediate

.seealso: PEPSolve(), PEPValuesView(), PEPVectorsView()
@*/
PetscErrorCode PEPErrorView(PEP pep,PEPErrorType etype,PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pep),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(pep,1,viewer,3);
  PEPCheckSolved(pep,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (!isascii) PetscFunctionReturn(0);

  CHKERRQ(PetscViewerGetFormat(viewer,&format));
  switch (format) {
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_ASCII_INFO:
      CHKERRQ(PEPErrorView_ASCII(pep,etype,viewer));
      break;
    case PETSC_VIEWER_ASCII_INFO_DETAIL:
      CHKERRQ(PEPErrorView_DETAIL(pep,etype,viewer));
      break;
    case PETSC_VIEWER_ASCII_MATLAB:
      CHKERRQ(PEPErrorView_MATLAB(pep,etype,viewer));
      break;
    default:
      CHKERRQ(PetscInfo(pep,"Unsupported viewer format %s\n",PetscViewerFormats[format]));
  }
  PetscFunctionReturn(0);
}

/*@
   PEPErrorViewFromOptions - Processes command line options to determine if/how
   the errors of the computed solution are to be viewed.

   Collective on pep

   Input Parameter:
.  pep - the eigensolver context

   Level: developer

.seealso: PEPErrorView()
@*/
PetscErrorCode PEPErrorViewFromOptions(PEP pep)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)pep),((PetscObject)pep)->options,((PetscObject)pep)->prefix,"-pep_error_absolute",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_ABSOLUTE,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)pep),((PetscObject)pep)->options,((PetscObject)pep)->prefix,"-pep_error_relative",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_RELATIVE,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)pep),((PetscObject)pep)->options,((PetscObject)pep)->prefix,"-pep_error_backward",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PEPErrorView(pep,PEP_ERROR_BACKWARD,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPValuesView_DRAW(PEP pep,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscDrawSP    drawsp;
  PetscReal      re,im;
  PetscInt       i,k;

  PetscFunctionBegin;
  if (!pep->nconv) PetscFunctionReturn(0);
  CHKERRQ(PetscViewerDrawGetDraw(viewer,0,&draw));
  CHKERRQ(PetscDrawSetTitle(draw,"Computed Eigenvalues"));
  CHKERRQ(PetscDrawSPCreate(draw,1,&drawsp));
  for (i=0;i<pep->nconv;i++) {
    k = pep->perm[i];
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(pep->eigr[k]);
    im = PetscImaginaryPart(pep->eigr[k]);
#else
    re = pep->eigr[k];
    im = pep->eigi[k];
#endif
    CHKERRQ(PetscDrawSPAddPoint(drawsp,&re,&im));
  }
  CHKERRQ(PetscDrawSPDraw(drawsp,PETSC_TRUE));
  CHKERRQ(PetscDrawSPSave(drawsp));
  CHKERRQ(PetscDrawSPDestroy(&drawsp));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPValuesView_BINARY(PEP pep,PetscViewer viewer)
{
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt       i,k;
  PetscComplex   *ev;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_COMPLEX)
  CHKERRQ(PetscMalloc1(pep->nconv,&ev));
  for (i=0;i<pep->nconv;i++) {
    k = pep->perm[i];
#if defined(PETSC_USE_COMPLEX)
    ev[i] = pep->eigr[k];
#else
    ev[i] = PetscCMPLX(pep->eigr[k],pep->eigi[k]);
#endif
  }
  CHKERRQ(PetscViewerBinaryWrite(viewer,ev,pep->nconv,PETSC_COMPLEX));
  CHKERRQ(PetscFree(ev));
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode PEPValuesView_HDF5(PEP pep,PetscViewer viewer)
{
  PetscInt       i,k,n,N;
  PetscMPIInt    rank;
  Vec            v;
  char           vname[30];
  const char     *ename;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pep),&rank));
  N = pep->nconv;
  n = rank? 0: N;
  /* create a vector containing the eigenvalues */
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)pep),n,N,&v));
  CHKERRQ(PetscObjectGetName((PetscObject)pep,&ename));
  CHKERRQ(PetscSNPrintf(vname,sizeof(vname),"eigr_%s",ename));
  CHKERRQ(PetscObjectSetName((PetscObject)v,vname));
  if (!rank) {
    for (i=0;i<pep->nconv;i++) {
      k = pep->perm[i];
      CHKERRQ(VecSetValue(v,i,pep->eigr[k],INSERT_VALUES));
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
    for (i=0;i<pep->nconv;i++) {
      k = pep->perm[i];
      CHKERRQ(VecSetValue(v,i,pep->eigi[k],INSERT_VALUES));
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

static PetscErrorCode PEPValuesView_ASCII(PEP pep,PetscViewer viewer)
{
  PetscInt       i,k;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Eigenvalues = \n"));
  for (i=0;i<pep->nconv;i++) {
    k = pep->perm[i];
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"   "));
    CHKERRQ(SlepcPrintEigenvalueASCII(viewer,pep->eigr[k],pep->eigi[k]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPValuesView_MATLAB(PEP pep,PetscViewer viewer)
{
  PetscInt       i,k;
  PetscReal      re,im;
  const char     *name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)pep,&name));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Lambda_%s = [\n",name));
  for (i=0;i<pep->nconv;i++) {
    k = pep->perm[i];
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(pep->eigr[k]);
    im = PetscImaginaryPart(pep->eigr[k]);
#else
    re = pep->eigr[k];
    im = pep->eigi[k];
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
   PEPValuesView - Displays the computed eigenvalues in a viewer.

   Collective on pep

   Input Parameters:
+  pep    - the eigensolver context
-  viewer - the viewer

   Options Database Key:
.  -pep_view_values - print computed eigenvalues

   Level: intermediate

.seealso: PEPSolve(), PEPVectorsView(), PEPErrorView()
@*/
PetscErrorCode PEPValuesView(PEP pep,PetscViewer viewer)
{
  PetscBool         isascii,isdraw,isbinary;
  PetscViewerFormat format;
#if defined(PETSC_HAVE_HDF5)
  PetscBool         ishdf5;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pep),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(pep,1,viewer,2);
  PEPCheckSolved(pep,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
#if defined(PETSC_HAVE_HDF5)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isdraw) {
    CHKERRQ(PEPValuesView_DRAW(pep,viewer));
  } else if (isbinary) {
    CHKERRQ(PEPValuesView_BINARY(pep,viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    CHKERRQ(PEPValuesView_HDF5(pep,viewer));
#endif
  } else if (isascii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    switch (format) {
      case PETSC_VIEWER_DEFAULT:
      case PETSC_VIEWER_ASCII_INFO:
      case PETSC_VIEWER_ASCII_INFO_DETAIL:
        CHKERRQ(PEPValuesView_ASCII(pep,viewer));
        break;
      case PETSC_VIEWER_ASCII_MATLAB:
        CHKERRQ(PEPValuesView_MATLAB(pep,viewer));
        break;
      default:
        CHKERRQ(PetscInfo(pep,"Unsupported viewer format %s\n",PetscViewerFormats[format]));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PEPValuesViewFromOptions - Processes command line options to determine if/how
   the computed eigenvalues are to be viewed.

   Collective on pep

   Input Parameter:
.  pep - the eigensolver context

   Level: developer

.seealso: PEPValuesView()
@*/
PetscErrorCode PEPValuesViewFromOptions(PEP pep)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)pep),((PetscObject)pep)->options,((PetscObject)pep)->prefix,"-pep_view_values",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PEPValuesView(pep,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PEPVectorsView - Outputs computed eigenvectors to a viewer.

   Collective on pep

   Input Parameters:
+  pep    - the eigensolver context
-  viewer - the viewer

   Options Database Keys:
.  -pep_view_vectors - output eigenvectors.

   Note:
   If PETSc was configured with real scalars, complex conjugate eigenvectors
   will be viewed as two separate real vectors, one containing the real part
   and another one containing the imaginary part.

   Level: intermediate

.seealso: PEPSolve(), PEPValuesView(), PEPErrorView()
@*/
PetscErrorCode PEPVectorsView(PEP pep,PetscViewer viewer)
{
  PetscInt       i,k;
  Vec            xr,xi=NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pep),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(pep,1,viewer,2);
  PEPCheckSolved(pep,1);
  if (pep->nconv) {
    CHKERRQ(PEPComputeVectors(pep));
    CHKERRQ(BVCreateVec(pep->V,&xr));
#if !defined(PETSC_USE_COMPLEX)
    CHKERRQ(BVCreateVec(pep->V,&xi));
#endif
    for (i=0;i<pep->nconv;i++) {
      k = pep->perm[i];
      CHKERRQ(BV_GetEigenvector(pep->V,k,pep->eigi[k],xr,xi));
      CHKERRQ(SlepcViewEigenvector(viewer,xr,xi,"X",i,(PetscObject)pep));
    }
    CHKERRQ(VecDestroy(&xr));
#if !defined(PETSC_USE_COMPLEX)
    CHKERRQ(VecDestroy(&xi));
#endif
  }
  PetscFunctionReturn(0);
}

/*@
   PEPVectorsViewFromOptions - Processes command line options to determine if/how
   the computed eigenvectors are to be viewed.

   Collective on pep

   Input Parameter:
.  pep - the eigensolver context

   Level: developer

.seealso: PEPVectorsView()
@*/
PetscErrorCode PEPVectorsViewFromOptions(PEP pep)
{
  PetscViewer       viewer;
  PetscBool         flg = PETSC_FALSE;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)pep),((PetscObject)pep)->options,((PetscObject)pep)->prefix,"-pep_view_vectors",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PEPVectorsView(pep,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}
