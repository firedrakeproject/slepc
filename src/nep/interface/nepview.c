/*
   The NEP routines related to various viewers.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/nepimpl.h>      /*I "slepcnep.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "NEPView"
/*@C
   NEPView - Prints the NEP data structure.

   Collective on NEP

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

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode NEPView(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  char           str[50];
  PetscBool      isascii,isslp,istrivial,nods;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(nep,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)nep,viewer);CHKERRQ(ierr);
    if (nep->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*nep->ops->view)(nep,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (nep->split) {
      ierr = PetscViewerASCIIPrintf(viewer,"  nonlinear operator in split form\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  nonlinear operator from user callbacks\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  iterative refinement: %s\n",NEPRefineTypes[nep->refine]);CHKERRQ(ierr);
    if (nep->refine) {
      ierr = PetscViewerASCIIPrintf(viewer,"  refinement stopping criterion: tol=%g, its=%D\n",(double)nep->reftol,nep->rits);CHKERRQ(ierr);
    }
      if (nep->npart>1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  splitting communicator in %D partitions for refinement\n",nep->npart);CHKERRQ(ierr);
      }
    ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: ");CHKERRQ(ierr);
    ierr = SlepcSNPrintfScalar(str,50,nep->target,PETSC_FALSE);CHKERRQ(ierr);
    if (!nep->which) {
      ierr = PetscViewerASCIIPrintf(viewer,"not yet set\n");CHKERRQ(ierr);
    } else switch (nep->which) {
      case NEP_TARGET_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (in magnitude)\n",str);CHKERRQ(ierr);
        break;
      case NEP_TARGET_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the real axis)\n",str);CHKERRQ(ierr);
        break;
      case NEP_TARGET_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the imaginary axis)\n",str);CHKERRQ(ierr);
        break;
      case NEP_LARGEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"largest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case NEP_SMALLEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case NEP_LARGEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"largest real parts\n");CHKERRQ(ierr);
        break;
      case NEP_SMALLEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest real parts\n");CHKERRQ(ierr);
        break;
      case NEP_LARGEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"largest imaginary parts\n");CHKERRQ(ierr);
        break;
      case NEP_SMALLEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest imaginary parts\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)nep),1,"Wrong value of nep->which");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %D\n",nep->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",nep->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %D\n",nep->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",nep->max_it);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of function evaluations: %D\n",nep->max_funcs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%g, absolute=%g, solution=%g\n",(double)nep->rtol,(double)nep->abstol,(double)nep->stol);CHKERRQ(ierr);
    if (nep->lag) {
      ierr = PetscViewerASCIIPrintf(viewer,"  updating the preconditioner every %D iterations\n",nep->lag);CHKERRQ(ierr);
    }
    if (nep->cctol) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using a constant tolerance for the linear solver\n");CHKERRQ(ierr);
    }
    if (nep->nini) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %D\n",PetscAbs(nep->nini));CHKERRQ(ierr);
    }
  } else {
    if (nep->ops->view) {
      ierr = (*nep->ops->view)(nep,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  if (!nep->V) { ierr = NEPGetBV(nep,&nep->V);CHKERRQ(ierr); }
  ierr = BVView(nep->V,viewer);CHKERRQ(ierr);
  if (!nep->rg) { ierr = NEPGetRG(nep,&nep->rg);CHKERRQ(ierr); }
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) { ierr = RGView(nep->rg,viewer);CHKERRQ(ierr); }
  ierr = PetscObjectTypeCompareAny((PetscObject)nep,&nods,NEPRII,NEPSLP,NEPINTERPOL,"");CHKERRQ(ierr);
  if (!nods) {
    if (!nep->ds) { ierr = NEPGetDS(nep,&nep->ds);CHKERRQ(ierr); }
    ierr = DSView(nep->ds,viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)nep,NEPSLP,&isslp);CHKERRQ(ierr);
  if (!isslp) {
    if (!nep->ksp) { ierr = NEPGetKSP(nep,&nep->ksp);CHKERRQ(ierr); }
    ierr = KSPView(nep->ksp,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPReasonView"
/*@C
   NEPReasonView - Displays the reason a NEP solve converged or diverged.

   Collective on NEP

   Parameter:
+  nep - the nonlinear eigensolver context
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -nep_converged_reason - print reason for convergence, and number of iterations

   Level: intermediate

.seealso: NEPSetConvergenceTest(), NEPSetTolerances(), NEPGetIterationNumber()
@*/
PetscErrorCode NEPReasonView(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isAscii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii);CHKERRQ(ierr);
  if (isAscii) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)nep)->tablevel);CHKERRQ(ierr);
    if (nep->reason > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"%s Nonlinear eigensolve converged (%d eigenpair%s) due to %s; iterations %D\n",((PetscObject)nep)->prefix?((PetscObject)nep)->prefix:"",nep->nconv,(nep->nconv>1)?"s":"",NEPConvergedReasons[nep->reason],nep->its);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"%s Nonlinear eigensolve did not converge due to %s; iterations %D\n",((PetscObject)nep)->prefix?((PetscObject)nep)->prefix:"",NEPConvergedReasons[nep->reason],nep->its);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)nep)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPReasonViewFromOptions"
/*@
   NEPReasonViewFromOptions - Processes command line options to determine if/how
   the NEP converged reason is to be viewed. 

   Collective on NEP

   Input Parameters:
.  nep - the nonlinear eigensolver context

   Level: developer
@*/
PetscErrorCode NEPReasonViewFromOptions(NEP nep)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->prefix,"-nep_converged_reason",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = NEPReasonView(nep,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPErrorView_ASCII"
static PetscErrorCode NEPErrorView_ASCII(NEP nep,NEPErrorType etype,PetscViewer viewer)
{
  PetscBool      errok;
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nep->nconv<nep->nev) {
    ierr = PetscViewerASCIIPrintf(viewer," Problem: less than %D eigenvalues converged\n\n",nep->nev);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  errok = PETSC_TRUE;
  for (i=0;i<nep->nev;i++) {
    ierr = NEPComputeError(nep,i,etype,&error);CHKERRQ(ierr);
    errok = (errok && error<5.0*nep->rtol)? PETSC_TRUE: PETSC_FALSE;
  }
  if (!errok) {
    ierr = PetscViewerASCIIPrintf(viewer," Problem: some of the first %D relative errors are higher than the tolerance\n\n",nep->nev);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscViewerASCIIPrintf(viewer," All requested eigenvalues computed up to the required tolerance:");CHKERRQ(ierr);
  for (i=0;i<=(nep->nev-1)/8;i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"\n     ");CHKERRQ(ierr);
    for (j=0;j<PetscMin(8,nep->nev-8*i);j++) {
      ierr = NEPGetEigenpair(nep,8*i+j,&kr,&ki,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
#else
      re = kr;
      im = ki;
#endif
      if (PetscAbs(re)/PetscAbs(im)<PETSC_SMALL) re = 0.0;
      if (PetscAbs(im)/PetscAbs(re)<PETSC_SMALL) im = 0.0;
      if (im!=0.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"%.5f%+.5fi",(double)re,(double)im);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"%.5f",(double)re);CHKERRQ(ierr);
      }
      if (8*i+j+1<nep->nev) { ierr = PetscViewerASCIIPrintf(viewer,", ");CHKERRQ(ierr); }
    }
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPErrorView_DETAIL"
static PetscErrorCode NEPErrorView_DETAIL(NEP nep,NEPErrorType etype,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i;
#define EXLEN 30
  char           ex[EXLEN],sep[]=" ---------------------- --------------------\n";

  PetscFunctionBegin;
  if (!nep->nconv) PetscFunctionReturn(0);
  switch (etype) {
    case NEP_ERROR_ABSOLUTE:
      ierr = PetscSNPrintf(ex,EXLEN,"    ||T(k)x||");CHKERRQ(ierr);
      break;
    case NEP_ERROR_RELATIVE:
      ierr = PetscSNPrintf(ex,EXLEN," ||T(k)x||/||kx||");CHKERRQ(ierr);
      break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"%s            k             %s\n%s",sep,ex,sep);CHKERRQ(ierr);
  for (i=0;i<nep->nconv;i++) {
    ierr = NEPGetEigenpair(nep,i,&kr,&ki,NULL,NULL);CHKERRQ(ierr);
    ierr = NEPComputeError(nep,i,etype,&error);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(kr);
    im = PetscImaginaryPart(kr);
#else
    re = kr;
    im = ki;
#endif
    if (im!=0.0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  % 9f%+9fi      %12g\n",(double)re,(double)im,(double)error);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"    % 12f           %12g\n",(double)re,(double)error);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerASCIIPrintf(viewer,sep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPErrorView_MATLAB"
static PetscErrorCode NEPErrorView_MATLAB(NEP nep,NEPErrorType etype,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      error;
  PetscInt       i;
  const char     *name;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)nep,&name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Error_%s = [\n",name);CHKERRQ(ierr);
  for (i=0;i<nep->nconv;i++) {
    ierr = NEPComputeError(nep,i,etype,&error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",error);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPErrorView"
/*@C
   NEPErrorView - Displays the errors associated with the computed solution
   (as well as the eigenvalues).

   Collective on NEP

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  etype  - error type
-  viewer - optional visualization context

   Options Database Key:
+  -nep_error_absolute - print absolute errors of each eigenpair
-  -nep_error_relative - print relative errors of each eigenpair

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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(nep,1,viewer,2);
  NEPCheckSolved(nep,1);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  switch (format) {
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_ASCII_INFO:
      ierr = NEPErrorView_ASCII(nep,etype,viewer);CHKERRQ(ierr);
      break;
    case PETSC_VIEWER_ASCII_INFO_DETAIL:
      ierr = NEPErrorView_DETAIL(nep,etype,viewer);CHKERRQ(ierr);
      break;
    case PETSC_VIEWER_ASCII_MATLAB:
      ierr = NEPErrorView_MATLAB(nep,etype,viewer);CHKERRQ(ierr);
      break;
    default:
      ierr = PetscInfo1(nep,"Unsupported viewer format %s\n",PetscViewerFormats[format]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPErrorViewFromOptions"
/*@
   NEPErrorViewFromOptions - Processes command line options to determine if/how
   the errors of the computed solution are to be viewed. 

   Collective on NEP

   Input Parameters:
.  nep - the nonlinear eigensolver context

   Level: developer
@*/
PetscErrorCode NEPErrorViewFromOptions(NEP nep)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->prefix,"-nep_error_absolute",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = NEPErrorView(nep,NEP_ERROR_ABSOLUTE,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->prefix,"-nep_error_relative",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPValuesView_DRAW"
static PetscErrorCode NEPValuesView_DRAW(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscDraw      draw;
  PetscDrawSP    drawsp;
  PetscReal      re,im;
  PetscInt       i,k;

  PetscFunctionBegin;
  if (!nep->nconv) PetscFunctionReturn(0);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
  for (i=0;i<nep->nconv;i++) {
    k = nep->perm[i];
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(nep->eigr[k]);
    im = PetscImaginaryPart(nep->eigr[k]);
#else
    re = nep->eigr[k];
    im = nep->eigi[k];
#endif
    ierr = PetscDrawSPAddPoint(drawsp,&re,&im);CHKERRQ(ierr);
  }
  ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPValuesView_ASCII"
static PetscErrorCode NEPValuesView_ASCII(NEP nep,PetscViewer viewer)
{
  PetscReal      re,im;
  PetscInt       i,k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"Eigenvalues = \n");CHKERRQ(ierr);
  for (i=0;i<nep->nconv;i++) {
    k = nep->perm[i];
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(nep->eigr[k]);
    im = PetscImaginaryPart(nep->eigr[k]);
#else
    re = nep->eigr[k];
    im = nep->eigi[k];
#endif
    if (PetscAbs(re)/PetscAbs(im)<PETSC_SMALL) re = 0.0;
    if (PetscAbs(im)/PetscAbs(re)<PETSC_SMALL) im = 0.0;
    if (im!=0.0) {
      ierr = PetscViewerASCIIPrintf(viewer,"   %.5f%+.5fi\n",(double)re,(double)im);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"   %.5f\n",(double)re);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPValuesView_MATLAB"
static PetscErrorCode NEPValuesView_MATLAB(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
  PetscReal      re,im;
  const char     *name;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)nep,&name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Lambda_%s = [\n",name);CHKERRQ(ierr);
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
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e%+18.16ei\n",(double)re,(double)im);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)re);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPValuesView"
/*@C
   NEPValuesView - Displays the computed eigenvalues in a viewer.

   Collective on NEP

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
  PetscBool         isascii,isdraw;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(nep,1,viewer,2);
  NEPCheckSolved(nep,1);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isdraw) {
    ierr = NEPValuesView_DRAW(nep,viewer);CHKERRQ(ierr);
  } else if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    switch (format) {
      case PETSC_VIEWER_DEFAULT:
      case PETSC_VIEWER_ASCII_INFO:
      case PETSC_VIEWER_ASCII_INFO_DETAIL:
        ierr = NEPValuesView_ASCII(nep,viewer);CHKERRQ(ierr);
        break;
      case PETSC_VIEWER_ASCII_MATLAB:
        ierr = NEPValuesView_MATLAB(nep,viewer);CHKERRQ(ierr);
        break;
      default:
        ierr = PetscInfo1(nep,"Unsupported viewer format %s\n",PetscViewerFormats[format]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPValuesViewFromOptions"
/*@
   NEPValuesViewFromOptions - Processes command line options to determine if/how
   the computed eigenvalues are to be viewed. 

   Collective on NEP

   Input Parameters:
.  nep - the nonlinear eigensolver context

   Level: developer
@*/
PetscErrorCode NEPValuesViewFromOptions(NEP nep)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->prefix,"-nep_view_values",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = NEPValuesView(nep,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPVectorsView"
/*@C
   NEPVectorsView - Outputs computed eigenvectors to a viewer.

   Collective on NEP

   Parameter:
+  nep    - the nonlinear eigensolver context
-  viewer - the viewer

   Options Database Keys:
.  -nep_view_vectors - output eigenvectors.

   Note:
   If PETSc was configured with real scalars, complex conjugate eigenvectors
   will be viewed as two separate real vectors, one containing the real part
   and another one containing the imaginary part.

   Level: intermediate

.seealso: NEPSolve(), NEPValuesView(), NEPErrorView()
@*/
PetscErrorCode NEPVectorsView(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
  Vec            x;
#define NMLEN 30
  char           vname[NMLEN];
  const char     *ename;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(nep,1,viewer,2);
  NEPCheckSolved(nep,1);
  if (nep->nconv) {
    ierr = PetscObjectGetName((PetscObject)nep,&ename);CHKERRQ(ierr);
    ierr = NEPComputeVectors(nep);CHKERRQ(ierr);
    for (i=0;i<nep->nconv;i++) {
      k = nep->perm[i];
      ierr = PetscSNPrintf(vname,NMLEN,"V%d_%s",i,ename);CHKERRQ(ierr);
      ierr = BVGetColumn(nep->V,k,&x);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)x,vname);CHKERRQ(ierr);
      ierr = VecView(x,viewer);CHKERRQ(ierr);
      ierr = BVRestoreColumn(nep->V,k,&x);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPVectorsViewFromOptions"
/*@
   NEPVectorsViewFromOptions - Processes command line options to determine if/how
   the computed eigenvectors are to be viewed. 

   Collective on NEP

   Input Parameters:
.  nep - the nonlinear eigensolver context

   Level: developer
@*/
PetscErrorCode NEPVectorsViewFromOptions(NEP nep)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg = PETSC_FALSE;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)nep),((PetscObject)nep)->prefix,"-nep_view_vectors",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = NEPVectorsView(nep,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

