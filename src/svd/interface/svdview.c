/*
   The SVD routines related to various viewers.

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

#include <slepc/private/svdimpl.h>      /*I "slepcsvd.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "SVDView"
/*@C
   SVDView - Prints the SVD data structure.

   Collective on SVD

   Input Parameters:
+  svd - the singular value solver context
-  viewer - optional visualization context

   Options Database Key:
.  -svd_view -  Calls SVDView() at end of SVDSolve()

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

.seealso: STView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode SVDView(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii,isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)svd));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)svd,viewer);CHKERRQ(ierr);
    if (svd->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*svd->ops->view)(svd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  transpose mode: %s\n",svd->impltrans?"implicit":"explicit");CHKERRQ(ierr);
    if (svd->which == SVD_LARGEST) {
      ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: largest\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: smallest\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of singular values (nsv): %D\n",svd->nsv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",svd->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %D\n",svd->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",svd->max_it);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)svd->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  convergence test: ");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    switch (svd->conv) {
    case SVD_CONV_ABS:
      ierr = PetscViewerASCIIPrintf(viewer,"absolute\n");CHKERRQ(ierr);break;
    case SVD_CONV_REL:
      ierr = PetscViewerASCIIPrintf(viewer,"relative to the singular value\n");CHKERRQ(ierr);break;
    case SVD_CONV_USER:
      ierr = PetscViewerASCIIPrintf(viewer,"user-defined\n");CHKERRQ(ierr);break;
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    if (svd->nini) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %D\n",PetscAbs(svd->nini));CHKERRQ(ierr);
    }
    if (svd->ninil) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial left space: %D\n",PetscAbs(svd->ninil));CHKERRQ(ierr);
    }
  } else {
    if (svd->ops->view) {
      ierr = (*svd->ops->view)(svd,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectTypeCompareAny((PetscObject)svd,&isshell,SVDCROSS,SVDCYCLIC,"");CHKERRQ(ierr);
  if (!isshell) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    if (!svd->V) { ierr = SVDGetBV(svd,&svd->V,NULL);CHKERRQ(ierr); }
    ierr = BVView(svd->V,viewer);CHKERRQ(ierr);
    if (!svd->ds) { ierr = SVDGetDS(svd,&svd->ds);CHKERRQ(ierr); }
    ierr = DSView(svd->ds,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDReasonView"
/*@C
   SVDReasonView - Displays the reason an SVD solve converged or diverged.

   Collective on SVD

   Parameter:
+  svd - the singular value solver context
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -svd_converged_reason - print reason for convergence, and number of iterations

   Level: intermediate

.seealso: SVDSetTolerances(), SVDGetIterationNumber()
@*/
PetscErrorCode SVDReasonView(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isAscii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii);CHKERRQ(ierr);
  if (isAscii) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)svd)->tablevel);CHKERRQ(ierr);
    if (svd->reason > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"%s SVD solve converged (%D singular triplet%s) due to %s; iterations %D\n",((PetscObject)svd)->prefix?((PetscObject)svd)->prefix:"",svd->nconv,(svd->nconv>1)?"s":"",SVDConvergedReasons[svd->reason],svd->its);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"%s SVD solve did not converge due to %s; iterations %D\n",((PetscObject)svd)->prefix?((PetscObject)svd)->prefix:"",SVDConvergedReasons[svd->reason],svd->its);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)svd)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDReasonViewFromOptions"
/*@
   SVDReasonViewFromOptions - Processes command line options to determine if/how
   the SVD converged reason is to be viewed. 

   Collective on SVD

   Input Parameters:
.  svd - the singular value solver context

   Level: developer
@*/
PetscErrorCode SVDReasonViewFromOptions(SVD svd)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->prefix,"-svd_converged_reason",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = SVDReasonView(svd,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDErrorView_ASCII"
static PetscErrorCode SVDErrorView_ASCII(SVD svd,SVDErrorType etype,PetscViewer viewer)
{
  PetscBool      errok;
  PetscReal      error,sigma;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (svd->nconv<svd->nsv) {
    ierr = PetscViewerASCIIPrintf(viewer," Problem: less than %D singular values converged\n\n",svd->nsv);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  errok = PETSC_TRUE;
  for (i=0;i<svd->nsv;i++) {
    ierr = SVDComputeError(svd,i,etype,&error);CHKERRQ(ierr);
    errok = (errok && error<5.0*svd->tol)? PETSC_TRUE: PETSC_FALSE;
  }
  if (!errok) {
    ierr = PetscViewerASCIIPrintf(viewer," Problem: some of the first %D relative errors are higher than the tolerance\n\n",svd->nsv);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscViewerASCIIPrintf(viewer," All requested singular values computed up to the required tolerance:");CHKERRQ(ierr);
  for (i=0;i<=(svd->nsv-1)/8;i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"\n     ");CHKERRQ(ierr);
    for (j=0;j<PetscMin(8,svd->nsv-8*i);j++) {
      ierr = SVDGetSingularTriplet(svd,8*i+j,&sigma,NULL,NULL);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%.5f",(double)sigma);CHKERRQ(ierr);
      if (8*i+j+1<svd->nsv) { ierr = PetscViewerASCIIPrintf(viewer,", ");CHKERRQ(ierr); }
    }
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDErrorView_DETAIL"
static PetscErrorCode SVDErrorView_DETAIL(SVD svd,SVDErrorType etype,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      error,sigma;
  PetscInt       i;
#define EXLEN 30
  char           ex[EXLEN],sep[]=" ---------------------- --------------------\n";

  PetscFunctionBegin;
  if (!svd->nconv) PetscFunctionReturn(0);
  switch (etype) {
    case SVD_ERROR_ABSOLUTE:
      ierr = PetscSNPrintf(ex,EXLEN," absolute error");CHKERRQ(ierr);
      break;
    case SVD_ERROR_RELATIVE:
      ierr = PetscSNPrintf(ex,EXLEN," relative error");CHKERRQ(ierr);
      break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"%s          sigma           %s\n%s",sep,ex,sep);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = SVDGetSingularTriplet(svd,i,&sigma,NULL,NULL);CHKERRQ(ierr);
    ierr = SVDComputeError(svd,i,etype,&error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"       % 6f          %12g\n",(double)sigma,(double)error);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,sep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDErrorView_MATLAB"
static PetscErrorCode SVDErrorView_MATLAB(SVD svd,SVDErrorType etype,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      error;
  PetscInt       i;
  const char     *name;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)svd,&name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Error_%s = [\n",name);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = SVDComputeError(svd,i,etype,&error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",error);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDErrorView"
/*@C
   SVDErrorView - Displays the errors associated with the computed solution
   (as well as the singular values).

   Collective on SVD

   Input Parameters:
+  svd    - the singular value solver context
.  etype  - error type
-  viewer - optional visualization context

   Options Database Key:
+  -svd_error_absolute - print absolute errors of each singular triplet
-  -svd_error_relative - print relative errors of each singular triplet

   Notes:
   By default, this function checks the error of all singular triplets and prints
   the singular values if all of them are below the requested tolerance.
   If the viewer has format=PETSC_VIEWER_ASCII_INFO_DETAIL then a table with
   singular values and corresponding errors is printed.

   Level: intermediate

.seealso: SVDSolve(), SVDValuesView(), SVDVectorsView()
@*/
PetscErrorCode SVDErrorView(SVD svd,SVDErrorType etype,PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)svd));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);
  SVDCheckSolved(svd,1);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  switch (format) {
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_ASCII_INFO:
      ierr = SVDErrorView_ASCII(svd,etype,viewer);CHKERRQ(ierr);
      break;
    case PETSC_VIEWER_ASCII_INFO_DETAIL:
      ierr = SVDErrorView_DETAIL(svd,etype,viewer);CHKERRQ(ierr);
      break;
    case PETSC_VIEWER_ASCII_MATLAB:
      ierr = SVDErrorView_MATLAB(svd,etype,viewer);CHKERRQ(ierr);
      break;
    default:
      ierr = PetscInfo1(svd,"Unsupported viewer format %s\n",PetscViewerFormats[format]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDErrorViewFromOptions"
/*@
   SVDErrorViewFromOptions - Processes command line options to determine if/how
   the errors of the computed solution are to be viewed. 

   Collective on SVD

   Input Parameters:
.  svd - the singular value solver context

   Level: developer
@*/
PetscErrorCode SVDErrorViewFromOptions(SVD svd)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->prefix,"-svd_error_absolute",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = SVDErrorView(svd,SVD_ERROR_ABSOLUTE,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->prefix,"-svd_error_relative",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDValuesView_DRAW"
static PetscErrorCode SVDValuesView_DRAW(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscDraw      draw;
  PetscDrawSP    drawsp;
  PetscReal      re,im=0.0;
  PetscInt       i;

  PetscFunctionBegin;
  if (!svd->nconv) PetscFunctionReturn(0);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed singular values",PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    re = svd->sigma[svd->perm[i]];
    ierr = PetscDrawSPAddPoint(drawsp,&re,&im);CHKERRQ(ierr);
  }
  ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDrawSPSave(drawsp);CHKERRQ(ierr);
  ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDValuesView_ASCII"
static PetscErrorCode SVDValuesView_ASCII(SVD svd,PetscViewer viewer)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"Singular values = \n");CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"   %.5f\n",(double)svd->sigma[svd->perm[i]]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDValuesView_MATLAB"
static PetscErrorCode SVDValuesView_MATLAB(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i;
  const char     *name;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)svd,&name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Sigma_%s = [\n",name);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)svd->sigma[svd->perm[i]]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDValuesView"
/*@C
   SVDValuesView - Displays the computed singular values in a viewer.

   Collective on SVD

   Input Parameters:
+  svd    - the singular value solver context
-  viewer - the viewer

   Options Database Key:
.  -svd_view_values - print computed singular values

   Level: intermediate

.seealso: SVDSolve(), SVDVectorsView(), SVDErrorView()
@*/
PetscErrorCode SVDValuesView(SVD svd,PetscViewer viewer)
{
  PetscBool         isascii,isdraw;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)svd));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);
  SVDCheckSolved(svd,1);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isdraw) {
    ierr = SVDValuesView_DRAW(svd,viewer);CHKERRQ(ierr);
  } else if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    switch (format) {
      case PETSC_VIEWER_DEFAULT:
      case PETSC_VIEWER_ASCII_INFO:
      case PETSC_VIEWER_ASCII_INFO_DETAIL:
        ierr = SVDValuesView_ASCII(svd,viewer);CHKERRQ(ierr);
        break;
      case PETSC_VIEWER_ASCII_MATLAB:
        ierr = SVDValuesView_MATLAB(svd,viewer);CHKERRQ(ierr);
        break;
      default:
        ierr = PetscInfo1(svd,"Unsupported viewer format %s\n",PetscViewerFormats[format]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDValuesViewFromOptions"
/*@
   SVDValuesViewFromOptions - Processes command line options to determine if/how
   the computed singular values are to be viewed. 

   Collective on SVD

   Input Parameters:
.  svd - the singular value solver context

   Level: developer
@*/
PetscErrorCode SVDValuesViewFromOptions(SVD svd)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->prefix,"-svd_view_values",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = SVDValuesView(svd,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDVectorsView"
/*@C
   SVDVectorsView - Outputs computed singular vectors to a viewer.

   Collective on SVD

   Parameter:
+  svd    - the singular value solver context
-  viewer - the viewer

   Options Database Keys:
.  -svd_view_vectors - output singular vectors

   Level: intermediate

.seealso: SVDSolve(), SVDValuesView(), SVDErrorView()
@*/
PetscErrorCode SVDVectorsView(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
  Vec            x;
#define NMLEN 30
  char           vname[NMLEN];
  const char     *ename;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)svd));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);
  SVDCheckSolved(svd,1);
  if (svd->nconv) {
    ierr = PetscObjectGetName((PetscObject)svd,&ename);CHKERRQ(ierr);
    ierr = SVDComputeVectors(svd);CHKERRQ(ierr);
    for (i=0;i<svd->nconv;i++) {
      k = svd->perm[i];
      ierr = PetscSNPrintf(vname,NMLEN,"V%d_%s",(int)i,ename);CHKERRQ(ierr);
      ierr = BVGetColumn(svd->V,k,&x);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)x,vname);CHKERRQ(ierr);
      ierr = VecView(x,viewer);CHKERRQ(ierr);
      ierr = BVRestoreColumn(svd->V,k,&x);CHKERRQ(ierr);
      ierr = PetscSNPrintf(vname,NMLEN,"U%d_%s",(int)i,ename);CHKERRQ(ierr);
      ierr = BVGetColumn(svd->U,k,&x);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)x,vname);CHKERRQ(ierr);
      ierr = VecView(x,viewer);CHKERRQ(ierr);
      ierr = BVRestoreColumn(svd->U,k,&x);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDVectorsViewFromOptions"
/*@
   SVDVectorsViewFromOptions - Processes command line options to determine if/how
   the computed singular vectors are to be viewed. 

   Collective on SVD

   Input Parameters:
.  svd - the singular value solver context

   Level: developer
@*/
PetscErrorCode SVDVectorsViewFromOptions(SVD svd)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg = PETSC_FALSE;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->prefix,"-svd_view_vectors",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = SVDVectorsView(svd,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

