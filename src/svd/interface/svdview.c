/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   The SVD routines related to various viewers
*/

#include <slepc/private/svdimpl.h>      /*I "slepcsvd.h" I*/
#include <petscdraw.h>

/*@C
   SVDView - Prints the SVD data structure.

   Collective on svd

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

.seealso: EPSView()
@*/
PetscErrorCode SVDView(SVD svd,PetscViewer viewer)
{
  const char     *type=NULL;
  PetscBool      isascii,isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)svd),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)svd,viewer));
    if (svd->ops->view) {
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ((*svd->ops->view)(svd,viewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
    if (svd->problem_type) {
      switch (svd->problem_type) {
        case SVD_STANDARD:    type = "(standard) singular value problem"; break;
        case SVD_GENERALIZED: type = "generalized singular value problem"; break;
      }
    } else type = "not yet set";
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  transpose mode: %s\n",svd->impltrans?"implicit":"explicit"));
    if (svd->which == SVD_LARGEST) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: largest\n"));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: smallest\n"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of singular values (nsv): %" PetscInt_FMT "\n",svd->nsv));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %" PetscInt_FMT "\n",svd->ncv));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %" PetscInt_FMT "\n",svd->mpd));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %" PetscInt_FMT "\n",svd->max_it));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)svd->tol));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  convergence test: "));
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    switch (svd->conv) {
    case SVD_CONV_ABS:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"absolute\n"));break;
    case SVD_CONV_REL:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"relative to the singular value\n"));break;
    case SVD_CONV_NORM:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"relative to the matrix norms\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  computed matrix norms: norm(A)=%g",(double)svd->nrma));
      if (svd->isgeneralized) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,", norm(B)=%g",(double)svd->nrmb));
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
      break;
    case SVD_CONV_MAXIT:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"maximum number of iterations\n"));break;
    case SVD_CONV_USER:
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"user-defined\n"));break;
    }
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    if (svd->nini) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %" PetscInt_FMT "\n",PetscAbs(svd->nini)));
    }
    if (svd->ninil) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial left space: %" PetscInt_FMT "\n",PetscAbs(svd->ninil)));
    }
  } else {
    if (svd->ops->view) {
      CHKERRQ((*svd->ops->view)(svd,viewer));
    }
  }
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)svd,&isshell,SVDCROSS,SVDCYCLIC,SVDSCALAPACK,SVDELEMENTAL,SVDPRIMME,""));
  if (!isshell) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
    if (!svd->V) CHKERRQ(SVDGetBV(svd,&svd->V,NULL));
    CHKERRQ(BVView(svd->V,viewer));
    if (!svd->ds) CHKERRQ(SVDGetDS(svd,&svd->ds));
    CHKERRQ(DSView(svd->ds,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
   SVDViewFromOptions - View from options

   Collective on SVD

   Input Parameters:
+  svd  - the singular value solver context
.  obj  - optional object
-  name - command line option

   Level: intermediate

.seealso: SVDView(), SVDCreate()
@*/
PetscErrorCode SVDViewFromOptions(SVD svd,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)svd,obj,name));
  PetscFunctionReturn(0);
}

/*@C
   SVDConvergedReasonView - Displays the reason an SVD solve converged or diverged.

   Collective on svd

   Input Parameters:
+  svd - the singular value solver context
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -svd_converged_reason - print reason for convergence, and number of iterations

   Note:
   To change the format of the output call PetscViewerPushFormat(viewer,format) before
   this call. Use PETSC_VIEWER_DEFAULT for the default, use PETSC_VIEWER_FAILED to only
   display a reason if it fails. The latter can be set in the command line with
   -svd_converged_reason ::failed

   Level: intermediate

.seealso: SVDSetTolerances(), SVDGetIterationNumber(), SVDConvergedReasonViewFromOptions()
@*/
PetscErrorCode SVDConvergedReasonView(SVD svd,PetscViewer viewer)
{
  PetscBool         isAscii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)svd));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii));
  if (isAscii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)svd)->tablevel));
    if (svd->reason > 0 && format != PETSC_VIEWER_FAILED) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s SVD solve converged (%" PetscInt_FMT " singular triplet%s) due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)svd)->prefix?((PetscObject)svd)->prefix:"",svd->nconv,(svd->nconv>1)?"s":"",SVDConvergedReasons[svd->reason],svd->its));
    } else if (svd->reason <= 0) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s SVD solve did not converge due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)svd)->prefix?((PetscObject)svd)->prefix:"",SVDConvergedReasons[svd->reason],svd->its));
    }
    CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)svd)->tablevel));
  }
  PetscFunctionReturn(0);
}

/*@
   SVDConvergedReasonViewFromOptions - Processes command line options to determine if/how
   the SVD converged reason is to be viewed.

   Collective on svd

   Input Parameter:
.  svd - the singular value solver context

   Level: developer

.seealso: SVDConvergedReasonView()
@*/
PetscErrorCode SVDConvergedReasonViewFromOptions(SVD svd)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->options,((PetscObject)svd)->prefix,"-svd_converged_reason",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(SVDConvergedReasonView(svd,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDErrorView_ASCII(SVD svd,SVDErrorType etype,PetscViewer viewer)
{
  PetscReal      error,sigma;
  PetscInt       i,j;

  PetscFunctionBegin;
  if (svd->nconv<svd->nsv) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," Problem: less than %" PetscInt_FMT " singular values converged\n\n",svd->nsv));
    PetscFunctionReturn(0);
  }
  for (i=0;i<svd->nsv;i++) {
    CHKERRQ(SVDComputeError(svd,i,etype,&error));
    if (error>=5.0*svd->tol) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer," Problem: some of the first %" PetscInt_FMT " relative errors are higher than the tolerance\n\n",svd->nsv));
      PetscFunctionReturn(0);
    }
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer," All requested singular values computed up to the required tolerance:"));
  for (i=0;i<=(svd->nsv-1)/8;i++) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n     "));
    for (j=0;j<PetscMin(8,svd->nsv-8*i);j++) {
      CHKERRQ(SVDGetSingularTriplet(svd,8*i+j,&sigma,NULL,NULL));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"%.5f",(double)sigma));
      if (8*i+j+1<svd->nsv) CHKERRQ(PetscViewerASCIIPrintf(viewer,", "));
    }
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDErrorView_DETAIL(SVD svd,SVDErrorType etype,PetscViewer viewer)
{
  PetscReal      error,sigma;
  PetscInt       i;
  char           ex[30],sep[]=" ---------------------- --------------------\n";

  PetscFunctionBegin;
  if (!svd->nconv) PetscFunctionReturn(0);
  switch (etype) {
    case SVD_ERROR_ABSOLUTE:
      CHKERRQ(PetscSNPrintf(ex,sizeof(ex)," absolute error"));
      break;
    case SVD_ERROR_RELATIVE:
      CHKERRQ(PetscSNPrintf(ex,sizeof(ex)," relative error"));
      break;
    case SVD_ERROR_NORM:
      if (svd->isgeneralized) CHKERRQ(PetscSNPrintf(ex,sizeof(ex)," ||r||/||[A;B]||"));
      else CHKERRQ(PetscSNPrintf(ex,sizeof(ex),"  ||r||/||A||"));
      break;
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s          sigma           %s\n%s",sep,ex,sep));
  for (i=0;i<svd->nconv;i++) {
    CHKERRQ(SVDGetSingularTriplet(svd,i,&sigma,NULL,NULL));
    CHKERRQ(SVDComputeError(svd,i,etype,&error));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"       % 6f          %12g\n",(double)sigma,(double)error));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s",sep));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDErrorView_MATLAB(SVD svd,SVDErrorType etype,PetscViewer viewer)
{
  PetscReal      error;
  PetscInt       i;
  const char     *name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)svd,&name));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Error_%s = [\n",name));
  for (i=0;i<svd->nconv;i++) {
    CHKERRQ(SVDComputeError(svd,i,etype,&error));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)error));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"];\n"));
  PetscFunctionReturn(0);
}

/*@C
   SVDErrorView - Displays the errors associated with the computed solution
   (as well as the singular values).

   Collective on svd

   Input Parameters:
+  svd    - the singular value solver context
.  etype  - error type
-  viewer - optional visualization context

   Options Database Key:
+  -svd_error_absolute - print absolute errors of each singular triplet
.  -svd_error_relative - print relative errors of each singular triplet
-  -svd_error_norm     - print errors relative to the matrix norms of each singular triplet

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)svd),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(svd,1,viewer,3);
  SVDCheckSolved(svd,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (!isascii) PetscFunctionReturn(0);

  CHKERRQ(PetscViewerGetFormat(viewer,&format));
  switch (format) {
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_ASCII_INFO:
      CHKERRQ(SVDErrorView_ASCII(svd,etype,viewer));
      break;
    case PETSC_VIEWER_ASCII_INFO_DETAIL:
      CHKERRQ(SVDErrorView_DETAIL(svd,etype,viewer));
      break;
    case PETSC_VIEWER_ASCII_MATLAB:
      CHKERRQ(SVDErrorView_MATLAB(svd,etype,viewer));
      break;
    default:
      CHKERRQ(PetscInfo(svd,"Unsupported viewer format %s\n",PetscViewerFormats[format]));
  }
  PetscFunctionReturn(0);
}

/*@
   SVDErrorViewFromOptions - Processes command line options to determine if/how
   the errors of the computed solution are to be viewed.

   Collective on svd

   Input Parameter:
.  svd - the singular value solver context

   Level: developer

.seealso: SVDErrorView()
@*/
PetscErrorCode SVDErrorViewFromOptions(SVD svd)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->options,((PetscObject)svd)->prefix,"-svd_error_absolute",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(SVDErrorView(svd,SVD_ERROR_ABSOLUTE,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->options,((PetscObject)svd)->prefix,"-svd_error_relative",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->options,((PetscObject)svd)->prefix,"-svd_error_norm",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(SVDErrorView(svd,SVD_ERROR_NORM,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDValuesView_DRAW(SVD svd,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscDrawSP    drawsp;
  PetscReal      re,im=0.0;
  PetscInt       i;

  PetscFunctionBegin;
  if (!svd->nconv) PetscFunctionReturn(0);
  CHKERRQ(PetscViewerDrawGetDraw(viewer,0,&draw));
  CHKERRQ(PetscDrawSetTitle(draw,"Computed singular values"));
  CHKERRQ(PetscDrawSPCreate(draw,1,&drawsp));
  for (i=0;i<svd->nconv;i++) {
    re = svd->sigma[svd->perm[i]];
    CHKERRQ(PetscDrawSPAddPoint(drawsp,&re,&im));
  }
  CHKERRQ(PetscDrawSPDraw(drawsp,PETSC_TRUE));
  CHKERRQ(PetscDrawSPSave(drawsp));
  CHKERRQ(PetscDrawSPDestroy(&drawsp));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDValuesView_BINARY(SVD svd,PetscViewer viewer)
{
  PetscInt       i,k;
  PetscReal      *sv;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(svd->nconv,&sv));
  for (i=0;i<svd->nconv;i++) {
    k = svd->perm[i];
    sv[i] = svd->sigma[k];
  }
  CHKERRQ(PetscViewerBinaryWrite(viewer,sv,svd->nconv,PETSC_REAL));
  CHKERRQ(PetscFree(sv));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode SVDValuesView_HDF5(SVD svd,PetscViewer viewer)
{
  PetscInt       i,k,n,N;
  PetscMPIInt    rank;
  Vec            v;
  char           vname[30];
  const char     *ename;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)svd),&rank));
  N = svd->nconv;
  n = rank? 0: N;
  /* create a vector containing the singular values */
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)svd),n,N,&v));
  CHKERRQ(PetscObjectGetName((PetscObject)svd,&ename));
  CHKERRQ(PetscSNPrintf(vname,sizeof(vname),"sigma_%s",ename));
  CHKERRQ(PetscObjectSetName((PetscObject)v,vname));
  if (!rank) {
    for (i=0;i<svd->nconv;i++) {
      k = svd->perm[i];
      CHKERRQ(VecSetValue(v,i,svd->sigma[k],INSERT_VALUES));
    }
  }
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  CHKERRQ(VecView(v,viewer));
  CHKERRQ(VecDestroy(&v));
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode SVDValuesView_ASCII(SVD svd,PetscViewer viewer)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Singular values = \n"));
  for (i=0;i<svd->nconv;i++) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"   %.5f\n",(double)svd->sigma[svd->perm[i]]));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDValuesView_MATLAB(SVD svd,PetscViewer viewer)
{
  PetscInt       i;
  const char     *name;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)svd,&name));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Sigma_%s = [\n",name));
  for (i=0;i<svd->nconv;i++) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e\n",(double)svd->sigma[svd->perm[i]]));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"];\n"));
  PetscFunctionReturn(0);
}

/*@C
   SVDValuesView - Displays the computed singular values in a viewer.

   Collective on svd

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
  PetscBool         isascii,isdraw,isbinary;
  PetscViewerFormat format;
#if defined(PETSC_HAVE_HDF5)
  PetscBool         ishdf5;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)svd),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);
  SVDCheckSolved(svd,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
#if defined(PETSC_HAVE_HDF5)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
#endif
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isdraw) {
    CHKERRQ(SVDValuesView_DRAW(svd,viewer));
  } else if (isbinary) {
    CHKERRQ(SVDValuesView_BINARY(svd,viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    CHKERRQ(SVDValuesView_HDF5(svd,viewer));
#endif
  } else if (isascii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    switch (format) {
      case PETSC_VIEWER_DEFAULT:
      case PETSC_VIEWER_ASCII_INFO:
      case PETSC_VIEWER_ASCII_INFO_DETAIL:
        CHKERRQ(SVDValuesView_ASCII(svd,viewer));
        break;
      case PETSC_VIEWER_ASCII_MATLAB:
        CHKERRQ(SVDValuesView_MATLAB(svd,viewer));
        break;
      default:
        CHKERRQ(PetscInfo(svd,"Unsupported viewer format %s\n",PetscViewerFormats[format]));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SVDValuesViewFromOptions - Processes command line options to determine if/how
   the computed singular values are to be viewed.

   Collective on svd

   Input Parameter:
.  svd - the singular value solver context

   Level: developer

.seealso: SVDValuesView()
@*/
PetscErrorCode SVDValuesViewFromOptions(SVD svd)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->options,((PetscObject)svd)->prefix,"-svd_view_values",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(SVDValuesView(svd,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   SVDVectorsView - Outputs computed singular vectors to a viewer.

   Collective on svd

   Input Parameters:
+  svd    - the singular value solver context
-  viewer - the viewer

   Options Database Keys:
.  -svd_view_vectors - output singular vectors

   Note:
   Right and left singular vectors are interleaved, that is, the vectors are
   output in the following order V0, U0, V1, U1, V2, U2, ...

   Level: intermediate

.seealso: SVDSolve(), SVDValuesView(), SVDErrorView()
@*/
PetscErrorCode SVDVectorsView(SVD svd,PetscViewer viewer)
{
  PetscInt       i,k;
  Vec            x;
  char           vname[30];
  const char     *ename;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)svd),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);
  SVDCheckSolved(svd,1);
  if (svd->nconv) {
    CHKERRQ(PetscObjectGetName((PetscObject)svd,&ename));
    CHKERRQ(SVDComputeVectors(svd));
    for (i=0;i<svd->nconv;i++) {
      k = svd->perm[i];
      CHKERRQ(PetscSNPrintf(vname,sizeof(vname),"V%" PetscInt_FMT "_%s",i,ename));
      CHKERRQ(BVGetColumn(svd->V,k,&x));
      CHKERRQ(PetscObjectSetName((PetscObject)x,vname));
      CHKERRQ(VecView(x,viewer));
      CHKERRQ(BVRestoreColumn(svd->V,k,&x));
      CHKERRQ(PetscSNPrintf(vname,sizeof(vname),"U%" PetscInt_FMT "_%s",i,ename));
      CHKERRQ(BVGetColumn(svd->U,k,&x));
      CHKERRQ(PetscObjectSetName((PetscObject)x,vname));
      CHKERRQ(VecView(x,viewer));
      CHKERRQ(BVRestoreColumn(svd->U,k,&x));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SVDVectorsViewFromOptions - Processes command line options to determine if/how
   the computed singular vectors are to be viewed.

   Collective on svd

   Input Parameter:
.  svd - the singular value solver context

   Level: developer

.seealso: SVDVectorsView()
@*/
PetscErrorCode SVDVectorsViewFromOptions(SVD svd)
{
  PetscViewer       viewer;
  PetscBool         flg = PETSC_FALSE;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject)svd),((PetscObject)svd)->options,((PetscObject)svd)->prefix,"-svd_view_vectors",&viewer,&format,&flg));
  if (flg) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(SVDVectorsView(svd,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}
