/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Basic LME routines
*/

#include <slepc/private/lmeimpl.h>      /*I "slepclme.h" I*/

/* Logging support */
PetscClassId      LME_CLASSID = 0;
PetscLogEvent     LME_SetUp = 0,LME_Solve = 0,LME_ComputeError = 0;

/* List of registered LME routines */
PetscFunctionList LMEList = NULL;
PetscBool         LMERegisterAllCalled = PETSC_FALSE;

/* List of registered LME monitors */
PetscFunctionList LMEMonitorList              = NULL;
PetscFunctionList LMEMonitorCreateList        = NULL;
PetscFunctionList LMEMonitorDestroyList       = NULL;
PetscBool         LMEMonitorRegisterAllCalled = PETSC_FALSE;

/*@C
   LMEView - Prints the LME data structure.

   Collective on lme

   Input Parameters:
+  lme - the linear matrix equation solver context
-  viewer - optional visualization context

   Options Database Key:
.  -lme_view -  Calls LMEView() at end of LMESolve()

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

.seealso: LMECreate()
@*/
PetscErrorCode LMEView(LME lme,PetscViewer viewer)
{
  PetscBool      isascii;
  const char     *eqname[] = {
                   "continuous-time Lyapunov",
                   "continuous-time Sylvester",
                   "generalized Lyapunov",
                   "generalized Sylvester",
                   "Stein",
                   "discrete-time Lyapunov"
  };

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)lme),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(lme,1,viewer,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)lme,viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(lme,view,viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  equation type: %s\n",eqname[lme->problem_type]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %" PetscInt_FMT "\n",lme->ncv));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %" PetscInt_FMT "\n",lme->max_it));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)lme->tol));
  } else PetscTryTypeMethod(lme,view,viewer);
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
  if (!lme->V) PetscCall(LMEGetBV(lme,&lme->V));
  PetscCall(BVView(lme->V,viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
   LMEViewFromOptions - View from options

   Collective on LME

   Input Parameters:
+  lme  - the linear matrix equation context
.  obj  - optional object
-  name - command line option

   Level: intermediate

.seealso: LMEView(), LMECreate()
@*/
PetscErrorCode LMEViewFromOptions(LME lme,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)lme,obj,name));
  PetscFunctionReturn(0);
}
/*@C
   LMEConvergedReasonView - Displays the reason an LME solve converged or diverged.

   Collective on lme

   Input Parameters:
+  lme - the linear matrix equation context
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -lme_converged_reason - print reason for convergence, and number of iterations

   Note:
   To change the format of the output call PetscViewerPushFormat(viewer,format) before
   this call. Use PETSC_VIEWER_DEFAULT for the default, use PETSC_VIEWER_FAILED to only
   display a reason if it fails. The latter can be set in the command line with
   -lme_converged_reason ::failed

   Level: intermediate

.seealso: LMESetTolerances(), LMEGetIterationNumber(), LMEConvergedReasonViewFromOptions()
@*/
PetscErrorCode LMEConvergedReasonView(LME lme,PetscViewer viewer)
{
  PetscBool         isAscii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)lme));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii));
  if (isAscii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    PetscCall(PetscViewerASCIIAddTab(viewer,((PetscObject)lme)->tablevel));
    if (lme->reason > 0 && format != PETSC_VIEWER_FAILED) PetscCall(PetscViewerASCIIPrintf(viewer,"%s Linear matrix equation solve converged due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)lme)->prefix?((PetscObject)lme)->prefix:"",LMEConvergedReasons[lme->reason],lme->its));
    else if (lme->reason <= 0) PetscCall(PetscViewerASCIIPrintf(viewer,"%s Linear matrix equation solve did not converge due to %s; iterations %" PetscInt_FMT "\n",((PetscObject)lme)->prefix?((PetscObject)lme)->prefix:"",LMEConvergedReasons[lme->reason],lme->its));
    PetscCall(PetscViewerASCIISubtractTab(viewer,((PetscObject)lme)->tablevel));
  }
  PetscFunctionReturn(0);
}

/*@
   LMEConvergedReasonViewFromOptions - Processes command line options to determine if/how
   the LME converged reason is to be viewed.

   Collective on lme

   Input Parameter:
.  lme - the linear matrix equation context

   Level: developer

.seealso: LMEConvergedReasonView()
@*/
PetscErrorCode LMEConvergedReasonViewFromOptions(LME lme)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)lme),((PetscObject)lme)->options,((PetscObject)lme)->prefix,"-lme_converged_reason",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(LMEConvergedReasonView(lme,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   LMECreate - Creates the default LME context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  outlme - location to put the LME context

   Note:
   The default LME type is LMEKRYLOV

   Level: beginner

.seealso: LMESetUp(), LMESolve(), LMEDestroy(), LME
@*/
PetscErrorCode LMECreate(MPI_Comm comm,LME *outlme)
{
  LME            lme;

  PetscFunctionBegin;
  PetscValidPointer(outlme,2);
  *outlme = NULL;
  PetscCall(LMEInitializePackage());
  PetscCall(SlepcHeaderCreate(lme,LME_CLASSID,"LME","Linear Matrix Equation","LME",comm,LMEDestroy,LMEView));

  lme->A               = NULL;
  lme->B               = NULL;
  lme->D               = NULL;
  lme->E               = NULL;
  lme->C               = NULL;
  lme->X               = NULL;
  lme->problem_type    = LME_LYAPUNOV;
  lme->max_it          = PETSC_DEFAULT;
  lme->ncv             = PETSC_DEFAULT;
  lme->tol             = PETSC_DEFAULT;
  lme->errorifnotconverged = PETSC_FALSE;

  lme->numbermonitors  = 0;

  lme->V               = NULL;
  lme->nwork           = 0;
  lme->work            = NULL;
  lme->data            = NULL;

  lme->its             = 0;
  lme->errest          = 0;
  lme->setupcalled     = 0;
  lme->reason          = LME_CONVERGED_ITERATING;

  *outlme = lme;
  PetscFunctionReturn(0);
}

/*@C
   LMESetType - Selects the particular solver to be used in the LME object.

   Logically Collective on lme

   Input Parameters:
+  lme  - the linear matrix equation context
-  type - a known method

   Options Database Key:
.  -lme_type <method> - Sets the method; use -help for a list
    of available methods

   Notes:
   See "slepc/include/slepclme.h" for available methods. The default
   is LMEKRYLOV

   Normally, it is best to use the LMESetFromOptions() command and
   then set the LME type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The LMESetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database.

   Level: intermediate

.seealso: LMEType
@*/
PetscErrorCode LMESetType(LME lme,LMEType type)
{
  PetscErrorCode (*r)(LME);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidCharPointer(type,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)lme,type,&match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(LMEList,type,&r));
  PetscCheck(r,PetscObjectComm((PetscObject)lme),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown LME type given: %s",type);

  PetscTryTypeMethod(lme,destroy);
  PetscCall(PetscMemzero(lme->ops,sizeof(struct _LMEOps)));

  lme->setupcalled = 0;
  PetscCall(PetscObjectChangeTypeName((PetscObject)lme,type));
  PetscCall((*r)(lme));
  PetscFunctionReturn(0);
}

/*@C
   LMEGetType - Gets the LME type as a string from the LME object.

   Not Collective

   Input Parameter:
.  lme - the linear matrix equation context

   Output Parameter:
.  type - name of LME method

   Level: intermediate

.seealso: LMESetType()
@*/
PetscErrorCode LMEGetType(LME lme,LMEType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)lme)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   LMERegister - Adds a method to the linear matrix equation solver package.

   Not Collective

   Input Parameters:
+  name - name of a new user-defined solver
-  function - routine to create the solver context

   Notes:
   LMERegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
    LMERegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     LMESetType(lme,"my_solver")
   or at runtime via the option
$     -lme_type my_solver

   Level: advanced

.seealso: LMERegisterAll()
@*/
PetscErrorCode LMERegister(const char *name,PetscErrorCode (*function)(LME))
{
  PetscFunctionBegin;
  PetscCall(LMEInitializePackage());
  PetscCall(PetscFunctionListAdd(&LMEList,name,function));
  PetscFunctionReturn(0);
}

/*@C
   LMEMonitorRegister - Adds LME monitor routine.

   Not Collective

   Input Parameters:
+  name    - name of a new monitor routine
.  vtype   - a PetscViewerType for the output
.  format  - a PetscViewerFormat for the output
.  monitor - monitor routine
.  create  - creation routine, or NULL
-  destroy - destruction routine, or NULL

   Notes:
   LMEMonitorRegister() may be called multiple times to add several user-defined monitors.

   Sample usage:
.vb
   LMEMonitorRegister("my_monitor",PETSCVIEWERASCII,PETSC_VIEWER_ASCII_INFO_DETAIL,MyMonitor,NULL,NULL);
.ve

   Then, your monitor can be chosen with the procedural interface via
$      LMEMonitorSetFromOptions(lme,"-lme_monitor_my_monitor","my_monitor",NULL)
   or at runtime via the option
$      -lme_monitor_my_monitor

   Level: advanced

.seealso: LMEMonitorRegisterAll()
@*/
PetscErrorCode LMEMonitorRegister(const char name[],PetscViewerType vtype,PetscViewerFormat format,PetscErrorCode (*monitor)(LME,PetscInt,PetscReal,PetscViewerAndFormat*),PetscErrorCode (*create)(PetscViewer,PetscViewerFormat,void*,PetscViewerAndFormat**),PetscErrorCode (*destroy)(PetscViewerAndFormat**))
{
  char           key[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(LMEInitializePackage());
  PetscCall(SlepcMonitorMakeKey_Internal(name,vtype,format,key));
  PetscCall(PetscFunctionListAdd(&LMEMonitorList,key,monitor));
  if (create)  PetscCall(PetscFunctionListAdd(&LMEMonitorCreateList,key,create));
  if (destroy) PetscCall(PetscFunctionListAdd(&LMEMonitorDestroyList,key,destroy));
  PetscFunctionReturn(0);
}

/*@
   LMEReset - Resets the LME context to the initial state (prior to setup)
   and destroys any allocated Vecs and Mats.

   Collective on lme

   Input Parameter:
.  lme - linear matrix equation context obtained from LMECreate()

   Level: advanced

.seealso: LMEDestroy()
@*/
PetscErrorCode LMEReset(LME lme)
{
  PetscFunctionBegin;
  if (lme) PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  if (!lme) PetscFunctionReturn(0);
  PetscTryTypeMethod(lme,reset);
  PetscCall(MatDestroy(&lme->A));
  PetscCall(MatDestroy(&lme->B));
  PetscCall(MatDestroy(&lme->D));
  PetscCall(MatDestroy(&lme->E));
  PetscCall(MatDestroy(&lme->C));
  PetscCall(MatDestroy(&lme->X));
  PetscCall(BVDestroy(&lme->V));
  PetscCall(VecDestroyVecs(lme->nwork,&lme->work));
  lme->nwork = 0;
  lme->setupcalled = 0;
  PetscFunctionReturn(0);
}

/*@C
   LMEDestroy - Destroys the LME context.

   Collective on lme

   Input Parameter:
.  lme - linear matrix equation context obtained from LMECreate()

   Level: beginner

.seealso: LMECreate(), LMESetUp(), LMESolve()
@*/
PetscErrorCode LMEDestroy(LME *lme)
{
  PetscFunctionBegin;
  if (!*lme) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*lme,LME_CLASSID,1);
  if (--((PetscObject)(*lme))->refct > 0) { *lme = NULL; PetscFunctionReturn(0); }
  PetscCall(LMEReset(*lme));
  PetscTryTypeMethod(*lme,destroy);
  PetscCall(LMEMonitorCancel(*lme));
  PetscCall(PetscHeaderDestroy(lme));
  PetscFunctionReturn(0);
}

/*@
   LMESetBV - Associates a basis vectors object to the linear matrix equation solver.

   Collective on lme

   Input Parameters:
+  lme - linear matrix equation context obtained from LMECreate()
-  bv  - the basis vectors object

   Note:
   Use LMEGetBV() to retrieve the basis vectors context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: LMEGetBV()
@*/
PetscErrorCode LMESetBV(LME lme,BV bv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidHeaderSpecific(bv,BV_CLASSID,2);
  PetscCheckSameComm(lme,1,bv,2);
  PetscCall(PetscObjectReference((PetscObject)bv));
  PetscCall(BVDestroy(&lme->V));
  lme->V = bv;
  PetscFunctionReturn(0);
}

/*@
   LMEGetBV - Obtain the basis vectors object associated to the matrix
   function solver.

   Not Collective

   Input Parameters:
.  lme - linear matrix equation context obtained from LMECreate()

   Output Parameter:
.  bv - basis vectors context

   Level: advanced

.seealso: LMESetBV()
@*/
PetscErrorCode LMEGetBV(LME lme,BV *bv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(lme,LME_CLASSID,1);
  PetscValidPointer(bv,2);
  if (!lme->V) {
    PetscCall(BVCreate(PetscObjectComm((PetscObject)lme),&lme->V));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)lme->V,(PetscObject)lme,0));
    PetscCall(PetscObjectSetOptions((PetscObject)lme->V,((PetscObject)lme)->options));
  }
  *bv = lme->V;
  PetscFunctionReturn(0);
}
