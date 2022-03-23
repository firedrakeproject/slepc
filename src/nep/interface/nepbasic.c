/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Basic NEP routines
*/

#include <slepc/private/nepimpl.h>      /*I "slepcnep.h" I*/

/* Logging support */
PetscClassId      NEP_CLASSID = 0;
PetscLogEvent     NEP_SetUp = 0,NEP_Solve = 0,NEP_Refine = 0,NEP_FunctionEval = 0,NEP_JacobianEval = 0,NEP_Resolvent = 0,NEP_CISS_SVD = 0;

/* List of registered NEP routines */
PetscFunctionList NEPList = 0;
PetscBool         NEPRegisterAllCalled = PETSC_FALSE;

/* List of registered NEP monitors */
PetscFunctionList NEPMonitorList              = NULL;
PetscFunctionList NEPMonitorCreateList        = NULL;
PetscFunctionList NEPMonitorDestroyList       = NULL;
PetscBool         NEPMonitorRegisterAllCalled = PETSC_FALSE;

/*@
   NEPCreate - Creates the default NEP context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  outnep - location to put the NEP context

   Level: beginner

.seealso: NEPSetUp(), NEPSolve(), NEPDestroy(), NEP
@*/
PetscErrorCode NEPCreate(MPI_Comm comm,NEP *outnep)
{
  NEP            nep;

  PetscFunctionBegin;
  PetscValidPointer(outnep,2);
  *outnep = 0;
  CHKERRQ(NEPInitializePackage());
  CHKERRQ(SlepcHeaderCreate(nep,NEP_CLASSID,"NEP","Nonlinear Eigenvalue Problem","NEP",comm,NEPDestroy,NEPView));

  nep->max_it          = PETSC_DEFAULT;
  nep->nev             = 1;
  nep->ncv             = PETSC_DEFAULT;
  nep->mpd             = PETSC_DEFAULT;
  nep->nini            = 0;
  nep->target          = 0.0;
  nep->tol             = PETSC_DEFAULT;
  nep->conv            = NEP_CONV_REL;
  nep->stop            = NEP_STOP_BASIC;
  nep->which           = (NEPWhich)0;
  nep->problem_type    = (NEPProblemType)0;
  nep->refine          = NEP_REFINE_NONE;
  nep->npart           = 1;
  nep->rtol            = PETSC_DEFAULT;
  nep->rits            = PETSC_DEFAULT;
  nep->scheme          = (NEPRefineScheme)0;
  nep->trackall        = PETSC_FALSE;
  nep->twosided        = PETSC_FALSE;

  nep->computefunction = NULL;
  nep->computejacobian = NULL;
  nep->functionctx     = NULL;
  nep->jacobianctx     = NULL;
  nep->converged       = NEPConvergedRelative;
  nep->convergeduser   = NULL;
  nep->convergeddestroy= NULL;
  nep->stopping        = NEPStoppingBasic;
  nep->stoppinguser    = NULL;
  nep->stoppingdestroy = NULL;
  nep->convergedctx    = NULL;
  nep->stoppingctx     = NULL;
  nep->numbermonitors  = 0;

  nep->ds              = NULL;
  nep->V               = NULL;
  nep->W               = NULL;
  nep->rg              = NULL;
  nep->function        = NULL;
  nep->function_pre    = NULL;
  nep->jacobian        = NULL;
  nep->A               = NULL;
  nep->f               = NULL;
  nep->nt              = 0;
  nep->mstr            = UNKNOWN_NONZERO_PATTERN;
  nep->P               = NULL;
  nep->mstrp           = UNKNOWN_NONZERO_PATTERN;
  nep->IS              = NULL;
  nep->eigr            = NULL;
  nep->eigi            = NULL;
  nep->errest          = NULL;
  nep->perm            = NULL;
  nep->nwork           = 0;
  nep->work            = NULL;
  nep->data            = NULL;

  nep->state           = NEP_STATE_INITIAL;
  nep->nconv           = 0;
  nep->its             = 0;
  nep->n               = 0;
  nep->nloc            = 0;
  nep->nrma            = NULL;
  nep->fui             = (NEPUserInterface)0;
  nep->useds           = PETSC_FALSE;
  nep->resolvent       = NULL;
  nep->reason          = NEP_CONVERGED_ITERATING;

  CHKERRQ(PetscNewLog(nep,&nep->sc));
  *outnep = nep;
  PetscFunctionReturn(0);
}

/*@C
   NEPSetType - Selects the particular solver to be used in the NEP object.

   Logically Collective on nep

   Input Parameters:
+  nep      - the nonlinear eigensolver context
-  type     - a known method

   Options Database Key:
.  -nep_type <method> - Sets the method; use -help for a list
    of available methods

   Notes:
   See "slepc/include/slepcnep.h" for available methods.

   Normally, it is best to use the NEPSetFromOptions() command and
   then set the NEP type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The NEPSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database.

   Level: intermediate

.seealso: NEPType
@*/
PetscErrorCode NEPSetType(NEP nep,NEPType type)
{
  PetscErrorCode (*r)(NEP);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidCharPointer(type,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)nep,type,&match));
  if (match) PetscFunctionReturn(0);

  CHKERRQ(PetscFunctionListFind(NEPList,type,&r));
  PetscCheck(r,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown NEP type given: %s",type);

  if (nep->ops->destroy) CHKERRQ((*nep->ops->destroy)(nep));
  CHKERRQ(PetscMemzero(nep->ops,sizeof(struct _NEPOps)));

  nep->state = NEP_STATE_INITIAL;
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)nep,type));
  CHKERRQ((*r)(nep));
  PetscFunctionReturn(0);
}

/*@C
   NEPGetType - Gets the NEP type as a string from the NEP object.

   Not Collective

   Input Parameter:
.  nep - the eigensolver context

   Output Parameter:
.  type - name of NEP method

   Level: intermediate

.seealso: NEPSetType()
@*/
PetscErrorCode NEPGetType(NEP nep,NEPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)nep)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   NEPRegister - Adds a method to the nonlinear eigenproblem solver package.

   Not Collective

   Input Parameters:
+  name - name of a new user-defined solver
-  function - routine to create the solver context

   Notes:
   NEPRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
    NEPRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     NEPSetType(nep,"my_solver")
   or at runtime via the option
$     -nep_type my_solver

   Level: advanced

.seealso: NEPRegisterAll()
@*/
PetscErrorCode NEPRegister(const char *name,PetscErrorCode (*function)(NEP))
{
  PetscFunctionBegin;
  CHKERRQ(NEPInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&NEPList,name,function));
  PetscFunctionReturn(0);
}

/*@C
   NEPMonitorRegister - Adds NEP monitor routine.

   Not Collective

   Input Parameters:
+  name    - name of a new monitor routine
.  vtype   - a PetscViewerType for the output
.  format  - a PetscViewerFormat for the output
.  monitor - monitor routine
.  create  - creation routine, or NULL
-  destroy - destruction routine, or NULL

   Notes:
   NEPMonitorRegister() may be called multiple times to add several user-defined monitors.

   Sample usage:
.vb
   NEPMonitorRegister("my_monitor",PETSCVIEWERASCII,PETSC_VIEWER_ASCII_INFO_DETAIL,MyMonitor,NULL,NULL);
.ve

   Then, your monitor can be chosen with the procedural interface via
$      NEPMonitorSetFromOptions(nep,"-nep_monitor_my_monitor","my_monitor",NULL)
   or at runtime via the option
$      -nep_monitor_my_monitor

   Level: advanced

.seealso: NEPMonitorRegisterAll()
@*/
PetscErrorCode NEPMonitorRegister(const char name[],PetscViewerType vtype,PetscViewerFormat format,PetscErrorCode (*monitor)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*),PetscErrorCode (*create)(PetscViewer,PetscViewerFormat,void*,PetscViewerAndFormat**),PetscErrorCode (*destroy)(PetscViewerAndFormat**))
{
  char           key[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  CHKERRQ(NEPInitializePackage());
  CHKERRQ(SlepcMonitorMakeKey_Internal(name,vtype,format,key));
  CHKERRQ(PetscFunctionListAdd(&NEPMonitorList,key,monitor));
  if (create)  CHKERRQ(PetscFunctionListAdd(&NEPMonitorCreateList,key,create));
  if (destroy) CHKERRQ(PetscFunctionListAdd(&NEPMonitorDestroyList,key,destroy));
  PetscFunctionReturn(0);
}

/*
   NEPReset_Problem - Destroys the problem matrices.
*/
PetscErrorCode NEPReset_Problem(NEP nep)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  CHKERRQ(MatDestroy(&nep->function));
  CHKERRQ(MatDestroy(&nep->function_pre));
  CHKERRQ(MatDestroy(&nep->jacobian));
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    CHKERRQ(MatDestroyMatrices(nep->nt,&nep->A));
    for (i=0;i<nep->nt;i++) CHKERRQ(FNDestroy(&nep->f[i]));
    CHKERRQ(PetscFree(nep->f));
    CHKERRQ(PetscFree(nep->nrma));
    if (nep->P) CHKERRQ(MatDestroyMatrices(nep->nt,&nep->P));
    nep->nt = 0;
  }
  PetscFunctionReturn(0);
}
/*@
   NEPReset - Resets the NEP context to the initial state (prior to setup)
   and destroys any allocated Vecs and Mats.

   Collective on nep

   Input Parameter:
.  nep - eigensolver context obtained from NEPCreate()

   Level: advanced

.seealso: NEPDestroy()
@*/
PetscErrorCode NEPReset(NEP nep)
{
  PetscFunctionBegin;
  if (nep) PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!nep) PetscFunctionReturn(0);
  if (nep->ops->reset) CHKERRQ((nep->ops->reset)(nep));
  if (nep->refineksp) CHKERRQ(KSPReset(nep->refineksp));
  CHKERRQ(NEPReset_Problem(nep));
  CHKERRQ(BVDestroy(&nep->V));
  CHKERRQ(BVDestroy(&nep->W));
  CHKERRQ(VecDestroyVecs(nep->nwork,&nep->work));
  CHKERRQ(MatDestroy(&nep->resolvent));
  nep->nwork = 0;
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@C
   NEPDestroy - Destroys the NEP context.

   Collective on nep

   Input Parameter:
.  nep - eigensolver context obtained from NEPCreate()

   Level: beginner

.seealso: NEPCreate(), NEPSetUp(), NEPSolve()
@*/
PetscErrorCode NEPDestroy(NEP *nep)
{
  PetscFunctionBegin;
  if (!*nep) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*nep,NEP_CLASSID,1);
  if (--((PetscObject)(*nep))->refct > 0) { *nep = 0; PetscFunctionReturn(0); }
  CHKERRQ(NEPReset(*nep));
  if ((*nep)->ops->destroy) CHKERRQ((*(*nep)->ops->destroy)(*nep));
  if ((*nep)->eigr) CHKERRQ(PetscFree4((*nep)->eigr,(*nep)->eigi,(*nep)->errest,(*nep)->perm));
  CHKERRQ(RGDestroy(&(*nep)->rg));
  CHKERRQ(DSDestroy(&(*nep)->ds));
  CHKERRQ(KSPDestroy(&(*nep)->refineksp));
  CHKERRQ(PetscSubcommDestroy(&(*nep)->refinesubc));
  CHKERRQ(PetscFree((*nep)->sc));
  /* just in case the initial vectors have not been used */
  CHKERRQ(SlepcBasisDestroy_Private(&(*nep)->nini,&(*nep)->IS));
  if ((*nep)->convergeddestroy) CHKERRQ((*(*nep)->convergeddestroy)((*nep)->convergedctx));
  CHKERRQ(NEPMonitorCancel(*nep));
  CHKERRQ(PetscHeaderDestroy(nep));
  PetscFunctionReturn(0);
}

/*@
   NEPSetBV - Associates a basis vectors object to the nonlinear eigensolver.

   Collective on nep

   Input Parameters:
+  nep - eigensolver context obtained from NEPCreate()
-  bv  - the basis vectors object

   Note:
   Use NEPGetBV() to retrieve the basis vectors context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: NEPGetBV()
@*/
PetscErrorCode NEPSetBV(NEP nep,BV bv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(bv,BV_CLASSID,2);
  PetscCheckSameComm(nep,1,bv,2);
  CHKERRQ(PetscObjectReference((PetscObject)bv));
  CHKERRQ(BVDestroy(&nep->V));
  nep->V = bv;
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->V));
  PetscFunctionReturn(0);
}

/*@
   NEPGetBV - Obtain the basis vectors object associated to the nonlinear
   eigensolver object.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  bv - basis vectors context

   Level: advanced

.seealso: NEPSetBV()
@*/
PetscErrorCode NEPGetBV(NEP nep,BV *bv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(bv,2);
  if (!nep->V) {
    CHKERRQ(BVCreate(PetscObjectComm((PetscObject)nep),&nep->V));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)nep->V,(PetscObject)nep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->V));
    CHKERRQ(PetscObjectSetOptions((PetscObject)nep->V,((PetscObject)nep)->options));
  }
  *bv = nep->V;
  PetscFunctionReturn(0);
}

/*@
   NEPSetRG - Associates a region object to the nonlinear eigensolver.

   Collective on nep

   Input Parameters:
+  nep - eigensolver context obtained from NEPCreate()
-  rg  - the region object

   Note:
   Use NEPGetRG() to retrieve the region context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: NEPGetRG()
@*/
PetscErrorCode NEPSetRG(NEP nep,RG rg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (rg) {
    PetscValidHeaderSpecific(rg,RG_CLASSID,2);
    PetscCheckSameComm(nep,1,rg,2);
  }
  CHKERRQ(PetscObjectReference((PetscObject)rg));
  CHKERRQ(RGDestroy(&nep->rg));
  nep->rg = rg;
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->rg));
  PetscFunctionReturn(0);
}

/*@
   NEPGetRG - Obtain the region object associated to the
   nonlinear eigensolver object.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  rg - region context

   Level: advanced

.seealso: NEPSetRG()
@*/
PetscErrorCode NEPGetRG(NEP nep,RG *rg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(rg,2);
  if (!nep->rg) {
    CHKERRQ(RGCreate(PetscObjectComm((PetscObject)nep),&nep->rg));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)nep->rg,(PetscObject)nep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->rg));
    CHKERRQ(PetscObjectSetOptions((PetscObject)nep->rg,((PetscObject)nep)->options));
  }
  *rg = nep->rg;
  PetscFunctionReturn(0);
}

/*@
   NEPSetDS - Associates a direct solver object to the nonlinear eigensolver.

   Collective on nep

   Input Parameters:
+  nep - eigensolver context obtained from NEPCreate()
-  ds  - the direct solver object

   Note:
   Use NEPGetDS() to retrieve the direct solver context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: NEPGetDS()
@*/
PetscErrorCode NEPSetDS(NEP nep,DS ds)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(ds,DS_CLASSID,2);
  PetscCheckSameComm(nep,1,ds,2);
  CHKERRQ(PetscObjectReference((PetscObject)ds));
  CHKERRQ(DSDestroy(&nep->ds));
  nep->ds = ds;
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->ds));
  PetscFunctionReturn(0);
}

/*@
   NEPGetDS - Obtain the direct solver object associated to the
   nonlinear eigensolver object.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  ds - direct solver context

   Level: advanced

.seealso: NEPSetDS()
@*/
PetscErrorCode NEPGetDS(NEP nep,DS *ds)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ds,2);
  if (!nep->ds) {
    CHKERRQ(DSCreate(PetscObjectComm((PetscObject)nep),&nep->ds));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)nep->ds,(PetscObject)nep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->ds));
    CHKERRQ(PetscObjectSetOptions((PetscObject)nep->ds,((PetscObject)nep)->options));
  }
  *ds = nep->ds;
  PetscFunctionReturn(0);
}

/*@
   NEPRefineGetKSP - Obtain the ksp object used by the eigensolver
   object in the refinement phase.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  ksp - ksp context

   Level: advanced

.seealso: NEPSetRefine()
@*/
PetscErrorCode NEPRefineGetKSP(NEP nep,KSP *ksp)
{
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  if (!nep->refineksp) {
    if (nep->npart>1) {
      /* Split in subcomunicators */
      CHKERRQ(PetscSubcommCreate(PetscObjectComm((PetscObject)nep),&nep->refinesubc));
      CHKERRQ(PetscSubcommSetNumber(nep->refinesubc,nep->npart));
      CHKERRQ(PetscSubcommSetType(nep->refinesubc,PETSC_SUBCOMM_CONTIGUOUS));
      CHKERRQ(PetscLogObjectMemory((PetscObject)nep,sizeof(PetscSubcomm)));
      CHKERRQ(PetscSubcommGetChild(nep->refinesubc,&comm));
    } else CHKERRQ(PetscObjectGetComm((PetscObject)nep,&comm));
    CHKERRQ(KSPCreate(comm,&nep->refineksp));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)nep->refineksp,(PetscObject)nep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->refineksp));
    CHKERRQ(PetscObjectSetOptions((PetscObject)nep->refineksp,((PetscObject)nep)->options));
    CHKERRQ(KSPSetOptionsPrefix(*ksp,((PetscObject)nep)->prefix));
    CHKERRQ(KSPAppendOptionsPrefix(*ksp,"nep_refine_"));
    CHKERRQ(KSPSetTolerances(nep->refineksp,SlepcDefaultTol(nep->rtol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }
  *ksp = nep->refineksp;
  PetscFunctionReturn(0);
}

/*@
   NEPSetTarget - Sets the value of the target.

   Logically Collective on nep

   Input Parameters:
+  nep    - eigensolver context
-  target - the value of the target

   Options Database Key:
.  -nep_target <scalar> - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest. It is used in combination with NEPSetWhichEigenpairs().

   In the case of complex scalars, a complex value can be provided in the
   command line with [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -nep_target 1.0+2.0i

   Level: intermediate

.seealso: NEPGetTarget(), NEPSetWhichEigenpairs()
@*/
PetscErrorCode NEPSetTarget(NEP nep,PetscScalar target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(nep,target,2);
  nep->target = target;
  PetscFunctionReturn(0);
}

/*@
   NEPGetTarget - Gets the value of the target.

   Not Collective

   Input Parameter:
.  nep - eigensolver context

   Output Parameter:
.  target - the value of the target

   Note:
   If the target was not set by the user, then zero is returned.

   Level: intermediate

.seealso: NEPSetTarget()
@*/
PetscErrorCode NEPGetTarget(NEP nep,PetscScalar* target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidScalarPointer(target,2);
  *target = nep->target;
  PetscFunctionReturn(0);
}

/*@C
   NEPSetFunction - Sets the function to compute the nonlinear Function T(lambda)
   as well as the location to store the matrix.

   Logically Collective on nep

   Input Parameters:
+  nep - the NEP context
.  A   - Function matrix
.  B   - preconditioner matrix (usually same as A)
.  fun - Function evaluation routine (if NULL then NEP retains any
         previously set value)
-  ctx - [optional] user-defined context for private data for the Function
         evaluation routine (may be NULL) (if NULL then NEP retains any
         previously set value)

   Calling Sequence of fun:
$   fun(NEP nep,PetscScalar lambda,Mat F,Mat P,void *ctx)

+  nep    - the NEP context
.  lambda - the scalar argument where T(.) must be evaluated
.  T      - matrix that will contain T(lambda)
.  P      - (optional) different matrix to build the preconditioner
-  ctx    - (optional) user-defined context, as set by NEPSetFunction()

   Level: beginner

.seealso: NEPGetFunction(), NEPSetJacobian()
@*/
PetscErrorCode NEPSetFunction(NEP nep,Mat A,Mat B,PetscErrorCode (*fun)(NEP,PetscScalar,Mat,Mat,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  if (A) PetscCheckSameComm(nep,1,A,2);
  if (B) PetscCheckSameComm(nep,1,B,3);

  if (nep->state) CHKERRQ(NEPReset(nep));
  else if (nep->fui && nep->fui!=NEP_USER_INTERFACE_CALLBACK) CHKERRQ(NEPReset_Problem(nep));

  if (fun) nep->computefunction = fun;
  if (ctx) nep->functionctx     = ctx;
  if (A) {
    CHKERRQ(PetscObjectReference((PetscObject)A));
    CHKERRQ(MatDestroy(&nep->function));
    nep->function = A;
  }
  if (B) {
    CHKERRQ(PetscObjectReference((PetscObject)B));
    CHKERRQ(MatDestroy(&nep->function_pre));
    nep->function_pre = B;
  }
  nep->fui   = NEP_USER_INTERFACE_CALLBACK;
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@C
   NEPGetFunction - Returns the Function matrix and optionally the user
   provided context for evaluating the Function.

   Not Collective, but Mat object will be parallel if NEP object is

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  A   - location to stash Function matrix (or NULL)
.  B   - location to stash preconditioner matrix (or NULL)
.  fun - location to put Function function (or NULL)
-  ctx - location to stash Function context (or NULL)

   Level: advanced

.seealso: NEPSetFunction()
@*/
PetscErrorCode NEPGetFunction(NEP nep,Mat *A,Mat *B,PetscErrorCode (**fun)(NEP,PetscScalar,Mat,Mat,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  NEPCheckCallback(nep,1);
  if (A)   *A   = nep->function;
  if (B)   *B   = nep->function_pre;
  if (fun) *fun = nep->computefunction;
  if (ctx) *ctx = nep->functionctx;
  PetscFunctionReturn(0);
}

/*@C
   NEPSetJacobian - Sets the function to compute the Jacobian T'(lambda) as well
   as the location to store the matrix.

   Logically Collective on nep

   Input Parameters:
+  nep - the NEP context
.  A   - Jacobian matrix
.  jac - Jacobian evaluation routine (if NULL then NEP retains any
         previously set value)
-  ctx - [optional] user-defined context for private data for the Jacobian
         evaluation routine (may be NULL) (if NULL then NEP retains any
         previously set value)

   Calling Sequence of jac:
$   jac(NEP nep,PetscScalar lambda,Mat J,void *ctx)

+  nep    - the NEP context
.  lambda - the scalar argument where T'(.) must be evaluated
.  J      - matrix that will contain T'(lambda)
-  ctx    - (optional) user-defined context, as set by NEPSetJacobian()

   Level: beginner

.seealso: NEPSetFunction(), NEPGetJacobian()
@*/
PetscErrorCode NEPSetJacobian(NEP nep,Mat A,PetscErrorCode (*jac)(NEP,PetscScalar,Mat,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (A) PetscCheckSameComm(nep,1,A,2);

  if (nep->state) CHKERRQ(NEPReset(nep));
  else if (nep->fui && nep->fui!=NEP_USER_INTERFACE_CALLBACK) CHKERRQ(NEPReset_Problem(nep));

  if (jac) nep->computejacobian = jac;
  if (ctx) nep->jacobianctx     = ctx;
  if (A) {
    CHKERRQ(PetscObjectReference((PetscObject)A));
    CHKERRQ(MatDestroy(&nep->jacobian));
    nep->jacobian = A;
  }
  nep->fui   = NEP_USER_INTERFACE_CALLBACK;
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@C
   NEPGetJacobian - Returns the Jacobian matrix and optionally the user
   provided routine and context for evaluating the Jacobian.

   Not Collective, but Mat object will be parallel if NEP object is

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  A   - location to stash Jacobian matrix (or NULL)
.  jac - location to put Jacobian function (or NULL)
-  ctx - location to stash Jacobian context (or NULL)

   Level: advanced

.seealso: NEPSetJacobian()
@*/
PetscErrorCode NEPGetJacobian(NEP nep,Mat *A,PetscErrorCode (**jac)(NEP,PetscScalar,Mat,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  NEPCheckCallback(nep,1);
  if (A)   *A   = nep->jacobian;
  if (jac) *jac = nep->computejacobian;
  if (ctx) *ctx = nep->jacobianctx;
  PetscFunctionReturn(0);
}

/*@
   NEPSetSplitOperator - Sets the operator of the nonlinear eigenvalue problem
   in split form.

   Collective on nep

   Input Parameters:
+  nep - the nonlinear eigensolver context
.  nt  - number of terms in the split form
.  A   - array of matrices
.  f   - array of functions
-  str - structure flag for matrices

   Notes:
   The nonlinear operator is written as T(lambda) = sum_i A_i*f_i(lambda),
   for i=1,...,n. The derivative T'(lambda) can be obtained using the
   derivatives of f_i.

   The structure flag provides information about A_i's nonzero pattern
   (see MatStructure enum). If all matrices have the same pattern, then
   use SAME_NONZERO_PATTERN. If the patterns are different but contained
   in the pattern of the first one, then use SUBSET_NONZERO_PATTERN. If
   patterns are known to be different, use DIFFERENT_NONZERO_PATTERN.
   If set to UNKNOWN_NONZERO_PATTERN, the patterns will be compared to
   determine if they are equal.

   This function must be called before NEPSetUp(). If it is called again
   after NEPSetUp() then the NEP object is reset.

   Level: beginner

.seealso: NEPGetSplitOperatorTerm(), NEPGetSplitOperatorInfo(), NEPSetSplitPreconditioner()
 @*/
PetscErrorCode NEPSetSplitOperator(NEP nep,PetscInt nt,Mat A[],FN f[],MatStructure str)
{
  PetscInt       i,n=0,m,m0=0,mloc,nloc,mloc0=0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,nt,2);
  PetscCheck(nt>0,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Must have one or more terms, you have %" PetscInt_FMT,nt);
  PetscValidPointer(A,3);
  PetscValidPointer(f,4);
  PetscValidLogicalCollectiveEnum(nep,str,5);

  for (i=0;i<nt;i++) {
    PetscValidHeaderSpecific(A[i],MAT_CLASSID,3);
    PetscCheckSameComm(nep,1,A[i],3);
    PetscValidHeaderSpecific(f[i],FN_CLASSID,4);
    PetscCheckSameComm(nep,1,f[i],4);
    CHKERRQ(MatGetSize(A[i],&m,&n));
    CHKERRQ(MatGetLocalSize(A[i],&mloc,&nloc));
    PetscCheck(m==n,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_WRONG,"A[%" PetscInt_FMT "] is a non-square matrix (%" PetscInt_FMT " rows, %" PetscInt_FMT " cols)",i,m,n);
    PetscCheck(mloc==nloc,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_WRONG,"A[%" PetscInt_FMT "] does not have equal row and column local sizes (%" PetscInt_FMT ", %" PetscInt_FMT ")",i,mloc,nloc);
    if (!i) { m0 = m; mloc0 = mloc; }
    PetscCheck(m==m0,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_INCOMP,"Dimensions of A[%" PetscInt_FMT "] do not match with previous matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")",i,m,m0);
    PetscCheck(mloc==mloc0,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_INCOMP,"Local dimensions of A[%" PetscInt_FMT "] do not match with previous matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")",i,mloc,mloc0);
    CHKERRQ(PetscObjectReference((PetscObject)A[i]));
    CHKERRQ(PetscObjectReference((PetscObject)f[i]));
  }

  if (nep->state && (n!=nep->n || nloc!=nep->nloc)) CHKERRQ(NEPReset(nep));
  else CHKERRQ(NEPReset_Problem(nep));

  /* allocate space and copy matrices and functions */
  CHKERRQ(PetscMalloc1(nt,&nep->A));
  CHKERRQ(PetscLogObjectMemory((PetscObject)nep,nt*sizeof(Mat)));
  for (i=0;i<nt;i++) nep->A[i] = A[i];
  CHKERRQ(PetscMalloc1(nt,&nep->f));
  CHKERRQ(PetscLogObjectMemory((PetscObject)nep,nt*sizeof(FN)));
  for (i=0;i<nt;i++) nep->f[i] = f[i];
  CHKERRQ(PetscCalloc1(nt,&nep->nrma));
  CHKERRQ(PetscLogObjectMemory((PetscObject)nep,nt*sizeof(PetscReal)));
  nep->nt    = nt;
  nep->mstr  = str;
  nep->fui   = NEP_USER_INTERFACE_SPLIT;
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPGetSplitOperatorTerm - Gets the matrices and functions associated with
   the nonlinear operator in split form.

   Not collective, though parallel Mats and FNs are returned if the NEP is parallel

   Input Parameters:
+  nep - the nonlinear eigensolver context
-  k   - the index of the requested term (starting in 0)

   Output Parameters:
+  A - the matrix of the requested term
-  f - the function of the requested term

   Level: intermediate

.seealso: NEPSetSplitOperator(), NEPGetSplitOperatorInfo()
@*/
PetscErrorCode NEPGetSplitOperatorTerm(NEP nep,PetscInt k,Mat *A,FN *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,k,2);
  NEPCheckSplit(nep,1);
  PetscCheck(k>=0 && k<nep->nt,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %" PetscInt_FMT,nep->nt-1);
  if (A) *A = nep->A[k];
  if (f) *f = nep->f[k];
  PetscFunctionReturn(0);
}

/*@
   NEPGetSplitOperatorInfo - Returns the number of terms of the split form of
   the nonlinear operator, as well as the structure flag for matrices.

   Not collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  n   - the number of terms passed in NEPSetSplitOperator()
-  str - the matrix structure flag passed in NEPSetSplitOperator()

   Level: intermediate

.seealso: NEPSetSplitOperator(), NEPGetSplitOperatorTerm()
@*/
PetscErrorCode NEPGetSplitOperatorInfo(NEP nep,PetscInt *n,MatStructure *str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  NEPCheckSplit(nep,1);
  if (n)   *n = nep->nt;
  if (str) *str = nep->mstr;
  PetscFunctionReturn(0);
}

/*@
   NEPSetSplitPreconditioner - Sets an operator in split form from which
   to build the preconditioner to be used when solving the nonlinear
   eigenvalue problem in split form.

   Collective on nep

   Input Parameters:
+  nep  - the nonlinear eigensolver context
.  ntp  - number of terms in the split preconditioner
.  P    - array of matrices
-  strp - structure flag for matrices

   Notes:
   The matrix for the preconditioner is expressed as P(lambda) =
   sum_i P_i*f_i(lambda), for i=1,...,n, where the f_i functions
   are the same as in NEPSetSplitOperator(). It is not necessary to call
   this function. If it is not invoked, then the preconditioner is
   built from T(lambda), i.e., both matrices and functions passed in
   NEPSetSplitOperator().

   The structure flag provides information about P_i's nonzero pattern
   in the same way as in NEPSetSplitOperator().

   If the functions defining the preconditioner operator were different
   from the ones given in NEPSetSplitOperator(), then the split form
   cannot be used. Use the callback interface instead.

   Use ntp=0 to reset a previously set split preconditioner.

   Level: advanced

.seealso: NEPGetSplitPreconditionerTerm(), NEPGetSplitPreconditionerInfo(), NEPSetSplitOperator()
 @*/
PetscErrorCode NEPSetSplitPreconditioner(NEP nep,PetscInt ntp,Mat P[],MatStructure strp)
{
  PetscInt       i,n=0,m,m0=0,mloc,nloc,mloc0=0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,ntp,2);
  PetscCheck(ntp>=0,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Negative value of ntp = %" PetscInt_FMT,ntp);
  PetscCheck(nep->fui==NEP_USER_INTERFACE_SPLIT,PetscObjectComm((PetscObject)nep),PETSC_ERR_ORDER,"Must call NEPSetSplitOperator first");
  PetscCheck(ntp==0 || nep->nt==ntp,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"The number of terms must be the same as in NEPSetSplitOperator()");
  if (ntp) PetscValidPointer(P,3);
  PetscValidLogicalCollectiveEnum(nep,strp,4);

  for (i=0;i<ntp;i++) {
    PetscValidHeaderSpecific(P[i],MAT_CLASSID,3);
    PetscCheckSameComm(nep,1,P[i],3);
    CHKERRQ(MatGetSize(P[i],&m,&n));
    CHKERRQ(MatGetLocalSize(P[i],&mloc,&nloc));
    PetscCheck(m==n,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_WRONG,"P[%" PetscInt_FMT "] is a non-square matrix (%" PetscInt_FMT " rows, %" PetscInt_FMT " cols)",i,m,n);
    PetscCheck(mloc==nloc,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_WRONG,"P[%" PetscInt_FMT "] does not have equal row and column local sizes (%" PetscInt_FMT ", %" PetscInt_FMT ")",i,mloc,nloc);
    if (!i) { m0 = m; mloc0 = mloc; }
    PetscCheck(m==m0,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_INCOMP,"Dimensions of P[%" PetscInt_FMT "] do not match with previous matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")",i,m,m0);
    PetscCheck(mloc==mloc0,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_INCOMP,"Local dimensions of P[%" PetscInt_FMT "] do not match with previous matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")",i,mloc,mloc0);
    CHKERRQ(PetscObjectReference((PetscObject)P[i]));
  }

  PetscCheck(!nep->state,PetscObjectComm((PetscObject)nep),PETSC_ERR_ORDER,"To call this function after NEPSetUp(), you must call NEPSetSplitOperator() again");
  if (nep->P) CHKERRQ(MatDestroyMatrices(nep->nt,&nep->P));

  /* allocate space and copy matrices */
  if (ntp) {
    CHKERRQ(PetscMalloc1(ntp,&nep->P));
    CHKERRQ(PetscLogObjectMemory((PetscObject)nep,ntp*sizeof(Mat)));
    for (i=0;i<ntp;i++) nep->P[i] = P[i];
  }
  nep->mstrp = strp;
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPGetSplitPreconditionerTerm - Gets the matrices associated with
   the split preconditioner.

   Not collective, though parallel Mats are returned if the NEP is parallel

   Input Parameters:
+  nep - the nonlinear eigensolver context
-  k   - the index of the requested term (starting in 0)

   Output Parameter:
.  P  - the matrix of the requested term

   Level: advanced

.seealso: NEPSetSplitPreconditioner(), NEPGetSplitPreconditionerInfo()
@*/
PetscErrorCode NEPGetSplitPreconditionerTerm(NEP nep,PetscInt k,Mat *P)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,k,2);
  PetscValidPointer(P,3);
  NEPCheckSplit(nep,1);
  PetscCheck(k>=0 && k<nep->nt,PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %" PetscInt_FMT,nep->nt-1);
  PetscCheck(nep->P,PetscObjectComm((PetscObject)nep),PETSC_ERR_ORDER,"You have not called NEPSetSplitPreconditioner()");
  *P = nep->P[k];
  PetscFunctionReturn(0);
}

/*@
   NEPGetSplitPreconditionerInfo - Returns the number of terms of the split
   preconditioner, as well as the structure flag for matrices.

   Not collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  n    - the number of terms passed in NEPSetSplitPreconditioner()
-  strp - the matrix structure flag passed in NEPSetSplitPreconditioner()

   Level: advanced

.seealso: NEPSetSplitPreconditioner(), NEPGetSplitPreconditionerTerm()
@*/
PetscErrorCode NEPGetSplitPreconditionerInfo(NEP nep,PetscInt *n,MatStructure *strp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  NEPCheckSplit(nep,1);
  if (n)    *n    = nep->P? nep->nt: 0;
  if (strp) *strp = nep->mstrp;
  PetscFunctionReturn(0);
}
