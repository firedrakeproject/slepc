/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Basic PEP routines
*/

#include <slepc/private/pepimpl.h>      /*I "slepcpep.h" I*/

/* Logging support */
PetscClassId      PEP_CLASSID = 0;
PetscLogEvent     PEP_SetUp = 0,PEP_Solve = 0,PEP_Refine = 0,PEP_CISS_SVD = 0;

/* List of registered PEP routines */
PetscFunctionList PEPList = 0;
PetscBool         PEPRegisterAllCalled = PETSC_FALSE;

/* List of registered PEP monitors */
PetscFunctionList PEPMonitorList              = NULL;
PetscFunctionList PEPMonitorCreateList        = NULL;
PetscFunctionList PEPMonitorDestroyList       = NULL;
PetscBool         PEPMonitorRegisterAllCalled = PETSC_FALSE;

/*@
   PEPCreate - Creates the default PEP context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  outpep - location to put the PEP context

   Note:
   The default PEP type is PEPTOAR

   Level: beginner

.seealso: PEPSetUp(), PEPSolve(), PEPDestroy(), PEP
@*/
PetscErrorCode PEPCreate(MPI_Comm comm,PEP *outpep)
{
  PEP            pep;

  PetscFunctionBegin;
  PetscValidPointer(outpep,2);
  *outpep = 0;
  CHKERRQ(PEPInitializePackage());
  CHKERRQ(SlepcHeaderCreate(pep,PEP_CLASSID,"PEP","Polynomial Eigenvalue Problem","PEP",comm,PEPDestroy,PEPView));

  pep->max_it          = PETSC_DEFAULT;
  pep->nev             = 1;
  pep->ncv             = PETSC_DEFAULT;
  pep->mpd             = PETSC_DEFAULT;
  pep->nini            = 0;
  pep->target          = 0.0;
  pep->tol             = PETSC_DEFAULT;
  pep->conv            = PEP_CONV_REL;
  pep->stop            = PEP_STOP_BASIC;
  pep->which           = (PEPWhich)0;
  pep->basis           = PEP_BASIS_MONOMIAL;
  pep->problem_type    = (PEPProblemType)0;
  pep->scale           = PEP_SCALE_NONE;
  pep->sfactor         = 1.0;
  pep->dsfactor        = 1.0;
  pep->sits            = 5;
  pep->slambda         = 1.0;
  pep->refine          = PEP_REFINE_NONE;
  pep->npart           = 1;
  pep->rtol            = PETSC_DEFAULT;
  pep->rits            = PETSC_DEFAULT;
  pep->scheme          = (PEPRefineScheme)0;
  pep->extract         = (PEPExtract)0;
  pep->trackall        = PETSC_FALSE;

  pep->converged       = PEPConvergedRelative;
  pep->convergeduser   = NULL;
  pep->convergeddestroy= NULL;
  pep->stopping        = PEPStoppingBasic;
  pep->stoppinguser    = NULL;
  pep->stoppingdestroy = NULL;
  pep->convergedctx    = NULL;
  pep->stoppingctx     = NULL;
  pep->numbermonitors  = 0;

  pep->st              = NULL;
  pep->ds              = NULL;
  pep->V               = NULL;
  pep->rg              = NULL;
  pep->A               = NULL;
  pep->nmat            = 0;
  pep->Dl              = NULL;
  pep->Dr              = NULL;
  pep->IS              = NULL;
  pep->eigr            = NULL;
  pep->eigi            = NULL;
  pep->errest          = NULL;
  pep->perm            = NULL;
  pep->pbc             = NULL;
  pep->solvematcoeffs  = NULL;
  pep->nwork           = 0;
  pep->work            = NULL;
  pep->refineksp       = NULL;
  pep->refinesubc      = NULL;
  pep->data            = NULL;

  pep->state           = PEP_STATE_INITIAL;
  pep->nconv           = 0;
  pep->its             = 0;
  pep->n               = 0;
  pep->nloc            = 0;
  pep->nrma            = NULL;
  pep->sfactor_set     = PETSC_FALSE;
  pep->lineariz        = PETSC_FALSE;
  pep->reason          = PEP_CONVERGED_ITERATING;

  CHKERRQ(PetscNewLog(pep,&pep->sc));
  *outpep = pep;
  PetscFunctionReturn(0);
}

/*@C
   PEPSetType - Selects the particular solver to be used in the PEP object.

   Logically Collective on pep

   Input Parameters:
+  pep      - the polynomial eigensolver context
-  type     - a known method

   Options Database Key:
.  -pep_type <method> - Sets the method; use -help for a list
    of available methods

   Notes:
   See "slepc/include/slepcpep.h" for available methods. The default
   is PEPTOAR.

   Normally, it is best to use the PEPSetFromOptions() command and
   then set the PEP type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The PEPSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database.

   Level: intermediate

.seealso: PEPType
@*/
PetscErrorCode PEPSetType(PEP pep,PEPType type)
{
  PetscErrorCode (*r)(PEP);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidCharPointer(type,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)pep,type,&match));
  if (match) PetscFunctionReturn(0);

  CHKERRQ(PetscFunctionListFind(PEPList,type,&r));
  PetscCheck(r,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PEP type given: %s",type);

  if (pep->ops->destroy) CHKERRQ((*pep->ops->destroy)(pep));
  CHKERRQ(PetscMemzero(pep->ops,sizeof(struct _PEPOps)));

  pep->state = PEP_STATE_INITIAL;
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)pep,type));
  CHKERRQ((*r)(pep));
  PetscFunctionReturn(0);
}

/*@C
   PEPGetType - Gets the PEP type as a string from the PEP object.

   Not Collective

   Input Parameter:
.  pep - the eigensolver context

   Output Parameter:
.  type - name of PEP method

   Level: intermediate

.seealso: PEPSetType()
@*/
PetscErrorCode PEPGetType(PEP pep,PEPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)pep)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   PEPRegister - Adds a method to the polynomial eigenproblem solver package.

   Not Collective

   Input Parameters:
+  name - name of a new user-defined solver
-  function - routine to create the solver context

   Notes:
   PEPRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
    PEPRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PEPSetType(pep,"my_solver")
   or at runtime via the option
$     -pep_type my_solver

   Level: advanced

.seealso: PEPRegisterAll()
@*/
PetscErrorCode PEPRegister(const char *name,PetscErrorCode (*function)(PEP))
{
  PetscFunctionBegin;
  CHKERRQ(PEPInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&PEPList,name,function));
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorRegister - Adds PEP monitor routine.

   Not Collective

   Input Parameters:
+  name    - name of a new monitor routine
.  vtype   - a PetscViewerType for the output
.  format  - a PetscViewerFormat for the output
.  monitor - monitor routine
.  create  - creation routine, or NULL
-  destroy - destruction routine, or NULL

   Notes:
   PEPMonitorRegister() may be called multiple times to add several user-defined monitors.

   Sample usage:
.vb
   PEPMonitorRegister("my_monitor",PETSCVIEWERASCII,PETSC_VIEWER_ASCII_INFO_DETAIL,MyMonitor,NULL,NULL);
.ve

   Then, your monitor can be chosen with the procedural interface via
$      PEPMonitorSetFromOptions(pep,"-pep_monitor_my_monitor","my_monitor",NULL)
   or at runtime via the option
$      -pep_monitor_my_monitor

   Level: advanced

.seealso: PEPMonitorRegisterAll()
@*/
PetscErrorCode PEPMonitorRegister(const char name[],PetscViewerType vtype,PetscViewerFormat format,PetscErrorCode (*monitor)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*),PetscErrorCode (*create)(PetscViewer,PetscViewerFormat,void*,PetscViewerAndFormat**),PetscErrorCode (*destroy)(PetscViewerAndFormat**))
{
  char           key[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  CHKERRQ(PEPInitializePackage());
  CHKERRQ(SlepcMonitorMakeKey_Internal(name,vtype,format,key));
  CHKERRQ(PetscFunctionListAdd(&PEPMonitorList,key,monitor));
  if (create)  CHKERRQ(PetscFunctionListAdd(&PEPMonitorCreateList,key,create));
  if (destroy) CHKERRQ(PetscFunctionListAdd(&PEPMonitorDestroyList,key,destroy));
  PetscFunctionReturn(0);
}

/*@
   PEPReset - Resets the PEP context to the initial state (prior to setup)
   and destroys any allocated Vecs and Mats.

   Collective on pep

   Input Parameter:
.  pep - eigensolver context obtained from PEPCreate()

   Level: advanced

.seealso: PEPDestroy()
@*/
PetscErrorCode PEPReset(PEP pep)
{
  PetscFunctionBegin;
  if (pep) PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!pep) PetscFunctionReturn(0);
  if (pep->ops->reset) CHKERRQ((pep->ops->reset)(pep));
  if (pep->st) CHKERRQ(STReset(pep->st));
  if (pep->refineksp) CHKERRQ(KSPReset(pep->refineksp));
  if (pep->nmat) {
    CHKERRQ(MatDestroyMatrices(pep->nmat,&pep->A));
    CHKERRQ(PetscFree2(pep->pbc,pep->nrma));
    CHKERRQ(PetscFree(pep->solvematcoeffs));
    pep->nmat = 0;
  }
  CHKERRQ(VecDestroy(&pep->Dl));
  CHKERRQ(VecDestroy(&pep->Dr));
  CHKERRQ(BVDestroy(&pep->V));
  CHKERRQ(VecDestroyVecs(pep->nwork,&pep->work));
  pep->nwork = 0;
  pep->state = PEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@C
   PEPDestroy - Destroys the PEP context.

   Collective on pep

   Input Parameter:
.  pep - eigensolver context obtained from PEPCreate()

   Level: beginner

.seealso: PEPCreate(), PEPSetUp(), PEPSolve()
@*/
PetscErrorCode PEPDestroy(PEP *pep)
{
  PetscFunctionBegin;
  if (!*pep) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*pep,PEP_CLASSID,1);
  if (--((PetscObject)(*pep))->refct > 0) { *pep = 0; PetscFunctionReturn(0); }
  CHKERRQ(PEPReset(*pep));
  if ((*pep)->ops->destroy) CHKERRQ((*(*pep)->ops->destroy)(*pep));
  if ((*pep)->eigr) CHKERRQ(PetscFree4((*pep)->eigr,(*pep)->eigi,(*pep)->errest,(*pep)->perm));
  CHKERRQ(STDestroy(&(*pep)->st));
  CHKERRQ(RGDestroy(&(*pep)->rg));
  CHKERRQ(DSDestroy(&(*pep)->ds));
  CHKERRQ(KSPDestroy(&(*pep)->refineksp));
  CHKERRQ(PetscSubcommDestroy(&(*pep)->refinesubc));
  CHKERRQ(PetscFree((*pep)->sc));
  /* just in case the initial vectors have not been used */
  CHKERRQ(SlepcBasisDestroy_Private(&(*pep)->nini,&(*pep)->IS));
  if ((*pep)->convergeddestroy) CHKERRQ((*(*pep)->convergeddestroy)((*pep)->convergedctx));
  CHKERRQ(PEPMonitorCancel(*pep));
  CHKERRQ(PetscHeaderDestroy(pep));
  PetscFunctionReturn(0);
}

/*@
   PEPSetBV - Associates a basis vectors object to the polynomial eigensolver.

   Collective on pep

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  bv  - the basis vectors object

   Note:
   Use PEPGetBV() to retrieve the basis vectors context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: PEPGetBV()
@*/
PetscErrorCode PEPSetBV(PEP pep,BV bv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(bv,BV_CLASSID,2);
  PetscCheckSameComm(pep,1,bv,2);
  CHKERRQ(PetscObjectReference((PetscObject)bv));
  CHKERRQ(BVDestroy(&pep->V));
  pep->V = bv;
  CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->V));
  PetscFunctionReturn(0);
}

/*@
   PEPGetBV - Obtain the basis vectors object associated to the polynomial
   eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  bv - basis vectors context

   Level: advanced

.seealso: PEPSetBV()
@*/
PetscErrorCode PEPGetBV(PEP pep,BV *bv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(bv,2);
  if (!pep->V) {
    CHKERRQ(BVCreate(PetscObjectComm((PetscObject)pep),&pep->V));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pep->V,(PetscObject)pep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->V));
    CHKERRQ(PetscObjectSetOptions((PetscObject)pep->V,((PetscObject)pep)->options));
  }
  *bv = pep->V;
  PetscFunctionReturn(0);
}

/*@
   PEPSetRG - Associates a region object to the polynomial eigensolver.

   Collective on pep

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  rg  - the region object

   Note:
   Use PEPGetRG() to retrieve the region context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: PEPGetRG()
@*/
PetscErrorCode PEPSetRG(PEP pep,RG rg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (rg) {
    PetscValidHeaderSpecific(rg,RG_CLASSID,2);
    PetscCheckSameComm(pep,1,rg,2);
  }
  CHKERRQ(PetscObjectReference((PetscObject)rg));
  CHKERRQ(RGDestroy(&pep->rg));
  pep->rg = rg;
  CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->rg));
  PetscFunctionReturn(0);
}

/*@
   PEPGetRG - Obtain the region object associated to the
   polynomial eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  rg - region context

   Level: advanced

.seealso: PEPSetRG()
@*/
PetscErrorCode PEPGetRG(PEP pep,RG *rg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(rg,2);
  if (!pep->rg) {
    CHKERRQ(RGCreate(PetscObjectComm((PetscObject)pep),&pep->rg));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pep->rg,(PetscObject)pep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->rg));
    CHKERRQ(PetscObjectSetOptions((PetscObject)pep->rg,((PetscObject)pep)->options));
  }
  *rg = pep->rg;
  PetscFunctionReturn(0);
}

/*@
   PEPSetDS - Associates a direct solver object to the polynomial eigensolver.

   Collective on pep

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  ds  - the direct solver object

   Note:
   Use PEPGetDS() to retrieve the direct solver context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: PEPGetDS()
@*/
PetscErrorCode PEPSetDS(PEP pep,DS ds)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(ds,DS_CLASSID,2);
  PetscCheckSameComm(pep,1,ds,2);
  CHKERRQ(PetscObjectReference((PetscObject)ds));
  CHKERRQ(DSDestroy(&pep->ds));
  pep->ds = ds;
  CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->ds));
  PetscFunctionReturn(0);
}

/*@
   PEPGetDS - Obtain the direct solver object associated to the
   polynomial eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  ds - direct solver context

   Level: advanced

.seealso: PEPSetDS()
@*/
PetscErrorCode PEPGetDS(PEP pep,DS *ds)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(ds,2);
  if (!pep->ds) {
    CHKERRQ(DSCreate(PetscObjectComm((PetscObject)pep),&pep->ds));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pep->ds,(PetscObject)pep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->ds));
    CHKERRQ(PetscObjectSetOptions((PetscObject)pep->ds,((PetscObject)pep)->options));
  }
  *ds = pep->ds;
  PetscFunctionReturn(0);
}

/*@
   PEPSetST - Associates a spectral transformation object to the eigensolver.

   Collective on pep

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  st   - the spectral transformation object

   Note:
   Use PEPGetST() to retrieve the spectral transformation context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: PEPGetST()
@*/
PetscErrorCode PEPSetST(PEP pep,ST st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(st,ST_CLASSID,2);
  PetscCheckSameComm(pep,1,st,2);
  CHKERRQ(PetscObjectReference((PetscObject)st));
  CHKERRQ(STDestroy(&pep->st));
  pep->st = st;
  CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->st));
  PetscFunctionReturn(0);
}

/*@
   PEPGetST - Obtain the spectral transformation (ST) object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  st - spectral transformation context

   Level: intermediate

.seealso: PEPSetST()
@*/
PetscErrorCode PEPGetST(PEP pep,ST *st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(st,2);
  if (!pep->st) {
    CHKERRQ(STCreate(PetscObjectComm((PetscObject)pep),&pep->st));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pep->st,(PetscObject)pep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->st));
    CHKERRQ(PetscObjectSetOptions((PetscObject)pep->st,((PetscObject)pep)->options));
  }
  *st = pep->st;
  PetscFunctionReturn(0);
}

/*@
   PEPRefineGetKSP - Obtain the ksp object used by the eigensolver
   object in the refinement phase.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  ksp - ksp context

   Level: advanced

.seealso: PEPSetRefine()
@*/
PetscErrorCode PEPRefineGetKSP(PEP pep,KSP *ksp)
{
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  if (!pep->refineksp) {
    if (pep->npart>1) {
      /* Split in subcomunicators */
      CHKERRQ(PetscSubcommCreate(PetscObjectComm((PetscObject)pep),&pep->refinesubc));
      CHKERRQ(PetscSubcommSetNumber(pep->refinesubc,pep->npart));
      CHKERRQ(PetscSubcommSetType(pep->refinesubc,PETSC_SUBCOMM_CONTIGUOUS));
      CHKERRQ(PetscLogObjectMemory((PetscObject)pep,sizeof(PetscSubcomm)));
      CHKERRQ(PetscSubcommGetChild(pep->refinesubc,&comm));
    } else CHKERRQ(PetscObjectGetComm((PetscObject)pep,&comm));
    CHKERRQ(KSPCreate(comm,&pep->refineksp));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pep->refineksp,(PetscObject)pep,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->refineksp));
    CHKERRQ(PetscObjectSetOptions((PetscObject)pep->refineksp,((PetscObject)pep)->options));
    CHKERRQ(KSPSetOptionsPrefix(*ksp,((PetscObject)pep)->prefix));
    CHKERRQ(KSPAppendOptionsPrefix(*ksp,"pep_refine_"));
    CHKERRQ(KSPSetTolerances(pep->refineksp,SlepcDefaultTol(pep->rtol),PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }
  *ksp = pep->refineksp;
  PetscFunctionReturn(0);
}

/*@
   PEPSetTarget - Sets the value of the target.

   Logically Collective on pep

   Input Parameters:
+  pep    - eigensolver context
-  target - the value of the target

   Options Database Key:
.  -pep_target <scalar> - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest. It is used in combination with PEPSetWhichEigenpairs().

   In the case of complex scalars, a complex value can be provided in the
   command line with [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -pep_target 1.0+2.0i

   Level: intermediate

.seealso: PEPGetTarget(), PEPSetWhichEigenpairs()
@*/
PetscErrorCode PEPSetTarget(PEP pep,PetscScalar target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(pep,target,2);
  pep->target = target;
  if (!pep->st) CHKERRQ(PEPGetST(pep,&pep->st));
  CHKERRQ(STSetDefaultShift(pep->st,target));
  PetscFunctionReturn(0);
}

/*@
   PEPGetTarget - Gets the value of the target.

   Not Collective

   Input Parameter:
.  pep - eigensolver context

   Output Parameter:
.  target - the value of the target

   Note:
   If the target was not set by the user, then zero is returned.

   Level: intermediate

.seealso: PEPSetTarget()
@*/
PetscErrorCode PEPGetTarget(PEP pep,PetscScalar* target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidScalarPointer(target,2);
  *target = pep->target;
  PetscFunctionReturn(0);
}

/*@
   PEPSetInterval - Defines the computational interval for spectrum slicing.

   Logically Collective on pep

   Input Parameters:
+  pep  - eigensolver context
.  inta - left end of the interval
-  intb - right end of the interval

   Options Database Key:
.  -pep_interval <a,b> - set [a,b] as the interval of interest

   Notes:
   Spectrum slicing is a technique employed for computing all eigenvalues of
   symmetric eigenproblems in a given interval. This function provides the
   interval to be considered. It must be used in combination with PEP_ALL, see
   PEPSetWhichEigenpairs().

   In the command-line option, two values must be provided. For an open interval,
   one can give an infinite, e.g., -pep_interval 1.0,inf or -pep_interval -inf,1.0.
   An open interval in the programmatic interface can be specified with
   PETSC_MAX_REAL and -PETSC_MAX_REAL.

   Level: intermediate

.seealso: PEPGetInterval(), PEPSetWhichEigenpairs()
@*/
PetscErrorCode PEPSetInterval(PEP pep,PetscReal inta,PetscReal intb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,inta,2);
  PetscValidLogicalCollectiveReal(pep,intb,3);
  PetscCheck(inta<intb,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"Badly defined interval, must be inta<intb");
  if (pep->inta != inta || pep->intb != intb) {
    pep->inta = inta;
    pep->intb = intb;
    pep->state = PEP_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPGetInterval - Gets the computational interval for spectrum slicing.

   Not Collective

   Input Parameter:
.  pep - eigensolver context

   Output Parameters:
+  inta - left end of the interval
-  intb - right end of the interval

   Level: intermediate

   Note:
   If the interval was not set by the user, then zeros are returned.

.seealso: PEPSetInterval()
@*/
PetscErrorCode PEPGetInterval(PEP pep,PetscReal* inta,PetscReal* intb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (inta) *inta = pep->inta;
  if (intb) *intb = pep->intb;
  PetscFunctionReturn(0);
}
