/*
     The basic EPS routines, Create, View, etc. are here.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/eps/epsimpl.h"      /*I "slepceps.h" I*/

PetscFList EPSList = 0;
PetscCookie EPS_COOKIE = 0;
PetscLogEvent EPS_SetUp = 0, EPS_Solve = 0, EPS_Dense = 0;

#undef __FUNCT__  
#define __FUNCT__ "EPSInitializePackage"
/*@C
  EPSInitializePackage - This function initializes everything in the EPS package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to EPSCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode EPSInitializePackage(char *path) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Eigenproblem Solver",&EPS_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = EPSRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("EPSSetUp",EPS_COOKIE,&EPS_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("EPSSolve",EPS_COOKIE,&EPS_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("EPSDense",EPS_COOKIE,&EPS_Dense); CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "eps", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(EPS_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "eps", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(EPS_COOKIE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView"
/*@C
   EPSView - Prints the EPS data structure.

   Collective on EPS

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

.seealso: STView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode EPSView(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  const char     *type, *proj, *which;
  PetscTruth     isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)eps)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(eps,1,viewer,2);

#if defined(PETSC_USE_COMPLEX)
#define HERM "hermitian"
#else
#define HERM "symmetric"
#endif
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"EPS Object:\n");CHKERRQ(ierr);
    switch (eps->problem_type) {
      case EPS_HEP:   type = HERM " eigenvalue problem"; break;
      case EPS_GHEP:  type = "generalized " HERM " eigenvalue problem"; break;
      case EPS_NHEP:  type = "non-" HERM " eigenvalue problem"; break;
      case EPS_GNHEP: type = "generalized non-" HERM " eigenvalue problem"; break;
      case 0:         type = "not yet set"; break;
      default: SETERRQ(1,"Wrong value of eps->problem_type");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type);CHKERRQ(ierr);
    ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
    if (type) {
      ierr = PetscViewerASCIIPrintf(viewer,"  method: %s",type);CHKERRQ(ierr);
      switch (eps->solverclass) {
        case EPS_ONE_SIDE: 
          ierr = PetscViewerASCIIPrintf(viewer,"\n",type);CHKERRQ(ierr); break;
        case EPS_TWO_SIDE: 
          ierr = PetscViewerASCIIPrintf(viewer," (two-sided)\n",type);CHKERRQ(ierr); break;
        default: SETERRQ(1,"Wrong value of eps->solverclass");
      }
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  method: not yet set\n");CHKERRQ(ierr);
    }
    if (eps->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*eps->ops->view)(eps,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (eps->projection) {
      switch (eps->projection) {
        case EPS_RITZ:             proj = "Rayleigh-Ritz"; break;
        case EPS_HARMONIC:         proj = "harmonic Ritz"; break;
        case EPS_REFINED:          proj = "refined Ritz"; break;
        case EPS_REFINED_HARMONIC: proj = "refined harmonic Ritz"; break;
        default: SETERRQ(1,"Wrong value of eps->projection");
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  projection type: %s\n",proj);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: ");CHKERRQ(ierr);
    if (eps->target_set) {
#if !defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %g\n",eps->target);CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %g+%g i\n",PetscRealPart(eps->target),PetscImaginaryPart(eps->target));CHKERRQ(ierr);
#endif
    } else {
      switch (eps->which) {
        case EPS_LARGEST_MAGNITUDE:  which = "largest eigenvalues in magnitude"; break;
        case EPS_SMALLEST_MAGNITUDE: which = "smallest eigenvalues in magnitude"; break;
        case EPS_LARGEST_REAL:       which = "largest real parts"; break;
        case EPS_SMALLEST_REAL:      which = "smallest real parts"; break;
        case EPS_LARGEST_IMAGINARY:  which = "largest imaginary parts"; break;
        case EPS_SMALLEST_IMAGINARY: which = "smallest imaginary parts"; break;
        default: SETERRQ(1,"Wrong value of eps->which");
      }
      ierr = PetscViewerASCIIPrintf(viewer,"%s\n",which);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %d\n",eps->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %d\n",eps->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %d\n", eps->max_it);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",eps->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided deflation space: %d\n",eps->nds);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = IPView(eps->ip,viewer); CHKERRQ(ierr);
    ierr = STView(eps->OP,viewer); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    if (eps->ops->view) {
      ierr = (*eps->ops->view)(eps,viewer);CHKERRQ(ierr);
    }
    ierr = STView(eps->OP,viewer); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSCreate"
/*@C
   EPSCreate - Creates the default EPS context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  eps - location to put the EPS context

   Note:
   The default EPS type is EPSKRYLOVSCHUR

   Level: beginner

.seealso: EPSSetUp(), EPSSolve(), EPSDestroy(), EPS
@*/
PetscErrorCode EPSCreate(MPI_Comm comm,EPS *outeps)
{
  PetscErrorCode ierr;
  EPS            eps;

  PetscFunctionBegin;
  PetscValidPointer(outeps,2);
  *outeps = 0;

  ierr = PetscHeaderCreate(eps,_p_EPS,struct _EPSOps,EPS_COOKIE,-1,"EPS",comm,EPSDestroy,EPSView);CHKERRQ(ierr);
  *outeps = eps;

  ierr = PetscMemzero(eps->ops,sizeof(struct _EPSOps));CHKERRQ(ierr);

  eps->max_it          = 0;
  eps->nev             = 1;
  eps->ncv             = 0;
  eps->allocated_ncv   = 0;
  eps->nds             = 0;
  eps->tol             = 1e-7;
  eps->which           = EPS_LARGEST_MAGNITUDE;
  eps->target          = 0.0;
  eps->target_set      = PETSC_FALSE;
  eps->evecsavailable  = PETSC_FALSE;
  eps->problem_type    = (EPSProblemType)0;
  eps->projection      = (EPSProjection)0;
  eps->solverclass     = (EPSClass)0;

  eps->vec_initial     = 0;
  eps->vec_initial_left= 0;
  eps->V               = 0;
  eps->AV              = 0;
  eps->W               = 0;
  eps->AW              = 0;
  eps->T               = 0;
  eps->DS              = 0;
  eps->ds_ortho        = PETSC_TRUE;
  eps->eigr            = 0;
  eps->eigi            = 0;
  eps->errest          = 0;
  eps->errest_left     = 0;
  eps->OP              = 0;
  eps->ip              = 0;
  eps->data            = 0;
  eps->nconv           = 0;
  eps->its             = 0;
  eps->perm            = PETSC_NULL;

  eps->nwork           = 0;
  eps->work            = 0;
  eps->isgeneralized   = PETSC_FALSE;
  eps->ishermitian     = PETSC_FALSE;
  eps->ispositive      = PETSC_FALSE;
  eps->setupcalled     = 0;
  eps->reason          = EPS_CONVERGED_ITERATING;

  eps->numbermonitors  = 0;

  ierr = STCreate(comm,&eps->OP); CHKERRQ(ierr);
  PetscLogObjectParent(eps,eps->OP);
  ierr = IPCreate(comm,&eps->ip); CHKERRQ(ierr);
  ierr = IPSetOptionsPrefix(eps->ip,((PetscObject)eps)->prefix);
  ierr = IPAppendOptionsPrefix(eps->ip,"eps_");
  PetscLogObjectParent(eps,eps->ip);
  ierr = PetscPublishAll(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "EPSSetType"
/*@C
   EPSSetType - Selects the particular solver to be used in the EPS object. 

   Collective on EPS

   Input Parameters:
+  eps      - the eigensolver context
-  type     - a known method

   Options Database Key:
.  -eps_type <method> - Sets the method; use -help for a list 
    of available methods 
    
   Notes:  
   See "slepc/include/slepceps.h" for available methods. The default
   is EPSKRYLOVSCHUR.

   Normally, it is best to use the EPSSetFromOptions() command and
   then set the EPS type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The EPSSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database. 

   Level: intermediate

.seealso: STSetType(), EPSType
@*/
PetscErrorCode EPSSetType(EPS eps,EPSType type)
{
  PetscErrorCode ierr,(*r)(EPS);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)eps,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (eps->data) {
    /* destroy the old private EPS context */
    ierr = (*eps->ops->destroy)(eps); CHKERRQ(ierr);
    eps->data = 0;
  }

  ierr = PetscFListFind(EPSList,((PetscObject)eps)->comm,type,(void (**)(void)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,"Unknown EPS type given: %s",type);

  eps->setupcalled = 0;
  ierr = PetscMemzero(eps->ops,sizeof(struct _EPSOps));CHKERRQ(ierr);
  ierr = (*r)(eps); CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)eps,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetType"
/*@C
   EPSGetType - Gets the EPS type as a string from the EPS object.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context 

   Output Parameter:
.  name - name of EPS method 

   Level: intermediate

.seealso: EPSSetType()
@*/
PetscErrorCode EPSGetType(EPS eps,const EPSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)eps)->type_name;
  PetscFunctionReturn(0);
}

/*MC
   EPSRegisterDynamic - Adds a method to the eigenproblem solver package.

   Synopsis:
   EPSRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(EPS))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create the solver context
-  routine_create - routine to create the solver context

   Notes:
   EPSRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   EPSRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     EPSSetType(eps,"my_solver")
   or at runtime via the option
$     -eps_type my_solver

   Level: advanced

   Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
   and others of the form ${any_environmental_variable} occuring in pathname will be 
   replaced with appropriate values.

.seealso: EPSRegisterDestroy(), EPSRegisterAll()

M*/

#undef __FUNCT__  
#define __FUNCT__ "EPSRegister"
/*@C
  EPSRegister - See EPSRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode EPSRegister(const char *sname,const char *path,const char *name,int (*function)(EPS))
{
  PetscErrorCode ierr;
  char           fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&EPSList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSRegisterDestroy"
/*@
   EPSRegisterDestroy - Frees the list of EPS methods that were
   registered by EPSRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: EPSRegisterDynamic(), EPSRegisterAll()
@*/
PetscErrorCode EPSRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&EPSList);CHKERRQ(ierr);
  ierr = EPSRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy"
/*@
   EPSDestroy - Destroys the EPS context.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Level: beginner

.seealso: EPSCreate(), EPSSetUp(), EPSSolve()
@*/
PetscErrorCode EPSDestroy(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (--((PetscObject)eps)->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(eps);CHKERRQ(ierr);

  ierr = STDestroy(eps->OP);CHKERRQ(ierr);
  ierr = IPDestroy(eps->ip);CHKERRQ(ierr);

  if (eps->ops->destroy) {
    ierr = (*eps->ops->destroy)(eps); CHKERRQ(ierr);
  }
  
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscFree(eps->Tl);CHKERRQ(ierr);
  ierr = PetscFree(eps->perm);CHKERRQ(ierr);

  if (eps->vec_initial) {
    ierr = VecDestroy(eps->vec_initial);CHKERRQ(ierr);
  }

  if (eps->vec_initial_left) {
    ierr = VecDestroy(eps->vec_initial_left);CHKERRQ(ierr);
  }

  if (eps->nds > 0) {
    ierr = VecDestroyVecs(eps->DS, eps->nds);CHKERRQ(ierr);
  }
  
  ierr = PetscFree(eps->DSV);CHKERRQ(ierr);

  ierr = EPSMonitorCancel(eps);CHKERRQ(ierr);

  PetscHeaderDestroy(eps);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetTarget"
/*@
   EPSSetTarget - Sets the value of the target.

   Not collective

   Input Parameters:
+  eps    - eigensolver context
-  target - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest.

   If the target is not specified, then eigenvalues are computed according to
   the which parameter (see EPSSetWhichEigenpairs()).
   
   If the target is specified, then the sought-after eigenvalues are those
   closest to the target.

   Level: beginner

.seealso: EPSGetTarget(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSSetTarget(EPS eps,PetscScalar target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  eps->target = target;
  eps->target_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetTarget"
/*@
   EPSGetTarget - Gets the value of the target.

   Not collective

   Input Parameter:
.  eps - eigensolver context

   Output Parameter:
.  target - the value of the target

   Level: beginner

   Note:
   If the target was not set by the user, then zero is returned.

.seealso: EPSSetTarget()
@*/
PetscErrorCode EPSGetTarget(EPS eps,PetscScalar* target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (target) {
    if (eps->target_set) *target = eps->target;
    else *target = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetST"
/*@
   EPSSetST - Associates a spectral transformation object to the
   eigensolver. 

   Collective on EPS

   Input Parameters:
+  eps - eigensolver context obtained from EPSCreate()
-  st   - the spectral transformation object

   Note:
   Use EPSGetST() to retrieve the spectral transformation context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: EPSGetST()
@*/
PetscErrorCode EPSSetST(EPS eps,ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(st,ST_COOKIE,2);
  PetscCheckSameComm(eps,1,st,2);
  ierr = PetscObjectReference((PetscObject)st);CHKERRQ(ierr);
  ierr = STDestroy(eps->OP); CHKERRQ(ierr);
  eps->OP = st;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetST"
/*@C
   EPSGetST - Obtain the spectral transformation (ST) object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  st - spectral transformation context

   Level: beginner

.seealso: EPSSetST()
@*/
PetscErrorCode EPSGetST(EPS eps, ST *st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(st,2);
  *st = eps->OP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetIP"
/*@
   EPSSetIP - Associates an inner product object to the
   eigensolver. 

   Collective on EPS

   Input Parameters:
+  eps - eigensolver context obtained from EPSCreate()
-  ip  - the inner product object

   Note:
   Use EPSGetIP() to retrieve the inner product context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: EPSGetIP()
@*/
PetscErrorCode EPSSetIP(EPS eps,IP ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(ip,IP_COOKIE,2);
  PetscCheckSameComm(eps,1,ip,2);
  ierr = PetscObjectReference((PetscObject)ip);CHKERRQ(ierr);
  ierr = IPDestroy(eps->ip); CHKERRQ(ierr);
  eps->ip = ip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetIP"
/*@C
   EPSGetIP - Obtain the inner product object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  ip - inner product context

   Level: advanced

.seealso: EPSSetIP()
@*/
PetscErrorCode EPSGetIP(EPS eps,IP *ip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(ip,2);
  *ip = eps->ip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSIsGeneralized"
/*@
   EPSIsGeneralized - Ask if the EPS object corresponds to a generalized 
   eigenvalue problem.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

@*/
PetscErrorCode EPSIsGeneralized(EPS eps,PetscTruth* is)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = STGetOperators(eps->OP,PETSC_NULL,&B);CHKERRQ(ierr);
  if( B ) *is = PETSC_TRUE;
  else *is = PETSC_FALSE;
  if( eps->setupcalled ) {
    if( eps->isgeneralized != *is ) { 
      SETERRQ(0,"Warning: Inconsistent EPS state");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSIsHermitian"
/*@
   EPSIsHermitian - Ask if the EPS object corresponds to a Hermitian 
   eigenvalue problem.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

@*/
PetscErrorCode EPSIsHermitian(EPS eps,PetscTruth* is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if( eps->ishermitian ) *is = PETSC_TRUE;
  else *is = PETSC_FALSE;
  PetscFunctionReturn(0);
}
