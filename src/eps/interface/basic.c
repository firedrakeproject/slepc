/*
     The basic EPS routines, Create, View, etc. are here.
*/
#include "src/eps/epsimpl.h"      /*I "slepceps.h" I*/

PetscFList EPSList = 0;
PetscCookie EPS_COOKIE = 0;
PetscEvent EPS_SetUp = 0, EPS_Solve = 0, EPS_Orthogonalize = 0;

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
  ierr = PetscLogClassRegister(&EPS_COOKIE,"Eigenproblem Solver");CHKERRQ(ierr);
  /* Register Constructors */
  ierr = EPSRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&EPS_SetUp,"EPSSetUp",EPS_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&EPS_Solve,"EPSSolve",EPS_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&EPS_Orthogonalize,"EPSOrthogonalize",EPS_COOKIE); CHKERRQ(ierr);
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
  const char     *type, *which;
  PetscTruth     isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(eps->comm);
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
    switch (eps->which) {
      case EPS_LARGEST_MAGNITUDE:  which = "largest eigenvalues in magnitude"; break;
      case EPS_SMALLEST_MAGNITUDE: which = "smallest eigenvalues in magnitude"; break;
      case EPS_LARGEST_REAL:       which = "largest real parts"; break;
      case EPS_SMALLEST_REAL:      which = "smallest real parts"; break;
      case EPS_LARGEST_IMAGINARY:  which = "largest imaginary parts"; break;
      case EPS_SMALLEST_IMAGINARY: which = "smallest imaginary parts"; break;
      default: SETERRQ(1,"Wrong value of eps->which");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: %s\n",which);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %d\n",eps->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %d\n",eps->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %d\n", eps->max_it);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",eps->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  orthogonalization method: ");CHKERRQ(ierr);
    switch (eps->orthog_type) {
      case EPS_MGS_ORTH:
        ierr = PetscViewerASCIIPrintf(viewer,"modified Gram-Schmidt\n");CHKERRQ(ierr);
        break;
      case EPS_CGS_ORTH:
        ierr = PetscViewerASCIIPrintf(viewer,"classical Gram-Schmidt\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(1,"Wrong value of eps->orth_type");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  orthogonalization refinement: ");CHKERRQ(ierr);
    switch (eps->orthog_ref) {
      case EPS_ORTH_REFINE_NEVER:
        ierr = PetscViewerASCIIPrintf(viewer,"never\n");CHKERRQ(ierr);
        break;
      case EPS_ORTH_REFINE_IFNEEDED:
        ierr = PetscViewerASCIIPrintf(viewer,"if needed (eta: %f)\n",eps->orthog_eta);CHKERRQ(ierr);
        break;
      case EPS_ORTH_REFINE_ALWAYS:
        ierr = PetscViewerASCIIPrintf(viewer,"always\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(1,"Wrong value of eps->orth_ref");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided deflation space: %d\n",eps->nds);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
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
#define __FUNCT__ "EPSPublish_Petsc"
static PetscErrorCode EPSPublish_Petsc(PetscObject object)
{
  PetscFunctionBegin;
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

  PetscHeaderCreate(eps,_p_EPS,struct _EPSOps,EPS_COOKIE,-1,"EPS",comm,EPSDestroy,EPSView);
  PetscLogObjectCreate(eps);
  *outeps = eps;

  eps->bops->publish   = EPSPublish_Petsc;
  ierr = PetscMemzero(eps->ops,sizeof(struct _EPSOps));CHKERRQ(ierr);

  eps->type            = -1;
  eps->max_it          = 0;
  eps->nev             = 1;
  eps->ncv             = 0;
  eps->allocated_ncv   = 0;
  eps->nds             = 0;
  eps->tol             = 0.0;
  eps->which           = EPS_LARGEST_MAGNITUDE;
  eps->evecsavailable  = PETSC_FALSE;
  eps->problem_type    = (EPSProblemType)0;
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
  eps->data            = 0;
  eps->nconv           = 0;
  eps->its             = 0;
  eps->perm            = PETSC_NULL;

  eps->nwork           = 0;
  eps->work            = 0;
  eps->isgeneralized   = PETSC_FALSE;
  eps->ishermitian     = PETSC_FALSE;
  eps->setupcalled     = 0;
  eps->reason          = EPS_CONVERGED_ITERATING;

  eps->numbermonitors  = 0;

  eps->orthog_type     = EPS_CGS_ORTH;
  eps->orthog_ref      = EPS_ORTH_REFINE_IFNEEDED;
  eps->orthog_eta      = PETSC_DEFAULT;

  ierr = STCreate(comm,&eps->OP); CHKERRQ(ierr);
  PetscLogObjectParent(eps,eps->OP);
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

  ierr = PetscFListFind(eps->comm,EPSList,type,(void (**)(void)) &r);CHKERRQ(ierr);

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
PetscErrorCode EPSGetType(EPS eps,EPSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *type = eps->type_name;
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

   Environmental variables such as ${PETSC_ARCH}, ${SLEPC_DIR}, ${BOPT},
   and others of the form ${any_environmental_variable} occuring in pathname will be 
   replaced with appropriate values.

.seealso: EPSRegisterAll()

M*/

#undef __FUNCT__  
#define __FUNCT__ "EPSRegister"
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
  if (--eps->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(eps);CHKERRQ(ierr);

  ierr = STDestroy(eps->OP);CHKERRQ(ierr);

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
    ierr = VecDestroyVecs(eps->DS, eps->nds);
  }
  
  ierr = PetscFree(eps->DSV);CHKERRQ(ierr);

  PetscLogObjectDestroy(eps);
  PetscHeaderDestroy(eps);
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
  ierr = STDestroy(eps->OP); CHKERRQ(ierr);
  eps->OP = st;
  ierr = PetscObjectReference((PetscObject)eps->OP);CHKERRQ(ierr);
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
