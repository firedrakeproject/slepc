/*
     The basic EPS routines, Create, View, etc. are here.
*/
#include "src/eps/epsimpl.h"      /*I "slepceps.h" I*/
#include "slepcblaslapack.h" 
#include "petscsys.h"

PetscTruth EPSRegisterAllCalled = PETSC_FALSE;
PetscFList EPSList = 0;

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
int EPSView(EPS eps,PetscViewer viewer)
{
  char        *type, *which;
  int         ierr;
  PetscTruth  isascii;

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
      ierr = PetscViewerASCIIPrintf(viewer,"  method: %s\n",type);CHKERRQ(ierr);
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
      case EPS_LARGEST_ALGEBRAIC:  which = "largest (algebraic) eigenvalues"; break;
      case EPS_SMALLEST_ALGEBRAIC: which = "smallest (algebraic) eigenvalues"; break;
      case EPS_LARGEST_REAL:       which = "largest real parts"; break;
      case EPS_SMALLEST_REAL:      which = "smallest real parts"; break;
      case EPS_LARGEST_IMAGINARY:  which = "largest imaginary parts"; break;
      case EPS_SMALLEST_IMAGINARY: which = "smallest imaginary parts"; break;
      case EPS_BOTH_ENDS:          which = "eigenvalues from both ends of the spectrum"; break;
      default: SETERRQ(1,"Wrong value of eps->which");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: %s\n",which);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %d\n",eps->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of basis vectors (ncv): %d\n",eps->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %d\n", eps->max_it);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",eps->tol);CHKERRQ(ierr);
    if (eps->dropvectors) { ierr = PetscViewerASCIIPrintf(viewer,"  computing only eigenvalues\n");CHKERRQ(ierr);}
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
#define __FUNCT__ "EPSSetDropEigenvectors"
/*@C
   EPSSetDropEigenvectors - Sets the EPS solver not to compute the 
   eigenvectors. In some methods, this can reduce the number of operations 
   necessary for obtaining the eigenvalues.

   Collective on KSP

   Input Parameter:
.  eps - the eigensolver context

   Options Database Keys:
.   -eps_drop_eigenvectors - do not compute eigenvectors

   Level: advanced

.seealso: EPSSetUp(), EPSSolve(), EPSDestroy()
@*/
int EPSSetDropEigenvectors(EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  eps->dropvectors = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSPublish_Petsc"
static int EPSPublish_Petsc(PetscObject object)
{
#if defined(PETSC_HAVE_AMS)
  EPS          v = (EPS) object;
  int          ierr;
#endif
  
  PetscFunctionBegin;

#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0 ) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(object);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"Iteration",&v->its,1,AMS_INT,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(object);CHKERRQ(ierr);
#endif

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
   The default EPS type is EPSPOWER

   Level: beginner

.seealso: EPSSetUp(), EPSSolve(), EPSDestroy(), EPS
@*/
int EPSCreate(MPI_Comm comm,EPS *outeps)
{
  EPS   eps;
  int   ierr;

  PetscFunctionBegin;
  *outeps = 0;
  PetscHeaderCreate(eps,_p_EPS,struct _EPSOps,EPS_COOKIE,-1,"EPS",comm,EPSDestroy,EPSView);
  PetscLogObjectCreate(eps);
  *outeps = eps;

  eps->bops->publish       = EPSPublish_Petsc;
  eps->ops->setfromoptions = 0;
  eps->ops->solve          = 0;
  eps->ops->setup          = 0;
  eps->ops->destroy        = 0;

  eps->type            = -1;
  eps->max_it          = 0;
  eps->nev             = 1;
  eps->ncv             = 0;
  eps->tol             = 0.0;
  eps->which           = EPS_LARGEST_MAGNITUDE;
  eps->dropvectors     = PETSC_FALSE;
  eps->problem_type    = (EPSProblemType)0;

  eps->vec_initial     = 0;
  eps->V               = 0;
  eps->eigr            = 0;
  eps->eigi            = 0;
  eps->errest          = 0;
  eps->OP              = 0;
  eps->data            = 0;
  eps->nconv           = 0;
  eps->its             = 0;

  eps->nwork           = 0;
  eps->work            = 0;
  eps->isgeneralized   = PETSC_FALSE;
  eps->ishermitian     = PETSC_FALSE;
  eps->setupcalled     = 0;
  eps->reason          = EPS_CONVERGED_ITERATING;

  eps->numbermonitors  = 0;
  eps->numbervmonitors = 0;

  eps->orthog          = EPSIROrthogonalization;

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
   See "slepc/include/slepceps.h" for available methods.

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
int EPSSetType(EPS eps,EPSType type)
{
  int ierr,(*r)(EPS);
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
  /* Get the function pointers for the iterative method requested */
  if (!EPSRegisterAllCalled) {ierr = EPSRegisterAll(PETSC_NULL); CHKERRQ(ierr);}

  ierr = PetscFListFind(eps->comm,EPSList,type,(void (**)(void)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,"Unknown EPS type given: %s",type);

  eps->setupcalled = 0;
  ierr = (*r)(eps); CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)eps,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSRegisterDestroy"
/*@C
   EPSRegisterDestroy - Frees the list of EPS methods that were
   registered by EPSRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: EPSRegisterDynamic(), EPSRegisterAll()
@*/
int EPSRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (EPSList) {
    ierr = PetscFListDestroy(&EPSList);CHKERRQ(ierr);
    EPSList = 0;
  }
  EPSRegisterAllCalled = PETSC_FALSE;
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
int EPSGetType(EPS eps,EPSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *type = eps->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions"
/*@
   EPSSetFromOptions - Sets EPS options from the options database.
   This routine must be called before EPSSetUp() if the user is to be 
   allowed to set the solver type. 

   Collective on EPS

   Input Parameters:
.  eps - the eigensolver context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner

.seealso: 
@*/
int EPSSetFromOptions(EPS eps)
{
  int        ierr;
  char       type[256];
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (!EPSRegisterAllCalled) {ierr = EPSRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsBegin(eps->comm,eps->prefix,"Eigenproblem Solver (EPS) Options","EPS");CHKERRQ(ierr);
    ierr = PetscOptionsList("-eps_type","Eigenproblem Solver method","EPSSetType",EPSList,(char*)(eps->type_name?eps->type_name:EPSPOWER),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = EPSSetType(eps,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!eps->type_name) {
      ierr = EPSSetType(eps,EPSPOWER);CHKERRQ(ierr);
    }

    ierr = PetscOptionsLogicalGroupBegin("-eps_hermitian","hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_gen_hermitian","generalized hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_non_hermitian","non-hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupEnd("-eps_gen_non_hermitian","generalized non-hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);}

    ierr = PetscOptionsLogicalGroupBegin("-eps_mgs_orth","Modified Gram-Schmidt orthogonalization","EPSSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetOrthogonalization(eps,EPS_MGS_ORTH);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_cgs_orth","Classical Gram-Schmidt orthogonalization","EPSSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetOrthogonalization(eps,EPS_CGS_ORTH);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupEnd("-eps_ir_orth","Iterative refinement orthogonalization","EPSSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetOrthogonalization(eps,EPS_IR_ORTH);CHKERRQ(ierr);}

    ierr = PetscOptionsInt("-eps_max_it","Maximum number of iterations","EPSSetTolerances",eps->max_it,&eps->max_it,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps_tol","Tolerance","KSPSetTolerances",eps->tol,&eps->tol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_nev","Number of eigenvalues to compute","EPSSetDimensions",eps->nev,&eps->nev,&flg);CHKERRQ(ierr);
    if( eps->nev<1 ) SETERRQ(1,"Illegal value for option -eps_nev. Must be > 0");
    ierr = PetscOptionsInt("-eps_ncv","Number of basis vectors","EPSSetDimensions",eps->ncv,&eps->ncv,&flg);CHKERRQ(ierr);
    if( flg && eps->ncv<1 ) SETERRQ(1,"Illegal value for option -eps_ncv. Must be > 0");

    ierr = PetscOptionsName("-eps_drop_eigenvectors","Do not compute eigenvectors","EPSSetDropEigenvectors",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = EPSSetDropEigenvectors(eps);CHKERRQ(ierr);
    }

    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to EPSSetFromOptions()
    */
    ierr = PetscOptionsName("-eps_cancelmonitors","Remove any hardwired monitor routines","EPSClearMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = EPSClearMonitor(eps); CHKERRQ(ierr);
    }
    /*
      Prints error estimates at each iteration
    */
    ierr = PetscOptionsName("-eps_monitor","Monitor error estimates","EPSSetMonitor",&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = EPSSetMonitor(eps,EPSDefaultEstimatesMonitor,PETSC_NULL);CHKERRQ(ierr);
    }
    /*
      Prints approximate eigenvalues at each iteration
    */
    ierr = PetscOptionsName("-eps_monitor_values","Monitor approximate eigenvalues","EPSSetValuesMonitor",&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = EPSSetValuesMonitor(eps,EPSDefaultValuesMonitor,PETSC_NULL);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/
    ierr = PetscOptionsLogicalGroupBegin("-eps_largest_magnitude","compute largest eigenvalues in magnitude","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_smallest_magnitude","compute smallest eigenvalues in magnitude","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_largest_algebraic","compute largest (algebraic) eigenvalues","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_ALGEBRAIC);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_smallest_algebraic","compute smallest (algebraic) eigenvalues","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_ALGEBRAIC);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_largest_real","compute largest real parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_smallest_real","compute smallest real parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_largest_imaginary","compute largest imaginary parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_IMAGINARY);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_smallest_imaginary","compute smallest imaginary parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_IMAGINARY);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupEnd("-eps_both_ends","compute eigenvalues from both ends of the spectrum","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_BOTH_ENDS);CHKERRQ(ierr);}

    ierr = PetscOptionsName("-eps_view","Print detailed information on solver used","EPSView",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-eps_view_binary","Saves the matrices associated to the eigenproblem","EPSSetFromOptions",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-eps_plot_eigs","Makes a plot of the computed eigenvalues","EPSSolve",0);CHKERRQ(ierr);

    if (eps->ops->setfromoptions) {
      ierr = (*eps->ops->setfromoptions)(eps);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = STSetFromOptions(eps->OP); CHKERRQ(ierr);
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

.seealso: EPSRegisterAll(), EPSRegisterDestroy()

M*/

#undef __FUNCT__  
#define __FUNCT__ "EPSRegister"
int EPSRegister(char *sname,char *path,char *name,int (*function)(EPS))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&EPSList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

