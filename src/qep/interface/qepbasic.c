/*
     The basic QEP routines, Create, View, etc. are here.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "private/qepimpl.h"      /*I "slepcqep.h" I*/

PetscFList QEPList = 0;
PetscCookie QEP_COOKIE = 0;
PetscLogEvent QEP_SetUp = 0, QEP_Solve = 0, QEP_Dense = 0;
static PetscTruth QEPPackageInitialized = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "QEPFinalizePackage"
/*@C
  QEPFinalizePackage - This function destroys everything in the Slepc interface to the QEP package. It is
  called from SlepcFinalize().

  Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode QEPFinalizePackage(void) 
{
  PetscFunctionBegin;
  QEPPackageInitialized = PETSC_FALSE;
  QEPList               = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPInitializePackage"
/*@C
  QEPInitializePackage - This function initializes everything in the QEP package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to QEPCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode QEPInitializePackage(const char *path) {
  char           logList[256];
  char           *className;
  PetscTruth     opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (QEPPackageInitialized) PetscFunctionReturn(0);
  QEPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Quadratic Eigenproblem Solver",&QEP_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = QEPRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("QEPSetUp",QEP_COOKIE,&QEP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("QEPSolve",QEP_COOKIE,&QEP_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("QEPDense",QEP_COOKIE,&QEP_Dense);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"qep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(QEP_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"qep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(QEP_COOKIE);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(QEPFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPView"
/*@C
   QEPView - Prints the QEP data structure.

   Collective on QEP

   Input Parameters:
+  qep - the quadratic eigenproblem solver context
-  viewer - optional visualization context

   Options Database Key:
.  -qep_view -  Calls QEPView() at end of QEPSolve()

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
PetscErrorCode QEPView(QEP qep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  const char     *type;
  PetscTruth     isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)qep)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(qep,1,viewer,2);

#if defined(PETSC_USE_COMPLEX)
#define HERM "hermitian"
#else
#define HERM "symmetric"
#endif
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"QEP Object:\n");CHKERRQ(ierr);
    switch (qep->problem_type) {
      case QEP_GENERAL:    type = "general quadratic eigenvalue problem"; break;
      case QEP_HERMITIAN:  type = HERM " quadratic eigenvalue problem"; break;
      case QEP_GYROSCOPIC: type = "gyroscopic quadratic eigenvalue problem"; break;
      case 0:         type = "not yet set"; break;
      default: SETERRQ(1,"Wrong value of qep->problem_type");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type);CHKERRQ(ierr);
    ierr = QEPGetType(qep,&type);CHKERRQ(ierr);
    if (type) {
      ierr = PetscViewerASCIIPrintf(viewer,"  method: %s\n",type);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  method: not yet set\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: ");CHKERRQ(ierr);
    if (!qep->which) {
      ierr = PetscViewerASCIIPrintf(viewer,"not yet set\n");CHKERRQ(ierr);
    } else switch (qep->which) {
      case QEP_LARGEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"largest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case QEP_SMALLEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case QEP_LARGEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"largest real parts\n");CHKERRQ(ierr);
        break;
      case QEP_SMALLEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest real parts\n");CHKERRQ(ierr);
        break;
      case QEP_LARGEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"largest imaginary parts\n");CHKERRQ(ierr);
        break;
      case QEP_SMALLEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest imaginary parts\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(1,"Wrong value of qep->which");
    }    
    if (qep->leftvecs) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing left eigenvectors also\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %d\n",qep->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %d\n",qep->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %d\n",qep->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %d\n", qep->max_it);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",qep->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  scaling factor: %g\n",qep->sfactor);CHKERRQ(ierr);
    if (qep->nini!=0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %d\n",PetscAbs(qep->nini));CHKERRQ(ierr);
    }
    if (qep->ninil!=0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial left space: %d\n",PetscAbs(qep->ninil));CHKERRQ(ierr);
    }
    if (qep->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*qep->ops->view)(qep,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = IPView(qep->ip,viewer); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    if (qep->ops->view) {
      ierr = (*qep->ops->view)(qep,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPCreate"
/*@C
   QEPCreate - Creates the default QEP context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  qep - location to put the QEP context

   Note:
   The default QEP type is QEPLINEAR

   Level: beginner

.seealso: QEPSetUp(), QEPSolve(), QEPDestroy(), QEP
@*/
PetscErrorCode QEPCreate(MPI_Comm comm,QEP *outqep)
{
  PetscErrorCode ierr;
  QEP            qep;

  PetscFunctionBegin;
  PetscValidPointer(outqep,2);
  *outqep = 0;

  ierr = PetscHeaderCreate(qep,_p_QEP,struct _QEPOps,QEP_COOKIE,-1,"QEP",comm,QEPDestroy,QEPView);CHKERRQ(ierr);
  *outqep = qep;

  ierr = PetscMemzero(qep->ops,sizeof(struct _QEPOps));CHKERRQ(ierr);

  qep->max_it          = 0;
  qep->nev             = 1;
  qep->ncv             = 0;
  qep->mpd             = 0;
  qep->nini            = 0;
  qep->ninil           = 0;
  qep->tol             = 1e-7;
  qep->sfactor         = 0.0;
  qep->conv_func       = QEPDefaultConverged;
  qep->conv_ctx        = PETSC_NULL;
  qep->which           = (QEPWhich)0;
  qep->which_func      = PETSC_NULL;
  qep->which_ctx       = PETSC_NULL;
  qep->leftvecs        = PETSC_FALSE;
  qep->problem_type    = (QEPProblemType)0;
  qep->V               = PETSC_NULL;
  qep->IS              = PETSC_NULL;
  qep->ISL             = PETSC_NULL;
  qep->T               = PETSC_NULL;
  qep->eigr            = PETSC_NULL;
  qep->eigi            = PETSC_NULL;
  qep->errest          = PETSC_NULL;
  qep->data            = PETSC_NULL;
  qep->nconv           = 0;
  qep->its             = 0;
  qep->perm            = PETSC_NULL;
  qep->matvecs         = 0;
  qep->linits          = 0;
  qep->nwork           = 0;
  qep->work            = PETSC_NULL;
  qep->setupcalled     = 0;
  qep->reason          = QEP_CONVERGED_ITERATING;
  qep->numbermonitors  = 0;
  qep->trackall        = PETSC_FALSE;

  ierr = IPCreate(comm,&qep->ip); CHKERRQ(ierr);
  ierr = IPSetOptionsPrefix(qep->ip,((PetscObject)qep)->prefix);
  ierr = IPAppendOptionsPrefix(qep->ip,"qep_");
  PetscLogObjectParent(qep,qep->ip);
  ierr = PetscPublishAll(qep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "QEPSetType"
/*@C
   QEPSetType - Selects the particular solver to be used in the QEP object. 

   Collective on QEP

   Input Parameters:
+  qep      - the quadratic eigensolver context
-  type     - a known method

   Options Database Key:
.  -qep_type <method> - Sets the method; use -help for a list 
    of available methods 
    
   Notes:  
   See "slepc/include/slepcqep.h" for available methods. The default
   is QEPLINEAR.

   Normally, it is best to use the QEPSetFromOptions() command and
   then set the QEP type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The QEPSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database. 

   Level: intermediate

.seealso: QEPType
@*/
PetscErrorCode QEPSetType(QEP qep,const QEPType type)
{
  PetscErrorCode ierr,(*r)(QEP);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)qep,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (qep->data) {
    /* destroy the old private QEP context */
    ierr = (*qep->ops->destroy)(qep); CHKERRQ(ierr);
    qep->data = 0;
  }

  ierr = PetscFListFind(QEPList,((PetscObject)qep)->comm,type,(void (**)(void))&r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,"Unknown QEP type given: %s",type);

  qep->setupcalled = 0;
  ierr = PetscMemzero(qep->ops,sizeof(struct _QEPOps));CHKERRQ(ierr);
  ierr = (*r)(qep);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)qep,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetType"
/*@C
   QEPGetType - Gets the QEP type as a string from the QEP object.

   Not Collective

   Input Parameter:
.  qep - the eigensolver context 

   Output Parameter:
.  name - name of QEP method 

   Level: intermediate

.seealso: QEPSetType()
@*/
PetscErrorCode QEPGetType(QEP qep,const QEPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)qep)->type_name;
  PetscFunctionReturn(0);
}

/*MC
   QEPRegisterDynamic - Adds a method to the quadratic eigenproblem solver package.

   Synopsis:
   QEPRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(QEP))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create the solver context
-  routine_create - routine to create the solver context

   Notes:
   QEPRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   QEPRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     QEPSetType(qep,"my_solver")
   or at runtime via the option
$     -qep_type my_solver

   Level: advanced

   Environmental variables such as ${PETSC_ARCH}, ${SLEPC_DIR},
   and others of the form ${any_environmental_variable} occuring in pathname will be 
   replaced with appropriate values.

.seealso: QEPRegisterDestroy(), QEPRegisterAll()

M*/

#undef __FUNCT__  
#define __FUNCT__ "QEPRegister"
/*@C
  QEPRegister - See QEPRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode QEPRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(QEP))
{
  PetscErrorCode ierr;
  char           fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&QEPList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPRegisterDestroy"
/*@
   QEPRegisterDestroy - Frees the list of QEP methods that were
   registered by QEPRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: QEPRegisterDynamic(), QEPRegisterAll()
@*/
PetscErrorCode QEPRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&QEPList);CHKERRQ(ierr);
  ierr = QEPRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPDestroy"
/*@
   QEPDestroy - Destroys the QEP context.

   Collective on QEP

   Input Parameter:
.  qep - eigensolver context obtained from QEPCreate()

   Level: beginner

.seealso: QEPCreate(), QEPSetUp(), QEPSolve()
@*/
PetscErrorCode QEPDestroy(QEP qep)
{
  PetscErrorCode ierr;
  PetscScalar    *pV;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (--((PetscObject)qep)->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(qep);CHKERRQ(ierr);

  if (qep->ops->destroy) {
    ierr = (*qep->ops->destroy)(qep); CHKERRQ(ierr);
  }
  
  ierr = PetscFree(qep->T);CHKERRQ(ierr);

  if (qep->eigr) { 
    ierr = PetscFree(qep->eigr);CHKERRQ(ierr);
    ierr = PetscFree(qep->eigi);CHKERRQ(ierr);
    ierr = PetscFree(qep->perm);CHKERRQ(ierr);
    ierr = PetscFree(qep->errest);CHKERRQ(ierr);
    ierr = VecGetArray(qep->V[0],&pV);CHKERRQ(ierr);
    for (i=0;i<qep->ncv;i++) {
      ierr = VecDestroy(qep->V[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pV);CHKERRQ(ierr);
    ierr = PetscFree(qep->V);CHKERRQ(ierr);
  }

  ierr = QEPMonitorCancel(qep);CHKERRQ(ierr);

  ierr = IPDestroy(qep->ip);CHKERRQ(ierr);
  if (qep->rand) {
    ierr = PetscRandomDestroy(qep->rand);CHKERRQ(ierr);
  }

  if (qep->M) { ierr = MatDestroy(qep->M);CHKERRQ(ierr); }
  if (qep->C) { ierr = MatDestroy(qep->C);CHKERRQ(ierr); }
  if (qep->K) { ierr = MatDestroy(qep->K);CHKERRQ(ierr); }

  ierr = PetscHeaderDestroy(qep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetIP"
/*@
   QEPSetIP - Associates an inner product object to the quadratic eigensolver. 

   Collective on QEP and IP

   Input Parameters:
+  qep - eigensolver context obtained from QEPCreate()
-  ip  - the inner product object

   Note:
   Use QEPGetIP() to retrieve the inner product context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: QEPGetIP()
@*/
PetscErrorCode QEPSetIP(QEP qep,IP ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidHeaderSpecific(ip,IP_COOKIE,2);
  PetscCheckSameComm(qep,1,ip,2);
  ierr = PetscObjectReference((PetscObject)ip);CHKERRQ(ierr);
  ierr = IPDestroy(qep->ip); CHKERRQ(ierr);
  qep->ip = ip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetIP"
/*@C
   QEPGetIP - Obtain the inner product object associated
   to the quadratic eigensolver object.

   Not Collective

   Input Parameters:
.  qep - eigensolver context obtained from QEPCreate()

   Output Parameter:
.  ip - inner product context

   Level: advanced

.seealso: QEPSetIP()
@*/
PetscErrorCode QEPGetIP(QEP qep,IP *ip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidPointer(ip,2);
  *ip = qep->ip;
  PetscFunctionReturn(0);
}

