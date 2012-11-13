/*
     The basic QEP routines, Create, View, etc. are here.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/qepimpl.h>      /*I "slepcqep.h" I*/

PetscFList       QEPList = 0;
PetscBool        QEPRegisterAllCalled = PETSC_FALSE;
PetscClassId     QEP_CLASSID = 0;
PetscLogEvent    QEP_SetUp = 0,QEP_Solve = 0,QEP_Dense = 0;
static PetscBool QEPPackageInitialized = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "QEPFinalizePackage"
/*@C
   QEPFinalizePackage - This function destroys everything in the Slepc interface
   to the QEP package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode QEPFinalizePackage(void) 
{
  PetscFunctionBegin;
  QEPPackageInitialized = PETSC_FALSE;
  QEPList               = 0;
  QEPRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPInitializePackage"
/*@C
   QEPInitializePackage - This function initializes everything in the QEP package. It is called
   from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to QEPCreate()
   when using static libraries.

   Input Parameter:
.  path - The dynamic library path, or PETSC_NULL

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode QEPInitializePackage(const char *path)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (QEPPackageInitialized) PetscFunctionReturn(0);
  QEPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Quadratic Eigenproblem Solver",&QEP_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = QEPRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("QEPSetUp",QEP_CLASSID,&QEP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("QEPSolve",QEP_CLASSID,&QEP_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("QEPDense",QEP_CLASSID,&QEP_Dense);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"qep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(QEP_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"qep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(QEP_CLASSID);CHKERRQ(ierr);
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
  PetscBool      isascii,islinear;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)qep)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qep,1,viewer,2);

#if defined(PETSC_USE_COMPLEX)
#define HERM "hermitian"
#else
#define HERM "symmetric"
#endif
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)qep,viewer,"QEP Object");CHKERRQ(ierr);
    if (qep->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*qep->ops->view)(qep,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (qep->problem_type) {
      switch (qep->problem_type) {
        case QEP_GENERAL:    type = "general quadratic eigenvalue problem"; break;
        case QEP_HERMITIAN:  type = HERM " quadratic eigenvalue problem"; break;
        case QEP_GYROSCOPIC: type = "gyroscopic quadratic eigenvalue problem"; break;
        default: SETERRQ(((PetscObject)qep)->comm,1,"Wrong value of qep->problem_type");
      }
    } else type = "not yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type);CHKERRQ(ierr);
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
      default: SETERRQ(((PetscObject)qep)->comm,1,"Wrong value of qep->which");
    }    
    if (qep->leftvecs) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing left eigenvectors also\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %D\n",qep->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",qep->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %D\n",qep->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",qep->max_it);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %G\n",qep->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  scaling factor: %G\n",qep->sfactor);CHKERRQ(ierr);
    if (qep->nini!=0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %D\n",PetscAbs(qep->nini));CHKERRQ(ierr);
    }
    if (qep->ninil!=0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial left space: %D\n",PetscAbs(qep->ninil));CHKERRQ(ierr);
    }
  } else {
    if (qep->ops->view) {
      ierr = (*qep->ops->view)(qep,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectTypeCompare((PetscObject)qep,QEPLINEAR,&islinear);CHKERRQ(ierr);
  if (!islinear) {
    if (!qep->ip) { ierr = QEPGetIP(qep,&qep->ip);CHKERRQ(ierr); }
    ierr = IPView(qep->ip,viewer);CHKERRQ(ierr);
    if (!qep->ds) { ierr = QEPGetDS(qep,&qep->ds);CHKERRQ(ierr); }
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = DSView(qep->ds,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPPrintSolution"
/*@
   QEPPrintSolution - Prints the computed eigenvalues.

   Collective on QEP

   Input Parameters:
+  qep - the eigensolver context
-  viewer - optional visualization context

   Options Database:
.  -qep_terse - print only minimal information

   Note:
   By default, this function prints a table with eigenvalues and associated
   relative errors. With -qep_terse only the eigenvalues are printed.

   Level: intermediate

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode QEPPrintSolution(QEP qep,PetscViewer viewer)
{
  PetscBool      terse,errok,isascii;
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)qep)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qep,1,viewer,2);
  if (!qep->eigr || !qep->eigi || !qep->V) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_WRONGSTATE,"QEPSolve must be called first"); 
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);

  ierr = PetscOptionsHasName(PETSC_NULL,"-qep_terse",&terse);CHKERRQ(ierr);
  if (terse) {
    if (qep->nconv<qep->nev) {
      ierr = PetscViewerASCIIPrintf(viewer," Problem: less than %D eigenvalues converged\n\n",qep->nev);CHKERRQ(ierr);
    } else {
      errok = PETSC_TRUE;
      for (i=0;i<qep->nev;i++) {
        ierr = QEPComputeRelativeError(qep,i,&error);CHKERRQ(ierr);
        errok = (errok && error<qep->tol)? PETSC_TRUE: PETSC_FALSE;
      }
      if (errok) {
        ierr = PetscViewerASCIIPrintf(viewer," All requested eigenvalues computed up to the required tolerance:");CHKERRQ(ierr);
        for (i=0;i<=(qep->nev-1)/8;i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"\n     ");CHKERRQ(ierr);
          for (j=0;j<PetscMin(8,qep->nev-8*i);j++) {
            ierr = QEPGetEigenpair(qep,8*i+j,&kr,&ki,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
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
              ierr = PetscViewerASCIIPrintf(viewer,"%.5F%+.5Fi",re,im);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(viewer,"%.5F",re);CHKERRQ(ierr); 
            }
            if (8*i+j+1<qep->nev) { ierr = PetscViewerASCIIPrintf(viewer,", ");CHKERRQ(ierr); }
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer," Problem: some of the first %D relative errors are higher than the tolerance\n\n",qep->nev);CHKERRQ(ierr);
      }
    }
  } else {
    ierr = PetscViewerASCIIPrintf(viewer," Number of converged approximate eigenpairs: %D\n\n",qep->nconv);CHKERRQ(ierr);
    if (qep->nconv>0) {
      ierr = PetscViewerASCIIPrintf(viewer,
           "           k          ||(k^2M+Ck+K)x||/||kx||\n"
           "   ----------------- -------------------------\n");CHKERRQ(ierr);
      for (i=0;i<qep->nconv;i++) {
        ierr = QEPGetEigenpair(qep,i,&kr,&ki,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        ierr = QEPComputeRelativeError(qep,i,&error);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        re = PetscRealPart(kr);
        im = PetscImaginaryPart(kr);
#else
        re = kr;
        im = ki;
#endif 
        if (im!=0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," % 9F%+9F i     %12G\n",re,im,error);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"   % 12F           %12G\n",re,error);CHKERRQ(ierr); 
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
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
  ierr = SlepcHeaderCreate(qep,_p_QEP,struct _QEPOps,QEP_CLASSID,-1,"QEP","Quadratic Eigenvalue Problem","QEP",comm,QEPDestroy,QEPView);CHKERRQ(ierr);

  qep->M               = 0;
  qep->C               = 0;
  qep->K               = 0;
  qep->max_it          = 0;
  qep->nev             = 1;
  qep->ncv             = 0;
  qep->mpd             = 0;
  qep->nini            = 0;
  qep->ninil           = 0;
  qep->allocated_ncv   = 0;
  qep->ip              = 0;
  qep->ds              = 0;
  qep->tol             = PETSC_DEFAULT;
  qep->sfactor         = 0.0;
  qep->conv_func       = QEPConvergedDefault;
  qep->conv_ctx        = PETSC_NULL;
  qep->which           = (QEPWhich)0;
  qep->which_func      = PETSC_NULL;
  qep->which_ctx       = PETSC_NULL;
  qep->leftvecs        = PETSC_FALSE;
  qep->problem_type    = (QEPProblemType)0;
  qep->V               = PETSC_NULL;
  qep->W               = PETSC_NULL;
  qep->IS              = PETSC_NULL;
  qep->ISL             = PETSC_NULL;
  qep->eigr            = PETSC_NULL;
  qep->eigi            = PETSC_NULL;
  qep->errest          = PETSC_NULL;
  qep->data            = PETSC_NULL;
  qep->t               = PETSC_NULL;
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
  qep->rand            = 0;

  ierr = PetscRandomCreate(comm,&qep->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(qep->rand,0x12345678);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(qep,qep->rand);CHKERRQ(ierr);
  *outqep = qep;
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "QEPSetType"
/*@C
   QEPSetType - Selects the particular solver to be used in the QEP object. 

   Logically Collective on QEP

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
PetscErrorCode QEPSetType(QEP qep,QEPType type)
{
  PetscErrorCode ierr,(*r)(QEP);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)qep,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFListFind(QEPList,((PetscObject)qep)->comm,type,PETSC_TRUE,(void (**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(((PetscObject)qep)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown QEP type given: %s",type);

  if (qep->ops->destroy) { ierr = (*qep->ops->destroy)(qep);CHKERRQ(ierr); }
  ierr = PetscMemzero(qep->ops,sizeof(struct _QEPOps));CHKERRQ(ierr);

  qep->setupcalled = 0;
  ierr = PetscObjectChangeTypeName((PetscObject)qep,type);CHKERRQ(ierr);
  ierr = (*r)(qep);CHKERRQ(ierr);
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
PetscErrorCode QEPGetType(QEP qep,QEPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)qep)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPRegister"
/*@C
  QEPRegister - See QEPRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode QEPRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(QEP))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

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
  QEPRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPReset"
/*@
   QEPReset - Resets the QEP context to the setupcalled=0 state and removes any
   allocated objects.

   Collective on QEP

   Input Parameter:
.  qep - eigensolver context obtained from QEPCreate()

   Level: advanced

.seealso: QEPDestroy()
@*/
PetscErrorCode QEPReset(QEP qep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (qep->ops->reset) { ierr = (qep->ops->reset)(qep);CHKERRQ(ierr); }
  if (qep->ip) { ierr = IPReset(qep->ip);CHKERRQ(ierr); }
  if (qep->ds) { ierr = DSReset(qep->ds);CHKERRQ(ierr); }
  ierr = MatDestroy(&qep->M);CHKERRQ(ierr);
  ierr = MatDestroy(&qep->C);CHKERRQ(ierr);
  ierr = MatDestroy(&qep->K);CHKERRQ(ierr);
  ierr = VecDestroy(&qep->t);CHKERRQ(ierr);
  ierr = QEPFreeSolution(qep);CHKERRQ(ierr);
  qep->matvecs     = 0;
  qep->linits      = 0;
  qep->setupcalled = 0;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPDestroy"
/*@C
   QEPDestroy - Destroys the QEP context.

   Collective on QEP

   Input Parameter:
.  qep - eigensolver context obtained from QEPCreate()

   Level: beginner

.seealso: QEPCreate(), QEPSetUp(), QEPSolve()
@*/
PetscErrorCode QEPDestroy(QEP *qep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*qep) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*qep,QEP_CLASSID,1);
  if (--((PetscObject)(*qep))->refct > 0) { *qep = 0; PetscFunctionReturn(0); }
  ierr = QEPReset(*qep);CHKERRQ(ierr);
  ierr = PetscObjectDepublish(*qep);CHKERRQ(ierr);
  if ((*qep)->ops->destroy) { ierr = (*(*qep)->ops->destroy)(*qep);CHKERRQ(ierr); }
  ierr = STDestroy(&(*qep)->st);CHKERRQ(ierr);
  ierr = IPDestroy(&(*qep)->ip);CHKERRQ(ierr);
  ierr = DSDestroy(&(*qep)->ds);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&(*qep)->rand);CHKERRQ(ierr);
  ierr = QEPMonitorCancel(*qep);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(qep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetIP"
/*@
   QEPSetIP - Associates an inner product object to the quadratic eigensolver. 

   Collective on QEP

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
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidHeaderSpecific(ip,IP_CLASSID,2);
  PetscCheckSameComm(qep,1,ip,2);
  ierr = PetscObjectReference((PetscObject)ip);CHKERRQ(ierr);
  ierr = IPDestroy(&qep->ip);CHKERRQ(ierr);
  qep->ip = ip;
  ierr = PetscLogObjectParent(qep,qep->ip);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(ip,2);
  if (!qep->ip) {
    ierr = IPCreate(((PetscObject)qep)->comm,&qep->ip);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(qep,qep->ip);CHKERRQ(ierr);
  }
  *ip = qep->ip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetDS"
/*@
   QEPSetDS - Associates a direct solver object to the quadratic eigensolver. 

   Collective on QEP

   Input Parameters:
+  qep - eigensolver context obtained from QEPCreate()
-  ds  - the direct solver object

   Note:
   Use QEPGetDS() to retrieve the direct solver context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: QEPGetDS()
@*/
PetscErrorCode QEPSetDS(QEP qep,DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidHeaderSpecific(ds,DS_CLASSID,2);
  PetscCheckSameComm(qep,1,ds,2);
  ierr = PetscObjectReference((PetscObject)ds);CHKERRQ(ierr);
  ierr = DSDestroy(&qep->ds);CHKERRQ(ierr);
  qep->ds = ds;
  ierr = PetscLogObjectParent(qep,qep->ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetDS"
/*@C
   QEPGetDS - Obtain the direct solver object associated to the 
   quadratic eigensolver object.

   Not Collective

   Input Parameters:
.  qep - eigensolver context obtained from QEPCreate()

   Output Parameter:
.  ds - direct solver context

   Level: advanced

.seealso: QEPSetDS()
@*/
PetscErrorCode QEPGetDS(QEP qep,DS *ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(ds,2);
  if (!qep->ds) {
    ierr = DSCreate(((PetscObject)qep)->comm,&qep->ds);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(qep,qep->ds);CHKERRQ(ierr);
  }
  *ds = qep->ds;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetST"
/*@
   QEPSetST - Associates a spectral transformation object to the eigensolver. 

   Collective on QEP

   Input Parameters:
+  qep - eigensolver context obtained from QEPCreate()
-  st   - the spectral transformation object

   Note:
   Use QEPGetST() to retrieve the spectral transformation context (for example,
   to free it at the end of the computations).

   Level: developer

.seealso: QEPGetST()
@*/
PetscErrorCode QEPSetST(QEP qep,ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidHeaderSpecific(st,ST_CLASSID,2);
  PetscCheckSameComm(qep,1,st,2);
  ierr = PetscObjectReference((PetscObject)st);CHKERRQ(ierr);
  ierr = STDestroy(&qep->st);CHKERRQ(ierr);
  qep->st = st;
  ierr = PetscLogObjectParent(qep,qep->st);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetST"
/*@C
   QEPGetST - Obtain the spectral transformation (ST) object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  qep - eigensolver context obtained from QEPCreate()

   Output Parameter:
.  st - spectral transformation context

   Level: beginner

.seealso: QEPSetST()
@*/
PetscErrorCode QEPGetST(QEP qep,ST *st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidPointer(st,2);
  if (!qep->st) {
    ierr = STCreate(((PetscObject)qep)->comm,&qep->st);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(qep,qep->st);CHKERRQ(ierr);
  }
  *st = qep->st;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSetTarget"
/*@
   QEPSetTarget - Sets the value of the target.

   Logically Collective on QEP

   Input Parameters:
+  qep    - eigensolver context
-  target - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest. It is used in combination with QEPSetWhichEigenpairs().
   
   Level: beginner

.seealso: QEPGetTarget(), QEPSetWhichEigenpairs()
@*/
PetscErrorCode QEPSetTarget(QEP qep,PetscScalar target)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(qep,target,2);
  qep->target = target;
  if (!qep->st) { ierr = QEPGetST(qep,&qep->st);CHKERRQ(ierr); }
  ierr = STSetDefaultShift(qep->st,target);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetTarget"
/*@
   QEPGetTarget - Gets the value of the target.

   Not Collective

   Input Parameter:
.  qep - eigensolver context

   Output Parameter:
.  target - the value of the target

   Level: beginner

   Note:
   If the target was not set by the user, then zero is returned.

.seealso: QEPSetTarget()
@*/
PetscErrorCode QEPGetTarget(QEP qep,PetscScalar* target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidScalarPointer(target,2);
  *target = qep->target;
  PetscFunctionReturn(0);
}

