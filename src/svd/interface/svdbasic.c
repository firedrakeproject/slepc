/*
     The basic SVD routines, Create, View, etc. are here.

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

#include "private/svdimpl.h"      /*I "slepcsvd.h" I*/

PetscFList SVDList = 0;
PetscClassId SVD_CLASSID = 0;
PetscLogEvent SVD_SetUp = 0, SVD_Solve = 0, SVD_Dense = 0;
static PetscTruth SVDPackageInitialized = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "SVDFinalizePackage"
/*@C
  SVDFinalizePackage - This function destroys everything in the Slepc interface to the SVD package. It is
  called from SlepcFinalize().

  Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode SVDFinalizePackage(void) 
{
  PetscFunctionBegin;
  SVDPackageInitialized = PETSC_FALSE;
  SVDList               = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDInitializePackage"
/*@C
  SVDInitializePackage - This function initializes everything in the SVD package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to SVDCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode SVDInitializePackage(const char *path)
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (SVDPackageInitialized) PetscFunctionReturn(0);
  SVDPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Singular Value Solver",&SVD_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = SVDRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("SVDSetUp",SVD_CLASSID,&SVD_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SVDSolve",SVD_CLASSID,&SVD_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SVDDense",SVD_CLASSID,&SVD_Dense);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "svd", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SVD_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "svd", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SVD_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(SVDFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  const SVDType  type;
  PetscTruth     isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)svd)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"SVD Object:\n");CHKERRQ(ierr);
    ierr = SVDGetType(svd,&type);CHKERRQ(ierr);
    if (type) {
      ierr = PetscViewerASCIIPrintf(viewer,"  method: %s\n",type);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  method: not yet set\n");CHKERRQ(ierr);
    }
    switch (svd->transmode) {
      case SVD_TRANSPOSE_EXPLICIT:
        ierr = PetscViewerASCIIPrintf(viewer,"  transpose mode: explicit\n");CHKERRQ(ierr);
	break;
      case SVD_TRANSPOSE_IMPLICIT:
        ierr = PetscViewerASCIIPrintf(viewer,"  transpose mode: implicit\n");CHKERRQ(ierr);
	break;
      default:
        ierr = PetscViewerASCIIPrintf(viewer,"  transpose mode: not yet set\n");CHKERRQ(ierr);
    }
    if (svd->which == SVD_LARGEST) {
      ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: largest\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: smallest\n");CHKERRQ(ierr);
    }  
    ierr = PetscViewerASCIIPrintf(viewer,"  number of singular values (nsv): %d\n",svd->nsv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %d\n",svd->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %d\n",svd->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %d\n",svd->max_it);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",svd->tol);CHKERRQ(ierr);
    if (svd->nini!=0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %d\n",PetscAbs(svd->nini));CHKERRQ(ierr);
    }
    if (svd->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*svd->ops->view)(svd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = IPView(svd->ip,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    if (svd->ops->view) {
      ierr = (*svd->ops->view)(svd,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDCreate"
/*@C
   SVDCreate - Creates the default SVD context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  svd - location to put the SVD context

   Note:
   The default SVD type is SVDCROSS

   Level: beginner

.seealso: SVDSetUp(), SVDSolve(), SVDDestroy(), SVD
@*/
PetscErrorCode SVDCreate(MPI_Comm comm,SVD *outsvd)
{
  PetscErrorCode ierr;
  SVD            svd;

  PetscFunctionBegin;
  PetscValidPointer(outsvd,2);

  ierr = PetscHeaderCreate(svd,_p_SVD,struct _SVDOps,SVD_CLASSID,-1,"SVD",comm,SVDDestroy,SVDView);CHKERRQ(ierr);
  *outsvd = svd;

  ierr = PetscMemzero(svd->ops,sizeof(struct _SVDOps));CHKERRQ(ierr);

  svd->OP          = PETSC_NULL;
  svd->A           = PETSC_NULL;
  svd->AT          = PETSC_NULL;
  svd->transmode   = (SVDTransposeMode)PETSC_DECIDE;
  svd->sigma       = PETSC_NULL;
  svd->perm        = PETSC_NULL;
  svd->U           = PETSC_NULL;
  svd->V           = PETSC_NULL;
  svd->IS          = PETSC_NULL;
  svd->which       = SVD_LARGEST;
  svd->n           = 0;
  svd->nconv       = 0;
  svd->nsv         = 1;    
  svd->ncv         = 0;    
  svd->mpd         = 0;    
  svd->nini        = 0;
  svd->its         = 0;
  svd->max_it      = 0;  
  svd->tol         = 1e-7;    
  svd->errest      = PETSC_NULL;
  svd->data        = PETSC_NULL;
  svd->setupcalled = 0;
  svd->reason      = SVD_CONVERGED_ITERATING;
  svd->numbermonitors = 0;
  svd->matvecs = 0;
  svd->trackall    = PETSC_FALSE;

  ierr = IPCreate(comm,&svd->ip);CHKERRQ(ierr);
  ierr = IPSetOptionsPrefix(svd->ip,((PetscObject)svd)->prefix);
  ierr = IPAppendOptionsPrefix(svd->ip,"svd_");
  PetscLogObjectParent(svd,svd->ip);

  ierr = PetscPublishAll(svd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy"
/*@
   SVDDestroy - Destroys the SVD context.

   Collective on SVD

   Input Parameter:
.  svd - singular value solver context obtained from SVDCreate()

   Level: beginner

.seealso: SVDCreate(), SVDSetUp(), SVDSolve()
@*/
PetscErrorCode SVDDestroy(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *p;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (--((PetscObject)svd)->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(svd);CHKERRQ(ierr);

  if (svd->ops->destroy) {
    ierr = (*svd->ops->destroy)(svd); CHKERRQ(ierr);
  }

  if (svd->OP) { ierr = MatDestroy(svd->OP);CHKERRQ(ierr); }
  if (svd->A) { ierr = MatDestroy(svd->A);CHKERRQ(ierr); }
  if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
  if (svd->n) { 
    ierr = PetscFree(svd->sigma);CHKERRQ(ierr);
    ierr = PetscFree(svd->perm);CHKERRQ(ierr);
    ierr = PetscFree(svd->errest);CHKERRQ(ierr);
    if (svd->U) {
      ierr = VecGetArray(svd->U[0],&p);CHKERRQ(ierr);
      for (i=0;i<svd->n;i++) {
        ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr);
      }
      ierr = PetscFree(p);CHKERRQ(ierr);
      ierr = PetscFree(svd->U);CHKERRQ(ierr);
    }
    ierr = VecGetArray(svd->V[0],&p);CHKERRQ(ierr);
    for (i=0;i<svd->n;i++) {
      ierr = VecDestroy(svd->V[i]);CHKERRQ(ierr); 
    }
    ierr = PetscFree(p);CHKERRQ(ierr);
    ierr = PetscFree(svd->V);CHKERRQ(ierr);
  }
  ierr = SVDMonitorCancel(svd);CHKERRQ(ierr);
  
  ierr = IPDestroy(svd->ip);CHKERRQ(ierr);
  if (svd->rand) {
    ierr = PetscRandomDestroy(svd->rand);CHKERRQ(ierr);
  }
  
  ierr = PetscHeaderDestroy(svd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy_Default"
PetscErrorCode SVDDestroy_Default(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetType"
/*@C
   SVDSetType - Selects the particular solver to be used in the SVD object. 

   Collective on SVD

   Input Parameters:
+  svd      - the singular value solver context
-  type     - a known method

   Options Database Key:
.  -svd_type <method> - Sets the method; use -help for a list 
    of available methods 
    
   Notes:  
   See "slepc/include/slepcsvd.h" for available methods. The default
   is SVDCROSS.

   Normally, it is best to use the SVDSetFromOptions() command and
   then set the SVD type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The SVDSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database. 

   Level: intermediate

.seealso: SVDType
@*/
PetscErrorCode SVDSetType(SVD svd,const SVDType type)
{
  PetscErrorCode ierr,(*r)(SVD);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)svd,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (svd->data) {
    /* destroy the old private SVD context */
    ierr = (*svd->ops->destroy)(svd); CHKERRQ(ierr);
    svd->data = 0;
  }

  ierr = PetscFListFind(SVDList,((PetscObject)svd)->comm,type,(void (**)(void)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,"Unknown SVD type given: %s",type);

  svd->setupcalled = 0;
  ierr = PetscMemzero(svd->ops,sizeof(struct _SVDOps));CHKERRQ(ierr);
  ierr = (*r)(svd); CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)svd,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetType"
/*@C
   SVDGetType - Gets the SVD type as a string from the SVD object.

   Not Collective

   Input Parameter:
.  svd - the singular value solver context 

   Output Parameter:
.  name - name of SVD method 

   Level: intermediate

.seealso: SVDSetType()
@*/
PetscErrorCode SVDGetType(SVD svd,const SVDType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)svd)->type_name;
  PetscFunctionReturn(0);
}

/*MC
   SVDRegisterDynamic - Adds a method to the singular value solver package.

   Synopsis:
   SVDRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(SVD))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create the solver context
-  routine_create - routine to create the solver context

   Notes:
   SVDRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   SVDRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     SVDSetType(svd,"my_solver")
   or at runtime via the option
$     -svd_type my_solver

   Level: advanced

   Environmental variables such as ${PETSC_ARCH}, ${SLEPC_DIR},
   and others of the form ${any_environmental_variable} occuring in pathname will be 
   replaced with appropriate values.

.seealso: SVDRegisterDestroy(), SVDRegisterAll()

M*/

#undef __FUNCT__  
#define __FUNCT__ "SVDRegister"
/*@C
  SVDRegister - See SVDRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode SVDRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(SVD))
{
  PetscErrorCode ierr;
  char           fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&SVDList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDRegisterDestroy"
/*@
   SVDRegisterDestroy - Frees the list of SVD methods that were
   registered by SVDRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: SVDRegisterDynamic(), SVDRegisterAll()
@*/
PetscErrorCode SVDRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&SVDList);CHKERRQ(ierr);
  ierr = SVDRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetIP"
/*@
   SVDSetIP - Associates an inner product object to the
   singular value solver. 

   Collective on SVD

   Input Parameters:
+  svd - singular value solver context obtained from SVDCreate()
-  ip  - the inner product object

   Note:
   Use SVDGetIP() to retrieve the inner product context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: SVDGetIP()
@*/
PetscErrorCode SVDSetIP(SVD svd,IP ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(ip,IP_CLASSID,2);
  PetscCheckSameComm(svd,1,ip,2);
  ierr = PetscObjectReference((PetscObject)ip);CHKERRQ(ierr);
  ierr = IPDestroy(svd->ip); CHKERRQ(ierr);
  svd->ip = ip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetIP"
/*@C
   SVDGetIP - Obtain the inner product object associated
   to the singular value solver object.

   Not Collective

   Input Parameters:
.  svd - singular value solver context obtained from SVDCreate()

   Output Parameter:
.  ip - inner product context

   Level: advanced

.seealso: SVDSetIP()
@*/
PetscErrorCode SVDGetIP(SVD svd,IP *ip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(ip,2);
  *ip = svd->ip;
  PetscFunctionReturn(0);
}
