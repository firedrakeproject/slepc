/*
     The basic SVD routines, Create, View, etc. are here.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <private/svdimpl.h>      /*I "slepcsvd.h" I*/

PetscFList       SVDList = 0;
PetscBool        SVDRegisterAllCalled = PETSC_FALSE;
PetscClassId     SVD_CLASSID = 0;
PetscLogEvent    SVD_SetUp = 0,SVD_Solve = 0,SVD_Dense = 0;
static PetscBool SVDPackageInitialized = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "SVDFinalizePackage"
/*@C
   SVDFinalizePackage - This function destroys everything in the Slepc interface
   to the SVD package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode SVDFinalizePackage(void) 
{
  PetscFunctionBegin;
  SVDPackageInitialized = PETSC_FALSE;
  SVDList               = 0;
  SVDRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDInitializePackage"
/*@C
   SVDInitializePackage - This function initializes everything in the SVD package. It is called
   from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to SVDCreate()
   when using static libraries.

   Input Parameter:
.  path - The dynamic library path, or PETSC_NULL

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode SVDInitializePackage(const char *path)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

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
  ierr = PetscOptionsGetString(PETSC_NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"svd",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SVD_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"svd",&className);CHKERRQ(ierr);
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
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)svd)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)svd,viewer,"SVD Object");CHKERRQ(ierr);
    if (svd->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*svd->ops->view)(svd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer,"  number of singular values (nsv): %D\n",svd->nsv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",svd->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %D\n",svd->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",svd->max_it);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %G\n",svd->tol);CHKERRQ(ierr);
    if (svd->nini!=0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %D\n",PetscAbs(svd->nini));CHKERRQ(ierr);
    }
  } else {
    if (svd->ops->view) {
      ierr = (*svd->ops->view)(svd,viewer);CHKERRQ(ierr);
    }
  }
  if (!svd->ip) { ierr = SVDGetIP(svd,&svd->ip);CHKERRQ(ierr); }
  ierr = IPView(svd->ip,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDPrintSolution"
/*@
   SVDPrintSolution - Prints the computed singular values.

   Collective on SVD

   Input Parameters:
+  svd - the singular value solver context
-  viewer - optional visualization context

   Options Database:
.  -svd_terse - print only minimal information

   Note:
   By default, this function prints a table with singular values and associated
   relative errors. With -svd_terse only the singular values are printed.

   Level: intermediate

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode SVDPrintSolution(SVD svd,PetscViewer viewer)
{
  PetscBool      terse,errok,isascii;
  PetscReal      error,sigma;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)svd)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(svd,1,viewer,2);
  if (!svd->sigma) { 
    SETERRQ(((PetscObject)svd)->comm,PETSC_ERR_ARG_WRONGSTATE,"SVDSolve must be called first"); 
  }
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);

  ierr = PetscOptionsHasName(PETSC_NULL,"-svd_terse",&terse);CHKERRQ(ierr);
  if (terse) {
    if (svd->nconv<svd->nsv) {
      ierr = PetscViewerASCIIPrintf(viewer," Problem: less than %D singular values converged\n\n",svd->nsv);CHKERRQ(ierr);
    } else {
      errok = PETSC_TRUE;
      for (i=0;i<svd->nsv;i++) {
        ierr = SVDComputeRelativeError(svd,i,&error);CHKERRQ(ierr);
        errok = (errok && error<svd->tol)? PETSC_TRUE: PETSC_FALSE;
      }
      if (errok) {
        ierr = PetscViewerASCIIPrintf(viewer," All requested singular values computed up to the required tolerance:");CHKERRQ(ierr);
        for (i=0;i<=(svd->nsv-1)/8;i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"\n     ");CHKERRQ(ierr);
          for (j=0;j<PetscMin(8,svd->nsv-8*i);j++) {
            ierr = SVDGetSingularTriplet(svd,8*i+j,&sigma,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscViewerASCIIPrintf(viewer,"%.5F",sigma);CHKERRQ(ierr); 
            if (8*i+j+1<svd->nsv) { ierr = PetscViewerASCIIPrintf(viewer,", ");CHKERRQ(ierr); }
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer," Problem: some of the first %D relative errors are higher than the tolerance\n\n",svd->nsv);CHKERRQ(ierr);
      }
    }
  } else {
    ierr = PetscViewerASCIIPrintf(viewer," Number of converged approximate singular triplets: %D\n\n",svd->nconv);CHKERRQ(ierr);
    if (svd->nconv>0) {
      ierr = PetscViewerASCIIPrintf(viewer,
           "          sigma            relative error\n"
           "   --------------------- ------------------\n");CHKERRQ(ierr);
      for (i=0;i<svd->nconv;i++) {
        ierr = SVDGetSingularTriplet(svd,i,&sigma,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        ierr = SVDComputeRelativeError(svd,i,&error);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"       % 6F          %12G\n",sigma,error);CHKERRQ(ierr); 
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
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
  *outsvd = 0;
  ierr = PetscHeaderCreate(svd,_p_SVD,struct _SVDOps,SVD_CLASSID,-1,"SVD","Singular Value Decomposition","SVD",comm,SVDDestroy,SVDView);CHKERRQ(ierr);

  svd->OP             = PETSC_NULL;
  svd->A              = PETSC_NULL;
  svd->AT             = PETSC_NULL;
  svd->transmode      = (SVDTransposeMode)PETSC_DECIDE;
  svd->sigma          = PETSC_NULL;
  svd->perm           = PETSC_NULL;
  svd->U              = PETSC_NULL;
  svd->V              = PETSC_NULL;
  svd->IS             = PETSC_NULL;
  svd->tl             = PETSC_NULL;
  svd->tr             = PETSC_NULL;
  svd->rand           = PETSC_NULL;
  svd->which          = SVD_LARGEST;
  svd->n              = 0;
  svd->nconv          = 0;
  svd->nsv            = 1;    
  svd->ncv            = 0;    
  svd->mpd            = 0;    
  svd->nini           = 0;
  svd->its            = 0;
  svd->max_it         = 0;  
  svd->tol            = 1e-7;    
  svd->errest         = PETSC_NULL;
  svd->data           = PETSC_NULL;
  svd->setupcalled    = 0;
  svd->reason         = SVD_CONVERGED_ITERATING;
  svd->numbermonitors = 0;
  svd->matvecs        = 0;
  svd->trackall       = PETSC_FALSE;

  ierr = PetscRandomCreate(comm,&svd->rand);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(svd,svd->rand);CHKERRQ(ierr);
  *outsvd = svd;
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "SVDReset"
/*@
   SVDReset - Resets the SVD context to the setupcalled=0 state and removes any
   allocated objects.

   Collective on SVD

   Input Parameter:
.  svd - singular value solver context obtained from SVDCreate()

   Level: advanced

.seealso: SVDDestroy()
@*/
PetscErrorCode SVDReset(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (svd->ops->reset) { ierr = (svd->ops->reset)(svd);CHKERRQ(ierr); }
  if (svd->ip) { ierr = IPReset(svd->ip);CHKERRQ(ierr); }
  ierr = MatDestroy(&svd->OP);CHKERRQ(ierr);
  ierr = MatDestroy(&svd->A);CHKERRQ(ierr);
  ierr = MatDestroy(&svd->AT);CHKERRQ(ierr);
  ierr = VecDestroy(&svd->tl);CHKERRQ(ierr);
  ierr = VecDestroy(&svd->tr);CHKERRQ(ierr);
  if (svd->n) { 
    ierr = PetscFree(svd->sigma);CHKERRQ(ierr);
    ierr = PetscFree(svd->perm);CHKERRQ(ierr);
    ierr = PetscFree(svd->errest);CHKERRQ(ierr);
    ierr = VecDestroyVecs(svd->n,&svd->U);CHKERRQ(ierr);
    ierr = VecDestroyVecs(svd->n,&svd->V);CHKERRQ(ierr);
  }
  svd->transmode   = (SVDTransposeMode)PETSC_DECIDE;
  svd->matvecs     = 0;
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy"
/*@C
   SVDDestroy - Destroys the SVD context.

   Collective on SVD

   Input Parameter:
.  svd - singular value solver context obtained from SVDCreate()

   Level: beginner

.seealso: SVDCreate(), SVDSetUp(), SVDSolve()
@*/
PetscErrorCode SVDDestroy(SVD *svd)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (!*svd) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*svd,SVD_CLASSID,1);
  if (--((PetscObject)(*svd))->refct > 0) { *svd = 0; PetscFunctionReturn(0); }
  ierr = SVDReset(*svd);CHKERRQ(ierr);
  ierr = PetscObjectDepublish(*svd);CHKERRQ(ierr);
  if ((*svd)->ops->destroy) { ierr = (*(*svd)->ops->destroy)(*svd);CHKERRQ(ierr); }
  ierr = IPDestroy(&(*svd)->ip);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&(*svd)->rand);CHKERRQ(ierr);
  ierr = SVDMonitorCancel(*svd);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(svd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetType"
/*@C
   SVDSetType - Selects the particular solver to be used in the SVD object. 

   Logically Collective on SVD

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
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)svd,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFListFind(SVDList,((PetscObject)svd)->comm,type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(((PetscObject)svd)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown SVD type given: %s",type);

  if (svd->ops->destroy) { ierr = (*svd->ops->destroy)(svd);CHKERRQ(ierr); }
  ierr = PetscMemzero(svd->ops,sizeof(struct _SVDOps));CHKERRQ(ierr);

  svd->setupcalled = 0;
  ierr = PetscObjectChangeTypeName((PetscObject)svd,type);CHKERRQ(ierr);
  ierr = (*r)(svd);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "SVDRegister"
/*@C
   SVDRegister - See SVDRegisterDynamic()

   Level: advanced
@*/
PetscErrorCode SVDRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(SVD))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

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
  SVDRegisterAllCalled = PETSC_FALSE;
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
  ierr = IPDestroy(&svd->ip);CHKERRQ(ierr);
  svd->ip = ip;
  ierr = PetscLogObjectParent(svd,svd->ip);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(ip,2);
  if (!svd->ip) {
    ierr = IPCreate(((PetscObject)svd)->comm,&svd->ip);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(svd,svd->ip);CHKERRQ(ierr);
  }
  *ip = svd->ip;
  PetscFunctionReturn(0);
}
