/*
     SVD routines for setting solver options.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/svdimpl.h"      /*I "slepcsvd.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SVDSetTransposeMode"
/*@
   SVDSetTransposeMode - Sets how to handle the transpose of the matrix 
   associated with the singular value problem.

   Collective on SVD and Mat

   Input Parameters:
+  svd  - the singular value solver context
-  mode - how to compute the transpose, one of SVD_TRANSPOSE_EXPLICIT
          or SVD_TRANSPOSE_IMPLICIT (see notes below)

   Options Database Key:
.  -svd_transpose_mode <mode> - Indicates the mode flag, where <mode> 
    is one of 'explicit' or 'implicit'.

   Notes:
   In the SVD_TRANSPOSE_EXPLICIT mode, the transpose of the matrix is
   explicitly built.

   The option SVD_TRANSPOSE_IMPLICIT does not build the transpose, but
   handles it implicitly via MatMultTranspose() operations. This is 
   likely to be more inefficient than SVD_TRANSPOSE_EXPLICIT, both in
   sequential and in parallel, but requires less storage.

   The default is SVD_TRANSPOSE_EXPLICIT if the matrix has defined the
   MatTranspose operation, and SVD_TRANSPOSE_IMPLICIT otherwise.
   
   Level: advanced
   
.seealso: SVDGetTransposeMode(), SVDSolve(), SVDSetOperator(), 
   SVDGetOperator(), SVDTransposeMode
@*/
PetscErrorCode SVDSetTransposeMode(SVD svd,SVDTransposeMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  switch (mode) {
    case PETSC_DEFAULT:
      mode = (SVDTransposeMode)PETSC_DECIDE;
    case SVD_TRANSPOSE_EXPLICIT:
    case SVD_TRANSPOSE_IMPLICIT:
    case PETSC_DECIDE:
      svd->transmode = mode;
      svd->setupcalled = 0;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid transpose mode"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetTransposeMode"
/*@C
   SVDGetTransposeMode - Gets the mode use to compute the  transpose 
   of the matrix associated with the singular value problem.

   Not collective

   Input Parameter:
.  svd  - the singular value solver context

   Output paramter:
.  mode - how to compute the transpose, one of SVD_TRANSPOSE_EXPLICIT
          or SVD_TRANSPOSE_IMPLICIT
   
   Level: advanced
   
.seealso: SVDSetTransposeMode(), SVDSolve(), SVDSetOperator(), 
   SVDGetOperator(), SVDTransposeMode
@*/
PetscErrorCode SVDGetTransposeMode(SVD svd,SVDTransposeMode *mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidPointer(mode,2);
  *mode = svd->transmode;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetTolerances"
/*@
   SVDSetTolerances - Sets the tolerance and maximum
   iteration count used by the default SVD convergence testers. 

   Collective on SVD

   Input Parameters:
+  svd - the singular value solver context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -svd_tol <tol> - Sets the convergence tolerance
-  -svd_max_it <maxits> - Sets the maximum number of iterations allowed 
   (use PETSC_DECIDE to compute an educated guess based on basis and matrix sizes)

   Notes:
   Use PETSC_IGNORE to retain the previous value of any parameter. 

   Level: intermediate

.seealso: SVDGetTolerances()
@*/
PetscErrorCode SVDSetTolerances(SVD svd,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (tol != PETSC_IGNORE) {
    if (tol == PETSC_DEFAULT) {
      tol = 1e-7;
    } else {
      if (tol < 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
      svd->tol = tol;
    }
  }
  if (maxits != PETSC_IGNORE) {
    if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
      svd->max_it = PETSC_DECIDE;
      svd->setupcalled = 0;
    } else {
      if (maxits < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
      svd->max_it = maxits;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetTolerances"
/*@
   SVDGetTolerances - Gets the tolerance and maximum
   iteration count used by the default SVD convergence tests. 

   Not Collective

   Input Parameter:
.  svd - the singular value solver context
  
   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: SVDSetTolerances()
@*/
PetscErrorCode SVDGetTolerances(SVD svd,PetscReal *tol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (tol)    *tol    = svd->tol;
  if (maxits) *maxits = svd->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetDimensions"
/*@
   SVDSetDimensions - Sets the number of singular values to compute
   and the dimension of the subspace.

   Collective on SVD

   Input Parameters:
+  svd - the singular value solver context
.  nsv - number of singular values to compute
-  ncv - the maximum dimension of the subspace to be used by the solver

   Options Database Keys:
+  -svd_nsv <nsv> - Sets the number of singular values
-  -svd_ncv <ncv> - Sets the dimension of the subspace

   Notes:
   Use PETSC_IGNORE to retain the previous value of any parameter.

   Use PETSC_DECIDE for ncv to assign a reasonably good value, which is 
   dependent on the solution method and the number of singular values required.

   Level: intermediate

.seealso: SVDGetDimensions()
@*/
PetscErrorCode SVDSetDimensions(SVD svd,PetscInt nsv,PetscInt ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);

  if (nsv != PETSC_IGNORE) {
    if (nsv<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nsv. Must be > 0");
    svd->nsv = nsv;
  }
  if (ncv != PETSC_IGNORE) {
    if (ncv == PETSC_DEFAULT || ncv == PETSC_DECIDE) {
      svd->ncv = PETSC_DECIDE;
    } else {
      if (ncv<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
      svd->ncv = ncv;
    }
    svd->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetDimensions"
/*@
   SVDGetDimensions - Gets the number of singular values to compute
   and the dimension of the subspace.

   Not Collective

   Input Parameter:
.  svd - the singular value context
  
   Output Parameters:
+  nsv - number of singular values to compute
-  ncv - the maximum dimension of the subspace to be used by the solver

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: SVDSetDimensions()
@*/
PetscErrorCode SVDGetDimensions(SVD svd,PetscInt *nsv,PetscInt *ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (nsv) *nsv = svd->nsv;
  if (ncv) *ncv = svd->ncv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetWhichSingularTriplets"
/*@
    SVDSetWhichSingularTriplets - Specifies which singular triplets are 
    to be sought.

    Collective on SVD

    Input Parameter:
.   svd - singular value solver context obtained from SVDCreate()

    Output Parameter:
.   which - which singular triplets are to be sought

    Possible values:
    The parameter 'which' can have one of these values:
    
+     SVD_LARGEST  - largest singular values
-     SVD_SMALLEST - smallest singular values

    Options Database Keys:
+   -svd_largest  - Sets largest singular values
-   -svd_smallest - Sets smallest singular values
    
    Level: intermediate

.seealso: SVDGetWhichSingularTriplets(), SVDWhich
@*/
PetscErrorCode SVDSetWhichSingularTriplets(SVD svd,SVDWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  switch (which) {
    case SVD_LARGEST:
    case SVD_SMALLEST:
      if (svd->which != which) {
        svd->setupcalled = 0;
        svd->which = which;
      }
      break;
  default:
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'which' parameter");    
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetWhichSingularTriplets"
/*@C
    SVDGetWhichSingularTriplets - Returns which singular triplets are
    to be sought.

    Not Collective

    Input Parameter:
.   svd - singular value solver context obtained from SVDCreate()

    Output Parameter:
.   which - which singular triplets are to be sought

    Notes:
    See SVDSetWhichSingularTriplets() for possible values of which

    Level: intermediate

.seealso: SVDSetWhichSingularTriplets(), SVDWhich
@*/
PetscErrorCode SVDGetWhichSingularTriplets(SVD svd,SVDWhich *which) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidPointer(which,2);
  *which = svd->which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions"
/*@
   SVDSetFromOptions - Sets SVD options from the options database.
   This routine must be called before SVDSetUp() if the user is to be 
   allowed to set the solver type. 

   Collective on SVD

   Input Parameters:
.  svd - the singular value solver context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner

.seealso: 
@*/
PetscErrorCode SVDSetFromOptions(SVD svd)
{
  PetscErrorCode ierr;
  char           type[256],monfilename[PETSC_MAX_PATH_LEN];
  PetscTruth     flg;
  const char     *mode_list[2] = { "explicit", "implicit" };
  PetscInt       i,j;
  PetscReal      r;
  PetscViewer    monviewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  svd->setupcalled = 0;
  ierr = PetscOptionsBegin(((PetscObject)svd)->comm,((PetscObject)svd)->prefix,"Singular Value Solver (SVD) Options","SVD");CHKERRQ(ierr);

  ierr = PetscOptionsList("-svd_type","Singular Value Solver method","SVDSetType",SVDList,(char*)(((PetscObject)svd)->type_name?((PetscObject)svd)->type_name:SVDCROSS),type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SVDSetType(svd,type);CHKERRQ(ierr);
  } else if (!((PetscObject)svd)->type_name) {
    ierr = SVDSetType(svd,SVDCROSS);CHKERRQ(ierr);
  }

  ierr = PetscOptionsName("-svd_view","Print detailed information on solver used","SVDView",0);CHKERRQ(ierr);

  ierr = PetscOptionsEList("-svd_transpose_mode","Transpose SVD mode","SVDSetTransposeMode",mode_list,2,svd->transmode == PETSC_DECIDE ? "decide" : mode_list[svd->transmode],&i,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SVDSetTransposeMode(svd,(SVDTransposeMode)i);CHKERRQ(ierr);
  }   

  r = i = PETSC_IGNORE;
  ierr = PetscOptionsInt("-svd_max_it","Maximum number of iterations","SVDSetTolerances",svd->max_it,&i,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-svd_tol","Tolerance","SVDSetTolerances",svd->tol,&r,PETSC_NULL);CHKERRQ(ierr);
  ierr = SVDSetTolerances(svd,r,i);CHKERRQ(ierr);

  i = j = PETSC_IGNORE;
  ierr = PetscOptionsInt("-svd_nsv","Number of singular values to compute","SVDSetDimensions",svd->ncv,&i,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-svd_ncv","Number of basis vectors","SVDSetDimensions",svd->ncv,&j,PETSC_NULL);CHKERRQ(ierr);
  ierr = SVDSetDimensions(svd,i,j);CHKERRQ(ierr);

  ierr = PetscOptionsTruthGroupBegin("-svd_largest","compute largest singular values","SVDSetWhichSingularTriplets",&flg);CHKERRQ(ierr);
  if (flg) { ierr = SVDSetWhichSingularTriplets(svd,SVD_LARGEST);CHKERRQ(ierr); }
  ierr = PetscOptionsTruthGroupEnd("-svd_smallest","compute smallest singular values","SVDSetWhichSingularTriplets",&flg);CHKERRQ(ierr);
  if (flg) { ierr = SVDSetWhichSingularTriplets(svd,SVD_SMALLEST);CHKERRQ(ierr); }

  ierr = PetscOptionsName("-svd_monitor_cancel","Remove any hardwired monitor routines","SVDMonitorCancel",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SVDMonitorCancel(svd);CHKERRQ(ierr); 
  }

  ierr = PetscOptionsString("-svd_monitor","Monitor approximate singular values and error estimates","SVDMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr); 
  if (flg) {
    ierr = PetscViewerASCIIOpen(((PetscObject)svd)->comm,monfilename,&monviewer);CHKERRQ(ierr);
    ierr = SVDMonitorSet(svd,SVDMonitorDefault,monviewer,(PetscErrorCode (*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-svd_monitor_draw","Monitor error estimates graphically","SVDMonitorSet",&flg);CHKERRQ(ierr); 
  if (flg) {
    ierr = SVDMonitorSet(svd,SVDMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = IPSetFromOptions(svd->ip);CHKERRQ(ierr);
  if (svd->ops->setfromoptions) {
    ierr = (*svd->ops->setfromoptions)(svd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetOptionsPrefix"
/*@C
   SVDSetOptionsPrefix - Sets the prefix used for searching for all 
   SVD options in the database.

   Collective on SVD

   Input Parameters:
+  svd - the singular value solver context
-  prefix - the prefix string to prepend to all SVD option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different SVD contexts, one could call
.vb
      SVDSetOptionsPrefix(svd1,"svd1_")
      SVDSetOptionsPrefix(svd2,"svd2_")
.ve

   Level: advanced

.seealso: SVDAppendOptionsPrefix(), SVDGetOptionsPrefix()
@*/
PetscErrorCode SVDSetOptionsPrefix(SVD svd,const char *prefix)
{
  PetscErrorCode ierr;
  PetscTruth     flg1,flg2;
  EPS            eps;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)svd,prefix);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)svd,SVDCROSS,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)svd,SVDCYCLIC,&flg2);CHKERRQ(ierr);
  if (flg1) {
    ierr = SVDCrossGetEPS(svd,&eps);CHKERRQ(ierr);
  } else if (flg2) {
      ierr = SVDCyclicGetEPS(svd,&eps);CHKERRQ(ierr);
  }
  if (flg1 || flg2) {
    ierr = EPSSetOptionsPrefix(eps,prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(eps,"svd_");CHKERRQ(ierr);
  }
  ierr = IPSetOptionsPrefix(svd->ip,prefix);CHKERRQ(ierr);
  ierr = IPAppendOptionsPrefix(svd->ip,"svd_");CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "SVDAppendOptionsPrefix"
/*@C
   SVDAppendOptionsPrefix - Appends to the prefix used for searching for all 
   SVD options in the database.

   Collective on SVD

   Input Parameters:
+  svd - the singular value solver context
-  prefix - the prefix string to prepend to all SVD option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: SVDSetOptionsPrefix(), SVDGetOptionsPrefix()
@*/
PetscErrorCode SVDAppendOptionsPrefix(SVD svd,const char *prefix)
{
  PetscErrorCode ierr;
  PetscTruth     flg1,flg2;
  EPS            eps;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)svd,prefix);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)svd,SVDCROSS,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)svd,SVDCYCLIC,&flg2);CHKERRQ(ierr);
  if (flg1) {
    ierr = SVDCrossGetEPS(svd,&eps);CHKERRQ(ierr);
  } else if (flg2) {
    ierr = SVDCyclicGetEPS(svd,&eps);CHKERRQ(ierr);
  }
  if (flg1 || flg2) {
    ierr = EPSSetOptionsPrefix(eps,((PetscObject)svd)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(eps,"svd_");CHKERRQ(ierr);
  }
  ierr = IPSetOptionsPrefix(svd->ip,((PetscObject)svd)->prefix);CHKERRQ(ierr);
  ierr = IPAppendOptionsPrefix(svd->ip,"svd_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetOptionsPrefix"
/*@C
   SVDGetOptionsPrefix - Gets the prefix used for searching for all 
   SVD options in the database.

   Not Collective

   Input Parameters:
.  svd - the singular value solver context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: SVDSetOptionsPrefix(), SVDAppendOptionsPrefix()
@*/
PetscErrorCode SVDGetOptionsPrefix(SVD svd,const char *prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)svd,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

