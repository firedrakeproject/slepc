/*
      MFN routines related to options that can be set via the command-line 
      or procedurally.

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

#include <slepc-private/mfnimpl.h>   /*I "slepcmfn.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "MFNSetFromOptions"
/*@
   MFNSetFromOptions - Sets MFN options from the options database.
   This routine must be called before MFNSetUp() if the user is to be 
   allowed to set the solver type. 

   Collective on MFN

   Input Parameters:
.  mfn - the matrix function context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode MFNSetFromOptions(MFN mfn)
{
  PetscErrorCode   ierr;
  char             type[256],monfilename[PETSC_MAX_PATH_LEN];
  PetscBool        flg;
  PetscReal        r;
  PetscInt         i;
  PetscViewer      monviewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (!MFNRegisterAllCalled) { ierr = MFNRegisterAll(PETSC_NULL);CHKERRQ(ierr); }
  ierr = PetscObjectOptionsBegin((PetscObject)mfn);CHKERRQ(ierr);
    ierr = PetscOptionsList("-mfn_type","Matrix Function method","MFNSetType",MFNList,(char*)(((PetscObject)mfn)->type_name?((PetscObject)mfn)->type_name:MFNKRYLOV),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MFNSetType(mfn,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!((PetscObject)mfn)->type_name) {
      ierr = MFNSetType(mfn,MFNKRYLOV);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBoolGroupBegin("-mfn_exp","matrix exponential","MFNSetFunction",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MFNSetFunction(mfn,SLEPC_FUNCTION_EXP);CHKERRQ(ierr);
    }

    r = i = PETSC_IGNORE;
    ierr = PetscOptionsInt("-mfn_max_it","Maximum number of iterations","MFNSetTolerances",mfn->max_it,&i,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mfn_tol","Tolerance","MFNSetTolerances",mfn->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:mfn->tol,&r,PETSC_NULL);CHKERRQ(ierr);
    ierr = MFNSetTolerances(mfn,r,i);CHKERRQ(ierr);

    i = PETSC_IGNORE;
    ierr = PetscOptionsInt("-mfn_ncv","Number of basis vectors","MFNSetDimensions",mfn->ncv,&i,PETSC_NULL);CHKERRQ(ierr);
    ierr = MFNSetDimensions(mfn,i);CHKERRQ(ierr);
    
    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to MFNSetFromOptions()
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-mfn_monitor_cancel","Remove any hardwired monitor routines","MFNMonitorCancel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = MFNMonitorCancel(mfn);CHKERRQ(ierr);
    }
    /*
      Prints error estimate at each iteration
    */
    ierr = PetscOptionsString("-mfn_monitor","Monitor error estimate","MFNMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)mfn)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = MFNMonitorSet(mfn,MFNMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-mfn_monitor_draw","Monitor error estimate graphically","MFNMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr); 
    if (flg) {
      ierr = MFNMonitorSet(mfn,MFNMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/

    ierr = PetscOptionsName("-mfn_view","Print detailed information on solver used","MFNView",0);CHKERRQ(ierr);
   
    if (mfn->ops->setfromoptions) {
      ierr = (*mfn->ops->setfromoptions)(mfn);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)mfn);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!mfn->ip) { ierr = MFNGetIP(mfn,&mfn->ip);CHKERRQ(ierr); }
  ierr = IPSetFromOptions(mfn->ip);CHKERRQ(ierr);
  if (!mfn->ds) { ierr = MFNGetDS(mfn,&mfn->ds);CHKERRQ(ierr); }
  ierr = DSSetFromOptions(mfn->ds);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(mfn->rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNGetTolerances"
/*@
   MFNGetTolerances - Gets the tolerance and maximum iteration count used
   by the MFN convergence tests. 

   Not Collective

   Input Parameter:
.  mfn - the matrix function context
  
   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: MFNSetTolerances()
@*/
PetscErrorCode MFNGetTolerances(MFN mfn,PetscReal *tol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (tol)    *tol    = mfn->tol;
  if (maxits) *maxits = mfn->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNSetTolerances"
/*@
   MFNSetTolerances - Sets the tolerance and maximum iteration count used
   by the MFN convergence tests. 

   Logically Collective on MFN

   Input Parameters:
+  mfn - the matrix function context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -mfn_tol <tol> - Sets the convergence tolerance
-  -mfn_max_it <maxits> - Sets the maximum number of iterations allowed

   Notes:
   Use PETSC_IGNORE for an argument that need not be changed.

   Use PETSC_DECIDE for maxits to assign a reasonably good value, which is 
   dependent on the solution method.

   Level: intermediate

.seealso: MFNGetTolerances()
@*/
PetscErrorCode MFNSetTolerances(MFN mfn,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidLogicalCollectiveReal(mfn,tol,2);
  PetscValidLogicalCollectiveInt(mfn,maxits,3);
  if (tol != PETSC_IGNORE) {
    if (tol == PETSC_DEFAULT) {
      mfn->tol = PETSC_DEFAULT;
    } else {
      if (tol < 0.0) SETERRQ(((PetscObject)mfn)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
      mfn->tol = tol;
    }
  }
  if (maxits != PETSC_IGNORE) {
    if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
      mfn->max_it = 0;
      mfn->setupcalled = 0;
    } else {
      if (maxits < 0) SETERRQ(((PetscObject)mfn)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
      mfn->max_it = maxits;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNGetDimensions"
/*@
   MFNGetDimensions - Gets the dimension of the subspace used by the solver.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context
  
   Output Parameter:
.  ncv - the maximum dimension of the subspace to be used by the solver

   Level: intermediate

.seealso: MFNSetDimensions()
@*/
PetscErrorCode MFNGetDimensions(MFN mfn,PetscInt *ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(ncv,2);
  *ncv = mfn->ncv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNSetDimensions"
/*@
   MFNSetDimensions - Sets the dimension of the subspace to be used by the solver.

   Logically Collective on MFN

   Input Parameters:
+  mfn - the matrix function context
-  ncv - the maximum dimension of the subspace to be used by the solver

   Options Database Keys:
.  -mfn_ncv <ncv> - Sets the dimension of the subspace

   Notes:
   Use PETSC_DECIDE for ncv to assign a reasonably good value, which is
   dependent on the solution method.

   Level: intermediate

.seealso: MFNGetDimensions()
@*/
PetscErrorCode MFNSetDimensions(MFN mfn,PetscInt ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidLogicalCollectiveInt(mfn,ncv,2);
  if (ncv != PETSC_IGNORE) {
    if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
      mfn->ncv = 0;
    } else {
      if (ncv<1) SETERRQ(((PetscObject)mfn)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
      mfn->ncv = ncv;
    }
    mfn->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNSetFunction"
/*@
   MFNSetFunction - Specifies the function to be computed.

   Logically Collective on MFN

   Input Parameters:
+  mfn      - the matrix function context
-  fun      - a known function

   Options Database Keys:
.  -mfn_exp - matrix exponential
    
   Level: beginner

.seealso: MFNSetOperator(), MFNSetType(), MFNGetFunction(), SlepcFunction
@*/
PetscErrorCode MFNSetFunction(MFN mfn,SlepcFunction fun)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mfn,fun,2);
  switch (fun) {
    case SLEPC_FUNCTION_EXP:
      break;      
    default:
      SETERRQ(((PetscObject)mfn)->comm,PETSC_ERR_ARG_WRONG,"Unknown function");
  }
  mfn->function = fun;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNGetFunction"
/*@C
   MFNGetFunction - Gets the function from the MFN object.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context 

   Output Parameter:
.  fun - function

   Level: intermediate

.seealso: MFNSetFunction(), SlepcFunction
@*/
PetscErrorCode MFNGetFunction(MFN mfn,SlepcFunction *fun)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(fun,2);
  *fun = mfn->function;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNSetOptionsPrefix"
/*@C
   MFNSetOptionsPrefix - Sets the prefix used for searching for all 
   MFN options in the database.

   Logically Collective on MFN

   Input Parameters:
+  mfn - the matrix function context
-  prefix - the prefix string to prepend to all MFN option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different MFN contexts, one could call
.vb
      MFNSetOptionsPrefix(mfn1,"fun1_")
      MFNSetOptionsPrefix(mfn2,"fun2_")
.ve

   Level: advanced

.seealso: MFNAppendOptionsPrefix(), MFNGetOptionsPrefix()
@*/
PetscErrorCode MFNSetOptionsPrefix(MFN mfn,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (!mfn->ip) { ierr = MFNGetIP(mfn,&mfn->ip);CHKERRQ(ierr); }
  ierr = IPSetOptionsPrefix(mfn->ip,prefix);CHKERRQ(ierr);
  if (!mfn->ds) { ierr = MFNGetDS(mfn,&mfn->ds);CHKERRQ(ierr); }
  ierr = DSSetOptionsPrefix(mfn->ds,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)mfn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}
 
#undef __FUNCT__  
#define __FUNCT__ "MFNAppendOptionsPrefix"
/*@C
   MFNAppendOptionsPrefix - Appends to the prefix used for searching for all 
   MFN options in the database.

   Logically Collective on MFN

   Input Parameters:
+  mfn - the matrix function context
-  prefix - the prefix string to prepend to all MFN option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: MFNSetOptionsPrefix(), MFNGetOptionsPrefix()
@*/
PetscErrorCode MFNAppendOptionsPrefix(MFN mfn,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (!mfn->ip) { ierr = MFNGetIP(mfn,&mfn->ip);CHKERRQ(ierr); }
  ierr = IPSetOptionsPrefix(mfn->ip,prefix);CHKERRQ(ierr);
  if (!mfn->ds) { ierr = MFNGetDS(mfn,&mfn->ds);CHKERRQ(ierr); }
  ierr = DSSetOptionsPrefix(mfn->ds,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)mfn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MFNGetOptionsPrefix"
/*@C
   MFNGetOptionsPrefix - Gets the prefix used for searching for all 
   MFN options in the database.

   Not Collective

   Input Parameters:
.  mfn - the matrix function context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: MFNSetOptionsPrefix(), MFNAppendOptionsPrefix()
@*/
PetscErrorCode MFNGetOptionsPrefix(MFN mfn,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)mfn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
