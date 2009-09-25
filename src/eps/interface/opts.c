/*
      EPS routines related to options that can be set via the command-line 
      or procedurally.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/epsimpl.h"   /*I "slepceps.h" I*/

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
PetscErrorCode EPSSetFromOptions(EPS eps)
{
  PetscErrorCode ierr;
  char           type[256],monfilename[PETSC_MAX_PATH_LEN];
  PetscTruth     flg;
  PetscReal      r;
  PetscScalar    s;
  PetscInt       i,j,k;
  const char     *bal_list[3] = { "none", "oneside", "twoside" };
  PetscViewerASCIIMonitor monviewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscOptionsBegin(((PetscObject)eps)->comm,((PetscObject)eps)->prefix,"Eigenproblem Solver (EPS) Options","EPS");CHKERRQ(ierr);
    ierr = PetscOptionsList("-eps_type","Eigenproblem Solver method","EPSSetType",EPSList,(char*)(((PetscObject)eps)->type_name?((PetscObject)eps)->type_name:EPSKRYLOVSCHUR),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = EPSSetType(eps,type);CHKERRQ(ierr);
    }

    ierr = PetscOptionsTruthGroupBegin("-eps_hermitian","hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_gen_hermitian","generalized hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_non_hermitian","non-hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_gen_non_hermitian","generalized non-hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-eps_pos_gen_non_hermitian","generalized non-hermitian eigenvalue problem with positive semi-definite B","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_PGNHEP);CHKERRQ(ierr);}

    /*
      Set the type if it was never set.
    */
    if (!((PetscObject)eps)->type_name) {
      ierr = EPSSetType(eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
    }

    ierr = PetscOptionsTruthGroupBegin("-eps_ritz","Rayleigh-Ritz extraction","EPSSetExtraction",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_harmonic","harmonic Ritz extraction","EPSSetExtraction",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetExtraction(eps,EPS_HARMONIC);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_refined","refined Ritz extraction","EPSSetExtraction",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetExtraction(eps,EPS_REFINED);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-eps_refined_harmonic","refined harmonic Ritz extraction","EPSSetExtraction",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetExtraction(eps,EPS_REFINED_HARMONIC);CHKERRQ(ierr);}

    if (!eps->balance) eps->balance = EPSBALANCE_NONE;
    ierr = PetscOptionsEList("-eps_balance", "Balancing method","EPSSetBalance",bal_list,3,bal_list[eps->balance-EPSBALANCE_NONE],&i,&flg);CHKERRQ(ierr);
    if (flg) { eps->balance = (EPSBalance)(i+EPSBALANCE_NONE); }
    r = j = PETSC_IGNORE;
    ierr = PetscOptionsInt("-eps_balance_its","Number of iterations in balancing","EPSSetBalance",eps->balance_its,&j,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps_balance_cutoff","Cutoff value in balancing","EPSSetBalance",eps->balance_cutoff,&r,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSSetBalance(eps,PETSC_IGNORE,j,r);CHKERRQ(ierr);

    ierr = PetscOptionsTruthGroupBegin("-eps_oneside","one-sided eigensolver","EPSSetClass",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetClass(eps,EPS_ONE_SIDE);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-eps_twoside","two-sided eigensolver","EPSSetClass",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetClass(eps,EPS_TWO_SIDE);CHKERRQ(ierr);}

    r = i = PETSC_IGNORE;
    ierr = PetscOptionsInt("-eps_max_it","Maximum number of iterations","EPSSetTolerances",eps->max_it,&i,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps_tol","Tolerance","EPSSetTolerances",eps->tol,&r,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSSetTolerances(eps,r,i);CHKERRQ(ierr);
    ierr = PetscOptionsTruthGroupBegin("-eps_convergence_default","Default (relative error) convergence test","EPSSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetConvergenceTest(eps,EPSDefaultConverged,PETSC_NULL);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-eps_convergence_absolute","Absolute error convergence test","EPSSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetConvergenceTest(eps,EPSAbsoluteConverged,PETSC_NULL);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-eps_convergence_residual","Residual convergence test","EPSSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetConvergenceTest(eps,EPSResidualConverged,PETSC_NULL);CHKERRQ(ierr);}

    i = j = k = PETSC_IGNORE;
    ierr = PetscOptionsInt("-eps_nev","Number of eigenvalues to compute","EPSSetDimensions",eps->nev,&i,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_ncv","Number of basis vectors","EPSSetDimensions",eps->ncv,&j,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_mpd","Maximum dimension of projected problem","EPSSetDimensions",eps->mpd,&k,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSSetDimensions(eps,i,j,k);CHKERRQ(ierr);
    
    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to EPSSetFromOptions()
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-eps_monitor_cancel","Remove any hardwired monitor routines","EPSMonitorCancel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = EPSMonitorCancel(eps); CHKERRQ(ierr);
    }
    /*
      Prints approximate eigenvalues and error estimates at each iteration
    */
    ierr = PetscOptionsString("-eps_monitor","Monitor approximate eigenvalues and error estimates","EPSMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)eps)->comm,monfilename,((PetscObject)eps)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = EPSMonitorSet(eps,EPSMonitorDefault,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-eps_monitor_conv","Monitor approximate eigenvalues and error estimates as they converge","EPSMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)eps)->comm,monfilename,((PetscObject)eps)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = EPSMonitorSet(eps,EPSMonitorConverged,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-eps_monitor_first","Monitor first unconverged approximate eigenvalue and error estimate","EPSMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)eps)->comm,monfilename,((PetscObject)eps)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = EPSMonitorSet(eps,EPSMonitorFirst,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsTruth("-eps_monitor_draw","Monitor error estimates graphically","EPSMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr); 
    if (flg) {
      ierr = EPSMonitorSet(eps,EPSMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/
    ierr = PetscOptionsScalar("-eps_target","Value of the target","EPSSetTarget",eps->target,&s,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
      ierr = EPSSetTarget(eps,s);CHKERRQ(ierr);
    }

    ierr = PetscOptionsTruthGroupBegin("-eps_largest_magnitude","compute largest eigenvalues in magnitude","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_smallest_magnitude","compute smallest eigenvalues in magnitude","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_largest_real","compute largest real parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_smallest_real","compute smallest real parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_largest_imaginary","compute largest imaginary parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_IMAGINARY);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_smallest_imaginary","compute smallest imaginary parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_IMAGINARY);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_target_magnitude","compute nearest eigenvalues to target","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-eps_target_real","compute eigenvalues with real parts close to target","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-eps_target_imaginary","compute eigenvalues with imaginary parts close to target","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_IMAGINARY);CHKERRQ(ierr);}

    ierr = PetscOptionsName("-eps_view","Print detailed information on solver used","EPSView",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-eps_view_binary","Save the matrices associated to the eigenproblem","EPSSetFromOptions",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-eps_plot_eigs","Make a plot of the computed eigenvalues","EPSSolve",0);CHKERRQ(ierr);
   
    if (eps->ops->setfromoptions) {
      ierr = (*eps->ops->setfromoptions)(eps);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = IPSetFromOptions(eps->ip); CHKERRQ(ierr);
  ierr = STSetFromOptions(eps->OP); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetTolerances"
/*@
   EPSGetTolerances - Gets the tolerance and maximum
   iteration count used by the default EPS convergence tests. 

   Not Collective

   Input Parameter:
.  eps - the eigensolver context
  
   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: EPSSetTolerances()
@*/
PetscErrorCode EPSGetTolerances(EPS eps,PetscReal *tol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (tol)    *tol    = eps->tol;
  if (maxits) *maxits = eps->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetTolerances"
/*@
   EPSSetTolerances - Sets the tolerance and maximum
   iteration count used by the default EPS convergence testers. 

   Collective on EPS

   Input Parameters:
+  eps - the eigensolver context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -eps_tol <tol> - Sets the convergence tolerance
-  -eps_max_it <maxits> - Sets the maximum number of iterations allowed

   Notes:
   Use PETSC_IGNORE for an argument that need not be changed.

   Use PETSC_DECIDE for maxits to assign a reasonably good value, which is 
   dependent on the solution method.

   Level: intermediate

.seealso: EPSGetTolerances()
@*/
PetscErrorCode EPSSetTolerances(EPS eps,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (tol != PETSC_IGNORE) {
    if (tol == PETSC_DEFAULT) {
      eps->tol = 1e-7;
    } else {
      if (tol < 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
      eps->tol = tol;
    }
  }
  if (maxits != PETSC_IGNORE) {
    if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
      eps->max_it = 0;
    } else {
      if (maxits < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
      eps->max_it = maxits;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetDimensions"
/*@
   EPSGetDimensions - Gets the number of eigenvalues to compute
   and the dimension of the subspace.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context
  
   Output Parameters:
+  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: EPSSetDimensions()
@*/
PetscErrorCode EPSGetDimensions(EPS eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (nev) *nev = eps->nev;
  if (ncv) *ncv = eps->ncv;
  if (mpd) *mpd = eps->mpd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetDimensions"
/*@
   EPSSetDimensions - Sets the number of eigenvalues to compute
   and the dimension of the subspace.

   Collective on EPS

   Input Parameters:
+  eps - the eigensolver context
.  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Options Database Keys:
+  -eps_nev <nev> - Sets the number of eigenvalues
.  -eps_ncv <ncv> - Sets the dimension of the subspace
-  -eps_mpd <mpd> - Sets the maximum projected dimension

   Notes:
   Use PETSC_IGNORE to retain the previous value of any parameter.

   Use PETSC_DECIDE for ncv and mpd to assign a reasonably good value, which is
   dependent on the solution method.

   The parameters ncv and mpd are intimately related, so that the user is advised
   to set one of them at most. Normal usage is the following
+  - In cases where nev is small, the user sets ncv (a reasonable default is 2*nev).
-  - In cases where nev is large, the user sets mpd.

   The value of ncv should always be between nev and (nev+mpd), typically
   ncv=nev+mpd. If nev is not too large, mpd=nev is a reasonable choice, otherwise
   a smaller value should be used.

   Level: intermediate

.seealso: EPSGetDimensions()
@*/
PetscErrorCode EPSSetDimensions(EPS eps,PetscInt nev,PetscInt ncv,PetscInt mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if( nev != PETSC_IGNORE ) {
    if (nev<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nev. Must be > 0");
    eps->nev = nev;
    eps->setupcalled = 0;
  }
  if( ncv != PETSC_IGNORE ) {
    if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
      eps->ncv = 0;
    } else {
      if (ncv<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
      eps->ncv = ncv;
    }
    eps->setupcalled = 0;
  }
  if( mpd != PETSC_IGNORE ) {
    if (mpd == PETSC_DECIDE || mpd == PETSC_DEFAULT) {
      eps->mpd = 0;
    } else {
      if (mpd<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of mpd. Must be > 0");
      eps->mpd = mpd;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetWhichEigenpairs"
/*@
    EPSSetWhichEigenpairs - Specifies which portion of the spectrum is 
    to be sought.

    Collective on EPS

    Input Parameters:
+   eps - eigensolver context obtained from EPSCreate()
-   which - the portion of the spectrum to be sought

    Possible values:
    The parameter 'which' can have one of these values
    
+     EPS_LARGEST_MAGNITUDE - largest eigenvalues in magnitude (default)
.     EPS_SMALLEST_MAGNITUDE - smallest eigenvalues in magnitude
.     EPS_LARGEST_REAL - largest real parts
.     EPS_SMALLEST_REAL - smallest real parts
.     EPS_LARGEST_IMAGINARY - largest imaginary parts
.     EPS_SMALLEST_IMAGINARY - smallest imaginary parts
.     EPS_TARGET_MAGNITUDE - nearest eigenvalues to the target set by EPSSetTarget
.     EPS_TARGET_REAL - eigenvalues with real part near to target
.     EPS_TARGET_IMAGINARY - eigenvalues with imaginary part near to target
-     EPS_USER - user defined order set by EPSSetEigenvalueComparison

    Options Database Keys:
+   -eps_largest_magnitude - Sets largest eigenvalues in magnitude
.   -eps_smallest_magnitude - Sets smallest eigenvalues in magnitude
.   -eps_largest_real - Sets largest real parts
.   -eps_smallest_real - Sets smallest real parts
.   -eps_largest_imaginary - Sets largest imaginary parts in magnitude
.   -eps_smallest_imaginary - Sets smallest imaginary parts in magnitude
.   -eps_target_magnitude - Sets nearest eigenvalues to target
.   -eps_target_real - Sets real parts nearest to target
-   -eps_target_imaginary - Sets imaginary parts nearest to target

    Notes:
    Not all eigensolvers implemented in EPS account for all the possible values
    stated above. Also, some values make sense only for certain types of 
    problems. If SLEPc is compiled for real numbers EPS_LARGEST_IMAGINARY
    and EPS_SMALLEST_IMAGINARY use the absolute value of the imaginary part 
    for eigenvalue selection.     
    
    Level: intermediate

.seealso: EPSGetWhichEigenpairs(), EPSSetTarget(), EPSSetEigenvalueComparison, EPSSortEigenvalues(), EPSWhich
@*/
PetscErrorCode EPSSetWhichEigenpairs(EPS eps,EPSWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  switch (which) {
    case EPS_LARGEST_MAGNITUDE:
    case EPS_SMALLEST_MAGNITUDE:
    case EPS_LARGEST_REAL:
    case EPS_SMALLEST_REAL:
    case EPS_LARGEST_IMAGINARY:
    case EPS_SMALLEST_IMAGINARY:
    case EPS_TARGET_MAGNITUDE:
    case EPS_TARGET_REAL:
    case EPS_TARGET_IMAGINARY:
    case EPS_USER:
      if (eps->which != which) {
        eps->setupcalled = 0;
        eps->which = which;
      }
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'which' value"); 
  }  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetWhichEigenpairs"
/*@C
    EPSGetWhichEigenpairs - Returns which portion of the spectrum is to be 
    sought.

    Not Collective

    Input Parameter:
.   eps - eigensolver context obtained from EPSCreate()

    Output Parameter:
.   which - the portion of the spectrum to be sought

    Notes:
    See EPSSetWhichEigenpairs() for possible values of which

    Level: intermediate

.seealso: EPSSetWhichEigenpairs(), EPSWhich
@*/
PetscErrorCode EPSGetWhichEigenpairs(EPS eps,EPSWhich *which) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(which,2);
  *which = eps->which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetEigenvalueComparison"
/*@C
    EPSSetCompareEigenvalues - Specifies the eigenvalue comparison function
    when EPSSetWhichEigenpairs is set to EPS_USER.
    Collective on EPS

    Input Parameters:
+   eps  - eigensolver context obtained from EPSCreate()
.   func - a pointer to the comparison function
-   ctx  - a context pointer (the last parameter to the comparison function)

    Notes:
     The comparison function must return an integer less than, equal to, or
     greater than zero if the first eigenvalue is considered to be respectively
     less than, equal to, or greater than the second.
    
    Level: advanced

.seealso: EPSSetWhichEigenpairs(), EPSSortEigenvalues(), EPSWhich
@*/
PetscErrorCode EPSSetEigenvalueComparison(EPS eps,PetscErrorCode (*func)(EPS,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void* ctx)
{
  PetscFunctionBegin;
  eps->which_func = func;
  eps->which_ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetConvergenceTest"
/*@C
    EPSSetConvergenceTest - Specifies the convergence test.
    Collective on EPS

    Input Parameters:
+   eps  - eigensolver context obtained from EPSCreate()
.   func - a pointer to the convergence test function
-   ctx  - a context pointer (the last parameter to the convergence test function)

    Notes:
    The convergence function sets an element of the flag array for each eigenvalue.
    
    Level: advanced

.seealso: EPSSetTolerances()
@*/
EXTERN PetscErrorCode EPSSetConvergenceTest(EPS eps,PetscErrorCode (*func)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*),void* ctx)
{
  PetscFunctionBegin;
  eps->conv_func = func;
  eps->conv_ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetProblemType"
/*@
   EPSSetProblemType - Specifies the type of the eigenvalue problem.

   Collective on EPS

   Input Parameters:
+  eps      - the eigensolver context
-  type     - a known type of eigenvalue problem 

   Options Database Keys:
+  -eps_hermitian - Hermitian eigenvalue problem
.  -eps_gen_hermitian - generalized Hermitian eigenvalue problem
.  -eps_non_hermitian - non-Hermitian eigenvalue problem
.  -eps_gen_non_hermitian - generalized non-Hermitian eigenvalue problem 
-  -eps_pos_gen_non_hermitian - generalized non-Hermitian eigenvalue problem 
   with positive semi-definite B
    
   Notes:  
   Allowed values for the problem type are: Hermitian (EPS_HEP), non-Hermitian
   (EPS_NHEP), generalized Hermitian (EPS_GHEP), generalized non-Hermitian 
   (EPS_GNHEP), and generalized non-Hermitian with positive semi-definite B
   (EPS_PGNHEP).

   This function must be used to instruct SLEPc to exploit symmetry. If no
   problem type is specified, by default a non-Hermitian problem is assumed
   (either standard or generalized). If the user knows that the problem is
   Hermitian (i.e. A=A^H) or generalized Hermitian (i.e. A=A^H, B=B^H, and 
   B positive definite) then it is recommended to set the problem type so
   that eigensolver can exploit these properties. 

   Level: beginner

.seealso: EPSSetOperators(), EPSSetType(), EPSGetProblemType(), EPSProblemType
@*/
PetscErrorCode EPSSetProblemType(EPS eps,EPSProblemType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  switch (type) {
    case EPS_HEP:
      eps->isgeneralized = PETSC_FALSE;
      eps->ishermitian = PETSC_TRUE;
      eps->ispositive = PETSC_FALSE;
      break;      
    case EPS_NHEP:
      eps->isgeneralized = PETSC_FALSE;
      eps->ishermitian = PETSC_FALSE;
      eps->ispositive = PETSC_FALSE;
      break;
    case EPS_GHEP:
      eps->isgeneralized = PETSC_TRUE;
      eps->ishermitian = PETSC_TRUE;
      eps->ispositive = PETSC_TRUE;
      break;
    case EPS_GNHEP:
      eps->isgeneralized = PETSC_TRUE;
      eps->ishermitian = PETSC_FALSE;
      eps->ispositive = PETSC_FALSE;
      break;
    case EPS_PGNHEP:
      eps->isgeneralized = PETSC_TRUE;
      eps->ishermitian = PETSC_FALSE;
      eps->ispositive = PETSC_TRUE;
      break;
/*
    case EPS_CSEP: 
      eps->isgeneralized = PETSC_FALSE;
      eps->ishermitian = PETSC_FALSE;
      ierr = STSetBilinearForm(eps->OP,STINNER_SYMMETRIC);CHKERRQ(ierr);
      break;
    case EPS_GCSEP:
      eps->isgeneralized = PETSC_TRUE;
      eps->ishermitian = PETSC_FALSE;
      ierr = STSetBilinearForm(eps->OP,STINNER_B_SYMMETRIC);CHKERRQ(ierr);
      break;
*/
    default:
      SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown eigenvalue problem type");
  }
  eps->problem_type = type;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetProblemType"
/*@C
   EPSGetProblemType - Gets the problem type from the EPS object.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context 

   Output Parameter:
.  type - name of EPS problem type 

   Level: intermediate

.seealso: EPSSetProblemType(), EPSProblemType
@*/
PetscErrorCode EPSGetProblemType(EPS eps,EPSProblemType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(type,2);
  *type = eps->problem_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetExtraction"
/*@
   EPSSetExtraction - Specifies the type of extraction technique to be employed 
   by the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps  - the eigensolver context
-  extr - a known type of extraction

   Options Database Keys:
+  -eps_ritz - Rayleigh-Ritz extraction
.  -eps_harmonic - hamonic Ritz extraction
.  -eps_refined - refined Ritz extraction
-  -eps_refined_harmonic - refined harmonic Ritz extraction
    
   Notes:  
   Not all eigensolvers support all types of extraction. See the SLEPc
   Users Manual for details.

   By default, a standard Rayleigh-Ritz extraction is used. Other extractions
   may be useful when computing interior eigenvalues.

   Harmonic-type extractions are used in combination with a 'target'.

   Level: beginner

.seealso: EPSSetTarget(), EPSGetExtraction(), EPSExtraction
@*/
PetscErrorCode EPSSetExtraction(EPS eps,EPSExtraction extr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  eps->extraction = extr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetExtraction"
/*@C
   EPSGetExtraction - Gets the extraction type used by the EPS object.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context 

   Output Parameter:
.  extr - name of extraction type 

   Level: intermediate

.seealso: EPSSetExtraction(), EPSExtraction
@*/
PetscErrorCode EPSGetExtraction(EPS eps,EPSExtraction *extr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(extr,2);
  *extr = eps->extraction;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetBalance"
/*@
   EPSSetBalance - Specifies the balancing technique to be employed by the
   eigensolver, and some parameters associated to it.

   Collective on EPS

   Input Parameters:
+  eps    - the eigensolver context
.  bal    - the balancing method, one of EPSBALANCE_NONE, EPSBALANCE_ONESIDE,
            or EPSBALANCE_TWOSIDE
.  its    - number of iterations of the balancing algorithm
-  cutoff - cutoff value

   Options Database Keys:
+  -eps_balance <method> - the balancing method, where <method> is one of
                           'none', 'oneside', or 'twoside'
.  -eps_balance_its <its> - number of iterations
-  -eps_balance_cutoff <cutoff> - cutoff value
    
   Notes:
   When balancing is enabled, the solver works implicitly with matrix DAD^-1,
   where D is an appropriate diagonal matrix. This improves the accuracy of
   the computed results in some cases. See the SLEPc Users Manual for details.

   Balancing makes sense only for non-Hermitian problems when the required
   precision is high (i.e. a small tolerance such as 1e-15).

   By default, balancing is disabled. The two-sided method is much more
   effective than the one-sided counterpart, but it requires the system
   matrices to have the MatMultTranspose operation defined.

   The parameter 'its' is the number of iterations performed by the method. The
   cutoff value is used only in the two-side variant. Use PETSC_IGNORE for an
   argument that need not be changed. Use PETSC_DECIDE to assign a reasonably
   good value.

   Level: intermediate

.seealso: EPSGetBalance(), EPSBalance
@*/
PetscErrorCode EPSSetBalance(EPS eps,EPSBalance bal,PetscInt its,PetscReal cutoff)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (bal!=PETSC_IGNORE) {
    if (bal==PETSC_DECIDE || bal==PETSC_DEFAULT) eps->balance = EPSBALANCE_TWOSIDE;
    eps->balance = bal;
  }
  if (its!=PETSC_IGNORE) {
    if (its==PETSC_DECIDE || its==PETSC_DEFAULT) eps->balance_its = 5;
    eps->balance_its = its;
  }
  if (cutoff!=PETSC_IGNORE) {
    if (cutoff==PETSC_DECIDE || cutoff==PETSC_DEFAULT) eps->balance_cutoff = 1e-8;
    eps->balance_cutoff = cutoff;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetBalance"
/*@C
   EPSGetBalance - Gets the balancing type used by the EPS object, and the associated
   parameters.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context 

   Output Parameters:
+  bal    - the balancing method
.  its    - number of iterations of the balancing algorithm
-  cutoff - cutoff value

   Level: intermediate

   Note:
   The user can specify PETSC_NULL for any parameter that is not needed.

.seealso: EPSSetBalance(), EPSBalance
@*/
PetscErrorCode EPSGetBalance(EPS eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(bal,2);
  PetscValidPointer(its,3);
  PetscValidPointer(cutoff,4);
  if (bal)    *bal = eps->balance;
  if (its)    *its = eps->balance_its;
  if (cutoff) *cutoff = eps->balance_cutoff;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetClass"
/*@
   EPSSetClass - Specifies the eigensolver class: either one-sided or two-sided.

   Collective on EPS

   Input Parameters:
+  eps      - the eigensolver context
-  class    - the class of solver

   Options Database Keys:
+  -eps_oneside - one-sided solver
-  -eps_twoside - two-sided solver
    
   Note:  
   Allowed solver classes are: one-sided (EPS_ONE_SIDE) and two-sided (EPS_TWO_SIDE).
   One-sided eigensolvers are the standard ones, which allow the computation of
   eigenvalues and (right) eigenvectors, whereas two-sided eigensolvers compute
   left eigenvectors as well.

   Level: intermediate

.seealso: EPSGetLeftVector(), EPSComputeRelativeErrorLeft(), EPSSetLeftInitialVector(),
   EPSGetClass(), EPSClass
@*/
PetscErrorCode EPSSetClass(EPS eps,EPSClass cl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if (cl != EPS_ONE_SIDE && cl != EPS_TWO_SIDE) SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown eigensolver class");
  if (eps->solverclass!=cl) {
    if (eps->solverclass == EPS_TWO_SIDE) { ierr = EPSFreeSolution(eps);CHKERRQ(ierr); }
    eps->solverclass = cl;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetClass"
/*@C
   EPSGetClass - Gets the eigensolver class from the EPS object.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context 

   Output Parameter:
.  class - class of EPS solver (either one-sided or two-sided)

   Level: intermediate

.seealso: EPSSetClass(), EPSClass
@*/
PetscErrorCode EPSGetClass(EPS eps,EPSClass *cl)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(cl,2);
  *cl = eps->solverclass;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetOptionsPrefix"
/*@C
   EPSSetOptionsPrefix - Sets the prefix used for searching for all 
   EPS options in the database.

   Collective on EPS

   Input Parameters:
+  eps - the eigensolver context
-  prefix - the prefix string to prepend to all EPS option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different EPS contexts, one could call
.vb
      EPSSetOptionsPrefix(eps1,"eig1_")
      EPSSetOptionsPrefix(eps2,"eig2_")
.ve

   Level: advanced

.seealso: EPSAppendOptionsPrefix(), EPSGetOptionsPrefix()
@*/
PetscErrorCode EPSSetOptionsPrefix(EPS eps,const char *prefix)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)eps, prefix);CHKERRQ(ierr);
  ierr = STSetOptionsPrefix(eps->OP,prefix);CHKERRQ(ierr);
  ierr = IPSetOptionsPrefix(eps->ip,prefix);CHKERRQ(ierr);
  ierr = IPAppendOptionsPrefix(eps->ip,"eps_");CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}
 
#undef __FUNCT__  
#define __FUNCT__ "EPSAppendOptionsPrefix"
/*@C
   EPSAppendOptionsPrefix - Appends to the prefix used for searching for all 
   EPS options in the database.

   Collective on EPS

   Input Parameters:
+  eps - the eigensolver context
-  prefix - the prefix string to prepend to all EPS option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: EPSSetOptionsPrefix(), EPSGetOptionsPrefix()
@*/
PetscErrorCode EPSAppendOptionsPrefix(EPS eps,const char *prefix)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)eps, prefix);CHKERRQ(ierr);
  ierr = STAppendOptionsPrefix(eps->OP,prefix); CHKERRQ(ierr);
  ierr = IPSetOptionsPrefix(eps->ip,prefix);CHKERRQ(ierr);
  ierr = IPAppendOptionsPrefix(eps->ip,"eps_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetOptionsPrefix"
/*@C
   EPSGetOptionsPrefix - Gets the prefix used for searching for all 
   EPS options in the database.

   Not Collective

   Input Parameters:
.  eps - the eigensolver context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: EPSSetOptionsPrefix(), EPSAppendOptionsPrefix()
@*/
PetscErrorCode EPSGetOptionsPrefix(EPS eps,const char *prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)eps, prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
