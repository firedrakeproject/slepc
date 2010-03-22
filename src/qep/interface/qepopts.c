/*
      QEP routines related to options that can be set via the command-line 
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

#include "private/qepimpl.h"   /*I "slepcqep.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "QEPSetFromOptions"
/*@
   QEPSetFromOptions - Sets QEP options from the options database.
   This routine must be called before QEPSetUp() if the user is to be 
   allowed to set the solver type. 

   Collective on QEP

   Input Parameters:
.  qep - the quadratic eigensolver context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode QEPSetFromOptions(QEP qep)
{
  PetscErrorCode ierr;
  char           type[256],monfilename[PETSC_MAX_PATH_LEN];
  PetscTruth     flg;
  PetscReal      r;
  PetscInt       i,j,k;
  PetscViewerASCIIMonitor monviewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  ierr = PetscOptionsBegin(((PetscObject)qep)->comm,((PetscObject)qep)->prefix,"Quadratic Eigenvalue Problem (QEP) Solver Options","QEP");CHKERRQ(ierr);
    ierr = PetscOptionsList("-qep_type","Quadratic Eigenvalue Problem method","QEPSetType",QEPList,(char*)(((PetscObject)qep)->type_name?((PetscObject)qep)->type_name:QEPLINEAR),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = QEPSetType(qep,type);CHKERRQ(ierr);
    } else if (!((PetscObject)qep)->type_name) {
      ierr = QEPSetType(qep,QEPLINEAR);CHKERRQ(ierr);
    }

    r = i = PETSC_IGNORE;
    ierr = PetscOptionsInt("-qep_max_it","Maximum number of iterations","QEPSetTolerances",qep->max_it,&i,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-qep_tol","Tolerance","QEPSetTolerances",qep->tol,&r,PETSC_NULL);CHKERRQ(ierr);
    ierr = QEPSetTolerances(qep,r,i);CHKERRQ(ierr);
    ierr = PetscOptionsTruthGroupBegin("-qep_convergence_default","Default (relative error) convergence test","QEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) {ierr = QEPSetConvergenceTest(qep,QEPDefaultConverged,PETSC_NULL);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-qep_convergence_absolute","Absolute error convergence test","QEPSetConvergenceTest",&flg);CHKERRQ(ierr);
    if (flg) {ierr = QEPSetConvergenceTest(qep,QEPAbsoluteConverged,PETSC_NULL);CHKERRQ(ierr);}

    i = j = k = PETSC_IGNORE;
    ierr = PetscOptionsInt("-qep_nev","Number of eigenvalues to compute","QEPSetDimensions",qep->nev,&i,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-qep_ncv","Number of basis vectors","QEPSetDimensions",qep->ncv,&j,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-qep_mpd","Maximum dimension of projected problem","QEPSetDimensions",qep->mpd,&k,PETSC_NULL);CHKERRQ(ierr);
    ierr = QEPSetDimensions(qep,i,j,k);CHKERRQ(ierr);
    
    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to QEPSetFromOptions()
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-qep_monitor_cancel","Remove any hardwired monitor routines","QEPMonitorCancel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = QEPMonitorCancel(qep); CHKERRQ(ierr);
    }
    /*
      Prints approximate eigenvalues and error estimates at each iteration
    */
    ierr = PetscOptionsString("-qep_monitor","Monitor approximate eigenvalues and error estimates","QEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)qep)->comm,monfilename,((PetscObject)qep)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = QEPMonitorSet(qep,QEPMonitorDefault,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-qep_monitor_conv","Monitor approximate eigenvalues and error estimates as they converge","QEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)qep)->comm,monfilename,((PetscObject)qep)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = QEPMonitorSet(qep,QEPMonitorConverged,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsString("-qep_monitor_first","Monitor first unconverged approximate eigenvalue and error estimate","QEPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)qep)->comm,monfilename,((PetscObject)qep)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = QEPMonitorSet(qep,QEPMonitorFirst,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    flg = PETSC_FALSE;
    ierr = PetscOptionsTruth("-qep_monitor_draw","Monitor error estimates graphically","QEPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr); 
    if (flg) {
      ierr = QEPMonitorSet(qep,QEPMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/

    ierr = PetscOptionsTruthGroupBegin("-qep_largest_magnitude","compute largest eigenvalues in magnitude","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = QEPSetWhichEigenpairs(qep,QEP_LARGEST_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-qep_smallest_magnitude","compute smallest eigenvalues in magnitude","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = QEPSetWhichEigenpairs(qep,QEP_SMALLEST_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-qep_largest_real","compute largest real parts","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = QEPSetWhichEigenpairs(qep,QEP_LARGEST_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-qep_smallest_real","compute smallest real parts","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = QEPSetWhichEigenpairs(qep,QEP_SMALLEST_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroup("-qep_largest_imaginary","compute largest imaginary parts","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = QEPSetWhichEigenpairs(qep,QEP_LARGEST_IMAGINARY);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-qep_smallest_imaginary","compute smallest imaginary parts","QEPSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = QEPSetWhichEigenpairs(qep,QEP_SMALLEST_IMAGINARY);CHKERRQ(ierr);}

    ierr = PetscOptionsName("-qep_view","Print detailed information on solver used","QEPView",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-qep_view_binary","Save the matrices associated to the eigenproblem","QEPSetFromOptions",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-qep_plot_eigs","Make a plot of the computed eigenvalues","QEPSolve",0);CHKERRQ(ierr);
   
    if (qep->ops->setfromoptions) {
      ierr = (*qep->ops->setfromoptions)(qep);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = IPSetFromOptions(qep->ip); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetTolerances"
/*@
   QEPGetTolerances - Gets the tolerance and maximum iteration count used
   by the QEP convergence tests. 

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context
  
   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: QEPSetTolerances()
@*/
PetscErrorCode QEPGetTolerances(QEP qep,PetscReal *tol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (tol)    *tol    = qep->tol;
  if (maxits) *maxits = qep->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPESetTolerances"
/*@
   QEPSetTolerances - Sets the tolerance and maximum iteration count used
   by the QEP convergence tests. 

   Collective on QEP

   Input Parameters:
+  qep - the quadratic eigensolver context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -qep_tol <tol> - Sets the convergence tolerance
-  -qep_max_it <maxits> - Sets the maximum number of iterations allowed

   Notes:
   Use PETSC_IGNORE for an argument that need not be changed.

   Use PETSC_DECIDE for maxits to assign a reasonably good value, which is 
   dependent on the solution method.

   Level: intermediate

.seealso: QEPGetTolerances()
@*/
PetscErrorCode QEPSetTolerances(QEP qep,PetscReal tol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (tol != PETSC_IGNORE) {
    if (tol == PETSC_DEFAULT) {
      qep->tol = 1e-7;
    } else {
      if (tol < 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
      qep->tol = tol;
    }
  }
  if (maxits != PETSC_IGNORE) {
    if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
      qep->max_it = 0;
      qep->setupcalled = 0;
    } else {
      if (maxits < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
      qep->max_it = maxits;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetDimensions"
/*@
   QEPGetDimensions - Gets the number of eigenvalues to compute
   and the dimension of the subspace.

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context
  
   Output Parameters:
+  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: QEPSetDimensions()
@*/
PetscErrorCode QEPGetDimensions(QEP qep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (nev) *nev = qep->nev;
  if (ncv) *ncv = qep->ncv;
  if (mpd) *mpd = qep->mpd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetDimensions"
/*@
   QEPSetDimensions - Sets the number of eigenvalues to compute
   and the dimension of the subspace.

   Collective on QEP

   Input Parameters:
+  qep - the quadratic eigensolver context
.  nev - number of eigenvalues to compute
.  ncv - the maximum dimension of the subspace to be used by the solver
-  mpd - the maximum dimension allowed for the projected problem

   Options Database Keys:
+  -qep_nev <nev> - Sets the number of eigenvalues
.  -qep_ncv <ncv> - Sets the dimension of the subspace
-  -qep_mpd <mpd> - Sets the maximum projected dimension

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

.seealso: QEPGetDimensions()
@*/
PetscErrorCode QEPSetDimensions(QEP qep,PetscInt nev,PetscInt ncv,PetscInt mpd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);

  if( nev != PETSC_IGNORE ) {
    if (nev<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of nev. Must be > 0");
    qep->nev = nev;
    qep->setupcalled = 0;
  }
  if( ncv != PETSC_IGNORE ) {
    if (ncv == PETSC_DECIDE || ncv == PETSC_DEFAULT) {
      qep->ncv = 0;
    } else {
      if (ncv<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of ncv. Must be > 0");
      qep->ncv = ncv;
    }
    qep->setupcalled = 0;
  }
  if( mpd != PETSC_IGNORE ) {
    if (mpd == PETSC_DECIDE || mpd == PETSC_DEFAULT) {
      qep->mpd = 0;
    } else {
      if (mpd<1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of mpd. Must be > 0");
      qep->mpd = mpd;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetWhichEigenpairs"
/*@
    QEPSetWhichEigenpairs - Specifies which portion of the spectrum is 
    to be sought.

    Collective on QEP

    Input Parameters:
+   qep - eigensolver context obtained from QEPCreate()
-   which - the portion of the spectrum to be sought

    Possible values:
    The parameter 'which' can have one of these values
    
+     QEP_LARGEST_MAGNITUDE - largest eigenvalues in magnitude (default)
.     QEP_SMALLEST_MAGNITUDE - smallest eigenvalues in magnitude
.     QEP_LARGEST_REAL - largest real parts
.     QEP_SMALLEST_REAL - smallest real parts
.     QEP_LARGEST_IMAGINARY - largest imaginary parts
-     QEP_SMALLEST_IMAGINARY - smallest imaginary parts

    Options Database Keys:
+   -qep_largest_magnitude - Sets largest eigenvalues in magnitude
.   -qep_smallest_magnitude - Sets smallest eigenvalues in magnitude
.   -qep_largest_real - Sets largest real parts
.   -qep_smallest_real - Sets smallest real parts
.   -qep_largest_imaginary - Sets largest imaginary parts
-   -qep_smallest_imaginary - Sets smallest imaginary parts

    Notes:
    Not all eigensolvers implemented in QEP account for all the possible values
    stated above. If SLEPc is compiled for real numbers QEP_LARGEST_IMAGINARY
    and QEP_SMALLEST_IMAGINARY use the absolute value of the imaginary part 
    for eigenvalue selection.
    
    Level: intermediate

.seealso: QEPGetWhichEigenpairs(), QEPWhich
@*/
PetscErrorCode QEPSetWhichEigenpairs(QEP qep,QEPWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  switch (which) {
    case QEP_LARGEST_MAGNITUDE:
    case QEP_SMALLEST_MAGNITUDE:
    case QEP_LARGEST_REAL:
    case QEP_SMALLEST_REAL:
    case QEP_LARGEST_IMAGINARY:
    case QEP_SMALLEST_IMAGINARY:
      if (qep->which != which) {
        qep->setupcalled = 0;
        qep->which = which;
      }
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid 'which' value"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetWhichEigenpairs"
/*@C
    QEPGetWhichEigenpairs - Returns which portion of the spectrum is to be 
    sought.

    Not Collective

    Input Parameter:
.   qep - eigensolver context obtained from QEPCreate()

    Output Parameter:
.   which - the portion of the spectrum to be sought

    Notes:
    See QEPSetWhichEigenpairs() for possible values of 'which'.

    Level: intermediate

.seealso: QEPSetWhichEigenpairs(), QEPWhich
@*/
PetscErrorCode QEPGetWhichEigenpairs(QEP qep,QEPWhich *which) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidPointer(which,2);
  *which = qep->which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetProblemType"
/*@
   QEPSetProblemType - Specifies the type of the quadratic eigenvalue problem.

   Collective on QEP

   Input Parameters:
+  qep      - the quadratic eigensolver context
-  type     - a known type of quadratic eigenvalue problem 

   Options Database Keys:
+  -qep_general - general problem with no particular structure
.  -qep_hermitian - problem whose coefficient matrices are Hermitian
-  -qep_gyroscopic - problem with Hamiltonian structure
    
   Notes:  
   Allowed values for the problem type are: general (QEP_GENERAL), Hermitian
   (QEP_HERMITIAN), and gyroscopic (QEP_GYROSCOPIC).

   This function is used to instruct SLEPc to exploit certain structure in
   the quadratic eigenproblem. By default, no particular structure is assumed.

   If the problem matrices are Hermitian (symmetric in the real case) or 
   Hermitian/skew-Hermitian then the solver can exploit this fact to perform
   less operations or provide better stability. 

   Level: intermediate

.seealso: QEPSetOperators(), QEPSetType(), QEPGetProblemType(), QEPProblemType
@*/
PetscErrorCode QEPSetProblemType(QEP qep,QEPProblemType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (type!=QEP_GENERAL && type!=QEP_HERMITIAN && type!=QEP_GYROSCOPIC)
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown eigenvalue problem type");
  qep->problem_type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetProblemType"
/*@C
   QEPGetProblemType - Gets the problem type from the QEP object.

   Not Collective

   Input Parameter:
.  qep - the quadratic eigensolver context 

   Output Parameter:
.  type - name of QEP problem type 

   Level: intermediate

.seealso: QEPSetProblemType(), QEPProblemType
@*/
PetscErrorCode QEPGetProblemType(QEP qep,QEPProblemType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidPointer(type,2);
  *type = qep->problem_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetConvergenceTest"
/*@C
    QEPSetConvergenceTest - Specifies the convergence test.

    Collective on QEP

    Input Parameters:
+   qep  - eigensolver context obtained from QEPCreate()
.   func - a pointer to the convergence test function
-   ctx  - a context pointer (the last parameter to the convergence test function)

    Calling Sequence of func:
$   func(QEP qep,PetscInt n,PetscInt k,PetscScalar* eigr,PetscScalar* eigi,PetscReal* errest,PetscTruth *conv,void *ctx)

+   qep    - eigensolver context obtained from QEPCreate()
.   n      - length of the arrays
.   k      - first position of the array to be considered
.   eigr   - array containing real parts of the eigenvalues
.   eigi   - array containing imaginary parts of the eigenvalues
.   errest - array containing the error estimates (residuals)
.   conv   - (output) boolean array with the result of the test
-   ctx    - optional context, as set by QEPSetConvergenceTest()

    Note:
    The convergence function sets an element of the flag array for each eigenvalue.
    
    Level: advanced

.seealso: QEPSetTolerances()
@*/
EXTERN PetscErrorCode QEPSetConvergenceTest(QEP qep,PetscErrorCode (*func)(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*),void* ctx)
{
  PetscFunctionBegin;
  qep->conv_func = func;
  qep->conv_ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetOptionsPrefix"
/*@C
   QEPSetOptionsPrefix - Sets the prefix used for searching for all 
   QEP options in the database.

   Collective on QEP

   Input Parameters:
+  qep - the quadratic eigensolver context
-  prefix - the prefix string to prepend to all QEP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different QEP contexts, one could call
.vb
      QEPSetOptionsPrefix(qep1,"qeig1_")
      QEPSetOptionsPrefix(qep2,"qeig2_")
.ve

   Level: advanced

.seealso: QEPAppendOptionsPrefix(), QEPGetOptionsPrefix()
@*/
PetscErrorCode QEPSetOptionsPrefix(QEP qep,const char *prefix)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)qep,prefix);CHKERRQ(ierr);
  ierr = IPSetOptionsPrefix(qep->ip,prefix);CHKERRQ(ierr);
  ierr = IPAppendOptionsPrefix(qep->ip,"qep_");CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}
 
#undef __FUNCT__  
#define __FUNCT__ "QEPAppendOptionsPrefix"
/*@C
   QEPAppendOptionsPrefix - Appends to the prefix used for searching for all 
   QEP options in the database.

   Collective on QEP

   Input Parameters:
+  qep - the quadratic eigensolver context
-  prefix - the prefix string to prepend to all QEP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: QEPSetOptionsPrefix(), QEPGetOptionsPrefix()
@*/
PetscErrorCode QEPAppendOptionsPrefix(QEP qep,const char *prefix)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)qep, prefix);CHKERRQ(ierr);
  ierr = IPSetOptionsPrefix(qep->ip,prefix);CHKERRQ(ierr);
  ierr = IPAppendOptionsPrefix(qep->ip,"qep_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetOptionsPrefix"
/*@C
   QEPGetOptionsPrefix - Gets the prefix used for searching for all 
   QEP options in the database.

   Not Collective

   Input Parameters:
.  qep - the quadratic eigensolver context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: QEPSetOptionsPrefix(), QEPAppendOptionsPrefix()
@*/
PetscErrorCode QEPGetOptionsPrefix(QEP qep,const char *prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)qep,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
