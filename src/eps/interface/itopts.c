#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

EXTERN PetscTruth EPSRegisterAllCalled;

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
  char           type[256];
  PetscTruth     flg;
  const char     *orth_list[2] = { "mgs" , "cgs" };
  const char     *ref_list[3] = { "never" , "ifneeded", "always" };
  PetscReal      eta;
  EPSOrthogonalizationType orth_type;
  EPSOrthogonalizationRefinementType ref_type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (!EPSRegisterAllCalled) {ierr = EPSRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsBegin(eps->comm,eps->prefix,"Eigenproblem Solver (EPS) Options","EPS");CHKERRQ(ierr);
    ierr = PetscOptionsList("-eps_type","Eigenproblem Solver method","EPSSetType",EPSList,(char*)(eps->type_name?eps->type_name:EPSARNOLDI),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = EPSSetType(eps,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!eps->type_name) {
      ierr = EPSSetType(eps,EPSARNOLDI);CHKERRQ(ierr);
    }

    ierr = PetscOptionsLogicalGroupBegin("-eps_hermitian","hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_gen_hermitian","generalized hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_non_hermitian","non-hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupEnd("-eps_gen_non_hermitian","generalized non-hermitian eigenvalue problem","EPSSetProblemType",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);}

    orth_type = eps->orthog_type;
    ierr = PetscOptionsEList("-eps_orthog_type","Orthogonalization method","EPSSetOrthogonalization",orth_list,2,orth_list[orth_type],(int*)&orth_type,&flg);CHKERRQ(ierr);
    ref_type = eps->orthog_ref;
    ierr = PetscOptionsEList("-eps_orthog_refinement","Iterative refinement mode during orthogonalization","EPSSetOrthogonalizationRefinement",ref_list,3,ref_list[ref_type],(int*)&ref_type,&flg);CHKERRQ(ierr);
    eta = eps->orthog_eta;
    ierr = PetscOptionsReal("-eps_orthog_eta","Parameter of iterative refinement during orthogonalization","EPSSetOrthogonalizationRefinement",eta,&eta,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSSetOrthogonalization(eps,orth_type,ref_type,eta);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-eps_max_it","Maximum number of iterations","EPSSetTolerances",eps->max_it,&eps->max_it,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps_tol","Tolerance","KSPSetTolerances",eps->tol,&eps->tol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_nev","Number of eigenvalues to compute","EPSSetDimensions",eps->nev,&eps->nev,&flg);CHKERRQ(ierr);
    if( eps->nev<1 ) SETERRQ(1,"Illegal value for option -eps_nev. Must be > 0");
    ierr = PetscOptionsInt("-eps_ncv","Number of basis vectors","EPSSetDimensions",eps->ncv,&eps->ncv,&flg);CHKERRQ(ierr);
    if( flg && eps->ncv<1 ) SETERRQ(1,"Illegal value for option -eps_ncv. Must be > 0");

    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to EPSSetFromOptions()
    */
    ierr = PetscOptionsName("-eps_cancelmonitors","Remove any hardwired monitor routines","EPSClearMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = EPSClearMonitor(eps); CHKERRQ(ierr);
    }
    /*
      Prints approximate eigenvalues and error estimates at each iteration
    */
    ierr = PetscOptionsName("-eps_monitor","Monitor approximate eigenvalues and error estimates","EPSSetMonitor",&flg);CHKERRQ(ierr); 
    if (flg) {
      ierr = EPSSetMonitor(eps,EPSDefaultMonitor,PETSC_NULL);CHKERRQ(ierr);
    }
  /* -----------------------------------------------------------------------*/
    ierr = PetscOptionsLogicalGroupBegin("-eps_largest_magnitude","compute largest eigenvalues in magnitude","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_smallest_magnitude","compute smallest eigenvalues in magnitude","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_largest_real","compute largest real parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_smallest_real","compute smallest real parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-eps_largest_imaginary","compute largest imaginary parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_IMAGINARY);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupEnd("-eps_smallest_imaginary","compute smallest imaginary parts","EPSSetWhichEigenpairs",&flg);CHKERRQ(ierr);
    if (flg) {ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_IMAGINARY);CHKERRQ(ierr);}

    ierr = PetscOptionsName("-eps_view","Print detailed information on solver used","EPSView",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-eps_view_binary","Save the matrices associated to the eigenproblem","EPSSetFromOptions",0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-eps_plot_eigs","Make a plot of the computed eigenvalues","EPSSolve",0);CHKERRQ(ierr);

    if (eps->ops->setfromoptions) {
      ierr = (*eps->ops->setfromoptions)(eps);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

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
PetscErrorCode EPSGetTolerances(EPS eps,PetscReal *tol,int *maxits)
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
   Use PETSC_DEFAULT to retain the default value of any of the tolerances.

   Level: intermediate

.seealso: EPSGetTolerances()
@*/
PetscErrorCode EPSSetTolerances(EPS eps,PetscReal tol,int maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (tol != PETSC_DEFAULT)    eps->tol    = tol;
  if (maxits != PETSC_DEFAULT) eps->max_it = maxits;
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
-  ncv - the maximum dimension of the subspace to be used by the solver

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: EPSSetDimensions()
@*/
PetscErrorCode EPSGetDimensions(EPS eps,int *nev,int *ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if( nev )   *nev = eps->nev;
  if( ncv )   *ncv = eps->ncv;
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
-  ncv - the maximum dimension of the subspace to be used by the solver

   Options Database Keys:
+  -eps_nev <nev> - Sets the number of eigenvalues
-  -eps_ncv <ncv> - Sets the dimension of the subspace

   Notes:
   Use PETSC_DEFAULT to retain the previous value of any parameter.

   Use PETSC_DECIDE for ncv to assign a reasonably good value, which is 
   dependent on the solution method.

   Level: intermediate

.seealso: EPSGetDimensions()
@*/
PetscErrorCode EPSSetDimensions(EPS eps,int nev,int ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if( nev != PETSC_DEFAULT ) {
    if (nev<1) SETERRQ(1,"Illegal value of nev. Must be > 0");
    eps->nev = nev;
    eps->setupcalled = 0;
  }
  if( ncv != PETSC_DEFAULT ) {
    if( ncv == PETSC_DECIDE ) eps->ncv = 0;
    else { 
      if (ncv<1) SETERRQ(1,"Illegal value of ncv. Must be > 0");
      eps->ncv = ncv;
    }
    eps->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetWhichEigenpairs"
/*@
    EPSSetWhichEigenpairs - Specifies which portion of the spectrum is 
    to be sought.

    Collective on EPS

    Input Parameter:
.   eps - eigensolver context obtained from EPSCreate()

    Output Parameter:
.   which - the portion of the spectrum to be sought

    Possible values:
    The parameter 'which' can have one of these values:
    
+     EPS_LARGEST_MAGNITUDE - largest eigenvalues in magnitude (default)
.     EPS_SMALLEST_MAGNITUDE - smallest eigenvalues in magnitude
.     EPS_LARGEST_REAL - largest real parts
.     EPS_SMALLEST_REAL - smallest real parts
.     EPS_LARGEST_IMAGINARY - largest imaginary parts
-     EPS_SMALLEST_IMAGINARY - smallest imaginary parts

    Options Database Keys:
+   -eps_largest_magnitude - Sets largest eigenvalues in magnitude
.   -eps_smallest_magnitude - Sets smallest eigenvalues in magnitude
.   -eps_largest_real - Sets largest real parts
.   -eps_smallest_real - Sets smallest real parts
.   -eps_largest_imaginary - Sets largest imaginary parts in magnitude
-   -eps_smallest_imaginary - Sets smallest imaginary parts in magnitude

    Notes:
    Not all eigensolvers implemented in EPS account for all the possible values
    stated above. Also, some values make sense only for certain types of 
    problems. If SLEPc is compiled for real numbers EPS_LARGEST_IMAGINARY
    and EPS_SMALLEST_IMAGINARY use the absolute value of the imaginary part 
    for eigenvalue selection.     
    
    Level: intermediate

.seealso: EPSGetWhichEigenpairs(), EPSSortEigenvalues()
@*/
PetscErrorCode EPSSetWhichEigenpairs(EPS eps,EPSWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  eps->which = which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetWhichEigenpairs"
/*@
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

.seealso: EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSGetWhichEigenpairs(EPS eps,EPSWhich *which) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *which = eps->which;
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
-  -eps_gen_non_hermitian - generalized non-Hermitian eigenvalue problem 
    
   Note:  
   Allowed values for the problem type are: Hermitian (EPS_HEP), non-Hermitian
   (EPS_NHEP), generalized Hermitian (EPS_GHEP) and generalized non-Hermitian 
   (EPS_GNHEP).

   This function must be used to instruct SLEPc to exploit symmetry. If no
   problem type is specified, by default a non-Hermitian problem is assumed
   (either standard or generalized). If the user knows that the problem is
   Hermitian (i.e. A=A^H) of generalized Hermitian (i.e. A=A^H, B=B^H, and 
   B positive definite) then it is recommended to set the problem type so
   that eigensolver can exploit these properties. 

   Level: beginner

.seealso: EPSSetOperators(), EPSSetType(), EPSProblemType
@*/
PetscErrorCode EPSSetProblemType(EPS eps,EPSProblemType type)
{
  PetscErrorCode ierr;
  Mat            A,B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  if (!A) { SETERRQ(1,"Must call EPSSetOperators() first"); }

  switch (type) {
    case EPS_HEP:
      eps->isgeneralized = PETSC_FALSE;
      eps->ishermitian = PETSC_TRUE;
      ierr = STSetBilinearForm(eps->OP,STINNER_HERMITIAN);CHKERRQ(ierr);
      break;      
    case EPS_NHEP:
      eps->isgeneralized = PETSC_FALSE;
      eps->ishermitian = PETSC_FALSE;
      ierr = STSetBilinearForm(eps->OP,STINNER_HERMITIAN);CHKERRQ(ierr);
      break;
    case EPS_GHEP:
      eps->isgeneralized = PETSC_TRUE;
      eps->ishermitian = PETSC_TRUE;
      ierr = STSetBilinearForm(eps->OP,STINNER_B_HERMITIAN);CHKERRQ(ierr);
      break;
    case EPS_GNHEP:
      eps->isgeneralized = PETSC_TRUE;
      eps->ishermitian = PETSC_FALSE;
      ierr = STSetBilinearForm(eps->OP,STINNER_HERMITIAN);CHKERRQ(ierr);
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
/*@
   EPSGetProblemType - Gets the problem type from the EPS object.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context 

   Output Parameter:
.  type - name of EPS problem type 

   Level: intermediate

.seealso: EPSSetProblemType()
@*/
PetscErrorCode EPSGetProblemType(EPS eps,EPSProblemType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *type = eps->problem_type;
  PetscFunctionReturn(0);
}

