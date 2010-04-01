/*
       This file implements a wrapper to the ARPACK package

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

#include "src/eps/impls/external/arpack/arpackp.h"
#include "private/stimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_ARPACK"
PetscErrorCode EPSSetUp_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       ncv;
  EPS_ARPACK     *ar = (EPS_ARPACK *)eps->data;

  PetscFunctionBegin;
  if (eps->ncv) {
    if (eps->ncv<eps->nev+2) SETERRQ(1,"The value of ncv must be at least nev+2"); 
  } else /* set default value of ncv */
    eps->ncv = PetscMin(PetscMax(20,2*eps->nev+1),eps->n);
  if (eps->mpd) PetscInfo(eps,"Warning: parameter mpd ignored\n");
  if (!eps->max_it) eps->max_it = PetscMax(300,(PetscInt)(2*eps->n/eps->ncv));

  ncv = eps->ncv;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(ar->rwork);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&ar->rwork);CHKERRQ(ierr);
  ar->lworkl = PetscBLASIntCast(3*ncv*ncv+5*ncv);
  ierr = PetscFree(ar->workev);CHKERRQ(ierr); 
  ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&ar->workev);CHKERRQ(ierr);
#else
  if( eps->ishermitian ) {
    ar->lworkl = PetscBLASIntCast(ncv*(ncv+8));
  } else {
    ar->lworkl = PetscBLASIntCast(3*ncv*ncv+6*ncv);
    ierr = PetscFree(ar->workev);CHKERRQ(ierr); 
    ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&ar->workev);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(ar->workl);CHKERRQ(ierr); 
  ierr = PetscMalloc(ar->lworkl*sizeof(PetscScalar),&ar->workl);CHKERRQ(ierr);
  ierr = PetscFree(ar->select);CHKERRQ(ierr); 
  ierr = PetscMalloc(ncv*sizeof(PetscTruth),&ar->select);CHKERRQ(ierr);
  ierr = PetscFree(ar->workd);CHKERRQ(ierr); 
  ierr = PetscMalloc(3*eps->nloc*sizeof(PetscScalar),&ar->workd);CHKERRQ(ierr);

  if (eps->extraction) {
     ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr);
  }

  if (eps->balance!=EPSBALANCE_NONE)
    SETERRQ(PETSC_ERR_SUP,"Balancing not supported in the Arpack interface");

  ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARPACK"
PetscErrorCode EPSSolve_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARPACK  	 *ar = (EPS_ARPACK *)eps->data;
  char        	 bmat[1], howmny[] = "A";
  const char  	 *which;
  PetscBLASInt   n, iparam[11], ipntr[14], ido, info,
		 nev, ncv;
  PetscScalar 	 sigmar, *pV, *resid;
  Vec         	 x, y, w = eps->work[0];
  Mat         	 A;
  PetscTruth  	 isSinv, isShift, rvec;
  PetscBLASInt   fcomm;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    sigmai = 0.0;
#endif
  PetscFunctionBegin;
  
  nev = PetscBLASIntCast(eps->nev);
  ncv = PetscBLASIntCast(eps->ncv);
  fcomm = PetscBLASIntCast(MPI_Comm_c2f(((PetscObject)eps)->comm));
  n = PetscBLASIntCast(eps->nloc);
  ierr = VecCreateMPIWithArray(((PetscObject)eps)->comm,eps->nloc,PETSC_DECIDE,PETSC_NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)eps)->comm,eps->nloc,PETSC_DECIDE,PETSC_NULL,&y);CHKERRQ(ierr);
  ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
  ierr = VecCopy(eps->vec_initial,eps->work[1]);CHKERRQ(ierr);
  ierr = VecGetArray(eps->work[1],&resid);CHKERRQ(ierr);
  
  ido  = 0;            /* first call to reverse communication interface */
  info = 1;            /* indicates a initial vector is provided */
  iparam[0] = 1;       /* use exact shifts */
  iparam[2] = PetscBLASIntCast(eps->max_it);  /* maximum number of Arnoldi update iterations */
  iparam[3] = 1;       /* blocksize */
  iparam[4] = 0;       /* number of converged Ritz values */
 
  /*
     Computational modes ([]=not supported):
            symmetric    non-symmetric    complex
        1     1  'I'        1  'I'         1  'I'
        2     3  'I'        3  'I'         3  'I'
        3     2  'G'        2  'G'         2  'G'
        4     3  'G'        3  'G'         3  'G'
        5   [ 4  'G' ]    [ 3  'G' ]
        6   [ 5  'G' ]    [ 4  'G' ]
   */
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSINV,&isSinv);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSHIFT,&isShift);CHKERRQ(ierr);
  ierr = STGetShift(eps->OP,&sigmar);CHKERRQ(ierr);
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);

  if (isSinv) { 
    /* shift-and-invert mode */
    iparam[6] = 3;
    if (eps->ispositive) bmat[0] = 'G';
    else bmat[0] = 'I';
  } else if (isShift && eps->ispositive) {
    /* generalized shift mode with B positive definite */
    iparam[6] = 2;
    bmat[0] = 'G';
  } else { 
    /* regular mode */
    if (eps->ishermitian && eps->isgeneralized)
      SETERRQ(PETSC_ERR_SUP,"Spectral transformation not supported by ARPACK hermitian solver");
    iparam[6] = 1;
    bmat[0] = 'I';
  }
 
#if !defined(PETSC_USE_COMPLEX)
    if (eps->ishermitian) {
      switch(eps->which) {
        case EPS_LARGEST_MAGNITUDE:  which = "LM"; break;
        case EPS_SMALLEST_MAGNITUDE: which = "SM"; break;
        case EPS_LARGEST_REAL:       which = "LA"; break;
        case EPS_SMALLEST_REAL:      which = "SA"; break;
        default: SETERRQ(1,"Wrong value of eps->which");
      }
    } else {
#endif
      switch(eps->which) {
        case EPS_LARGEST_MAGNITUDE:  which = "LM"; break;
        case EPS_SMALLEST_MAGNITUDE: which = "SM"; break;
        case EPS_LARGEST_REAL:       which = "LR"; break;
        case EPS_SMALLEST_REAL:      which = "SR"; break;
        case EPS_LARGEST_IMAGINARY:  which = "LI"; break;
        case EPS_SMALLEST_IMAGINARY: which = "SI"; break;
        default: SETERRQ(1,"Wrong value of eps->which");
      }
#if !defined(PETSC_USE_COMPLEX)
    }
#endif

  do {

#if !defined(PETSC_USE_COMPLEX)
    if (eps->ishermitian) {
      ARsaupd_( &fcomm, &ido, bmat, &n, which, &nev, &eps->tol,
                resid, &ncv, pV, &n, iparam, ipntr, ar->workd, 
                ar->workl, &ar->lworkl, &info, 1, 2 );
    }
    else {
      ARnaupd_( &fcomm, &ido, bmat, &n, which, &nev, &eps->tol,
                resid, &ncv, pV, &n, iparam, ipntr, ar->workd, 
                ar->workl, &ar->lworkl, &info, 1, 2 );
    }
#else
    ARnaupd_( &fcomm, &ido, bmat, &n, which, &nev, &eps->tol,
              resid, &ncv, pV, &n, iparam, ipntr, ar->workd, 
              ar->workl, &ar->lworkl, ar->rwork, &info, 1, 2 );
#endif
    
    if (ido == -1 || ido == 1 || ido == 2) {
      if (ido == 1 && iparam[6] == 3 && bmat[0] == 'G') {
        /* special case for shift-and-invert with B semi-positive definite*/
        ierr = VecPlaceArray(x,&ar->workd[ipntr[2]-1]); CHKERRQ(ierr);
      } else {
        ierr = VecPlaceArray(x,&ar->workd[ipntr[0]-1]); CHKERRQ(ierr);
      }
      ierr = VecPlaceArray(y,&ar->workd[ipntr[1]-1]); CHKERRQ(ierr);
      
      if (ido == -1) { 
        /* Y = OP * X for for the initialization phase to 
	   force the starting vector into the range of OP */
	ierr = STApply(eps->OP,x,y); CHKERRQ(ierr);
      } else if (ido == 2) {
        /* Y = B * X */
	ierr = IPApplyMatrix(eps->ip,x,y); CHKERRQ(ierr);
      } else { /* ido == 1 */
        if (iparam[6] == 3 && bmat[0] == 'G') {
          /* Y = OP * X for shift-and-invert with B semi-positive definite */
	  ierr = STAssociatedKSPSolve(eps->OP,x,y);CHKERRQ(ierr);
	} else if (iparam[6] == 2) {
          /* X=A*X Y=B^-1*X for shift with B positive definite */
	  ierr = MatMult(A,x,y);CHKERRQ(ierr);
	  if (sigmar != 0.0) {
	    ierr = IPApplyMatrix(eps->ip,x,w);CHKERRQ(ierr);
            ierr = VecAXPY(y,sigmar,w);CHKERRQ(ierr);
	  }
          ierr = VecCopy(y,x); CHKERRQ(ierr);
          ierr = STAssociatedKSPSolve(eps->OP,x,y);CHKERRQ(ierr);
	} else  {
          /* Y = OP * X */
	  ierr = STApply(eps->OP,x,y); CHKERRQ(ierr);        
	}
        ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,eps->nds,PETSC_NULL,eps->DS,y,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      }
            
      ierr = VecResetArray(x); CHKERRQ(ierr);
      ierr = VecResetArray(y); CHKERRQ(ierr);
    } else if (ido != 99) {
      SETERRQ1(1,"Internal error in ARPACK reverse comunication interface (ido=%i)\n",ido);
    }
    
  } while (ido != 99);

  eps->nconv = iparam[4];
  eps->its = iparam[2];
  
  if (info==3) { SETERRQ(1,"No shift could be applied in xxAUPD.\n"
                           "Try increasing the size of NCV relative to NEV."); }
  else if (info!=0 && info!=1) { SETERRQ1(PETSC_ERR_LIB,"Error reported by ARPACK subroutine xxAUPD (%d)",info);}

  rvec = PETSC_TRUE;

  if (eps->nconv > 0) {
#if !defined(PETSC_USE_COMPLEX)
    if (eps->ishermitian) {
      EPSMonitor(eps,iparam[2],iparam[4],&ar->workl[ipntr[5]-1],eps->eigi,&ar->workl[ipntr[6]-1],eps->ncv); 
      ARseupd_ ( &fcomm, &rvec, howmny, ar->select, eps->eigr,  
        	 pV, &n, &sigmar, 
        	 bmat, &n, which, &nev, &eps->tol,
        	 resid, &ncv, pV, &n, iparam, ipntr, ar->workd, 
        	 ar->workl, &ar->lworkl, &info, 1, 1, 2 );
    }
    else {
      EPSMonitor(eps,iparam[2],iparam[4],&ar->workl[ipntr[5]-1],&ar->workl[ipntr[6]-1],&ar->workl[ipntr[7]-1],eps->ncv); 
      ARneupd_ ( &fcomm, &rvec, howmny, ar->select, eps->eigr, eps->eigi, 
        	 pV, &n, &sigmar, &sigmai, ar->workev, 
        	 bmat, &n, which, &nev, &eps->tol,
        	 resid, &ncv, pV, &n, iparam, ipntr, ar->workd, 
        	 ar->workl, &ar->lworkl, &info, 1, 1, 2 );
    }
#else
    EPSMonitor(eps,eps->its,iparam[4],&ar->workl[ipntr[5]-1],eps->eigi,(PetscReal*)&ar->workl[ipntr[7]-1],eps->ncv); 
    ARneupd_ ( &fcomm, &rvec, howmny, ar->select, eps->eigr,
               pV, &n, &sigmar, ar->workev, 
               bmat, &n, which, &nev, &eps->tol,
               resid, &ncv, pV, &n, iparam, ipntr, ar->workd, 
               ar->workl, &ar->lworkl, ar->rwork, &info, 1, 1, 2 );
#endif
    if (info!=0) { SETERRQ1(PETSC_ERR_LIB,"Error reported by ARPACK subroutine xxEUPD (%d)",info); }
  }

  ierr = VecRestoreArray( eps->V[0], &pV ); CHKERRQ(ierr);
  ierr = VecRestoreArray( eps->work[1], &resid ); CHKERRQ(ierr);
  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  if (eps->ishermitian) {
    ierr = PetscMemcpy(eps->errest,&ar->workl[ipntr[8]-1],eps->nconv);CHKERRQ(ierr);
  } else {
    ierr = PetscMemcpy(eps->errest,&ar->workl[ipntr[10]-1],eps->nconv);CHKERRQ(ierr);
  }

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_ARPACK"
PetscErrorCode EPSBackTransform_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  PetscTruth     isSinv;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSINV,&isSinv);CHKERRQ(ierr);
  if (!isSinv) {
    ierr = EPSBackTransform_Default(eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_ARPACK"
PetscErrorCode EPSDestroy_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARPACK     *ar = (EPS_ARPACK *)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscFree(ar->workev);CHKERRQ(ierr); 
  ierr = PetscFree(ar->workl);CHKERRQ(ierr); 
  ierr = PetscFree(ar->select);CHKERRQ(ierr); 
  ierr = PetscFree(ar->workd);CHKERRQ(ierr); 
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(ar->rwork);CHKERRQ(ierr); 
#endif
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSDefaultFreeWork(eps);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARPACK"
PetscErrorCode EPSCreate_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARPACK     *arpack;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_ARPACK,&arpack);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_ARPACK));
  eps->data                      = (void *) arpack;
  eps->ops->solve                = EPSSolve_ARPACK;
  eps->ops->setup                = EPSSetUp_ARPACK;
  eps->ops->destroy              = EPSDestroy_ARPACK;
  eps->ops->backtransform        = EPSBackTransform_ARPACK;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END
