/*
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

#include <petsc-private/fortranimpl.h>
#include <slepc-private/epsimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define epsdestroy_                 EPSDESTROY
#define epsview_                    EPSVIEW
#define epssetoptionsprefix_        EPSSETOPTIONSPREFIX
#define epsappendoptionsprefix_     EPSAPPENDOPTIONSPREFIX
#define epsgetoptionsprefix_        EPSGETOPTIONSPREFIX
#define epscreate_                  EPSCREATE
#define epssettype_                 EPSSETTYPE           
#define epsgettype_                 EPSGETTYPE
#define epsgetoperators_            EPSGETOPERATORS
#define epsmonitorall_              EPSMONITORALL
#define epsmonitorlg_               EPSMONITORLG
#define epsmonitorlgall_            EPSMONITORLGALL
#define epsmonitorset_              EPSMONITORSET
#define epsmonitorconverged_        EPSMONITORCONVERGED
#define epsmonitorfirst_            EPSMONITORFIRST
#define epsgetst_                   EPSGETST
#define epsgetip_                   EPSGETIP
#define epsgetds_                   EPSGETDS
#define epsgetwhicheigenpairs_      EPSGETWHICHEIGENPAIRS
#define epsgetproblemtype_          EPSGETPROBLEMTYPE
#define epsgetextraction_           EPSGETEXTRACTION
#define epsgetconvergedreason_      EPSGETCONVERGEDREASON
#define epspowergetshifttype_       EPSPOWERGETSHIFTTYPE
#define epslanczosgetreorthog_      EPSLANCZOSGETREORTHOG
#define epsconvergedabsolute_       EPSCONVERGEDABSOLUTE
#define epsconvergedeigrelative_    EPSCONVERGEDEIGRELATIVE
#define epsconvergednormrelative_   EPSCONVERGEDNORMRELATIVE
#define epssetconvergencetestfunction_ EPSSETCONVERGENCETESTFUNCTION
#define epsseteigenvaluecomparison_ EPSSETEIGENVALUECOMPARISON
#define epssetarbitraryselection_   EPSSETARBITRARYSELECTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsdestroy_                 epsdestroy
#define epsview_                    epsview
#define epssetoptionsprefix_        epssetoptionsprefix
#define epsappendoptionsprefix_     epsappendoptionsprefix
#define epsgetoptionsprefix_        epsgetoptionsprefix
#define epscreate_                  epscreate
#define epssettype_                 epssettype           
#define epsgettype_                 epsgettype
#define epsgetoperators_            epsgetoperators
#define epsmonitorall_              epsmonitorall
#define epsmonitorlg_               epsmonitorlg
#define epsmonitorlgall_            epsmonitorlgall
#define epsmonitorset_              epsmonitorset
#define epsmonitorconverged_        epsmonitorconverged
#define epsmonitorfirst_            epsmonitorfirst
#define epsgetst_                   epsgetst
#define epsgetip_                   epsgetip
#define epsgetds_                   epsgetds
#define epsgetwhicheigenpairs_      epsgetwhicheigenpairs
#define epsgetproblemtype_          epsgetproblemtype
#define epsgetextraction_           epsgetextraction
#define epsgetconvergedreason_      epsgetconvergedreason
#define epspowergetshifttype_       epspowergetshifttype
#define epslanczosgetreorthog_      epslanczosgetreorthog
#define epsconvergedabsolute_       epsconvergedabsolute
#define epsconvergedeigrelative_    epsconvergedeigrelative
#define epsconvergednormrelative_   epsconvergednormrelative
#define epssetconvergencetestfunction_ epssetconvergencetestfunction
#define epsseteigenvaluecomparison_ epsseteigenvaluecomparison
#define epssetarbitraryselection_   epssetarbitraryselection
#endif

EXTERN_C_BEGIN

/*
   These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code, hence no STDCALL
*/
void epsmonitorall_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorAll(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void epsmonitorlg_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorLG(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void epsmonitorlgall_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorLGAll(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void epsmonitorconverged_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorConverged(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void epsmonitorfirst_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorFirst(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

EXTERN_C_END
 
static struct {
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId monitordestroy;
  PetscFortranCallbackId convergence;
  PetscFortranCallbackId comparison;
  PetscFortranCallbackId arbitrary;
} _cb;

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmonitor(EPS eps,PetscInt i,PetscInt nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,PetscInt l,void* ctx)
{
  PetscObjectUseFortranCallback(eps,_cb.monitor,(EPS*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),(&eps,&i,&nc,er,ei,d,&l,_ctx,&ierr));
  return 0;
}

static PetscErrorCode ourdestroy(void** ctx)
{
  EPS eps = (EPS)*ctx;
  PetscObjectUseFortranCallback(eps,_cb.monitordestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL epsdestroy_(EPS *eps,PetscErrorCode *ierr)
{
  *ierr = EPSDestroy(eps);
}

void PETSC_STDCALL epsview_(EPS *eps,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = EPSView(*eps,v);
}

void PETSC_STDCALL epssettype_(EPS *eps,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = EPSSetType(*eps,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL epsgettype_(EPS *eps,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  EPSType tname;

  *ierr = EPSGetType(*eps,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

void PETSC_STDCALL epsgetoperators_(EPS *eps,Mat *A,Mat *B,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(A);
  CHKFORTRANNULLOBJECT(B);
  *ierr = EPSGetOperators(*eps,A,B);
}

void PETSC_STDCALL epssetoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSSetOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL epsappendoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSAppendOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL epscreate_(MPI_Fint *comm,EPS *eps,PetscErrorCode *ierr)
{
  *ierr = EPSCreate(MPI_Comm_f2c(*(comm)),eps);
}

void PETSC_STDCALL epsmonitorset_(EPS *eps,void (PETSC_STDCALL *monitor)(EPS*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode*),PetscErrorCode *ierr)
{
  SlepcConvMonitor ctx;

  CHKFORTRANNULLOBJECT(mctx);
  CHKFORTRANNULLFUNCTION(monitordestroy);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorall_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorAll,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorlg_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorLG,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorlgall_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorLGAll,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorconverged_) {
    if (!FORTRANNULLOBJECT(mctx)) {
      PetscError(((PetscObject)*eps)->comm,__LINE__,"epsmonitorset_",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL,"Must provide PETSC_NULL_OBJECT as a context in the Fortran interface to EPSMonitorSet");
      *ierr = 1;
      return;
    }
    *ierr = PetscNew(struct _n_SlepcConvMonitor,&ctx);
    if (*ierr) return;
    ctx->viewer = PETSC_NULL;
    *ierr = EPSMonitorSet(*eps,EPSMonitorConverged,ctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorfirst_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorFirst,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitor,(PetscVoidFunction)monitor,mctx); if (*ierr) return;
    if (!monitordestroy) {
      *ierr = EPSMonitorSet(*eps,ourmonitor,*eps,0);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitordestroy,(PetscVoidFunction)monitordestroy,mctx); if (*ierr) return;
      *ierr = EPSMonitorSet(*eps,ourmonitor,*eps,ourdestroy);
    }
  }
}

void PETSC_STDCALL epsgetoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = EPSGetOptionsPrefix(*eps,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
}

void PETSC_STDCALL epsgetst_(EPS *eps,ST *st,PetscErrorCode *ierr)
{
  *ierr = EPSGetST(*eps,st);
}

void PETSC_STDCALL epsgetip_(EPS *eps,IP *ip,PetscErrorCode *ierr)
{
  *ierr = EPSGetIP(*eps,ip);
}

void PETSC_STDCALL epsgetds_(EPS *eps,DS *ds,PetscErrorCode *ierr)
{
  *ierr = EPSGetDS(*eps,ds);
}

void PETSC_STDCALL epsgetwhicheigenpairs_(EPS *eps,EPSWhich *which,PetscErrorCode *ierr)
{
  *ierr = EPSGetWhichEigenpairs(*eps,which);
}

void PETSC_STDCALL epsgetproblemtype_(EPS *eps,EPSProblemType *type,PetscErrorCode *ierr)
{
  *ierr = EPSGetProblemType(*eps,type);
}

void PETSC_STDCALL epsgetextraction_(EPS *eps,EPSExtraction *proj,PetscErrorCode *ierr)
{
  *ierr = EPSGetExtraction(*eps,proj);
}

void PETSC_STDCALL epsgetconvergedreason_(EPS *eps,EPSConvergedReason *reason,PetscErrorCode *ierr)
{
  *ierr = EPSGetConvergedReason(*eps,reason);
}

void PETSC_STDCALL epspowergetshifttype_(EPS *eps,EPSPowerShiftType *shift,PetscErrorCode *ierr)
{
  *ierr = EPSPowerGetShiftType(*eps,shift);
}

void PETSC_STDCALL epslanczosgetreorthog_(EPS *eps,EPSLanczosReorthogType *reorthog,PetscErrorCode *ierr)
{
  *ierr = EPSLanczosGetReorthog(*eps,reorthog);
}

void PETSC_STDCALL epsconvergedabsolute_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSConvergedAbsolute(*eps,*eigr,*eigi,*res,errest,ctx);
}

void PETSC_STDCALL epsconvergedeigrelative_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSConvergedEigRelative(*eps,*eigr,*eigi,*res,errest,ctx);
}

void PETSC_STDCALL epsconvergednormrelative_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSConvergedNormRelative(*eps,*eigr,*eigi,*res,errest,ctx);
}

EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourconvergence(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscObjectUseFortranCallback(eps,_cb.convergence,(EPS*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),(&eps,&eigr,&eigi,&res,errest,_ctx,&ierr));
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL epssetconvergencetestfunction_(EPS *eps,void (PETSC_STDCALL *func)(EPS*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void* ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  if ((PetscVoidFunction)func == (PetscVoidFunction)epsconvergedabsolute_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_ABS);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)epsconvergedeigrelative_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_EIG);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)epsconvergednormrelative_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_NORM);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.convergence,(PetscVoidFunction)func,ctx); if (*ierr) return;
    *ierr = EPSSetConvergenceTestFunction(*eps,ourconvergence,PETSC_NULL);
  }
}

EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode oureigenvaluecomparison(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  EPS eps = (EPS)ctx;
  PetscObjectUseFortranCallback(eps,_cb.comparison,(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*),(&ar,&ai,&br,&bi,r,_ctx,&ierr));
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL epsseteigenvaluecomparison_(EPS *eps,void (PETSC_STDCALL *func)(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*),void* ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.comparison,(PetscVoidFunction)func,ctx); if (*ierr) return;
  *ierr = EPSSetEigenvalueComparison(*eps,oureigenvaluecomparison,eps);
}

EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourarbitraryfunc(PetscScalar er,PetscScalar ei,Vec xr,Vec xi,PetscScalar *rr,PetscScalar *ri,void *ctx)
{
  EPS eps = (EPS)ctx;
  PetscObjectUseFortranCallback(eps,_cb.arbitrary,(PetscScalar*,PetscScalar*,Vec*,Vec*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*),(&er,&ei,&xr,&xi,rr,ri,_ctx,&ierr));
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL epssetarbitraryselection_(EPS *eps,void (PETSC_STDCALL *func)(PetscScalar*,PetscScalar*,Vec*,Vec*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.arbitrary,(PetscVoidFunction)func,ctx); if (*ierr) return;
  *ierr = EPSSetArbitrarySelection(*eps,ourarbitraryfunc,*eps);
}

EXTERN_C_END
