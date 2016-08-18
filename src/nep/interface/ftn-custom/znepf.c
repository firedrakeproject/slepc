/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <petsc/private/fortranimpl.h>
#include <slepc/private/slepcimpl.h>
#include <slepc/private/nepimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define nepview_                    NEPVIEW
#define neperrorview_               NEPERRORVIEW
#define nepreasonview_              NEPREASONVIEW
#define nepvaluesview_              NEPVALUESVIEW
#define nepvectorsview_             NEPVECTORSVIEW
#define nepsetoptionsprefix_        NEPSETOPTIONSPREFIX
#define nepappendoptionsprefix_     NEPAPPENDOPTIONSPREFIX
#define nepgetoptionsprefix_        NEPGETOPTIONSPREFIX
#define nepsettype_                 NEPSETTYPE
#define nepgettype_                 NEPGETTYPE
#define nepmonitorall_              NEPMONITORALL
#define nepmonitorlg_               NEPMONITORLG
#define nepmonitorlgall_            NEPMONITORLGALL
#define nepmonitorset_              NEPMONITORSET
#define nepmonitorconverged_        NEPMONITORCONVERGED
#define nepmonitorfirst_            NEPMONITORFIRST
#define nepconvergedabsolute_       NEPCONVERGEDABSOLUTE
#define nepconvergedrelative_       NEPCONVERGEDRELATIVE
#define nepsetconvergencetestfunction_ NEPSETCONVERGENCETESTFUNCTION
#define nepsetstoppingtestfunction_ NEPSETSTOPPINGTESTFUNCTION
#define nepseteigenvaluecomparison_ NEPSETEIGENVALUECOMPARISON
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define nepview_                    nepview
#define neperrorview_               neperrorview
#define nepreasonview_              nepreasonview
#define nepvaluesview_              nepvaluesview
#define nepvectorsview_             nepvectorsview
#define nepsetoptionsprefix_        nepsetoptionsprefix
#define nepappendoptionsprefix_     nepappendoptionsprefix
#define nepgetoptionsprefix_        nepgetoptionsprefix
#define nepsettype_                 nepsettype
#define nepgettype_                 nepgettype
#define nepmonitorall_              nepmonitorall
#define nepmonitorlg_               nepmonitorlg
#define nepmonitorlgall_            nepmonitorlgall
#define nepmonitorset_              nepmonitorset
#define nepmonitorconverged_        nepmonitorconverged
#define nepmonitorfirst_            nepmonitorfirst
#define nepconvergedabsolute_       nepconvergedabsolute
#define nepconvergedrelative_       nepconvergedrelative
#define nepsetconvergencetestfunction_ nepsetconvergencetestfunction
#define nepsetstoppingtestfunction_ nepsetstoppingtestfunction
#define nepseteigenvaluecomparison_ nepseteigenvaluecomparison
#endif

/*
   These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code, hence no STDCALL
*/
PETSC_EXTERN void nepmonitorall_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,PetscViewerAndFormat **ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorAll(*nep,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void nepmonitorconverged_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,SlepcConvMonitor *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorConverged(*nep,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void nepmonitorfirst_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,PetscViewerAndFormat **ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorFirst(*nep,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void nepmonitorlg_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorLG(*nep,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

PETSC_EXTERN void nepmonitorlgall_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorLGAll(*nep,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

static struct {
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId monitordestroy;
  PetscFortranCallbackId convergence;
  PetscFortranCallbackId convdestroy;
  PetscFortranCallbackId stopping;
  PetscFortranCallbackId stopdestroy;
  PetscFortranCallbackId comparison;
} _cb;

/* These are not extern C because they are passed into non-extern C user level functions */
#undef __FUNCT__
#define __FUNCT__ "ourmonitor"
static PetscErrorCode ourmonitor(NEP nep,PetscInt i,PetscInt nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,PetscInt l,void* ctx)
{
  PetscObjectUseFortranCallback(nep,_cb.monitor,(NEP*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),(&nep,&i,&nc,er,ei,d,&l,_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourdestroy"
static PetscErrorCode ourdestroy(void** ctx)
{
  NEP nep = (NEP)*ctx;
  PetscObjectUseFortranCallback(nep,_cb.monitordestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourconvergence"
static PetscErrorCode ourconvergence(NEP nep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscObjectUseFortranCallback(nep,_cb.convergence,(NEP*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),(&nep,&eigr,&eigi,&res,errest,_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourconvdestroy"
static PetscErrorCode ourconvdestroy(void *ctx)
{
  NEP nep = (NEP)ctx;
  PetscObjectUseFortranCallback(nep,_cb.convdestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourstopping"
static PetscErrorCode ourstopping(NEP nep,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,NEPConvergedReason *reason,void *ctx)
{
  PetscObjectUseFortranCallback(nep,_cb.stopping,(NEP*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,NEPConvergedReason*,void*,PetscErrorCode*),(&nep,&its,&max_it,&nconv,&nev,reason,_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourstopdestroy"
static PetscErrorCode ourstopdestroy(void *ctx)
{
  NEP nep = (NEP)ctx;
  PetscObjectUseFortranCallback(nep,_cb.stopdestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "oureigenvaluecomparison"
static PetscErrorCode oureigenvaluecomparison(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  NEP eps = (NEP)ctx;
  PetscObjectUseFortranCallback(eps,_cb.comparison,(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*),(&ar,&ai,&br,&bi,r,_ctx,&ierr));
}

PETSC_EXTERN void PETSC_STDCALL nepview_(NEP *nep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = NEPView(*nep,v);
}

PETSC_EXTERN void PETSC_STDCALL nepreasonview_(NEP *nep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = NEPReasonView(*nep,v);
}

PETSC_EXTERN void PETSC_STDCALL neperrorview_(NEP *nep,NEPErrorType *etype,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = NEPErrorView(*nep,*etype,v);
}

PETSC_EXTERN void PETSC_STDCALL nepvaluesview_(NEP *nep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = NEPValuesView(*nep,v);
}

PETSC_EXTERN void PETSC_STDCALL nepvectorsview_(NEP *nep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = NEPVectorsView(*nep,v);
}

PETSC_EXTERN void PETSC_STDCALL nepsettype_(NEP *nep,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = NEPSetType(*nep,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL nepgettype_(NEP *nep,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  NEPType tname;

  *ierr = NEPGetType(*nep,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL nepsetoptionsprefix_(NEP *nep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = NEPSetOptionsPrefix(*nep,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL nepappendoptionsprefix_(NEP *nep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = NEPAppendOptionsPrefix(*nep,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL nepgetoptionsprefix_(NEP *nep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = NEPGetOptionsPrefix(*nep,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void PETSC_STDCALL nepmonitorset_(NEP *nep,void (PETSC_STDCALL *monitor)(NEP*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mctx);
  CHKFORTRANNULLFUNCTION(monitordestroy);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorall_) {
    *ierr = NEPMonitorSet(*nep,(PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))NEPMonitorAll,*(PetscViewerAndFormat**)mctx,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorconverged_) {
    *ierr = NEPMonitorSet(*nep,(PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))NEPMonitorConverged,*(SlepcConvMonitor*)mctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorfirst_) {
    *ierr = NEPMonitorSet(*nep,(PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))NEPMonitorFirst,*(PetscViewerAndFormat**)mctx,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorlg_) {
    *ierr = NEPMonitorSet(*nep,NEPMonitorLG,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorlgall_) {
    *ierr = NEPMonitorSet(*nep,NEPMonitorLGAll,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitor,(PetscVoidFunction)monitor,mctx); if (*ierr) return;
    if (!monitordestroy) {
      *ierr = NEPMonitorSet(*nep,ourmonitor,*nep,0);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitordestroy,(PetscVoidFunction)monitordestroy,mctx); if (*ierr) return;
      *ierr = NEPMonitorSet(*nep,ourmonitor,*nep,ourdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL nepconvergedabsolute_(NEP *nep,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPConvergedAbsolute(*nep,*eigr,*eigi,*res,errest,ctx);
}

PETSC_EXTERN void PETSC_STDCALL nepconvergedrelative_(NEP *nep,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPConvergedRelative(*nep,*eigr,*eigi,*res,errest,ctx);
}

PETSC_EXTERN void PETSC_STDCALL nepsetconvergencetestfunction_(NEP *nep,void (PETSC_STDCALL *func)(NEP*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *destroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(destroy);
  if ((PetscVoidFunction)func == (PetscVoidFunction)nepconvergedabsolute_) {
    *ierr = NEPSetConvergenceTest(*nep,NEP_CONV_ABS);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)nepconvergedrelative_) {
    *ierr = NEPSetConvergenceTest(*nep,NEP_CONV_REL);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.convergence,(PetscVoidFunction)func,ctx); if (*ierr) return;
    if (!destroy) {
      *ierr = NEPSetConvergenceTestFunction(*nep,ourconvergence,*nep,NULL);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.convdestroy,(PetscVoidFunction)destroy,ctx); if (*ierr) return;
      *ierr = NEPSetConvergenceTestFunction(*nep,ourconvergence,*nep,ourconvdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL nepstoppingbasic_(NEP *nep,PetscInt *its,PetscInt *max_it,PetscInt *nconv,PetscInt *nev,NEPConvergedReason *reason,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPStoppingBasic(*nep,*its,*max_it,*nconv,*nev,reason,ctx);
}

PETSC_EXTERN void PETSC_STDCALL nepsetstoppingtestfunction_(NEP *nep,void (PETSC_STDCALL *func)(NEP*,PetscInt,PetscInt,PetscInt,PetscInt,NEPConvergedReason*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *destroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(destroy);
  if ((PetscVoidFunction)func == (PetscVoidFunction)nepstoppingbasic_) {
    *ierr = NEPSetStoppingTest(*nep,NEP_STOP_BASIC);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.stopping,(PetscVoidFunction)func,ctx); if (*ierr) return;
    if (!destroy) {
      *ierr = NEPSetStoppingTestFunction(*nep,ourstopping,*nep,NULL);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.stopdestroy,(PetscVoidFunction)destroy,ctx); if (*ierr) return;
      *ierr = NEPSetStoppingTestFunction(*nep,ourstopping,*nep,ourstopdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL nepseteigenvaluecomparison_(NEP *nep,void (PETSC_STDCALL *func)(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*),void* ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.comparison,(PetscVoidFunction)func,ctx); if (*ierr) return;
  *ierr = NEPSetEigenvalueComparison(*nep,oureigenvaluecomparison,*nep);
}

