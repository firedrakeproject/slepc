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
#include <slepc/private/pepimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pepview_                    PEPVIEW
#define peperrorview_               PEPERRORVIEW
#define pepreasonview_              PEPREASONVIEW
#define pepvaluesview_              PEPVALUESVIEW
#define pepvectorsview_             PEPVECTORSVIEW
#define pepsetoptionsprefix_        PEPSETOPTIONSPREFIX
#define pepappendoptionsprefix_     PEPAPPENDOPTIONSPREFIX
#define pepgetoptionsprefix_        PEPGETOPTIONSPREFIX
#define pepsettype_                 PEPSETTYPE
#define pepgettype_                 PEPGETTYPE
#define pepmonitorall_              PEPMONITORALL
#define pepmonitorlg_               PEPMONITORLG
#define pepmonitorlgall_            PEPMONITORLGALL
#define pepmonitorset_              PEPMONITORSET
#define pepmonitorconverged_        PEPMONITORCONVERGED
#define pepmonitorfirst_            PEPMONITORFIRST
#define pepconvergedabsolute_       PEPCONVERGEDABSOLUTE
#define pepconvergedrelative_       PEPCONVERGEDRELATIVE
#define pepsetconvergencetestfunction_ PEPSETCONVERGENCETESTFUNCTION
#define pepsetstoppingtestfunction_ PEPSETSTOPPINGTESTFUNCTION
#define pepseteigenvaluecomparison_ PEPSETEIGENVALUECOMPARISON
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pepview_                    pepview
#define peperrorview_               peperrorview
#define pepreasonview_              pepreasonview
#define pepvaluesview_              pepvaluesview
#define pepvectorsview_             pepvectorsview
#define pepsetoptionsprefix_        pepsetoptionsprefix
#define pepappendoptionsprefix_     pepappendoptionsprefix
#define pepgetoptionsprefix_        pepgetoptionsprefix
#define pepsettype_                 pepsettype
#define pepgettype_                 pepgettype
#define pepmonitorall_              pepmonitorall
#define pepmonitorlg_               pepmonitorlg
#define pepmonitorlgall_            pepmonitorlgall
#define pepmonitorset_              pepmonitorset
#define pepmonitorconverged_        pepmonitorconverged
#define pepmonitorfirst_            pepmonitorfirst
#define pepconvergedabsolute_       pepconvergedabsolute
#define pepconvergedrelative_       pepconvergedrelative
#define pepsetconvergencetestfunction_ pepsetconvergencetestfunction
#define pepsetstoppingtestfunction_ pepsetstoppingtestfunction
#define pepseteigenvaluecomparison_ pepseteigenvaluecomparison
#endif

/*
   These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code, hence no STDCALL
*/
PETSC_EXTERN void pepmonitorall_(PEP *pep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,PetscViewerAndFormat **ctx,PetscErrorCode *ierr)
{
  *ierr = PEPMonitorAll(*pep,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void pepmonitorconverged_(PEP *pep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,SlepcConvMonitor *ctx,PetscErrorCode *ierr)
{
  *ierr = PEPMonitorConverged(*pep,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void pepmonitorfirst_(PEP *pep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,PetscViewerAndFormat **ctx,PetscErrorCode *ierr)
{
  *ierr = PEPMonitorFirst(*pep,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void pepmonitorlg_(PEP *pep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PEPMonitorLG(*pep,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

PETSC_EXTERN void pepmonitorlgall_(PEP *pep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PEPMonitorLGAll(*pep,*it,*nconv,eigr,eigi,errest,*nest,ctx);
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
static PetscErrorCode ourmonitor(PEP pep,PetscInt i,PetscInt nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,PetscInt l,void* ctx)
{
  PetscObjectUseFortranCallback(pep,_cb.monitor,(PEP*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),(&pep,&i,&nc,er,ei,d,&l,_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourdestroy"
static PetscErrorCode ourdestroy(void** ctx)
{
  PEP pep = (PEP)*ctx;
  PetscObjectUseFortranCallback(pep,_cb.monitordestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourconvergence"
static PetscErrorCode ourconvergence(PEP pep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscObjectUseFortranCallback(pep,_cb.convergence,(PEP*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),(&pep,&eigr,&eigi,&res,errest,_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourconvdestroy"
static PetscErrorCode ourconvdestroy(void *ctx)
{
  PEP pep = (PEP)ctx;
  PetscObjectUseFortranCallback(pep,_cb.convdestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourstopping"
static PetscErrorCode ourstopping(PEP pep,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,PEPConvergedReason *reason,void *ctx)
{
  PetscObjectUseFortranCallback(pep,_cb.stopping,(PEP*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PEPConvergedReason*,void*,PetscErrorCode*),(&pep,&its,&max_it,&nconv,&nev,reason,_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourstopdestroy"
static PetscErrorCode ourstopdestroy(void *ctx)
{
  PEP pep = (PEP)ctx;
  PetscObjectUseFortranCallback(pep,_cb.stopdestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "oureigenvaluecomparison"
static PetscErrorCode oureigenvaluecomparison(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  PEP eps = (PEP)ctx;
  PetscObjectUseFortranCallback(eps,_cb.comparison,(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*),(&ar,&ai,&br,&bi,r,_ctx,&ierr));
}

PETSC_EXTERN void PETSC_STDCALL pepview_(PEP *pep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PEPView(*pep,v);
}

PETSC_EXTERN void PETSC_STDCALL pepreasonview_(PEP *pep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PEPReasonView(*pep,v);
}

PETSC_EXTERN void PETSC_STDCALL peperrorview_(PEP *pep,PEPErrorType *etype,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PEPErrorView(*pep,*etype,v);
}

PETSC_EXTERN void PETSC_STDCALL pepvaluesview_(PEP *pep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PEPValuesView(*pep,v);
}

PETSC_EXTERN void PETSC_STDCALL pepvectorsview_(PEP *pep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PEPVectorsView(*pep,v);
}

PETSC_EXTERN void PETSC_STDCALL pepsettype_(PEP *pep,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PEPSetType(*pep,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL pepgettype_(PEP *pep,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  PEPType tname;

  *ierr = PEPGetType(*pep,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL pepsetoptionsprefix_(PEP *pep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PEPSetOptionsPrefix(*pep,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL pepappendoptionsprefix_(PEP *pep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PEPAppendOptionsPrefix(*pep,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL pepgetoptionsprefix_(PEP *pep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PEPGetOptionsPrefix(*pep,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void PETSC_STDCALL pepmonitorset_(PEP *pep,void (PETSC_STDCALL *monitor)(PEP*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mctx);
  CHKFORTRANNULLFUNCTION(monitordestroy);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)pepmonitorall_) {
    *ierr = PEPMonitorSet(*pep,(PetscErrorCode (*)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))PEPMonitorAll,*(PetscViewerAndFormat**)mctx,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)pepmonitorconverged_) {
    *ierr = PEPMonitorSet(*pep,(PetscErrorCode (*)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))PEPMonitorConverged,*(SlepcConvMonitor*)mctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)pepmonitorfirst_) {
    *ierr = PEPMonitorSet(*pep,(PetscErrorCode (*)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))PEPMonitorFirst,*(PetscViewerAndFormat**)mctx,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)pepmonitorlg_) {
    *ierr = PEPMonitorSet(*pep,PEPMonitorLG,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)pepmonitorlgall_) {
    *ierr = PEPMonitorSet(*pep,PEPMonitorLGAll,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitor,(PetscVoidFunction)monitor,mctx); if (*ierr) return;
    if (!monitordestroy) {
      *ierr = PEPMonitorSet(*pep,ourmonitor,*pep,0);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitordestroy,(PetscVoidFunction)monitordestroy,mctx); if (*ierr) return;
      *ierr = PEPMonitorSet(*pep,ourmonitor,*pep,ourdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL pepconvergedabsolute_(PEP *pep,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PEPConvergedAbsolute(*pep,*eigr,*eigi,*res,errest,ctx);
}

PETSC_EXTERN void PETSC_STDCALL pepconvergedrelative_(PEP *pep,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PEPConvergedRelative(*pep,*eigr,*eigi,*res,errest,ctx);
}

PETSC_EXTERN void PETSC_STDCALL pepsetconvergencetestfunction_(PEP *pep,void (PETSC_STDCALL *func)(PEP*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *destroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(destroy);
  if ((PetscVoidFunction)func == (PetscVoidFunction)pepconvergedabsolute_) {
    *ierr = PEPSetConvergenceTest(*pep,PEP_CONV_ABS);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)pepconvergedrelative_) {
    *ierr = PEPSetConvergenceTest(*pep,PEP_CONV_REL);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.convergence,(PetscVoidFunction)func,ctx); if (*ierr) return;
    if (!destroy) {
      *ierr = PEPSetConvergenceTestFunction(*pep,ourconvergence,*pep,NULL);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.convdestroy,(PetscVoidFunction)destroy,ctx); if (*ierr) return;
      *ierr = PEPSetConvergenceTestFunction(*pep,ourconvergence,*pep,ourconvdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL pepstoppingbasic_(PEP *pep,PetscInt *its,PetscInt *max_it,PetscInt *nconv,PetscInt *nev,PEPConvergedReason *reason,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PEPStoppingBasic(*pep,*its,*max_it,*nconv,*nev,reason,ctx);
}

PETSC_EXTERN void PETSC_STDCALL pepsetstoppingtestfunction_(PEP *pep,void (PETSC_STDCALL *func)(PEP*,PetscInt,PetscInt,PetscInt,PetscInt,PEPConvergedReason*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *destroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(destroy);
  if ((PetscVoidFunction)func == (PetscVoidFunction)pepstoppingbasic_) {
    *ierr = PEPSetStoppingTest(*pep,PEP_STOP_BASIC);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.stopping,(PetscVoidFunction)func,ctx); if (*ierr) return;
    if (!destroy) {
      *ierr = PEPSetStoppingTestFunction(*pep,ourstopping,*pep,NULL);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.stopdestroy,(PetscVoidFunction)destroy,ctx); if (*ierr) return;
      *ierr = PEPSetStoppingTestFunction(*pep,ourstopping,*pep,ourstopdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL pepseteigenvaluecomparison_(PEP *pep,void (PETSC_STDCALL *func)(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*),void* ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.comparison,(PetscVoidFunction)func,ctx); if (*ierr) return;
  *ierr = PEPSetEigenvalueComparison(*pep,oureigenvaluecomparison,*pep);
}

