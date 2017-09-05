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
#define pepview_                          PEPVIEW
#define peperrorview_                     PEPERRORVIEW
#define pepreasonview_                    PEPREASONVIEW
#define pepvaluesview_                    PEPVALUESVIEW
#define pepvectorsview_                   PEPVECTORSVIEW
#define pepsetoptionsprefix_              PEPSETOPTIONSPREFIX
#define pepappendoptionsprefix_           PEPAPPENDOPTIONSPREFIX
#define pepgetoptionsprefix_              PEPGETOPTIONSPREFIX
#define pepsettype_                       PEPSETTYPE
#define pepgettype_                       PEPGETTYPE
#define pepmonitorall_                    PEPMONITORALL
#define pepmonitorlg_                     PEPMONITORLG
#define pepmonitorlgall_                  PEPMONITORLGALL
#define pepmonitorset_                    PEPMONITORSET
#define pepmonitorconverged_              PEPMONITORCONVERGED
#define pepmonitorfirst_                  PEPMONITORFIRST
#define pepconvergedabsolute_             PEPCONVERGEDABSOLUTE
#define pepconvergedrelative_             PEPCONVERGEDRELATIVE
#define pepsetconvergencetestfunction_    PEPSETCONVERGENCETESTFUNCTION
#define pepsetstoppingtestfunction_       PEPSETSTOPPINGTESTFUNCTION
#define pepseteigenvaluecomparison_       PEPSETEIGENVALUECOMPARISON
#define pepgetdimensions000_              PEPGETDIMENSIONS000
#define pepgetdimensions100_              PEPGETDIMENSIONS100
#define pepgetdimensions010_              PEPGETDIMENSIONS010
#define pepgetdimensions001_              PEPGETDIMENSIONS001
#define pepgetdimensions110_              PEPGETDIMENSIONS110
#define pepgetdimensions011_              PEPGETDIMENSIONS011
#define pepgetdimensions101_              PEPGETDIMENSIONS101
#define pepgeteigenpair00_                PEPGETEIGENPAIR00
#define pepgeteigenpair10_                PEPGETEIGENPAIR10
#define pepgeteigenpair01_                PEPGETEIGENPAIR01
#define pepgeteigenpair11_                PEPGETEIGENPAIR11
#define pepgettolerances00_               PEPGETTOLERANCES00
#define pepgettolerances10_               PEPGETTOLERANCES10
#define pepgettolerances01_               PEPGETTOLERANCES01
#define pepsetscale_                      PEPSETSCALE
#define pepgetscale000_                   PEPGETSCALE000
#define pepgetscale100_                   PEPGETSCALE100
#define pepgetscale010_                   PEPGETSCALE010
#define pepgetscale001_                   PEPGETSCALE001
#define pepgetscale110_                   PEPGETSCALE110
#define pepgetscale011_                   PEPGETSCALE011
#define pepgetscale101_                   PEPGETSCALE101
#define pepgetscale111_                   PEPGETSCALE111
#define pepgetrefine000_                  PEPGETREFINE000
#define pepgetrefine100_                  PEPGETREFINE100
#define pepgetrefine010_                  PEPGETREFINE010
#define pepgetrefine001_                  PEPGETREFINE001
#define pepgetrefine110_                  PEPGETREFINE110
#define pepgetrefine011_                  PEPGETREFINE011
#define pepgetrefine101_                  PEPGETREFINE101
#define pepgetrefine111_                  PEPGETREFINE111
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pepview_                          pepview
#define peperrorview_                     peperrorview
#define pepreasonview_                    pepreasonview
#define pepvaluesview_                    pepvaluesview
#define pepvectorsview_                   pepvectorsview
#define pepsetoptionsprefix_              pepsetoptionsprefix
#define pepappendoptionsprefix_           pepappendoptionsprefix
#define pepgetoptionsprefix_              pepgetoptionsprefix
#define pepsettype_                       pepsettype
#define pepgettype_                       pepgettype
#define pepmonitorall_                    pepmonitorall
#define pepmonitorlg_                     pepmonitorlg
#define pepmonitorlgall_                  pepmonitorlgall
#define pepmonitorset_                    pepmonitorset
#define pepmonitorconverged_              pepmonitorconverged
#define pepmonitorfirst_                  pepmonitorfirst
#define pepconvergedabsolute_             pepconvergedabsolute
#define pepconvergedrelative_             pepconvergedrelative
#define pepsetconvergencetestfunction_    pepsetconvergencetestfunction
#define pepsetstoppingtestfunction_       pepsetstoppingtestfunction
#define pepseteigenvaluecomparison_       pepseteigenvaluecomparison
#define pepgetdimensions000_              pepgetdimensions000
#define pepgetdimensions100_              pepgetdimensions100
#define pepgetdimensions010_              pepgetdimensions010
#define pepgetdimensions001_              pepgetdimensions001
#define pepgetdimensions110_              pepgetdimensions110
#define pepgetdimensions011_              pepgetdimensions011
#define pepgetdimensions101_              pepgetdimensions101
#define pepgeteigenpair00_                pepgeteigenpair00
#define pepgeteigenpair10_                pepgeteigenpair10
#define pepgeteigenpair01_                pepgeteigenpair01
#define pepgeteigenpair11_                pepgeteigenpair11
#define pepgettolerances00_               pepgettolerances00
#define pepgettolerances10_               pepgettolerances10
#define pepgettolerances01_               pepgettolerances01
#define pepsetscale_                      pepsetscale
#define pepgetscale000_                   pepgetscale000
#define pepgetscale100_                   pepgetscale100
#define pepgetscale010_                   pepgetscale010
#define pepgetscale001_                   pepgetscale001
#define pepgetscale110_                   pepgetscale110
#define pepgetscale011_                   pepgetscale011
#define pepgetscale101_                   pepgetscale101
#define pepgetscale111_                   pepgetscale111
#define pepgetrefine000_                  pepgetrefine000
#define pepgetrefine100_                  pepgetrefine100
#define pepgetrefine010_                  pepgetrefine010
#define pepgetrefine001_                  pepgetrefine001
#define pepgetrefine110_                  pepgetrefine110
#define pepgetrefine011_                  pepgetrefine011
#define pepgetrefine101_                  pepgetrefine101
#define pepgetrefine111_                  pepgetrefine111
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
static PetscErrorCode ourmonitor(PEP pep,PetscInt i,PetscInt nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,PetscInt l,void* ctx)
{
  PetscObjectUseFortranCallback(pep,_cb.monitor,(PEP*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),(&pep,&i,&nc,er,ei,d,&l,_ctx,&ierr));
}

static PetscErrorCode ourdestroy(void** ctx)
{
  PEP pep = (PEP)*ctx;
  PetscObjectUseFortranCallback(pep,_cb.monitordestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

static PetscErrorCode ourconvergence(PEP pep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscObjectUseFortranCallback(pep,_cb.convergence,(PEP*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),(&pep,&eigr,&eigi,&res,errest,_ctx,&ierr));
}

static PetscErrorCode ourconvdestroy(void *ctx)
{
  PEP pep = (PEP)ctx;
  PetscObjectUseFortranCallback(pep,_cb.convdestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

static PetscErrorCode ourstopping(PEP pep,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,PEPConvergedReason *reason,void *ctx)
{
  PetscObjectUseFortranCallback(pep,_cb.stopping,(PEP*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PEPConvergedReason*,void*,PetscErrorCode*),(&pep,&its,&max_it,&nconv,&nev,reason,_ctx,&ierr));
}

static PetscErrorCode ourstopdestroy(void *ctx)
{
  PEP pep = (PEP)ctx;
  PetscObjectUseFortranCallback(pep,_cb.stopdestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

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

PETSC_EXTERN void PETSC_STDCALL pepsettype_(PEP *pep,char *type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PEPSetType(*pep,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL pepgettype_(PEP *pep,char *name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  PEPType tname;

  *ierr = PEPGetType(*pep,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL pepsetoptionsprefix_(PEP *pep,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PEPSetOptionsPrefix(*pep,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL pepappendoptionsprefix_(PEP *pep,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PEPAppendOptionsPrefix(*pep,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL pepgetoptionsprefix_(PEP *pep,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
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
    *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitordestroy,(PetscVoidFunction)monitordestroy,mctx); if (*ierr) return;
    *ierr = PEPMonitorSet(*pep,ourmonitor,*pep,ourdestroy);
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
    *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.convdestroy,(PetscVoidFunction)destroy,ctx); if (*ierr) return;
    *ierr = PEPSetConvergenceTestFunction(*pep,ourconvergence,*pep,ourconvdestroy);
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
    *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.stopdestroy,(PetscVoidFunction)destroy,ctx); if (*ierr) return;
    *ierr = PEPSetStoppingTestFunction(*pep,ourstopping,*pep,ourstopdestroy);
  }
}

PETSC_EXTERN void PETSC_STDCALL pepseteigenvaluecomparison_(PEP *pep,void (PETSC_STDCALL *func)(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*),void* ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*pep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.comparison,(PetscVoidFunction)func,ctx); if (*ierr) return;
  *ierr = PEPSetEigenvalueComparison(*pep,oureigenvaluecomparison,*pep);
}

PETSC_EXTERN void PETSC_STDCALL pepgetdimensions_(PEP *pep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  CHKFORTRANNULLINTEGER(nev);
  CHKFORTRANNULLINTEGER(ncv);
  CHKFORTRANNULLINTEGER(mpd);
  *ierr = PEPGetDimensions(*pep,nev,ncv,mpd);
}

PETSC_EXTERN void PETSC_STDCALL pepgetdimensions000_(PEP *pep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  pepgetdimensions_(pep,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetdimensions100_(PEP *pep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  pepgetdimensions_(pep,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetdimensions010_(PEP *pep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  pepgetdimensions_(pep,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetdimensions001_(PEP *pep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  pepgetdimensions_(pep,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetdimensions110_(PEP *pep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  pepgetdimensions_(pep,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetdimensions011_(PEP *pep,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  pepgetdimensions_(pep,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgeteigenpair_(PEP *pep,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  CHKFORTRANNULLSCALAR(eigr);
  CHKFORTRANNULLSCALAR(eigi);
  CHKFORTRANNULLOBJECTDEREFERENCE(Vr);
  CHKFORTRANNULLOBJECTDEREFERENCE(Vi);
  *ierr = PEPGetEigenpair(*pep,*i,eigr,eigi,*Vr,*Vi);
}

PETSC_EXTERN void PETSC_STDCALL pepgeteigenpair00_(PEP *pep,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  pepgeteigenpair_(pep,i,eigr,eigi,Vr,Vi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgeteigenpair10_(PEP *pep,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  pepgeteigenpair_(pep,i,eigr,eigi,Vr,Vi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgeteigenpair01_(PEP *pep,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  pepgeteigenpair_(pep,i,eigr,eigi,Vr,Vi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgeteigenpair11_(PEP *pep,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  pepgeteigenpair_(pep,i,eigr,eigi,Vr,Vi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgettolerances_(PEP *pep,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  CHKFORTRANNULLREAL(tol);
  CHKFORTRANNULLINTEGER(maxits);
  *ierr = PEPGetTolerances(*pep,tol,maxits);
}

PETSC_EXTERN void PETSC_STDCALL pepgettolerances00_(PEP *pep,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  pepgettolerances_(pep,tol,maxits,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgettolerances10_(PEP *pep,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  pepgettolerances_(pep,tol,maxits,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgettolerances01_(PEP *pep,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  pepgettolerances_(pep,tol,maxits,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepsetscale_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(Dl);
  CHKFORTRANNULLOBJECTDEREFERENCE(Dr);
  *ierr = PEPSetScale(*pep,*scale,*alpha,*Dl,*Dr,*its,*lambda);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  CHKFORTRANNULLREAL(alpha);
  CHKFORTRANNULLINTEGER(its);
  CHKFORTRANNULLREAL(lambda);
  *ierr = PEPGetScale(*pep,scale,alpha,Dl,Dr,its,lambda);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale000_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  pepgetscale_(pep,scale,alpha,Dl,Dr,its,lambda,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale100_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  pepgetscale_(pep,scale,alpha,Dl,Dr,its,lambda,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale010_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  pepgetscale_(pep,scale,alpha,Dl,Dr,its,lambda,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale001_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  pepgetscale_(pep,scale,alpha,Dl,Dr,its,lambda,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale110_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  pepgetscale_(pep,scale,alpha,Dl,Dr,its,lambda,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale011_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  pepgetscale_(pep,scale,alpha,Dl,Dr,its,lambda,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale101_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  pepgetscale_(pep,scale,alpha,Dl,Dr,its,lambda,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetscale111_(PEP *pep,PEPScale *scale,PetscReal *alpha,Vec *Dl,Vec *Dr,PetscInt *its,PetscReal *lambda,int *ierr)
{
  pepgetscale_(pep,scale,alpha,Dl,Dr,its,lambda,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  CHKFORTRANNULLINTEGER(npart);
  CHKFORTRANNULLREAL(tol);
  CHKFORTRANNULLINTEGER(its);
  *ierr = PEPGetRefine(*pep,refine,npart,tol,its,scheme);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine000_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  pepgetrefine_(pep,refine,npart,tol,its,scheme,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine100_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  pepgetrefine_(pep,refine,npart,tol,its,scheme,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine010_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  pepgetrefine_(pep,refine,npart,tol,its,scheme,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine001_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  pepgetrefine_(pep,refine,npart,tol,its,scheme,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine110_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  pepgetrefine_(pep,refine,npart,tol,its,scheme,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine011_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  pepgetrefine_(pep,refine,npart,tol,its,scheme,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine101_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  pepgetrefine_(pep,refine,npart,tol,its,scheme,ierr);
}

PETSC_EXTERN void PETSC_STDCALL pepgetrefine111_(PEP *pep,PEPRefine *refine,PetscInt *npart,PetscReal *tol,PetscInt *its,PEPRefineScheme *scheme,int *ierr)
{
  pepgetrefine_(pep,refine,npart,tol,its,scheme,ierr);
}

