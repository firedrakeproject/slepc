/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <private/fortranimpl.h>
#include <private/epsimpl.h>        /*I "slepceps.h" I*/

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define epsdestroy_                 EPSDESTROY
#define epsview_                    EPSVIEW
#define epssetoptionsprefix_        EPSSETOPTIONSPREFIX
#define epsappendoptionsprefix_     EPSAPPENDOPTIONSPREFIX
#define epsgetoptionsprefix_        EPSGETOPTIONSPREFIX
#define epscreate_                  EPSCREATE
#define epssettype_                 EPSSETTYPE           
#define epsgettype_                 EPSGETTYPE
#define epsmonitorall_              EPSMONITORALL
#define epsmonitorlg_               EPSMONITORLG
#define epsmonitorlgall_            EPSMONITORLGALL
#define epsmonitorset_              EPSMONITORSET
#define epsmonitorconverged_        EPSMONITORCONVERGED
#define epsmonitorfirst_            EPSMONITORFIRST
#define epsgetst_                   EPSGETST
#define epsgetip_                   EPSGETIP
#define epsgetwhicheigenpairs_      EPSGETWHICHEIGENPAIRS
#define epsgetproblemtype_          EPSGETPROBLEMTYPE
#define epsgetextraction_           EPSGETEXTRACTION
#define epsgetbalance_              EPSGETBALANCE
#define epsgetconvergedreason_      EPSGETCONVERGEDREASON
#define epspowergetshifttype_       EPSPOWERGETSHIFTTYPE
#define epslanczosgetreorthog_      EPSLANCZOSGETREORTHOG
#define epsabsoluteconverged_       EPSABSOLUTECONVERGED
#define epseigrelativeconverged_    EPSEIGRELATIVECONVERGED
#define epsnormrelativeconverged_   EPSNORMRELATIVECONVERGED
#define epssetconvergencetestfunction_ EPSSETCONVERGENCETESTFUNCTION
#define epsseteigenvaluecomparison_ EPSSETEIGENVALUECOMPARISON
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsdestroy_                 epsdestroy
#define epsview_                    epsview
#define epssetoptionsprefix_        epssetoptionsprefix
#define epsappendoptionsprefix_     epsappendoptionsprefix
#define epsgetoptionsprefix_        epsgetoptionsprefix
#define epscreate_                  epscreate
#define epssettype_                 epssettype           
#define epsgettype_                 epsgettype
#define epsmonitorall_              epsmonitorall
#define epsmonitorlg_               epsmonitorlg
#define epsmonitorlgall_            epsmonitorlgall
#define epsmonitorset_              epsmonitorset
#define epsmonitorconverged_        epsmonitorconverged
#define epsmonitorfirst_            epsmonitorfirst
#define epsgetst_                   epsgetst
#define epsgetip_                   epsgetip
#define epsgetwhicheigenpairs_      epsgetwhicheigenpairs
#define epsgetproblemtype_          epsgetproblemtype
#define epsgetextraction_           epsgetextraction
#define epsgetbalance_              epsgetbalance
#define epsgetconvergedreason_      epsgetconvergedreason
#define epspowergetshifttype_       epspowergetshifttype
#define epslanczosgetreorthog_      epslanczosgetreorthog
#define epsabsoluteconverged_       epsabsoluteconverged
#define epseigrelativeconverged_    epseigrelativeconverged
#define epsnormrelativeconverged_   epsnormrelativeconverged
#define epssetconvergencetestfunction_ epssetconvergencetestfunction
#define epsseteigenvaluecomparison_ epsseteigenvaluecomparison
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
 
/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmonitor(EPS eps,PetscInt i,PetscInt nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,PetscInt l,void* ctx)
{
  PetscErrorCode ierr = 0;
  void           *mctx = (void*) ((PetscObject)eps)->fortran_func_pointers[1];
  (*(void (PETSC_STDCALL *)(EPS*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*))
    (((PetscObject)eps)->fortran_func_pointers[0]))(&eps,&i,&nc,er,ei,d,&l,mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourdestroy(void** ctx)
{
  PetscErrorCode ierr = 0;
  EPS            eps = *(EPS*)ctx;
  void           *mctx = (void*) ((PetscObject)eps)->fortran_func_pointers[1];
  (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)eps)->fortran_func_pointers[2]))(mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL epsdestroy_(EPS *eps, PetscErrorCode *ierr)
{
  *ierr = EPSDestroy(eps);
}

void PETSC_STDCALL epsview_(EPS *eps,PetscViewer *viewer, PetscErrorCode *ierr)
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
  const EPSType tname;

  *ierr = EPSGetType(*eps,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
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

void PETSC_STDCALL epsmonitorset_(EPS *eps,void (PETSC_STDCALL *monitor)(EPS*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),
                                  void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode *),PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(monitordestroy);
  PetscObjectAllocateFortranPointers(*eps,3);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorall_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorAll,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorlg_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorLG,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorlgall_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorLGAll,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorconverged_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorConverged,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorfirst_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorFirst,0,0);
  } else {
    ((PetscObject)*eps)->fortran_func_pointers[0] = (PetscVoidFunction)monitor;
    ((PetscObject)*eps)->fortran_func_pointers[1] = (PetscVoidFunction)mctx;
    if (FORTRANNULLFUNCTION(monitordestroy)) {
      *ierr = EPSMonitorSet(*eps,ourmonitor,*eps,0);
    } else {
      ((PetscObject)*eps)->fortran_func_pointers[2] = (PetscVoidFunction)monitordestroy;
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

void PETSC_STDCALL epsabsoluteconverged_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSAbsoluteConverged(*eps,*eigr,*eigi,*res,errest,ctx);
}

void PETSC_STDCALL epseigrelativeconverged_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSEigRelativeConverged(*eps,*eigr,*eigi,*res,errest,ctx);
}

void PETSC_STDCALL epsnormrelativeconverged_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSNormRelativeConverged(*eps,*eigr,*eigi,*res,errest,ctx);
}

EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourconvergence(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscErrorCode ierr = 0;
  void           *mctx = (void*) ((PetscObject)eps)->fortran_func_pointers[4];
  (*(void (PETSC_STDCALL *)(EPS*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*))
   (((PetscObject)eps)->fortran_func_pointers[3]))(&eps,&eigr,&eigi,&res,errest,mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL epssetconvergencetestfunction_(EPS *eps,void (PETSC_STDCALL *func)(EPS*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void* ctx,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*eps,5);
  if ((PetscVoidFunction)func == (PetscVoidFunction)epsabsoluteconverged_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_ABS);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)epseigrelativeconverged_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_EIG);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)epsnormrelativeconverged_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_NORM);
  } else {
    ((PetscObject)*eps)->fortran_func_pointers[3] = (PetscVoidFunction)func;
    ((PetscObject)*eps)->fortran_func_pointers[4] = (PetscVoidFunction)ctx;
    *ierr = EPSSetConvergenceTestFunction(*eps,ourconvergence,PETSC_NULL);
  }
}

EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode oureigenvaluecomparison(EPS eps,PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  PetscErrorCode ierr = 0;
  void           *mctx = (void*) ((PetscObject)eps)->fortran_func_pointers[6];
  (*(void (PETSC_STDCALL *)(EPS*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*))
   (((PetscObject)eps)->fortran_func_pointers[5]))(&eps,&ar,&ai,&br,&bi,r,mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL epsseteigenvaluecomparison_(EPS *eps,void (PETSC_STDCALL *func)(EPS*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*),void* ctx,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*eps,7);
  ((PetscObject)*eps)->fortran_func_pointers[5] = (PetscVoidFunction)func;
  ((PetscObject)*eps)->fortran_func_pointers[6] = (PetscVoidFunction)ctx;
  *ierr = EPSSetEigenvalueComparison(*eps,oureigenvaluecomparison,PETSC_NULL);
}

EXTERN_C_END
