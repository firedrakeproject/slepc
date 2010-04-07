/*
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

#include "private/fortranimpl.h"
#include "slepceps.h"
#include "private/epsimpl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define epsview_                    EPSVIEW
#define epssetoptionsprefix_        EPSSETOPTIONSPREFIX
#define epsappendoptionsprefix_     EPSAPPENDOPTIONSPREFIX
#define epsgetoptionsprefix_        EPSGETOPTIONSPREFIX
#define epscreate_                  EPSCREATE
#define epssettype_                 EPSSETTYPE           
#define epsgettype_                 EPSGETTYPE
#define epsmonitordefault_          EPSMONITORDEFAULT
#define epsmonitorlg_               EPSMONITORLG
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
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsview_                    epsview
#define epssetoptionsprefix_        epssetoptionsprefix
#define epsappendoptionsprefix_     epsappendoptionsprefix
#define epsgetoptionsprefix_        epsgetoptionsprefix
#define epscreate_                  epscreate
#define epssettype_                 epssettype           
#define epsgettype_                 epsgettype
#define epsmonitordefault_          epsmonitordefault
#define epsmonitorlg_               epsmonitorlg
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
#endif

EXTERN_C_BEGIN

/*
   These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code, hence no STDCALL
*/
void epsmonitordefault_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorDefault(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void epsmonitorlg_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorLG(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
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

static PetscErrorCode ourdestroy(void* ctx)
{
  PetscErrorCode ierr = 0;
  EPS            eps = (EPS)ctx;
  void           *mctx = (void*) ((PetscObject)eps)->fortran_func_pointers[1];
  (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)eps)->fortran_func_pointers[2]))(mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

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
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitordefault_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorDefault,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorlg_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorLG,0,0);
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

void PETSC_STDCALL epsgetbalance_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,PetscErrorCode *ierr)
{
  *ierr = EPSGetBalance(*eps,bal,its,cutoff);
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
EXTERN_C_END

