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
#include "slepcqep.h"
#include "private/qepimpl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define qepview_                    QEPVIEW
#define qepsetoptionsprefix_        QEPSETOPTIONSPREFIX
#define qepappendoptionsprefix_     QEPAPPENDOPTIONSPREFIX
#define qepgetoptionsprefix_        QEPGETOPTIONSPREFIX
#define qepcreate_                  QEPCREATE
#define qepsettype_                 QEPSETTYPE           
#define qepgettype_                 QEPGETTYPE
#define qepmonitordefault_          QEPMONITORDEFAULT
#define qepmonitorlg_               QEPMONITORLG
#define qepmonitorset_              QEPMONITORSET
#define qepmonitorconverged_        QEPMONITORCONVERGED
#define qepmonitorfirst_            QEPMONITORFIRST
#define qepgetip_                   QEPGETIP
#define qepgetwhicheigenpairs_      QEPGETWHICHEIGENPAIRS
#define qepgetproblemtype_          QEPGETPROBLEMTYPE
#define qepgetconvergedreason_      QEPGETCONVERGEDREASON
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define qepview_                    qepview
#define qepsetoptionsprefix_        qepsetoptionsprefix
#define qepappendoptionsprefix_     qepappendoptionsprefix
#define qepgetoptionsprefix_        qepgetoptionsprefix
#define qepcreate_                  qepcreate
#define qepsettype_                 qepsettype           
#define qepgettype_                 qepgettype
#define qepmonitordefault_          qepmonitordefault
#define qepmonitorset_              qepmonitorset
#define qepmonitorconverged_        qepmonitorconverged
#define qepmonitorfirst_            qepmonitorfirst
#define qepgetip_                   qepgetip
#define qepgetwhicheigenpairs_      qepgetwhicheigenpairs
#define qepgetproblemtype_          qepgetproblemtype
#define qepgetconvergedreason_      qepgetconvergedreason
#endif

EXTERN_C_BEGIN

/*
   These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code, hence no STDCALL
*/
void qepmonitordefault_(QEP *qep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = QEPMonitorDefault(*qep,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void qepmonitorlg_(QEP *qep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = QEPMonitorLG(*qep,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void qepmonitorconverged_(QEP *qep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = QEPMonitorConverged(*qep,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void qepmonitorfirst_(QEP *qep,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = QEPMonitorFirst(*qep,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

EXTERN_C_END
 
/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmonitor(QEP qep,PetscInt i,PetscInt nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,PetscInt l,void* ctx)
{
  PetscErrorCode ierr = 0;
  void           *mctx = (void*) ((PetscObject)qep)->fortran_func_pointers[1];
  (*(void (PETSC_STDCALL *)(QEP*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*))
    (((PetscObject)qep)->fortran_func_pointers[0]))(&qep,&i,&nc,er,ei,d,&l,mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourdestroy(void* ctx)
{
  PetscErrorCode ierr = 0;
  QEP            qep = (QEP)ctx;
  void           *mctx = (void*) ((PetscObject)qep)->fortran_func_pointers[1];
  (*(void (PETSC_STDCALL *)(void*,PetscErrorCode*))(((PetscObject)qep)->fortran_func_pointers[2]))(mctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL qepview_(QEP *qep,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = QEPView(*qep,v);
}

void PETSC_STDCALL qepsettype_(QEP *qep,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = QEPSetType(*qep,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL qepgettype_(QEP *qep,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const QEPType tname;

  *ierr = QEPGetType(*qep,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

void PETSC_STDCALL qepsetoptionsprefix_(QEP *qep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = QEPSetOptionsPrefix(*qep,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL qepappendoptionsprefix_(QEP *qep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = QEPAppendOptionsPrefix(*qep,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL qepcreate_(MPI_Fint *comm,QEP *qep,PetscErrorCode *ierr)
{
  *ierr = QEPCreate(MPI_Comm_f2c(*(comm)),qep);
}

void PETSC_STDCALL qepmonitorset_(QEP *qep,void (PETSC_STDCALL *monitor)(QEP*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),
                                  void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode *),PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(monitordestroy);
  PetscObjectAllocateFortranPointers(*qep,3);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)qepmonitordefault_) {
    *ierr = QEPMonitorSet(*qep,QEPMonitorDefault,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)qepmonitorlg_) {
    *ierr = QEPMonitorSet(*qep,QEPMonitorLG,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)qepmonitorconverged_) {
    *ierr = QEPMonitorSet(*qep,QEPMonitorConverged,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)qepmonitorfirst_) {
    *ierr = QEPMonitorSet(*qep,QEPMonitorFirst,0,0);
  } else {
    ((PetscObject)*qep)->fortran_func_pointers[0] = (PetscVoidFunction)monitor;
    ((PetscObject)*qep)->fortran_func_pointers[1] = (PetscVoidFunction)mctx;
    if (FORTRANNULLFUNCTION(monitordestroy)) {
      *ierr = QEPMonitorSet(*qep,ourmonitor,*qep,0);
    } else {
      ((PetscObject)*qep)->fortran_func_pointers[2] = (PetscVoidFunction)monitordestroy;
      *ierr = QEPMonitorSet(*qep,ourmonitor,*qep,ourdestroy);
    }
  }
}

void PETSC_STDCALL qepgetoptionsprefix_(QEP *qep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = QEPGetOptionsPrefix(*qep,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
}

void PETSC_STDCALL qepgetip_(QEP *qep,IP *ip,PetscErrorCode *ierr)
{
  *ierr = QEPGetIP(*qep,ip);
}

void PETSC_STDCALL qepgetwhicheigenpairs_(QEP *qep,QEPWhich *which,PetscErrorCode *ierr)
{
  *ierr = QEPGetWhichEigenpairs(*qep,which);
}

void PETSC_STDCALL qepgetproblemtype_(QEP *qep,QEPProblemType *type,PetscErrorCode *ierr)
{
  *ierr = QEPGetProblemType(*qep,type);
}

void PETSC_STDCALL qepgetconvergedreason_(QEP *qep,QEPConvergedReason *reason,PetscErrorCode *ierr)
{
  *ierr = QEPGetConvergedReason(*qep,reason);
}

EXTERN_C_END

