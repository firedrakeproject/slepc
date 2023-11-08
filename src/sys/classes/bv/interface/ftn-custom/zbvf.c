/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepcbv.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define bvsettype_                BVSETTYPE
#define bvgettype_                BVGETTYPE
#define bvsetoptionsprefix_       BVSETOPTIONSPREFIX
#define bvappendoptionsprefix_    BVAPPENDOPTIONSPREFIX
#define bvgetoptionsprefix_       BVGETOPTIONSPREFIX
#define bvdestroy_                BVDESTROY
#define bvview_                   BVVIEW
#define bvviewfromoptions_        BVVIEWFROMOPTIONS
#define bvsetvectype_             BVSETVECTYPE
#define bvgetvectype_             BVGETVECTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define bvsettype_                bvsettype
#define bvgettype_                bvgettype
#define bvsetoptionsprefix_       bvsetoptionsprefix
#define bvappendoptionsprefix_    bvappendoptionsprefix
#define bvgetoptionsprefix_       bvgetoptionsprefix
#define bvdestroy_                bvdestroy
#define bvview_                   bvview
#define bvviewfromoptions_        bvviewfromoptions
#define bvsetvectype_             bvsetvectype
#define bvgetvectype_             bvgetvectype
#endif

SLEPC_EXTERN void bvsettype_(BV *bv,char *type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = BVSetType(*bv,t);if (*ierr) return;
  FREECHAR(type,t);
}

SLEPC_EXTERN void bvgettype_(BV *bv,char *name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  BVType tname;

  *ierr = BVGetType(*bv,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

SLEPC_EXTERN void bvsetoptionsprefix_(BV *bv,char *prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = BVSetOptionsPrefix(*bv,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

SLEPC_EXTERN void bvappendoptionsprefix_(BV *bv,char *prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = BVAppendOptionsPrefix(*bv,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

SLEPC_EXTERN void bvgetoptionsprefix_(BV *bv,char *prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = BVGetOptionsPrefix(*bv,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

SLEPC_EXTERN void bvdestroy_(BV *bv,PetscErrorCode *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(bv);
  *ierr = BVDestroy(bv); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(bv);
}

SLEPC_EXTERN void bvview_(BV *bv,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = BVView(*bv,v);
}

SLEPC_EXTERN void bvviewfromoptions_(BV *bv,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = BVViewFromOptions(*bv,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

SLEPC_EXTERN void bvsetvectype_(BV *bv,char *vtype,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(vtype,len,t);
  *ierr = BVSetVecType(*bv,t);if (*ierr) return;
  FREECHAR(vtype, t);
}

SLEPC_EXTERN void bvgetvectype_(BV *bv,char *vtype,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = BVGetVecType(*bv,&tname);if (*ierr) return;
  if (vtype!=PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(vtype,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,vtype,len);
}
