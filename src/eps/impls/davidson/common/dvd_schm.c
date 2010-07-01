
#include "davidson.h"

#define DVD_CHECKSUM(b) \
  ( (b)->max_size_V + (b)->max_size_auxV + (b)->max_size_auxS + \
    (b)->own_vecs + (b)->own_scalars + (b)->max_size_oldX )

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_schm_basic_preconf"
PetscErrorCode dvd_schm_basic_preconf(dvdDashboard *d, dvdBlackboard *b,
  PetscInt max_size_V, PetscInt min_size_V, PetscInt bs, PetscInt ini_size_V,
  Vec *initV, PetscInt size_initV, PetscInt plusk, PC pc, HarmType_t harmMode,
  KSP ksp, InitType_t init)
{
  PetscErrorCode ierr;
  PetscInt        check_sum0, check_sum1;

  PetscFunctionBegin;

  ierr = PetscMemzero(b, sizeof(dvdBlackboard)); CHKERRQ(ierr);
  b->state = DVD_STATE_PRECONF;

  for(check_sum0=-1,check_sum1=DVD_CHECKSUM(b); check_sum0 != check_sum1;
      check_sum0 = check_sum1, check_sum1 = DVD_CHECKSUM(b)) {
    b->own_vecs = b->own_scalars = 0;

    /* Setup basic management of V */
    dvd_managementV_basic(d, b, bs, max_size_V, min_size_V, plusk,
                          harmMode==DVD_HARM_NONE?PETSC_FALSE:PETSC_TRUE);
  
    /* Setup the initial subspace for V */
    if (initV) dvd_initV_user(d, b, initV, size_initV, ini_size_V);
    else switch(init) {
    case DVD_INITV_CLASSIC:		dvd_initV_classic(d, b, ini_size_V); break;
    case DVD_INITV_KRYLOV:    dvd_initV_krylov(d, b, ini_size_V); break;
		}
  
    /* Setup the convergence in order to use the SLEPc convergence test */
    dvd_testconv_slepc(d, b);
  
    /* Setup Raileigh-Ritz for selecting the best eigenpairs in V */
    dvd_calcpairs_qz(d, b, PETSC_NULL);
    if (harmMode != DVD_HARM_NONE)
      dvd_harm_conf(d, b, harmMode, PETSC_FALSE, 0.0);
  
    /* Setup the preconditioner */
    dvd_static_precond_PC(d, b, pc);

    /* Setup the method for improving the eigenvectors */
    dvd_improvex_jd(d, b, ksp, bs);
    dvd_improvex_jd_proj_uv(d, b, 0);
    dvd_improvex_jd_lit_const(d, b, 0, 0.0, 0.0);

    /* Setup the profiler */
    dvd_profiler(d, b);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_schm_basic_conf"
PetscErrorCode dvd_schm_basic_conf(dvdDashboard *d, dvdBlackboard *b,
  PetscInt max_size_V, PetscInt min_size_V, PetscInt bs, PetscInt ini_size_V,
  Vec *initV, PetscInt size_initV, PetscInt plusk, PC pc, IP ip,
  HarmType_t harmMode, PetscTruth fixedTarget, PetscScalar t, KSP ksp,
  PetscReal fix, InitType_t init)
{
  PetscInt        check_sum0, check_sum1, maxits;
  Vec             *fv;
  PetscScalar     *fs;
  PetscReal       tol;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  b->state = DVD_STATE_CONF;
  check_sum0 = DVD_CHECKSUM(b);
  b->own_vecs = 0; b->own_scalars = 0;
  fv = b->free_vecs; fs = b->free_scalars;

  /* Setup basic management of V */
  dvd_managementV_basic(d, b, bs, max_size_V, min_size_V, plusk,
                        harmMode==DVD_HARM_NONE?PETSC_FALSE:PETSC_TRUE);

  /* Setup the initial subspace for V */
  if (initV) dvd_initV_user(d, b, initV, size_initV, ini_size_V);
  else switch(init) {
  case DVD_INITV_CLASSIC:     dvd_initV_classic(d, b, ini_size_V); break;
  case DVD_INITV_KRYLOV:      dvd_initV_krylov(d, b, ini_size_V); break;
  }

  /* Setup the convergence in order to use the SLEPc convergence test */
  dvd_testconv_slepc(d, b);

  /* Setup Raileigh-Ritz for selecting the best eigenpairs in V */
  dvd_calcpairs_qz(d, b, ip);
  if (harmMode != DVD_HARM_NONE)
    dvd_harm_conf(d, b, harmMode, fixedTarget, t);

  /* Setup the preconditioner */
  dvd_static_precond_PC(d, b, pc);

  /* Setup the method for improving the eigenvectors */
  dvd_improvex_jd(d, b, ksp, bs);
  dvd_improvex_jd_proj_uv(d, b, DVD_IS(d->sEP, DVD_EP_HERMITIAN)?
                                                DVD_PROJ_KBXZ:DVD_PROJ_KBXY);
  ierr = KSPGetTolerances(ksp, &tol, PETSC_NULL, PETSC_NULL, &maxits);
  CHKERRQ(ierr);
  dvd_improvex_jd_lit_const(d, b, maxits, tol, fix);

  /* Setup the profiler */
  dvd_profiler(d, b);

  check_sum1 = DVD_CHECKSUM(b);
  if ((check_sum0 != check_sum1) ||
      (b->free_vecs - fv > b->own_vecs) ||
      (b->free_scalars - fs > b->own_scalars))
    SETERRQ(1, "Something awful happened!");
    
  PetscFunctionReturn(0);
}
EXTERN_C_END

