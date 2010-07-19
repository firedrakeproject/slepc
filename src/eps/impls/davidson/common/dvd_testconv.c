/*
  SLEPc eigensolver: "davidson"

  Step: test for convergence

*/

#include "davidson.h"

PetscTruth dvd_testconv_basic_0(dvdDashboard *d, PetscScalar eigvr,
                                PetscScalar eigvi, PetscReal r,
                                PetscReal *err);
PetscTruth dvd_testconv_slepc_0(dvdDashboard *d, PetscScalar eigvr,
                                PetscScalar eigvi, PetscReal r,
                                PetscReal *err);

#undef __FUNCT__  
#define __FUNCT__ "dvd_testconv_basic"
PetscErrorCode dvd_testconv_basic(dvdDashboard *d, dvdBlackboard *b)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    if (d->testConv_data) {
      ierr = PetscFree(d->testConv_data); CHKERRQ(ierr);
    }
    d->testConv_data = PETSC_NULL;
    d->testConv = dvd_testconv_basic_0;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_testconv_basic_0"
PetscTruth dvd_testconv_basic_0(dvdDashboard *d, PetscScalar eigvr,
                                PetscScalar eigvi, PetscReal r,
                                PetscReal *err)
{
  PetscTruth      conv;
  PetscReal       eig_norm, errest;

  PetscFunctionBegin;

  eig_norm = SlepcAbsEigenvalue(eigvr, eigvi);
  //errest = (r < eig_norm) ? r/eig_norm : r;
  errest = r/eig_norm;
  conv = (errest <= d->tol) ? PETSC_TRUE : PETSC_FALSE;
  if (err) *err = errest;

  PetscFunctionReturn(conv);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_testconv_slepc"
PetscErrorCode dvd_testconv_slepc(dvdDashboard *d, dvdBlackboard *b)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    if (d->testConv_data) {
      ierr = PetscFree(d->testConv_data); CHKERRQ(ierr);
    }
    d->testConv_data = PETSC_NULL;
    d->testConv = dvd_testconv_slepc_0;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_testconv_slepc_0"
PetscTruth dvd_testconv_slepc_0(dvdDashboard *d, PetscScalar eigvr,
                                PetscScalar eigvi, PetscReal r,
                                PetscReal *err)
{
  PetscErrorCode  ierr;
  PetscTruth      conv;

  PetscFunctionBegin;

  *err = r;
  ierr = (*d->eps->conv_func)(d->eps, eigvr, eigvi, err, &conv,
                              d->eps->conv_ctx);
  CHKERRABORT(((PetscObject)d->eps)->comm, ierr);

  PetscFunctionReturn(conv);
}

