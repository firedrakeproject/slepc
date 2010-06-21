/*
  SLEPc eigensolver: "davidson"

  Step: test for restarting, updateV, restartV

*/

#include "davidson.h"

PetscTruth dvd_isrestarting_fullV(dvdDashboard *d);
PetscInt dvd_managementV_basic_d(dvdDashboard *d);
PetscInt dvd_updateV_extrapol(dvdDashboard *d);
PetscErrorCode dvd_updateV_conv_gen(dvdDashboard *d);
PetscErrorCode dvd_updateV_restart_gen(dvdDashboard *d);
PetscErrorCode dvd_updateV_update_gen(dvdDashboard *d);
PetscInt dvd_updateV_testConv(dvdDashboard *d, PetscInt s, PetscInt pre,
                              PetscInt e, Vec *auxV, PetscScalar *auxS,
                              PetscInt *nConv);
PetscErrorCode dvd_updateV_restartV_aux(Vec *V, PetscInt size_V,
                                        PetscScalar *U, PetscInt ldU,
                                        PetscScalar *pX, PetscInt ldpX,
                                        PetscInt cpX, PetscScalar *oldpX,
                                        PetscInt ldoldpX, PetscInt roldpX,
                                        PetscInt coldpX, PetscScalar *auxS,
                                        PetscInt size_auxS,
                                        PetscInt *new_size_V);
PetscErrorCode dvd_updateV_YtWx(PetscScalar *S, PetscInt ldS,
                                Vec *Y, PetscInt cY, Vec *y, PetscInt cy,
                                Vec *W, PetscInt cW, PetscScalar *x,
                                PetscInt ldx,
                                PetscInt rx, PetscInt cx, Vec *auxV);
typedef struct {
  PetscInt bs,      /* common number of approximated eigenpairs obtained */
    real_max_size_V,
                    /* real max size of V */
    min_size_V,     /* restart with this number of eigenvectors */
    plusk;          /* when restart, save plusk vectors from last iteration */
  Vec *real_V,      /* real start vectors V */
    *new_cY;        /* new left converged eigenvectors from the last iter */
  void
    *old_updateV_data;
                    /* old updateV data */
  isRestarting_type
    old_isRestarting;
                    /* old isRestarting */
  PetscScalar
    *oldU,          /* previous projected right igenvectors */
    *oldV;          /* previous projected left eigenvectors */
  PetscInt
    ldoldU,         /* leading dimension of oldU */
    size_oldU,      /* size of oldU */
    size_new_cY;    /* size of new_cY */
} dvdManagV_basic;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_managementV_basic"
PetscInt dvd_managementV_basic(dvdDashboard *d, dvdBlackboard *b,
                               PetscInt bs, PetscInt max_size_V,
                               PetscInt min_size_V, PetscInt plusk,
                               PetscTruth harm)
{
  PetscErrorCode  ierr;
  dvdManagV_basic *data;
  PetscInt        max_conv = max_size_V;
  PetscInt        i;

  PetscFunctionBegin;

  /* Setting configuration constrains */
  b->max_size_auxV = PetscMax(PetscMax(b->max_size_auxV,
                                       max_conv /* updateV_conv_gen */ ),
                                       2 /* testConv */ );
  b->max_size_X = PetscMax(b->max_size_X, PetscMax(bs, min_size_V));
  b->max_size_auxS = PetscMax(PetscMax(PetscMax(b->max_size_auxS,
                              max_size_V*max_conv /* YtWx */ ),
                              max_size_V*2 /* SlepcDenseOrth  */ ), 
                              max_size_V*b->max_size_X /* testConv:res_0 */ );
  b->max_size_V = max_size_V;
  b->own_vecs+= max_size_V*(harm==PETSC_TRUE?2:1);      /* V, W? */
  b->own_scalars+= b->max_size_V*2 /* eigr, eigr */ +
                   b->max_size_V /* nR */   +
                   b->max_size_V /* nX */   +
                   b->max_size_V /* errest */ +
                   2*b->max_size_V*b->max_size_V*(harm==PETSC_TRUE?2:1)
                                 /* MTX,MTY?,oldU,oldV? */;
//  b->max_size_oldX = plusk;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdManagV_basic), &data); CHKERRQ(ierr);
    data->real_max_size_V = max_size_V;
    data->min_size_V = min_size_V;
    data->real_V = b->free_vecs; b->free_vecs+= max_size_V;
    data->bs = bs;
    data->plusk = plusk;
    data->new_cY = PETSC_NULL;
    data->size_new_cY = 0;

    d->V = data->real_V;
    d->max_size_V = data->real_max_size_V;
    d->cX = data->real_V;
    d->eigr = b->free_scalars; b->free_scalars+= b->max_size_V;
    d->eigi = b->free_scalars; b->free_scalars+= b->max_size_V;
#ifdef PETSC_USE_COMPLEX
    for(i=0; i<b->max_size_V; i++) d->eigi[i] = 0.0;
#endif
    d->nR = (PetscReal*)b->free_scalars;
    b->free_scalars = (PetscScalar*)(d->nR + b->max_size_V);
    for(i=0; i<b->max_size_V; i++) d->nR[i] = PETSC_MAX;
    d->nX = (PetscReal*)b->free_scalars;
    b->free_scalars = (PetscScalar*)(d->nX + b->max_size_V);
    d->errest = (PetscReal*)b->free_scalars;
    b->free_scalars = (PetscScalar*)(d->errest + b->max_size_V);
    d->ceigr = d->eigr;
    d->ceigi = d->eigi;
    d->MTX = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    data->oldU = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    data->ldoldU = 0;
    data->oldV = PETSC_NULL;
    d->W = PETSC_NULL;
    d->MTY = PETSC_NULL;
    d->ldMTY = 0;
    if (harm == PETSC_TRUE) {
      d->W = b->free_vecs; b->free_vecs+= max_size_V;
      d->MTY = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
      data->oldV = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    }

    data->size_oldU = 0;
    data->old_updateV_data = d->updateV_data;
    d->updateV_data = data;
    data->old_isRestarting = d->isRestarting;
    d->isRestarting = dvd_isrestarting_fullV;
    d->updateV = dvd_updateV_extrapol;
    DVD_FL_ADD(d->destroyList, dvd_managementV_basic_d);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_isrestarting_fullV"
PetscTruth dvd_isrestarting_fullV(dvdDashboard *d)
{
  PetscTruth      restart;
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;

  PetscFunctionBegin;

  restart = (d->size_V + data->bs > d->max_size_V)?PETSC_TRUE:PETSC_FALSE;

  /* Check old isRestarting function */
  if ((restart == PETSC_FALSE) && (data->old_isRestarting))
    restart = data->old_isRestarting(d);

  PetscFunctionReturn(restart);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_managementV_basic_d"
PetscInt dvd_managementV_basic_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;

  PetscFunctionBegin;

  /* Restore changes in dvdDashboard */
  d->updateV_data = data->old_updateV_data;
  
  /* Free local data */
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_extrapol"
PetscInt dvd_updateV_extrapol(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /// Temporal! Copy the converged left eigenvectors to cY
  if (data->size_new_cY > 0) {
    for (i=0; i<data->size_new_cY; i++) {
      ierr = VecCopy(data->new_cY[i], d->cY[d->size_cY+i]); CHKERRQ(ierr);
    }
    d->size_cY+= data->size_new_cY;
    data->size_new_cY = 0;
  }

  /* If some eigenpairs were converged, lock them  */
  if (d->npreconv > 0) {
    i = d->npreconv;
    ierr = dvd_updateV_conv_gen(d); CHKERRQ(ierr);

    /* If some eigenpair was locked, exit */
    if (i > d->npreconv) { PetscFunctionReturn(0); }
  }

  /* If the subspaces doesn't need restart, add new vector */
  if (d->isRestarting(d) == PETSC_FALSE) {
    i = d->size_V;
    ierr = dvd_updateV_update_gen(d); CHKERRQ(ierr);

    /* If some vector were add, exit */
    if (i < d->size_V) { PetscFunctionReturn(0); }
  }

  /* Else, a restarting is performed */
  ierr = dvd_updateV_restart_gen(d); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_conv_gen"
PetscErrorCode dvd_updateV_conv_gen(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        i, j, npreconv, ldpX, inc_V, cMT, size_cX, size_cy;
  PetscErrorCode  ierr;
  PetscScalar     *pX;
  PetscReal       norm;
  Vec             *new_cY=0, *cX, *cy;
  PetscTruth      lindep;

  PetscFunctionBegin;

  /* If the left subspace is present, constrains the converged pairs to the
     number of free vectors in V */
  if (d->cY && d->pY)
    npreconv = PetscMin(d->max_size_V-d->size_V, d->npreconv);
  else
    npreconv = d->npreconv;

  /* Constrains the converged pairs to nev */
#ifndef PETSC_USE_COMPLEX
  /* Tries to maintain together conjugate eigenpairs */
  for(i = 0;
      (i + (d->eigi[i]!=0.0?1:0) < npreconv) && (d->nconv + i < d->nev);
      i+= (d->eigi[i]!=0.0?2:1));
  npreconv = i;
#else
  npreconv = PetscMax(PetscMin(d->nev - d->nconv, npreconv), 0);
#endif

  if (d->npreconv == 0) { PetscFunctionReturn(0); }

  d->calcpairs_selectPairs(d, d->size_H);

  /* f.cY <- [f.cY f.V*f.pY(0:npreconv-1)] */
  if (d->cY && d->pY) {
    new_cY = &d->V[d->size_V];
    ierr = SlepcUpdateVectorsZ(new_cY, 0.0, 1.0, d->W?d->W:d->V, d->size_V,
                               d->pY, d->ldpY, d->size_H, npreconv);
    CHKERRQ(ierr);

    /// Temporal! Postpone the copy of new_cY to cY
    data->new_cY = new_cY;
    data->size_new_cY = npreconv;
  }

  /* BcX <- orth(BcX, B*cX),
     auxV = B*X, being X the last converged eigenvectors */
  if (d->BcX) for(i=0; i<npreconv; i++) {
    /* BcX <- [BcX auxV(i)] */
    ierr = VecCopy(d->auxV[i], d->BcX[d->size_cX+i]); CHKERRQ(ierr);
    ierr = IPOrthogonalize(d->ipI, 0, PETSC_NULL, d->size_cX+i, PETSC_NULL,
                           d->BcX, d->BcX[d->size_cX+i], PETSC_NULL,
                           &norm, &lindep); CHKERRQ(ierr);
    if(lindep == PETSC_TRUE) {
        SETERRQ(1, "Error during orth(BcX, B*cX(new))!");
    }
    ierr = VecScale(d->BcX[d->size_cX+i], 1.0/norm); CHKERRQ(ierr);
  }

  /* Harmonics restarts wiht right eigenvectors, and other with
     the left ones */
  pX = (d->W||!d->cY||d->BcX)?d->pX:d->pY;
  ldpX = (d->W||!d->cY||d->BcX)?d->ldpX:d->ldpY;

  /* If BcX, f.V <- orth(BcX, f.V) */ 
  if (d->BcX) cMT = 0;
  else        cMT = d->size_H - npreconv;

  /* f.MTX <- pY(npreconv:size_H-1), f.MTY <- f.pY(npreconv:size_H-1) */
  d->ldMTX = d->ldMTY = d->size_H;
  d->size_MT = d->size_H;
  d->MT_type = DVD_MT_ORTHO;
  ierr = SlepcDenseCopy(d->MTX, d->ldMTX, &pX[ldpX*npreconv], ldpX,
                        d->size_H, cMT); CHKERRQ(ierr);
  if (d->W && d->pY) {
    ierr = SlepcDenseCopy(d->MTY, d->ldMTY, &d->pY[d->ldpY*npreconv], d->ldpY,
                          d->size_H, cMT); CHKERRQ(ierr);
  }

  /* [f.cX(f.nconv) f.V] <- f.V*[f.pX(0:npreconv-1) f.pY(npreconv:f.size_V-1)] */
  if (&d->cX[d->nconv] == d->V) { /* cX and V are contiguous */
    ierr = SlepcDenseCopy(pX, ldpX, d->pX, d->ldpX, d->size_H, npreconv);
    CHKERRQ(ierr);
    ierr = SlepcUpdateVectorsZ(d->V, 0.0, 1.0, d->V, d->size_V, pX,
                               ldpX, d->size_H, d->size_H); CHKERRQ(ierr);
    d->V+= npreconv;
    inc_V = npreconv;
    d->max_size_V-= npreconv;
  } else {
    SETERRQ(1, "Untested case!");
    ierr = SlepcUpdateVectorsZ(&d->cX[d->nconv], 0.0, 1.0, d->V, d->size_V,
                               d->pX, d->ldpX, d->size_H, npreconv);
    CHKERRQ(ierr);
    ierr = SlepcUpdateVectorsZ(d->V, 0.0, 1.0, d->V, d->size_V,
                               &pX[ldpX*npreconv], ldpX,
                               d->size_H, cMT); CHKERRQ(ierr);
    inc_V = 0;
  }
  d->size_cX+= npreconv;
  d->size_V -= npreconv;

  /* Udpate cS and cT, if needed */
  if (d->cS) {
    PetscInt size_cS = d->size_cX-npreconv;
    cX = d->cY?d->cY:d->cX; size_cX = d->cY?d->size_cY:d->size_cX;
    cy = d->cY?new_cY:0; size_cy = d->cY?npreconv:0;
    ierr = 
    dvd_updateV_YtWx(&d->cS[d->ldcS*size_cS], d->ldcS, cX, size_cX, cy,
                     size_cy, d->AV, d->size_AV,
                     d->pX, d->ldpX,
                     d->size_H, npreconv, d->auxV); CHKERRQ(ierr);

    if (DVD_ISNOT(d->sEP, DVD_EP_STD)) {if (d->BV) {
      ierr = 
      dvd_updateV_YtWx(&d->cT[d->ldcT*size_cS], d->ldcT, cX, size_cX, cy,
                       size_cy, d->BV, d->size_AV,
                       d->pX, d->ldpX,
                       d->size_H, npreconv, d->auxV); CHKERRQ(ierr);
    } else if (!d->B) {
      ierr = VecsMultIa(&d->cT[d->ldcT*size_cS], 0, d->ldcT, cX, 0, size_cX,
                        &d->cX[size_cS], 0, npreconv); CHKERRQ(ierr);
      ierr = VecsMultIa(&d->cT[d->ldcT*size_cS+size_cX], 0, d->ldcT, cy, 0,
                        size_cy, &d->cX[size_cS], 0, npreconv); CHKERRQ(ierr);
    } else {
      /* TODO: Only for nprecond==1 */
      ierr = MatMult(d->B, d->cX[d->size_cX-1], d->auxV[0]); CHKERRQ(ierr);
      ierr = VecsMultIa(&d->cT[d->ldcT*size_cS], 0, d->ldcT, cX, 0, size_cX,
                        &d->auxV[0], 0, npreconv); CHKERRQ(ierr);
      ierr = VecsMultIa(&d->cT[d->ldcT*size_cS+size_cX], 0, d->ldcT, cy, 0,
                        size_cy, &d->auxV[0], 0, npreconv); CHKERRQ(ierr);
    }}
  }

  /* f.W <- f.W * f.MTY */
  if (d->W) {
    ierr = SlepcUpdateVectorsZ(d->W, 0.0, 1.0, d->W, d->size_V+npreconv,
                               d->MTY, d->ldMTY, d->size_H, cMT);
    CHKERRQ(ierr);
  }

  /* Lock the converged pairs */
  d->eigr+= npreconv;
#ifndef PETSC_USE_COMPLEX
  if (d->eigi) d->eigi+= npreconv;
#endif
  d->nconv+= npreconv;
  d->errest+= npreconv;

  /* Notify the changes in V and update the other subspaces */
  d->V_imm_s = inc_V;             d->V_imm_e = inc_V;
  d->V_tra_s = 0;                 d->V_tra_e = cMT;
  d->V_new_s = d->V_tra_e;        d->V_new_e = d->size_V;

  /* Remove oldU */
  data->size_oldU = 0;

  /* f.pX <- I, f.pY <- I */
  for(i=0; i<d->size_H; i++) {
    for(j=0; j<d->size_H; j++) {
      d->pX[d->ldpX*i+j] = 0.0;
      if (d->pY) d->pY[d->ldpY*i+j] = 0.0;
    }
    d->pX[d->ldpX*i+i] = 1.0;
    if (d->pY) d->pY[d->ldpY*i+i] = 1.0;
  }

  d->npreconv-= npreconv;

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_restart_gen"
PetscErrorCode dvd_updateV_restart_gen(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        size_plusk, size_X, new_size_X, i, j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Select size_X desired pairs from V */
  size_X = PetscMin(PetscMin(data->min_size_V,
                             d->size_V ),
                             d->max_size_V );

  /* Add plusk eigenvectors from the previous iteration */
  size_plusk = PetscMax(0, PetscMin(PetscMin(data->plusk,
                                    data->size_oldU ),
                                    d->max_size_V - size_X ));

  /* Modify the subspaces */
  d->calcpairs_selectPairs(d, d->size_H);

  d->ldMTX = d->size_MT = d->size_V;
  ierr = dvd_updateV_restartV_aux(d->V, d->size_V, d->MTX, d->ldMTX, d->pX,
                                  d->ldpX, size_X, data->oldU, data->ldoldU,
                                  data->size_oldU, size_plusk, d->auxS,
                                  d->size_auxS, &new_size_X); CHKERRQ(ierr);
  if (d->W && d->pY) {
    PetscInt new_size_Y;
    d->ldMTY = d->size_V;
    ierr = dvd_updateV_restartV_aux(d->W, d->size_V, d->MTY, d->ldMTY, d->pY,
                                    d->ldpY, size_X, data->oldV, data->ldoldU,
                                    data->size_oldU, new_size_X-size_X,
                                    d->auxS, d->size_auxS, &new_size_Y);
    CHKERRQ(ierr);
    new_size_X = PetscMin(new_size_X, new_size_Y);
  }

  /* Notify the changes in V and update the other subspaces */
  d->size_V = new_size_X;
  d->MT_type = DVD_MT_ORTHO;
  d->V_imm_s = 0;                 d->V_imm_e = 0;
  d->V_tra_s = 0;                 d->V_tra_e = new_size_X;
  d->V_new_s = d->V_tra_e;        d->V_new_e = d->V_tra_e;

  /* Remove oldU */
  data->size_oldU = 0;

  /* Remove information about the eigenpairs in V */
  /* Delete: eigr, eigi, errest, npreconv */
  for(i=0; i<d->size_V; i++)
    d->nR[i] = d->errest[i] = PETSC_MAX;
  d->npreconv = 0;
    
  /* f.pX <- I, f.pY <- I */
  for(i=0; i<d->size_H; i++) {
    for(j=0; j<d->size_H; j++) {
      d->pX[d->ldpX*i+j] = 0.0;
      if (d->pY) d->pY[d->ldpY*i+j] = 0.0;
    }
    d->pX[d->ldpX*i+i] = 1.0;
    if (d->pY) d->pY[d->ldpY*i+i] = 1.0;
  }


  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_update_gen"
PetscErrorCode dvd_updateV_update_gen(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        size_D;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  d->calcpairs_selectPairs(d, d->size_H);

  /* Select the desired pairs */
  size_D = PetscMin(PetscMin(PetscMin(data->bs,
                                      d->size_V ),
                                      d->max_size_V-d->size_V ),
                                      d->size_H );
  if (size_D == 0) {
    PetscPrintf(PETSC_COMM_WORLD, "MON: D:%d H:%d\n", size_D, d->size_H);
    d->initV(d);
    d->calcPairs(d);
    //SETERRQ(1, "D == 0!\n");
    //PetscFunctionReturn(1);
  }

//  PetscPrintf(PETSC_COMM_WORLD, "EIGS: ");
//  for(i=0; i<d->size_H; i++) PetscPrintf(PETSC_COMM_WORLD, "%d:%g ", i, d->eigr[i]);
//  PetscPrintf(PETSC_COMM_WORLD, "\n");

  /* Fill V with D */
  d->improveX(d, d->V+d->size_V, d->max_size_V-d->size_V, 0, size_D, &size_D);

  /* If D is empty, exit */
  if (size_D == 0) { PetscFunctionReturn(0); }

  /* Get the converged pairs */
  dvd_updateV_testConv(d, 0, size_D, size_D, d->auxV, d->auxS, &d->npreconv);

  /* Notify the changes in V */
  d->size_V+= size_D;
  d->size_D = size_D;
  d->V_imm_s = 0;                 d->V_imm_e = d->size_V-size_D;
  d->V_tra_s = 0;                 d->V_tra_e = 0;
  d->V_new_s = d->size_V-size_D;  d->V_new_e = d->size_V;

  /* Save the projected eigenvectors */
  if (data->plusk > 0) {
    data->ldoldU = data->size_oldU = d->size_H;
    ierr = SlepcDenseCopy(data->oldU, data->ldoldU, d->pX, d->ldpX, d->size_H,
                          d->size_H); CHKERRQ(ierr);
    if (d->pY && d->W) {
      ierr = SlepcDenseCopy(data->oldV, data->ldoldU, d->pY, d->ldpY, d->size_H,
                            d->size_H); CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_testConv"
PetscInt dvd_updateV_testConv(dvdDashboard *d, PetscInt s, PetscInt pre,
                              PetscInt e, Vec *auxV, PetscScalar *auxS,
			                        PetscInt *nConv)
{
  PetscInt        i;
#ifndef PETSC_USE_COMPLEX
  PetscInt        j;
#endif
  PetscReal       norm;
  PetscErrorCode  ierr;
  PetscTruth      conv;

  PetscFunctionBegin;
  
  *nConv = s;
  for(i=s, conv=PETSC_TRUE; (conv == PETSC_TRUE) && (i < e); i++) {
    if (i >= pre) {
      if ((d->B) && DVD_IS(d->sEP, DVD_EP_STD)) {
        d->calcpairs_X(d, i, i+1, &auxV[1]);
        ierr = MatMult(d->B, auxV[1], auxV[0]); CHKERRQ(ierr); 
      }
      d->calcpairs_residual(d, i, i+1, auxV, auxS, auxV[1]);
    }
    norm = d->nR[i]/d->nX[i];
    conv = d->testConv(d, d->eigr[i], 0, norm, &d->errest[i]);
    if (conv == PETSC_TRUE) *nConv = i+1;
  }

#ifndef PETSC_USE_COMPLEX
  /* Enforce converged conjugate conjugate complex eigenpairs */
  for(j=0; j<*nConv; j++) if(d->eigi[j] != 0.0) j++;
  if(j > *nConv) (*nConv)--;
#endif
  for(; i < e; i++) d->errest[i] = -1.0;
  for(i=PetscMax(pre, *nConv); i<e; i++) d->nR[i] = -1.0;
  
  PetscFunctionReturn(0);
}
EXTERN_C_END


/*
  U <- [pX(0:size_X-1) gs(pX(0:size_X-1), oldpX(0:size_plusk-1))]
  V <- V * U,
  where
  new_size_V, return the new size of V
  auxS, auxiliar vector of size 2*ldpX, at least
  size_auxS, the size of auxS
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_restartV_aux"
PetscErrorCode dvd_updateV_restartV_aux(Vec *V, PetscInt size_V,
                                        PetscScalar *U, PetscInt ldU,
                                        PetscScalar *pX, PetscInt ldpX,
                                        PetscInt cpX, PetscScalar *oldpX,
                                        PetscInt ldoldpX, PetscInt roldpX,
                                        PetscInt coldpX, PetscScalar *auxS,
                                        PetscInt size_auxS,
                                        PetscInt *new_size_V)
{
  PetscInt        i, j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Add the size_X best eigenpairs */
  ierr = SlepcDenseCopy(U, ldU, pX, ldpX, size_V, cpX); CHKERRQ(ierr);

  /* Add plusk eigenvectors from the previous iteration */
  ierr = SlepcDenseCopy(&U[cpX*ldU], ldU, oldpX, ldoldpX, roldpX, coldpX);
  CHKERRQ(ierr);
  for(i=cpX; i<cpX+coldpX; i++)
    for(j=roldpX; j<size_V; j++)
        U[i*ldU+j] = 0.0;

  /* U <- orth(U) */
  /// Temporal! Correct sentence: U <- orth(U(0:size_X-1), U(size_X:size_X+size_plusk))
  if (coldpX > 0) {
    ierr = SlepcDenseOrth(U, ldU, size_V, cpX+coldpX, auxS, size_auxS,
                          new_size_V); CHKERRQ(ierr);
  } else
    *new_size_V = cpX;

  /* V <- V * U */
  ierr = SlepcUpdateVectorsZ(V, 0.0, 1.0, V, size_V, U, ldU, size_V,
                             *new_size_V); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
  Compute S = [ Y' * W * x
                y' * W * x ]
  where
  ldS, the leading dimension of S,
  cY, number of vectors of Y,
  ldy, the leading dimension of y,
  ry,cy, rows and columns of y,
  cW, number of vectors of W,
  ldH, the leading dimension of H,
  rH,cH, rows and columns of H,
  ldx, the leading dimension of y,
  rx,cx, rows and columns of x,
  r, a reduction,
  sr, a permanent space,
  auxV, array of auxiliar vectors of size cx (at the end, auxV <- W*x),
  auxS, auxiliar scalar vector of size rH*cx.
*/
#undef __FUNCT__ 
#define __FUNCT__ "dvd_updateV_YtWx"
PetscErrorCode dvd_updateV_YtWx(PetscScalar *S, PetscInt ldS,
                                Vec *Y, PetscInt cY, Vec *y, PetscInt cy,
                                Vec *W, PetscInt cW, PetscScalar *x,
                                PetscInt ldx,
                                PetscInt rx, PetscInt cx, Vec *auxV)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* auxV <- W * x */
  ierr = SlepcUpdateVectorsZ(auxV, 0.0, 1.0, W, cW, x, ldx, rx, cx);
  CHKERRQ(ierr);

  /* S(0:cY-1, 0:cx-1) <- Y' * auxV */
  ierr = VecsMultIa(S, 0, ldS, Y, 0, cY, auxV, 0, cx); CHKERRQ(ierr);

  /* S(cY:cY+cy-1, 0:cx-1) <- y' * auxV */
  ierr = VecsMultIa(&S[cY], 0, ldS, y, 0, cy, auxV, 0, cx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

