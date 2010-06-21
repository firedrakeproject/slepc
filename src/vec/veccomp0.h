
#include "petscmat.h"
#include "private/vecimpl.h"     /*I  "vec.h"  I*/

#ifdef __WITH_MPI__
#define __SUF__(A) A##_MPI
#else
#define __SUF__(A) A##_Seq
#endif
#define __QUOTEME(x) #x
#define __SUF_C__(A) __QUOTEME(__SUF__(A))


#undef __FUNCT__  
#define __FUNCT__ __SUF_C__(VecDot_Comp)
PetscErrorCode __SUF__(VecDot_Comp)(Vec a, Vec b, PetscScalar *z)
{
  PetscScalar    sum = 0.0, work;
  PetscInt       i;
  PetscErrorCode ierr;
  Vec_Comp       *as = (Vec_Comp*)a->data,
                 *bs = (Vec_Comp*)b->data;

  PetscFunctionBegin;

  PetscValidVecComp(a);
  PetscValidVecComp(b);

  if (as->x[0]->ops->dot_local) {
    for(i=0, sum=0.0; i<as->n->n; i++) {
      ierr = as->x[i]->ops->dot_local(as->x[i], bs->x[i], &work); CHKERRQ(ierr);
      sum+= work;
    }
#ifdef __WITH_MPI__
    work = sum;
    ierr = MPI_Allreduce(&work, &sum, 1, MPIU_SCALAR, MPIU_SUM,
                         ((PetscObject)a)->comm); CHKERRQ(ierr);
#endif
  } else {
    for(i=0, sum=0.0; i<as->n->n; i++) {
      ierr = VecDot(as->x[i], bs->x[i], &work); CHKERRQ(ierr);
      sum+= work;
    }
  }
  *z = sum;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ __SUF_C__(VecMDot_Comp)
PetscErrorCode __SUF__(VecMDot_Comp)(Vec a, PetscInt n, const Vec b[],
                                     PetscScalar *z)
{
  PetscScalar     *work, *work0, *r;
  PetscErrorCode  ierr;
  Vec_Comp        *as = (Vec_Comp*)a->data;
  Vec             *bx;
  PetscInt        i, j;

  PetscFunctionBegin;

  PetscValidVecComp(a);
  for(i=0; i<n; i++) PetscValidVecComp(b[i]);

  if(as->n->n == 0) {
    *z = 0;
    PetscFunctionReturn(0);
  }

  ierr = PetscMalloc(sizeof(PetscScalar)*n, &work0); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*n, &bx); CHKERRQ(ierr);

#ifdef __WITH_MPI__
  if (as->x[0]->ops->mdot_local) {
    r = work0; work = z;
  } else
#endif
  {
    r = z; work = work0;
  }

  /* z[i] <- a.x' * b[i].x */
  for(i=0; i<n; i++) bx[i] = ((Vec_Comp*)b[i]->data)->x[0];
  if (as->x[0]->ops->mdot_local) {
    ierr = as->x[0]->ops->mdot_local(as->x[0], n, bx, r); CHKERRQ(ierr);
  } else {
    ierr = VecMDot(as->x[0], n, bx, r); CHKERRQ(ierr);
  }
  for(j=0; j<as->n->n; j++) {
    for(i=0; i<n; i++) bx[i] = ((Vec_Comp*)b[i]->data)->x[j];
    if (as->x[0]->ops->mdot_local) {
      ierr = as->x[j]->ops->mdot_local(as->x[j], n, bx, work); CHKERRQ(ierr);
    } else {
      ierr = VecMDot(as->x[j], n, bx, work); CHKERRQ(ierr);
    }
    for(i=0; i<n; i++) r[i]+= work[i];
  }

  /* If def(__WITH_MPI__) and exists mdot_local */
#ifdef __WITH_MPI__
  if (as->x[0]->ops->mdot_local) {
    /* z[i] <- Allreduce(work[i]) */
    ierr = MPI_Allreduce(r, z, n, MPIU_SCALAR, MPIU_SUM,
                         ((PetscObject)a)->comm); CHKERRQ(ierr);
  }
#endif

  ierr = PetscFree(work0); CHKERRQ(ierr);
  ierr = PetscFree(bx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#define __UNNORM__(A,T) \
  switch(T) { \
  case NORM_2: case NORM_FROBENIUS: (A) = (A)*(A); break; \
  case NORM_1_AND_2: (&(A))[1] = ((&(A))[1])*((&(A))[1]); break; \
  default: ; \
  }

#define __NORM__(A,T) \
  switch(T) { \
  case NORM_2: case NORM_FROBENIUS: (A) = sqrt(A); break; \
  case NORM_1_AND_2: (&(A))[1] = sqrt((&(A))[1]); break; \
  default: ; \
  }

#define __SUM_NORMS__(A,B,T) \
  switch(T) { \
  case NORM_1: case NORM_2: case NORM_FROBENIUS: (A) = (A)+(B); break; \
  case NORM_1_AND_2: (A) = (A)+(B); (&(A))[1] = (&(A))[1]+(&(B))[1]; break; \
  case NORM_INFINITY: (A) = PetscMax((A), (B)); break; \
  default: ; \
  }

#undef __FUNCT__  
#define __FUNCT__ __SUF_C__(VecNorm_Comp)
PetscErrorCode __SUF__(VecNorm_Comp)(Vec a, NormType t, PetscReal *norm)
{
  PetscReal       work[2];
  PetscErrorCode  ierr;
  Vec_Comp        *as = (Vec_Comp*)a->data;
  PetscInt        i;
#ifdef __WITH_MPI__
  PetscScalar     work0, norm0;
#endif

  PetscFunctionBegin;

  PetscValidVecComp(a);

  *norm = 0.0;
  if (as->x[0]->ops->norm_local) {
    for(i=0; i<as->n->n; i++) {
      ierr = as->x[0]->ops->norm_local(as->x[i], t, work); CHKERRQ(ierr);
      __UNNORM__(*work, t);
      __SUM_NORMS__(*norm, *work, t);
    }
  } else {
    for(i=0; i<as->n->n; i++) {
      ierr = VecNorm(as->x[i], t, work); CHKERRQ(ierr);
      __UNNORM__(*work, t);
      __SUM_NORMS__(*norm, *work, t);
    }
  }

  /* If def(__WITH_MPI__) and exists norm_local */
#ifdef __WITH_MPI__
  if (as->x[0]->ops->norm_local) {
    /* norm <- Allreduce(work) */
    work0 = *norm;
    ierr = MPI_Allreduce(&work0, &norm0, t==NORM_1_AND_2?2:1, MPIU_SCALAR,
                         t==NORM_INFINITY?MPI_MAX:MPIU_SUM,
                         ((PetscObject)a)->comm); CHKERRQ(ierr);
    *norm = norm0;
  }
#endif

  /* Norm correction */
  __NORM__(*norm, t);

  PetscFunctionReturn(0);
}

#undef __NORM__
#undef __SUM_NORMS__
#undef __UNNORM__



#undef __SUF__
#undef __QUOTEME
#undef __SUF_C__

