#ifndef _IPIMPL
#define _IPIMPL

#include "slepcip.h"

extern PetscCookie IP_COOKIE;
extern PetscEvent IP_InnerProduct,IP_Orthogonalize;

typedef struct _IPOps *IPOps;

struct _IPOps {
  int dummy;
};

struct _p_IP {
  PETSCHEADER(struct _IPOps);
  IPOrthogonalizationType orthog_type; /* which orthogonalization to use */
  IPOrthogonalizationRefinementType orthog_ref;   /* refinement method */
  PetscReal orthog_eta;
  IPBilinearForm bilinear_form;
  int innerproducts;
  Vec work[2]; /* workspace */
};

/* check if workspace vector is compatible */
#define IPCHECKWORK(x,w) \
  if (w) { \
    PetscMPIInt _flg; \
    ierr = MPI_Comm_compare(x->comm,w->comm,&_flg);CHKERRQ(ierr); \
    if ((_flg != MPI_CONGRUENT && _flg != MPI_IDENT) || x->type != w->type || \
        x->map.N != w->map.N || x->map.n != w->map.n) { \
      ierr = VecDestroy(w);CHKERRQ(ierr); \
      ierr = VecDuplicate(x,&w);CHKERRQ(ierr); \
    } \
  } else { ierr = VecDuplicate(x,&w);CHKERRQ(ierr); }

#endif
