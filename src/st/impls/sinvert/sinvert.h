/*
      Implements the shift-and-invert technique for eigenvalue problems.
*/

#if !defined(__SINVERT_H)
#define __SINVERT_H

typedef struct {
  Mat         A, B;
  Vec         w;
  PetscScalar sigma;
} CTX_SINV;

extern int MatCreateMatSinvert(ST,Mat*);

#endif
