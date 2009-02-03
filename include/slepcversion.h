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

#if !defined(__SLEPCVERSION_H)
#define __SLEPCVERSION_H

/* ========================================================================== */
/* 
   Current SLEPC version number and release date
*/
#define SLEPC_VERSION_RELEASE    1
#define SLEPC_VERSION_MAJOR      3
#define SLEPC_VERSION_MINOR      0
#define SLEPC_VERSION_SUBMINOR   0
#define SLEPC_VERSION_PATCH      0
#define SLEPC_VERSION_DATE       "February 3, 2009"
#define SLEPC_VERSION_PATCH_DATE "February 3, 2009"
#define SLEPC_AUTHOR_INFO        "        The SLEPc Team\n\
   slepc-maint@grycap.upv.es\n\
 http://www.grycap.upv.es/slepc\n"

#define SLEPC_VERSION_(MAJOR,MINOR,SUBMINOR) \
 ((SLEPC_VERSION_MAJOR == (MAJOR)) &&       \
  (SLEPC_VERSION_MINOR == (MINOR)) &&       \
  (SLEPC_VERSION_SUBMINOR == (SUBMINOR)) && \
  (SLEPC_VERSION_RELEASE  == 1))

#endif

