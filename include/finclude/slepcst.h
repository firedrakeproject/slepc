
!
!  Include file for Fortran use of the ST object in SLEPc
!
#if !defined(__SLEPCST_H)
#define __SLEPCST_H

#define ST      PetscFortranAddr
#define STType  character*(80)

#define STNONE      'none'
#define STSHELL     'shell'
#define STSHIFT     'shift'
#define STSINV      'sinvert'

#endif
