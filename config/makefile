include ${PETSC_DIR}/bmake/common/base

checklink: checklink.o chkopts
	${CLINKER} -o checklink checklink.o ${TESTFLAGS} ${PETSC_KSP_LIB}
	@${RM} -f checklink checklink.o
