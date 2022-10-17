PYTHON=python

TESTFILES = $(filter-out ex13.py,$(wildcard ex*.py))
TESTS     = $(TESTFILES:.py=)
RUNRULES  = $(addprefix run_,$(TESTS))

.PHONY:test run ${RUNRULES}
test: run

run: ${RUNRULES} run_ex13

${RUNRULES}:
	${PYTHON} $(subst run_,,$@).py ${SLEPC_OPTIONS}

run_ex13:
	${PYTHON} ex13.py -ploteigs 0 ${SLEPC_OPTIONS}
