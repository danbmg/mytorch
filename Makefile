##
## my_torch — the programs are Python: "building" them writes small executable
## wrappers that put the repository on sys.path and call the real entry point.
##

ANALYZER	=	my_torch_analyzer

GENERATOR	=	my_torch_generator

# $(call wrapper,package): executable that runs package/main.py's main()
define wrapper
	@printf '%s\n' '#!/usr/bin/env python3' \
		'import os' \
		'import sys' \
		'sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))' \
		'from $(1).main import main' \
		'sys.exit(main())' > $@
	@chmod +x $@
endef

all:	$(ANALYZER) $(GENERATOR)

$(ANALYZER):
	$(call wrapper,analyzer)

$(GENERATOR):
	$(call wrapper,generator)

# pytest if installed, else unittest on every tests/<module>/ directory; one failing directory fails the target
tests_run:
	@if python3 -c "import pytest" 2>/dev/null; then \
		python3 -m pytest tests/ -q; \
	else \
		status=0; \
		for dir in tests/*/; do \
			ls "$$dir"test_*.py > /dev/null 2>&1 || continue; \
			PYTHONPATH=.:tests python3 -m unittest discover -s "$$dir" -p "test_*.py" || status=1; \
		done; \
		exit $$status; \
	fi

clean:
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@rm -rf .pytest_cache

fclean:	clean
	@rm -f $(ANALYZER) $(GENERATOR)

re:	fclean all

.PHONY:	all clean fclean re tests_run
