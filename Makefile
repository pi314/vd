ESC := $(shell echo "\e")
S := $(shell printf "\033[38;5;135m$$\033[m")

.PHONY: what
what:
	@echo $S make run [VERBOSE=1]
	@echo $S make clean
	@echo $S make publish


.PHONY: clean
clean:
	@echo $S rm -rf dist/
	@rm -rf dist/


.PHONY: publish
publish:
	@echo Check git branch ... "$$(git branch --show-current)"
	@[ "$$(git branch --show-current)" = "main" ]

	@echo "$S" hatch clean
	@hatch clean

	@echo "$S" hatch build
	@hatch build

	@echo "$S" hatch publish --repo "$$(basename "${PWD}")"
	@hatch publish --repo "$$(basename "${PWD}")"
