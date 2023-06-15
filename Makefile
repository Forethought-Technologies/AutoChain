PYTHON_FILES = ./autochain ./tests

TEST_ENV = . $(TEST_ENV_DIR)/bin/activate
TEST_ENV_DIR = $(CURDIR)/venv

.PHONY: black
black:
	$(TEST_ENV) && \
	black $(PYTHON_FILES)
