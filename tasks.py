from invoke import task


@task
def build_docs(context):
    context.run("sphinx-apidoc -o ./docs ./route_distances")
    context.run("sphinx-build -M html ./docs ./docs/build")


@task
def full_tests(context):
    cmd = "pytest --black --mccabe " \
          "--cov route_distances --cov-branch --cov-report html:coverage --cov-report xml " \
          "tests/"
    context.run(cmd)

@task
def run_mypy(context):
    context.run("mypy --ignore-missing-imports --show-error-codes route_distances")


@task
def run_linting(context):
    # print("Running mypy...")
    # context.run("mypy --ignore-missing-imports --show-error-codes route_distances")
    print("Running pylint...")
    context.run("pylint route_distances")