from behave import given, when, then



@given("we have behave installed")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    # raise NotImplementedError(u'STEP: Given we have behave installed')
    pass


@when('we implement {number:d} tests')
def step_impl(context, number):  # -- NOTE: number is converted into integer
    assert number > 1 or number == 0
    context.tests_count = number


@then("behave will test them for us!")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    assert context.failed is False
    assert context.tests_count >= 0
