from enum import Enum

class TestType(str, Enum):
    TESTEM = "testem"
    PRINTONE = "printone"
    TESTCASE = "testcase"
    TESTPVALUE = "testp"
    TESTTOKEN = "testtoken"

TEST_STR_LIST = [t.value for t in TestType]
