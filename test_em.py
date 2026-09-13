from argparse import ArgumentParser

from Log.NoLog import NoLog
from Log.FileLog import FileLog

from Test.TestContext import TestContext
from Test.TestEM import TestEM
from Test.TestPValue import TestPValue
from Test.TestRecoveryBlind import TestRecoveryBlind
from Test.TestMissingAnswer import TestMissingAnswer
from Test.TestType import TEST_STR_LIST, TestType
from File.FileFactory import FileFactory

import json

# 已接上 test_em.py 的 test mode
TEST_MAP = {
    TestType.TESTEM: TestEM,
    TestType.TESTPVALUE: TestPValue,
    TestType.TESTRECOVERYBLIND: TestRecoveryBlind,
    TestType.TESTMISSING: TestMissingAnswer,
}

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument("--log", action="store_true", help="log to terminal")

    parser.add_argument("-t", "--testmode", choices=TEST_STR_LIST, default='testem', help="choose your test stratey")
    parser.add_argument("--testfile", nargs="+", help="The file need to be test")
    parser.add_argument("--testdir", help="The dir need to be test")

    args = parser.parse_args()
    return args

def testExperiment(args):
    testType = TestType(args.testmode)
    if testType not in TEST_MAP:
        print(f"test mode '{args.testmode}' 尚未接上 test_em.py，可用：{[t.value for t in TEST_MAP]}")
        return
    if testType == TestType.TESTPVALUE and (not args.testfile or len(args.testfile) != 2):
        print("testp 需要用 --testfile 指定剛好兩個檔案")
        return

    fileFactory: FileFactory = FileFactory()
    log = FileLog() if args.log else NoLog()

    if args.testfile:
        files = []
        for f_temp in args.testfile:
            files.append(fileFactory.getFileByPath(f_temp))
    else:
        # 逐檔載入：整個資料夾一次讀進記憶體可能超過可用 RAM（result/challenge 約 4–6 GB）
        files = fileFactory.iterFileInDir(args.testdir)

    context: TestContext = TestContext()
    context.setTest(TEST_MAP[testType]())
    context.runTest(files, log)

def main():
    args = parseArgs()
    print("Test Performance")
    testExperiment(args)

if __name__ == '__main__':
    main()
