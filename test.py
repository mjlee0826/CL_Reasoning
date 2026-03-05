from argparse import ArgumentParser

from Log.NoLog import NoLog
from Log.FileLog import FileLog

from Test.TestContext import TestContext
from Test.TestEM import TestEM
from Test.TestType import TEST_STR_LIST, TestType
from File.FileFactory import FileFactory

import json

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument("--log", action="store_true", help="log to terminal")

    parser.add_argument("--dirpath", help="your dir path")
    parser.add_argument("--filepath", help="your file path")

    parser.add_argument("-t", "--testmode", choices=TEST_STR_LIST, default='testem', help="choose your test stratey")
    parser.add_argument("--testfile", nargs="+", help="The file need to be test")
    parser.add_argument("--testdir", help="The dir need to be test")

    args = parser.parse_args()
    return args

def testExperiment(args):
    fileFactory: FileFactory = FileFactory()
    log = FileLog() if args.log else NoLog()

    if args.testfile:
        files = []
        for f_temp in args.testfile:
            files.append(fileFactory.getFileByPath(f_temp))
    else:
        files = fileFactory.getFileInDir(args.testdir)

    context: TestContext = TestContext()
    if args.testmode == TestType.TESTEM:
        context.setTest(TestEM())
    context.runTest(files, log)

def main():
    args = parseArgs()
    print("Test Performance")
    testExperiment(args)

if __name__ == '__main__':
    main()
