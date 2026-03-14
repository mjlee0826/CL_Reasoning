from argparse import ArgumentParser

from Log.NoLog import NoLog
from Log.FileLog import FileLog

from Test.TestContext import TestContext
from Test.TestTokenNums import TestTokenNums
from Test.TestType import TestType
from File.FileFactory import FileFactory

import json

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument("--log", action="store_true", help="log to terminal")
    parser.add_argument("--testfile", nargs="+", help="The file need to be test")
    parser.add_argument("--testdir", help="The dir need to be test")
    parser.add_argument("-e", "--extension", default="*.json", help="File extension to look for")

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
        files = fileFactory.getFileInDir(args.testdir, args.extension)

    context: TestContext = TestContext()
    context.setTest(TestTokenNums())
    context.runTest(files, log)

def main():
    args = parseArgs()
    print("Test Performance")
    testExperiment(args)

if __name__ == '__main__':
    main()
