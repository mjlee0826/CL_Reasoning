from argparse import ArgumentParser
from Strategy.RunContext import RunContext

from Model.Model import Model
from Model.ModelFactory import ModelFactory
from Model.ModelType import MODEL_LIST

from Dataset.Dataset import Dataset
from Dataset.DatasetFactory import DatasetFactory
from Dataset.DatasetType import DATASET_LIST

from Strategy.StrategyType import STRATEGY_LIST, StrategyType
from Strategy.OnlyOneLanguage import OnlyOneLanguage
from Strategy.SelfReflection import SelfReflection
from Strategy.Repair import Repair
from Strategy.Challenge import Challenge
from Strategy.GetOneResult import GetOneOutput

from Log.Log import Log
from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog
from Log.TwoAgentLog import TwoAgentLog

from Test.TestContext import TestContext
from Test.TestEM import TestEM
from Test.PrintOne import PrintOne
from Test.TestCaseBase import TestCaseBase
from Test.TestPValue import TestPValue
from Test.TestTokenNums import TestTokenNums
from Test.TestType import TEST_LIST, TestType
from File.FileFactory import FileFactory

import json

def parseArgs():
    parser = ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run Experiment")
    parser.add_argument("--test", action="store_true", help="Test Experiment")
    parser.add_argument("--log", action="store_true", help="log to terminal")

    parser.add_argument("-m", "--model", choices=MODEL_LIST, help="choose your model")
    parser.add_argument("--tempature", default=0, type=float, help="Tempature")

    parser.add_argument("-d", "--dataset", choices=DATASET_LIST, help="choose your dataset")
    parser.add_argument("--nums", help="Data Nums", default=-1, type=int)
    parser.add_argument("--sample", help="Data Sample", default=1, type=int)

    parser.add_argument("-s", "--strategy", choices=STRATEGY_LIST, help="choose your strategy")
    parser.add_argument("--repairpath", help="The file you need to repair")
    parser.add_argument("--datapath1", help="Two Result Path 1")
    parser.add_argument("--datapath2", help="Two Result Path 2")
    parser.add_argument("--threshold", type=int, default=3, help="Challenge Threshold 3")

    parser.add_argument("--dirpath", help="your dir path")
    parser.add_argument("--filepath", help="your file path")


    parser.add_argument("-t", "--testmode", choices=TEST_LIST, help="choose your test stratey")
    parser.add_argument("--testfile", nargs="+", help="The file need to be test")
    parser.add_argument("--testdir", help="The dir need to be test")
    parser.add_argument("--testmodel", choices=MODEL_LIST, nargs="+", help="The model you want to test")
    parser.add_argument("--testdataset", choices=DATASET_LIST, nargs="+", help="The dataset you want to test")
    parser.add_argument("--teststrategy", choices=STRATEGY_LIST, nargs="+", help="The strategy you want to test")

    args = parser.parse_args()
    return args

def runExperiment(args):
    model, dataset = None, None
    log = NoLog()
    if args.log:
        if args.strategy == StrategyType.GETONEOUTPUT:
            log = Log()
        elif args.strategy == StrategyType.CHALLENGE:
            log = TwoAgentLog()
        else:
            log = OneAgentLog()

    if args.model:
        modelFactory = ModelFactory()
        model: Model = modelFactory.buildModel(args.model, tempature=args.tempature)
    if args.dataset:
        datasetFactory = DatasetFactory()
        dataset: Dataset = datasetFactory.buildDataset(args.dataset, nums = args.nums, sample = args.sample)

    context = RunContext()
    if args.strategy == StrategyType.ONLYCHINESE or args.strategy == StrategyType.ONLYENGLISH or args.strategy == StrategyType.ONLYSPANISH \
        or args.strategy == StrategyType.ONLYJAPANESE or args.strategy == StrategyType.ONLYRUSSIAN:
        context.setStrategy(OnlyOneLanguage(model, dataset, log, args.strategy))

    elif args.strategy == StrategyType.REPAIR:
        file = FileFactory().getFileByPath(args.repairpath)
        context.setStrategy(Repair(model, dataset, log, file))

    elif args.strategy == StrategyType.SELFREFLECTION:
        dataFile = FileFactory().getFileByPath(args.datapath1)
        context.setStrategy(SelfReflection(model, dataset, log, dataFile))

    elif args.strategy == StrategyType.GETONEOUTPUT:
        context.setStrategy(GetOneOutput(model, dataset, log))

    elif args.strategy == StrategyType.CHALLENGE:
        dataFile1, dataFile2 = None, None
        if args.datapath1 and args.datapath2:
            fileFactory = FileFactory()
            dataFile1 = fileFactory.getFileByPath(args.datapath1)
            dataFile2 = fileFactory.getFileByPath(args.datapath2)
        context.setStrategy(Challenge(model, dataset, log, args.threshold, dataFile1, dataFile2))

    result = context.runExperiment()

    if not result:
        return
    path = ""
    if args.dirpath:
        path = f'{args.dirpath}/{args.model}_{args.dataset}_{args.strategy}.json'
    elif args.filepath:
        path = f'{args.filepath}'
    else:
        path = f'{args.model}_{args.dataset}_{args.strategy}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def textExperiment(args):
    fileFactory: FileFactory = FileFactory()
    if args.testfile:
        file = []
        for f_temp in args.testfile:
            file.append(fileFactory.getFileByPath(f_temp))
    else:
        file = fileFactory.getFileBySetting(args.testdir, args.testmodel, args.testdataset, args.teststrategy)

    context: TestContext = TestContext()
    if args.testmode == TestType.TESTEM:
        context.setTest(TestEM())
    if args.testmode == TestType.TESTPVALUE:
        context.setTest(TestPValue())
    elif args.testmode == TestType.PRINTONE:
        context.setTest(PrintOne())
    elif args.testmode == TestType.TESTCASE:
        context.setTest(TestCaseBase())
    elif args.testmode == TestType.TESTTOKEN:
        context.setTest(TestTokenNums())
    context.runTest(file)

def main():
    args = parseArgs()
    if args.run:
        print("Run Experiment Prepare")
        runExperiment(args)
    if args.test:
        print("Test Performance")
        textExperiment(args)

if __name__ == '__main__':
    main()
