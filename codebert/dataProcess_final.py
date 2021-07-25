import argparse


# srcPath = '/data/Simfix/codebert/data/src.txt'
# tarPath = '/data/Simfix/codebert/data/batch_0.txt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default="", type=str)
    parser.add_argument("--bug_id", default="", type=str)
    args = parser.parse_args()

    basePath = "/data/Simfix/result/" + args.project_name + "/" + args.bug_id
    srcPath = basePath + "/src.txt"
    tarPath = basePath + "/batch_0.txt"
    # srcPath = "data/test.txt"
    # tarPath = "data/batch_test.txt"

    print("--------data Processing-------")
    resultList = []
    with open(srcPath, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # 加标注
            result = "1<CODESPLIT><CODESPLIT><CODESPLIT><CODESPLIT>" + line
            resultList.append(result)

    # 写结果
    with open(tarPath, 'w') as w:
        for eachResult in resultList:
            w.write(eachResult)
    print("------data processed--------")








