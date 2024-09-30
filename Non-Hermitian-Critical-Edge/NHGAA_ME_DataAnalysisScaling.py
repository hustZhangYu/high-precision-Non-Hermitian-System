import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = pd.read_csv(file)
    return data

def parse_complex_string(s):
    # 移除所有空白字符
    s=s.replace('(','')
    s=s.replace(')','')
    s = s.replace(' ', '')
    s=s.replace('NaN','')
    


def main():
    data=load_data(r'C:\Users\Lenovo\OneDrive\A_CodePython\NHGAA\SingleEigenstateScalingMulti.csv')
    data1=np.array(data.iloc[0:,1:].values)
    
    MeanData=list()
    for i in range(len(data1)):
        res=list()
        for j in range(len(data1[i])):
            if np.isnan(data1[i][j]):
                pass
            else:
                res.append(data1[i][j])
        MeanData.append(np.mean(res))
        plt.plot(np.linspace(0,1,len(res)),res,'.')
    plt.legend(['L=144','L=233','L=377','L=610','L=987'],loc='best')
    plt.xlabel('n')
    plt.ylabel(r'$-\frac{\ln(IPR(n))}{\ln(N/2)}$')
    plt.savefig('SingleEigenstateScalingMulti.png')
    
    plt.figure()
    plt.plot(np.linspace(0,1,len(MeanData)),MeanData,'.')
    plt.xlabel('n')
    plt.ylabel(r'$-\frac{\ln(IPR(n))}{\ln(N/2)}$')
    plt.savefig('SingleEigenstateScalingMultiMean.png') 
    plt.show()

if __name__=='__main__':
    main()