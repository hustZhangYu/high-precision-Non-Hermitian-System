# some functions to analyze the data

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import re

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = pd.read_csv(file)
    return data


def parse_complex_string(s):
    # 移除所有空白字符
    s=s.replace('(','')
    s=s.replace(')','')
    s = s.replace(' ', '')
    
    # 使用正则表达式匹配实部和虚部
    match = re.match(r'([-+]?\d*\.?\d*)?([-+]?\d*\.?\d*)[ji]?', s)
    
    if match:
        real = match.group(1)
        imag = match.group(2)
        
        # 处理只有虚部的情况
        if real == '' and imag.startswith(('+', '-')):
            real = '0'
        # 处理只有实部的情况
        elif imag == '':
            imag = '0'
        # 处理虚部没有系数的情况
        elif imag in ('+', '-'):
            imag += '1'
        
        return complex(float(real or 0), float(imag or 0))
    else:
        raise ValueError(f"无法解析的复数字符串: {s}")


def EigenvaluesWithBoundaryCondition():
    data = load_data('C:/Users/Lenovo/OneDrive/A_CodePython/NHGAA/EigenvalueData_BoundaryConditionH.csv')
    print(data.shape)
    for i in range(data.shape[0]):
        m = data.loc[i].values
        m1 = np.array([parse_complex_string(str(val).strip("'")) for val in m])
        colors = plt.cm.viridis(i / data.shape[0])  # 使用viridis颜色映射创建渐变色
        plt.plot(np.real(m1), np.imag(m1), 'o', color=colors,markersize=5)

    E=np.linspace(-1,1,100)*np.exp(1j*np.pi*0)
    plt.plot(np.real(E),np.imag(E),'-',linewidth=3)
    plt.xlabel('Re(E)')
    plt.ylabel('Im(E)')
    plt.title('Eigenvalues with Boundary Condition')
    plt.savefig('EigenvaluesWithBoundaryConditionL_89_v_1.png')
    plt.show()

def EigenvaluesWithBoundaryConditionSingleParameter():
    data = load_data('C:/Users/Lenovo/OneDrive/A_CodePython/NHGAA/EigenvalueData_BoundaryConditionH.csv')
    print(data.shape)
    i=0 
    m = data.loc[i].values
    m1 = np.array([parse_complex_string(str(val).strip("'")) for val in m])
    colors = plt.cm.viridis(i / data.shape[0])  # 使用viridis颜色映射创建渐变色
    plt.plot(np.real(m1), np.imag(m1), 'o', color=colors,markersize=5)

    E=np.linspace(-1,1,100)*np.exp(1j*np.pi*0)
    plt.plot(np.real(E),np.imag(E),'-',linewidth=3)
    plt.xlabel('Re(E)')
    plt.ylabel('Im(E)')
    plt.title('Eigenvalues with Boundary Condition')
    plt.savefig('EigenvaluesWithBoundaryConditionSingleParameter_v_1.png')
    plt.show()

if __name__ == "__main__":
    EigenvaluesWithBoundaryCondition()
#   EigenvaluesWithBoundaryConditionSingleParameter()

