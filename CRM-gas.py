#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on [Date]

@author: deepthisen
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.integrate import cumulative_trapezoid
import time
from os.path import join as join
from Modules.CRM_module import CRMP as crm

# -------------------------------
# 1. 定义伪压转换函数（针对气体）
# -------------------------------
def pseudopressure(p, mu=0.02, Z=1.0, p_ref=100):
    """
    计算伪压 m(p)
    对于气体，常用一种简化公式：
      m(p) = (p^2 - p_ref^2) / (mu)
    其中：
      p     : 实际压力 (psi)
      mu    : 气体粘度 (cp)（可根据数据调整）
      Z     : 偏差因子 (无量纲)
      p_ref : 参考压力 (psi)
    注：若需要更精确，可在分子中引入 Z 的影响，本示例中简化处理。
    """
    return (p**2 - p_ref**2) / mu

# -------------------------------
# 2. 读取数据（参考原液体部分代码）
# -------------------------------
filepath = r'Datasets/Streak'
parse_date = True  # 数据中日期格式

qi = pd.read_excel(join(filepath, 'Injection.xlsx'))
qp = pd.read_excel(join(filepath, 'Production.xlsx'))

percent_train = 0.7
time_colname = 'Time [days]'
if parse_date:
    qi[time_colname] = (qi['Date'] - qi['Date'].iloc[0]) / pd.to_timedelta(1, unit='D')

# 提取注入井（以 'I' 开头）和生产井（以 'P' 开头，排除井底压力列）的数据
InjList = [col for col in qi.columns if col.startswith('I')]
# 生产速率列（例如 "P1", "P2", ...），井底压力列则以 "_pwf" 结尾，如 "P1_pwf"
PrdList = [col for col in qp.columns if col.startswith('P') and not col.endswith('_pwf')]
t_arr = qi[time_colname].values

N_inj = len(InjList)
N_prd = len(PrdList)
qi_arr = qi[InjList].values
q_obs = qp[PrdList].values

# 划分训练集和测试集
n_train = int(percent_train * len(t_arr))
q_obs_train = q_obs[:n_train, :]
q_obs_test = q_obs[n_train:, :]
input_series_train = [t_arr[:n_train], qi_arr[:n_train, :]]
input_series_test = [t_arr[n_train:], qi_arr[n_train:, :]]

# -------------------------------
# 3. 构造针对气体的CRM模型（启用伪压函数）
# -------------------------------
# 参数设定（示例值，可根据气体实际情况调整）
tau = np.ones(N_prd)  # 初始时间常数数组（后续拟合中可更新）
gain_mat = np.ones([N_inj, N_prd])
gain_mat = gain_mat / (np.sum(gain_mat, axis=1, keepdims=True))
qp0 = np.zeros((1, N_prd))  # 初始生产速率
inputs_list = [tau, gain_mat, qp0]
crm_gas = crm(inputs_list, include_press=False)

# 拟合模型
init_guess = inputs_list
t_start = time.perf_counter()
params_fit = crm_gas.fit_model(input_series_train, q_obs_train, init_guess)
t_taken = time.perf_counter() - t_start
print('Time taken to fit: ' + str(t_taken))

# 预测生产速率
qp_pred_train = crm_gas.prod_pred(input_series_train)
qp_pred_test = crm_gas.prod_pred(input_series_test)

# -------------------------------
# 4. 计算并耦合 DOI 10.1016/j.petrol.2011.12.015 中的 IRP 和 tTMB 公式
# -------------------------------
# 针对气体，计算 IRP 时用伪压函数
# 公式: IRP(t) = ( m(pi) - m(pwf(t)) ) / q(t)
# 其中 pi 为初始压力（取井底压力数据第一项的实际压力，经过伪压转换），
# q(t) 为生产速率（单位应与 qi, qp 数据一致）。
# 同时，利用生产速率 q(t) 计算累计产量 Q(t) 作为 tTMB 的近似（单位 STB）。
# ck 表示总系统可压缩性（示例值，单位1/psi）
ck = 1e-6

IRP = {}   # 存储每口井的 IRP 时间序列
tTMB = {}  # 存储每口井的累计产量 Q(t)（作为 tTMB）
for well in PrdList:
    # 井底压力列名：例如 "P1_pwf" 对应井 "P1"
    pwf_col = well + '_pwf'
    if pwf_col not in qp.columns:
        print(f"Warning: Column {pwf_col} not found for well {well}. Skipping IRP computation.")
        continue
    # 生产速率 q(t)
    q = qp[well].values.astype(float)  # 单位：STB/d 或 scf/d，根据数据而定
    # 井底压力 p_wf(t)
    p_wf = qp[pwf_col].values.astype(float)  # 单位：psi
    # 初始压力 pi 取井底压力第一项（实际生产时可能需要转换为下井压力）
    pi = p_wf[0]
    # 计算伪压值
    m_pi = pseudopressure(pi)
    m_pwf = pseudopressure(p_wf)  # 对每个时间点计算伪压（向量化运算）
    # 计算 IRP(t)
    # 为避免除零错误，若 q 中有零值，则对其做适当处理
    IRP[well] = (m_pi - m_pwf) / (q + 1e-8)
    # 计算累计产量 Q(t) 作为 tTMB
    Q = cumulative_trapezoid(q, t_arr, initial=0)
    tTMB[well] = Q

# 对于每个井，在假设处于边界主导流（BDF）阶段时，对 IRP vs. tTMB 进行线性回归，
# 进而利用斜率 m_IRP-tTMB 来估计排水体积 Nk
Nk_estimates = {}
for well in IRP.keys():
    Q_data = tTMB[well]
    IRP_data = IRP[well]
    n = len(Q_data)
    # 选择后半段数据作为 BDF 阶段（实际中需根据数据判断）
    window = slice(n//2, n)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Q_data[window], IRP_data[window])
    m = slope  # 斜率 m_IRP-tTMB
    Nk = 1.0 / (m * ck)
    Nk_estimates[well] = Nk
    print(f"Well {well}: slope = {m:.6f}, estimated drainage volume Nk = {Nk:.2f} STB")

# 绘制每口井的 IRP vs. tTMB 曲线
for well in IRP.keys():
    plt.figure()
    plt.plot(tTMB[well], IRP[well], 'o-', label=f'IRP for {well}')
    plt.xlabel('tTMB (cumulative production, STB)')
    plt.ylabel('IRP ((m(pi) - m(pwf))/q)')
    plt.title(f'IRP vs. tTMB for well {well}')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# 5. 绘制 CRM 模型预测结果（与原始部分一致）
# -------------------------------
for i in range(N_prd):
    plt.figure()
    plt.plot(t_arr, q_obs[:, i], 'r-', label='Actual')
    plt.plot(t_arr[:n_train], qp_pred_train[:, i], 'b-', label='CRM Train')
    plt.plot(t_arr[n_train:], qp_pred_test[:, i], 'b--', label='CRM Test')
    plt.title('P' + str(i + 1), fontsize=15)
    plt.xlabel('Time (D)', fontsize=13)
    plt.ylabel('Production (RB or scf)', fontsize=13)
    plt.legend(fontsize=11)
    plt.show()

# 计算训练集和测试集的均方根误差
train_err_crm = np.sqrt(np.mean((q_obs_train - qp_pred_train) ** 2, axis=0))
test_err_crm = np.sqrt(np.mean((q_obs_test - qp_pred_test) ** 2, axis=0))
print("Train Errors:", train_err_crm)
print("Test Errors:", test_err_crm)

