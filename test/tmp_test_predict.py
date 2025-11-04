from Software_Feasibility_Analysis.src.jm_model_prediction import jm_model_parameter_estimation, jm_predict_future_failures

# 示例数据（使用较长的样本，确保参数估计成功）
train_times = [9,12,11,4,7,2,5,8,5,7,1,6,1,9,4,1,3,8,6,1,1,33,7,91,2,1,87,47,12,9,135,258,16,35]

# 估计参数
N0, phi = jm_model_parameter_estimation(train_times)
print(f"Estimated N0={N0}, phi={phi}")

for steps in [1,3,5,10]:
    res = jm_predict_future_failures(N0, phi, train_times, steps)
    print('\nsteps=', steps)
    # debug: show current failures and remaining faults
    current_failures = len(train_times)
    remaining_faults = N0 - current_failures
    print('current_failures=', current_failures)
    print('remaining_faults=', remaining_faults, 'int=', int(remaining_faults))
    print('predicted_intervals len=', len(res['predicted_intervals']))
    print('cumulative_times len=', len(res['cumulative_times']))
    if 'next_failure_time' in res:
        print('next_failure_time=', res['next_failure_time'])
    if 'warning' in res:
        print('warning=', res['warning'])
    # show first 3 predicted intervals
    print('predicted_intervals sample=', res['predicted_intervals'][:3])
