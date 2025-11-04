from Software_Feasibility_Analysis.src.jm_model_prediction import jm_model_parameter_estimation, jm_predict_future_failures

train_times = [9,12,11,4,7,2,5,8,5,7,1,6,1,9,4,1,3,8,6,1,1,33,7,91,2,1,87,47,12,9,135,258,16,35]
N0, phi = jm_model_parameter_estimation(train_times)
print('N0, phi =', N0, phi)
res = jm_predict_future_failures(N0, phi, train_times, 5)
print('returned predicted intervals count=', len(res['predicted_intervals']))
print('sample intervals =', res['predicted_intervals'])
print('next_failure_time =', res.get('next_failure_time'))
print('remaining_faults =', res.get('remaining_faults'))
