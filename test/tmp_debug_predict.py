from Software_Feasibility_Analysis.src.jm_model_prediction import jm_model_parameter_estimation, jm_predict_future_failures

sets = [
    [9,12,11,4,7,2,5,8],
    [9,12,11,4,7,2,5,8,5,7,1,6],
    [20,15,25,18,30,22,12]
]

for s in sets:
    try:
        N0, phi = jm_model_parameter_estimation(s)
        res = jm_predict_future_failures(N0, phi, s, 5)
        print('train_len=', len(s), 'N0=', round(N0,4), 'phi=', round(phi,6), 'pred_count=', len(res['predicted_intervals']))
        print('sample intervals=', res['predicted_intervals'])
    except Exception as e:
        print('error for set', s, e)
