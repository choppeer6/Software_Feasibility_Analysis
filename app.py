from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '软件可行性分析'))

# 导入JM模型函数
from jm_model_prediction import (
    jm_model_parameter_estimation, 
    jm_predict_future_failures, 
    plot_prediction_results,
    calculate_reliability,
    calculate_model_accuracy
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# 模拟用户数据库
users = {
    'admin': 'password123',
    'user1': 'pass456'
}

# 示例失效数据
SAMPLE_FAILURE_DATA = [9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 103, 109, 110, 111, 144, 151, 242, 244, 245, 332, 379, 391, 400, 535, 793, 809, 844]

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/models')
def models():
    if 'username' in session:
        return render_template('models.html')
    return redirect(url_for('login'))

@app.route('/model/jm', methods=['GET', 'POST'])
def jm_model():
    if 'username' in session:
        if request.method == 'POST':
            # 获取表单数据
            data_type = request.form.get('data_type')
            prediction_step = int(request.form.get('prediction_step', 5))
            ex = float(request.form.get('ex', 0.001))
            ey = float(request.form.get('ey', 0.001))
            
            # 获取合适的失效数据
            if data_type == 'id':
                failure_data = SAMPLE_FAILURE_DATA
            else:
                failure_data = SAMPLE_FAILURE_DATA
            
            # 执行JM模型预测
            try:
                # 估计模型参数
                N0, phi = jm_model_parameter_estimation(failure_data, ex, ey)
                
                if N0 is None or phi is None:
                    raise ValueError("JM模型参数估计失败")
                
                # 预测未来失效
                prediction_results = jm_predict_future_failures(N0, phi, failure_data, prediction_step)
                
                # 计算模型准确率
                accuracy_metrics = calculate_model_accuracy(N0, phi, failure_data)
                
                # 生成可靠度曲线数据点
                if 'reliability_curve' in prediction_results and len(prediction_results['reliability_curve'][0]) > 0:
                    time_points, reliability_values = prediction_results['reliability_curve']
                else:
                    time_points = np.array([])
                    reliability_values = np.array([])
                
                return render_template('jm_model.html', 
                                     success=True,
                                     message='预测成功完成！',
                                     N0=round(N0, 4),
                                     phi=round(phi, 6),
                                     remaining_faults=round(prediction_results['remaining_faults'], 4),
                                     next_failure_time=round(prediction_results['next_failure_time'], 2) if prediction_results['next_failure_time'] else 'N/A',
                                     predicted_intervals=[round(interval, 2) for interval in prediction_results['predicted_intervals']],
                                     cumulative_times=[round(time, 2) for time in prediction_results['cumulative_times']],
                                     reliability_curve=list(zip(time_points.round(2), reliability_values.round(4))),
                                     warning=prediction_results.get('warning'),
                                     mae=round(accuracy_metrics['mae'], 2),
                                     mse=round(accuracy_metrics['mse'], 2),
                                     rmse=round(accuracy_metrics['rmse'], 2),
                                     r2_score=round(accuracy_metrics['r2_score'], 4),
                                     accuracy=round(accuracy_metrics['accuracy'], 2))
            
            except Exception as e:
                return render_template('jm_model.html', 
                                     success=False,
                                     message=f'预测过程中发生错误: {str(e)}')
        
        # GET请求，显示页面
        return render_template('jm_model.html')
    return redirect(url_for('login'))

# 添加API端点用于AJAX请求
@app.route('/api/jm/predict', methods=['POST'])
def api_jm_predict():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        # 解析JSON数据
        if not request.is_json:
            return jsonify({'success': False, 'error': '请求必须是JSON格式'}), 400
        
        data = request.get_json()
        
        # 验证输入数据
        if not data:
            return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
        
        # 获取并验证参数
        data_type = data.get('data_type', 'id')
        prediction_step = data.get('prediction_step', 5)
        ex = data.get('ex', 0.001)
        ey = data.get('ey', 0.001)
        
        # 参数验证
        try:
            prediction_step = int(prediction_step)
            ex = float(ex)
            ey = float(ey)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400
        
        if prediction_step < 1 or prediction_step > 100:
            return jsonify({'success': False, 'error': '预测步长必须在1-100之间'}), 400
        
        if ex < 0 or ex > 1:
            return jsonify({'success': False, 'error': '预测精度ex必须在0-1之间'}), 400
        
        if ey < 0 or ey > 1:
            return jsonify({'success': False, 'error': '预测精度ey必须在0-1之间'}), 400
        
        # 优先使用请求中提供的 train_data（用于前百分比训练或自定义数据）
        train_data = data.get('train_data')
        test_data = data.get('test_data')
        if train_data and isinstance(train_data, list) and len(train_data) >= 2:
            failure_data = train_data
        else:
            # 否则退回到样例数据或通过 data_type 选择
            if data_type == 'id':
                failure_data = SAMPLE_FAILURE_DATA
            else:
                failure_data = SAMPLE_FAILURE_DATA
        
        # 执行JM模型预测
        try:
            N0, phi = jm_model_parameter_estimation(failure_data, ex, ey)
            
            if N0 is None or phi is None:
                return jsonify({'success': False, 'error': 'JM模型参数估计失败'}), 400
            
            prediction_results = jm_predict_future_failures(N0, phi, failure_data, prediction_step)
            
            # 计算模型准确率
            accuracy_metrics = calculate_model_accuracy(N0, phi, failure_data)
            
            # 准备响应数据
            response_data = {
                'success': True,
                'N0': round(N0, 4),
                'phi': round(phi, 6),
                'remaining_faults': round(prediction_results['remaining_faults'], 4),
                'next_failure_time': round(prediction_results['next_failure_time'], 2) if prediction_results['next_failure_time'] else None,
                'predicted_intervals': [round(interval, 2) for interval in prediction_results['predicted_intervals']],
                'cumulative_times': [round(time, 2) for time in prediction_results['cumulative_times']],
                'mae': round(accuracy_metrics['mae'], 2),
                'mse': round(accuracy_metrics['mse'], 2),
                'rmse': round(accuracy_metrics['rmse'], 2),
                'r2_score': round(accuracy_metrics['r2_score'], 4),
                'accuracy': round(accuracy_metrics['accuracy'], 2)
            }
            # 添加训练数据摘要以便调试前端/后端一致性
            try:
                response_data['used_train_count'] = len(failure_data)
                response_data['used_train_sum'] = float(np.sum(failure_data)) if len(failure_data) > 0 else 0.0
                response_data['used_train_preview'] = failure_data[:10]
            except Exception:
                pass
            
            # 添加可靠度曲线数据
            if 'reliability_curve' in prediction_results and len(prediction_results['reliability_curve'][0]) > 0:
                time_points, reliability_values = prediction_results['reliability_curve']
                response_data['reliability_curve'] = [{'time': round(t, 2), 'reliability': round(r, 4)} for t, r in zip(time_points, reliability_values)]
            
            # 添加警告信息
            if 'warning' in prediction_results:
                response_data['warning'] = prediction_results['warning']
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'模型预测错误: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500


@app.route('/api/jm/train', methods=['POST'])
def api_jm_train():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401

    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': '请求必须是JSON格式'}), 400

        data = request.get_json()
        train_data = data.get('train_data')
        ex = data.get('ex', 0.001)
        ey = data.get('ey', 0.001)

        if not train_data or not isinstance(train_data, list) or len(train_data) < 2:
            return jsonify({'success': False, 'error': '请提供至少2个失效时间点作为训练数据'}), 400

        try:
            ex = float(ex)
            ey = float(ey)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400

        N0, phi = jm_model_parameter_estimation(train_data, ex, ey)
        resp = {'success': True, 'N0': round(N0, 4), 'phi': round(phi, 6)}
        try:
            resp['used_train_count'] = len(train_data)
            resp['used_train_sum'] = float(np.sum(train_data))
            resp['used_train_preview'] = train_data[:10]
        except Exception:
            pass
        return jsonify(resp)

    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

if __name__ == '__main__':
    # 确保static目录存在用于保存图表
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=True, host='0.0.0.0', port=5000)