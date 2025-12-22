from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '软件可行性分析'))

# 导入JM模型函数
from model.jm_model_prediction import (
    jm_model_parameter_estimation, 
    jm_predict_future_failures, 
    plot_prediction_results,
    calculate_reliability,
    calculate_model_accuracy
)

# 导入GO模型函数
from model.go_model_prediction import (
    go_model_parameter_estimation,
    go_predict_future_failures,
    plot_prediction_results as go_plot_prediction_results,
    calculate_reliability as go_calculate_reliability,
    calculate_model_accuracy as go_calculate_model_accuracy
)

# 导入NHPP模型函数
from model.nhpp_model_prediction import (
    nhpp_model_parameter_estimation,
    nhpp_predict_future_failures,
    calculate_nhpp_model_accuracy,
    plot_nhpp_prediction_results
)

# 导入BP神经网络模型函数
from model.bp_model_prediction import (
    bp_train_model,
    bp_predict_future_failures,
    calculate_model_accuracy as bp_calculate_model_accuracy,
    plot_prediction_results as bp_plot_prediction_results
)

# 导入ARIMA模型函数
from model.arima_model_prediction import (
    arima_train_model,
    arima_predict_future_failures,
    calculate_arima_accuracy
)

# 导入GM11模型函数
from model.gm11_model_prediction import (
    gm11_train_model,
    gm11_predict_future_failures,
    calculate_gm11_accuracy
)

# 导入SVR模型函数
from model.svr_model_prediction import (
    svr_train_model,
    svr_predict_future_failures,
    calculate_svr_accuracy
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# 示例失效数据
SAMPLE_FAILURE_DATA = [9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 103, 109, 110, 111, 144, 151, 242, 244, 245, 332, 379, 391, 400, 535, 793, 809, 844]


#######################################################

# --- 在 app.py 顶部添加导入 ---
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os

# --- 配置数据库 (放在 app = Flask(__name__) 之后) ---
# 配置上传文件夹路径
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reliability_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- 定义数据表模型 ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)      # 存储在磁盘上的文件名 (避免重名)
    original_name = db.Column(db.String(100), nullable=False) # 用户上传时的原始文件名
    upload_time = db.Column(db.DateTime, default=datetime.now)
    data_count = db.Column(db.Integer, default=0)             # 数据点数量
    user_id = db.Column(db.String(50), nullable=False)        # 关联的用户

# 初始化数据库表并创建默认用户
# 初始化数据库表并创建默认用户
def sync_files_to_db():
    # 获取所有已存在于数据库中的文件名
    existing_db_filenames = {d.filename for d in Dataset.query.all()}
    
    # 扫描 uploads 文件夹
    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        if filename.endswith((".csv", ".txt")) and filename not in existing_db_filenames:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().replace("\n", ",")
                    data_points = [x for x in content.split(",") if x.strip()]
                    count = len(data_points)
                
                # 尝试从文件名中解析 user_id (假设格式为 user_id_timestamp_original_name.ext)
                parts = filename.split("_")
                inferred_user_id = "unknown" # 默认值
                if len(parts) >= 3: # 至少有 user_id_timestamp_name
                    inferred_user_id = parts[0]
                
                # 如果解析出的用户不存在，则使用 admin 或其他默认用户
                if not User.query.filter_by(username=inferred_user_id).first():
                    inferred_user_id = "admin" # fallback to admin

                # 尝试从文件名中恢复 original_name
                original_name = filename # 默认值
                if len(parts) > 2: # user_id_timestamp_originalName.ext
                    # 原始文件名可能是 timestamp_originalName.ext 或 originalName.ext
                    # 尝试去除 user_id 和 timestamp 部分
                    original_name_parts = filename.split("_", 2)
                    if len(original_name_parts) == 3: # 假设 user_id_timestamp_original_name.ext
                        original_name = original_name_parts[2]
                    else: # 假设 timestamp_original_name.ext
                        original_name_parts = filename.split("_", 1)
                        if len(original_name_parts) == 2:
                            original_name = original_name_parts[1]

                new_dataset = Dataset(
                    filename=filename,
                    original_name=original_name,
                    data_count=count,
                    upload_time=datetime.fromtimestamp(os.path.getmtime(file_path)), # 使用文件修改时间
                    user_id=inferred_user_id
                )
                db.session.add(new_dataset)
                print(f"Synced new file to DB: {filename} for user {inferred_user_id}")
            except Exception as e:
                print(f"Error syncing file {filename}: {e}")
    db.session.commit()

with app.app_context():
    db.create_all()
    # 迁移现有的模拟用户到数据库 (如果不存在)
    existing_users = {
        'admin': 'password123',
        'user1': 'pass456'
    }
    for uname, pword in existing_users.items():
        if not User.query.filter_by(username=uname).first():
            new_user = User(username=uname, password=pword)
            db.session.add(new_user)
    db.session.commit()
    sync_files_to_db() # 在应用启动时同步文件

# --- 添加数据管理路由 ---

# 1. 数据管理页面
@app.route('/data-management')
def data_management():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('data_management.html')

# 2. 上传数据 API
@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '无文件部分'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'})

    try:
        # 生成唯一文件名保存到磁盘
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        safe_name = secure_filename(file.filename)
        saved_filename = f"{session['username']}_{timestamp}_{safe_name}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        file.save(file_path)
        
        # 读取数据以获取点数 (简单解析逗号分隔或换行)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().replace('\n', ',')
            # 简单的清洗逻辑
            data_points = [x for x in content.split(',') if x.strip()]
            count = len(data_points)

        # 写入数据库
        new_dataset = Dataset(
            filename=saved_filename,
            original_name=file.filename,
            data_count=count,
            user_id=session['username']
        )
        db.session.add(new_dataset)
        db.session.commit()
        
        return jsonify({'success': True, 'message': '上传成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# 3. 获取列表 API
@app.route('/api/data/list', methods=['GET'])
def list_datasets():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
        
    # 只查询当前用户的数据
    datasets = Dataset.query.filter_by(user_id=session['username']).order_by(Dataset.upload_time.desc()).all()
    
    return jsonify({
        'success': True,
        'data': [{
            'id': d.id,
            'name': d.original_name,
            'count': d.data_count,
            'time': d.upload_time.strftime('%Y-%m-%d %H:%M')
        } for d in datasets]
    })

# 4. 删除数据 API
@app.route('/api/data/delete/<int:id>', methods=['POST'])
def delete_dataset(id):
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
        
    dataset = Dataset.query.get_or_404(id)
    
    # 权限检查
    if dataset.user_id != session['username']:
        return jsonify({'success': False, 'error': '无权删除此文件'}), 403
        
    try:
        # 删除磁盘文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # 删除数据库记录
        db.session.delete(dataset)
        db.session.commit()
        return jsonify({'success': True, 'message': '删除成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# 5. 获取具体数据内容 API (供模型页面调用)
@app.route('/api/data/get/<int:id>', methods=['GET'])
def get_dataset_content(id):
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
        
    dataset = Dataset.query.get_or_404(id)
    if dataset.user_id != session['username']:
        return jsonify({'success': False, 'error': '无权访问'}), 403
        
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 简单的解析逻辑：支持换行或逗号分隔
            content = content.replace('\n', ',')
            data_values = [float(x.strip()) for x in content.split(',') if x.strip() and x.strip().replace('.','',1).isdigit()]
            
        return jsonify({'success': True, 'data': data_values})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    

####################################################################

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    
    if User.query.filter_by(username=username).first():
        return render_template('login.html', error='Username already exists')
    
    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()
    
    session['username'] = username
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username, password=password).first()
        if user:
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
        # 尝试解析JSON数据（更稳健的方式，兼容不同Flask/Werkzeug版本或被意外覆盖的request对象）
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        # 如果 get_json 没有返回内容，尝试从原始数据解析
        if data is None:
            raw = None
            try:
                raw = request.get_data(as_text=True)
            except Exception:
                raw = None

            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400

            try:
                import json
                data = json.loads(raw)
            except Exception:
                return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
        
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
        # 尝试解析JSON数据（更稳健的方式）
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data is None:
            raw = None
            try:
                raw = request.get_data(as_text=True)
            except Exception:
                raw = None

            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400

            try:
                import json
                data = json.loads(raw)
            except Exception:
                return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
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

# GO模型路由
@app.route('/model/go', methods=['GET', 'POST'])
def go_model():
    if 'username' in session:
        if request.method == 'POST':
            # 获取表单数据
            data_type = request.form.get('data_type')
            prediction_step = int(request.form.get('prediction_step', 5))
            
            # 获取合适的失效数据
            if data_type == 'id':
                failure_data = SAMPLE_FAILURE_DATA
            else:
                failure_data = SAMPLE_FAILURE_DATA
            
            # 执行GO模型预测
            try:
                # 估计模型参数
                a, b = go_model_parameter_estimation(failure_data)
                
                if a is None or b is None:
                    raise ValueError("GO模型参数估计失败")
                
                # 预测未来失效
                prediction_results = go_predict_future_failures(a, b, failure_data, prediction_step)
                
                # 计算模型准确率
                accuracy_metrics = go_calculate_model_accuracy(a, b, failure_data)
                
                # 生成可靠度曲线数据点
                if 'reliability_curve' in prediction_results and len(prediction_results['reliability_curve'][0]) > 0:
                    time_points, reliability_values = prediction_results['reliability_curve']
                    # 转换为numpy数组并取整，然后转换为列表
                    time_points = np.array(time_points).round(2).tolist()
                    reliability_values = np.array(reliability_values).round(4).tolist()
                    reliability_curve = list(zip(time_points, reliability_values))
                else:
                    reliability_curve = []
                
                # 确保predicted_intervals和cumulative_times是列表
                predicted_intervals = [round(float(interval), 2) for interval in prediction_results['predicted_intervals']]
                cumulative_times = [round(float(time), 2) for time in prediction_results['cumulative_times']]
                
                return render_template('go_model.html', 
                                     success=True,
                                     message='预测成功完成！',
                                     a=round(float(a), 4),
                                     b=round(float(b), 6),
                                     remaining_faults=round(float(prediction_results['remaining_faults']), 4),
                                     next_failure_time=round(float(prediction_results['next_failure_time']), 2) if prediction_results['next_failure_time'] else 'N/A',
                                     predicted_intervals=predicted_intervals,
                                     cumulative_times=cumulative_times,
                                     reliability_curve=reliability_curve,
                                     warning=prediction_results.get('warning'),
                                     mae=round(float(accuracy_metrics['mae']), 2),
                                     mse=round(float(accuracy_metrics['mse']), 2),
                                     rmse=round(float(accuracy_metrics['rmse']), 2),
                                     r2_score=round(float(accuracy_metrics['r2_score']), 4),
                                     accuracy=round(float(accuracy_metrics['accuracy']), 2))
            
            except Exception as e:
                return render_template('go_model.html', 
                                     success=False,
                                     message=f'预测过程中发生错误: {str(e)}')
        
        # GET请求，显示页面
        return render_template('go_model.html')
    return redirect(url_for('login'))


# NHPP模型路由
@app.route('/model/nhpp', methods=['GET', 'POST'])
def nhpp_model():
    if 'username' in session:
        if request.method == 'POST':
            data_type = request.form.get('data_type', 'id')
            prediction_step = int(request.form.get('prediction_step', 5))
            model_type = request.form.get('model_type', 'exponential')

            if data_type == 'id':
                failure_data = SAMPLE_FAILURE_DATA
            else:
                failure_data = SAMPLE_FAILURE_DATA

            try:
                params, used_model_type = nhpp_model_parameter_estimation(
                    failure_data, model_type=model_type
                )

                prediction_results = nhpp_predict_future_failures(
                    params, used_model_type, failure_data, prediction_step=prediction_step
                )

                accuracy_metrics = calculate_nhpp_model_accuracy(
                    params, used_model_type, failure_data
                )

                time_points, reliability_values = prediction_results['reliability_curve']
                reliability_curve = list(zip(
                    np.array(time_points).round(2).tolist(),
                    np.array(reliability_values).round(4).tolist()
                ))

                return render_template(
                    'nhpp_model.html',
                    success=True,
                    message='预测成功完成！',
                    model_type=used_model_type,
                    params=[round(float(p), 4) for p in params],
                    remaining_faults=round(float(prediction_results['remaining_faults']), 4),
                    total_faults=round(float(prediction_results['total_faults']), 4),
                    next_failure_time=round(float(prediction_results['next_failure_time']), 2) if prediction_results['next_failure_time'] else 'N/A',
                    predicted_intervals=[round(float(x), 2) for x in prediction_results['predicted_intervals']],
                    cumulative_times=[round(float(x), 2) for x in prediction_results['cumulative_times']],
                    reliability_curve=reliability_curve,
                    warning=prediction_results.get('warning'),
                    mae=round(float(accuracy_metrics['mae']), 2),
                    mse=round(float(accuracy_metrics['mse']), 2),
                    rmse=round(float(accuracy_metrics['rmse']), 2),
                    r2_score=round(float(accuracy_metrics['r2_score']), 4),
                    accuracy=round(float(accuracy_metrics['accuracy']), 2)
                )

            except Exception as e:
                return render_template(
                    'nhpp_model.html',
                    success=False,
                    message=f'预测过程中发生错误: {str(e)}'
                )

        return render_template('nhpp_model.html')
    return redirect(url_for('login'))

# GO模型API端点
@app.route('/api/go/predict', methods=['POST'])
def api_go_predict():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        # 尝试解析JSON数据
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data is None:
            raw = None
            try:
                raw = request.get_data(as_text=True)
            except Exception:
                raw = None

            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400

            try:
                import json
                data = json.loads(raw)
            except Exception:
                return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
        
        # 验证输入数据
        if not data:
            return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
        
        # 获取并验证参数
        data_type = data.get('data_type', 'id')
        prediction_step = data.get('prediction_step', 5)
        
        # 参数验证
        try:
            prediction_step = int(prediction_step)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400
        
        if prediction_step < 1 or prediction_step > 100:
            return jsonify({'success': False, 'error': '预测步长必须在1-100之间'}), 400
        
        # 优先使用请求中提供的 train_data
        train_data = data.get('train_data')
        test_data = data.get('test_data')
        if train_data and isinstance(train_data, list) and len(train_data) >= 2:
            failure_data = train_data
        else:
            if data_type == 'id':
                failure_data = SAMPLE_FAILURE_DATA
            else:
                failure_data = SAMPLE_FAILURE_DATA
        
        # 执行GO模型预测
        try:
            a, b = go_model_parameter_estimation(failure_data)
            
            if a is None or b is None:
                return jsonify({'success': False, 'error': 'GO模型参数估计失败'}), 400
            
            prediction_results = go_predict_future_failures(a, b, failure_data, prediction_step)
            
            # 计算模型准确率
            accuracy_metrics = go_calculate_model_accuracy(a, b, failure_data)
            
            # 准备响应数据
            # 确保所有数值都是Python标量类型
            predicted_intervals = [round(float(interval), 2) for interval in prediction_results['predicted_intervals']]
            cumulative_times = [round(float(time), 2) for time in prediction_results['cumulative_times']]
            
            response_data = {
                'success': True,
                'a': round(float(a), 4),
                'b': round(float(b), 6),
                'remaining_faults': round(float(prediction_results['remaining_faults']), 4),
                'next_failure_time': round(float(prediction_results['next_failure_time']), 2) if prediction_results['next_failure_time'] else None,
                'predicted_intervals': predicted_intervals,
                'cumulative_times': cumulative_times,
                'mae': round(float(accuracy_metrics['mae']), 2),
                'mse': round(float(accuracy_metrics['mse']), 2),
                'rmse': round(float(accuracy_metrics['rmse']), 2),
                'r2_score': round(float(accuracy_metrics['r2_score']), 4),
                'accuracy': round(float(accuracy_metrics['accuracy']), 2)
            }
            
            try:
                response_data['used_train_count'] = len(failure_data)
                response_data['used_train_sum'] = float(np.sum(failure_data)) if len(failure_data) > 0 else 0.0
                response_data['used_train_preview'] = failure_data[:10]
            except Exception:
                pass
            
            # 添加可靠度曲线数据
            if 'reliability_curve' in prediction_results and len(prediction_results['reliability_curve'][0]) > 0:
                time_points, reliability_values = prediction_results['reliability_curve']
                # 转换为numpy数组并取整，然后转换为列表
                time_points = np.array(time_points).round(2).tolist()
                reliability_values = np.array(reliability_values).round(4).tolist()
                response_data['reliability_curve'] = [{'time': float(t), 'reliability': float(r)} for t, r in zip(time_points, reliability_values)]
            
            # 添加警告信息
            if 'warning' in prediction_results:
                response_data['warning'] = prediction_results['warning']
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'模型预测错误: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

@app.route('/api/go/train', methods=['POST'])
def api_go_train():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401

    try:
        # 尝试解析JSON数据
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data is None:
            raw = None
            try:
                raw = request.get_data(as_text=True)
            except Exception:
                raw = None

            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400

            try:
                import json
                data = json.loads(raw)
            except Exception:
                return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
        
        train_data = data.get('train_data')

        if not train_data or not isinstance(train_data, list) or len(train_data) < 2:
            return jsonify({'success': False, 'error': '请提供至少2个失效时间点作为训练数据'}), 400

        a, b = go_model_parameter_estimation(train_data)
        resp = {'success': True, 'a': round(float(a), 4), 'b': round(float(b), 6)}
        try:
            resp['used_train_count'] = len(train_data)
            resp['used_train_sum'] = float(np.sum(train_data))
            resp['used_train_preview'] = train_data[:10]
        except Exception:
            pass
        return jsonify(resp)

    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

# BP神经网络模型路由
@app.route('/model/bp', methods=['GET', 'POST'])
def bp_model():
    if 'username' in session:
        if request.method == 'POST':
            # 获取表单数据
            data_type = request.form.get('data_type')
            prediction_step = int(request.form.get('prediction_step', 5))
            look_back = int(request.form.get('look_back', 5))
            hidden_size = int(request.form.get('hidden_size', 10))
            lr = float(request.form.get('lr', 0.05))
            epochs = int(request.form.get('epochs', 1500))
            
            # 获取合适的失效数据
            if data_type == 'id':
                failure_data = SAMPLE_FAILURE_DATA
            else:
                failure_data = SAMPLE_FAILURE_DATA
            
            # 执行BP模型预测
            try:
                # 训练模型
                model, min_val, max_val, train_losses = bp_train_model(
                    failure_data, look_back=look_back, hidden_size=hidden_size,
                    lr=lr, epochs=epochs, verbose=False
                )
                
                # 预测未来失效
                prediction_results = bp_predict_future_failures(
                    model, failure_data, prediction_step=prediction_step, look_back=look_back
                )
                
                # 计算模型准确率
                accuracy_metrics = bp_calculate_model_accuracy(
                    model, failure_data, look_back=look_back
                )
                
                return render_template('bp_model.html', 
                                     success=True,
                                     message='预测成功完成！',
                                     predicted_times=[round(t, 2) for t in prediction_results['predicted_times']],
                                     predicted_intervals=[round(interval, 2) for interval in prediction_results['predicted_intervals']],
                                     cumulative_times=[round(time, 2) for time in prediction_results['cumulative_times']],
                                     next_failure_time=round(prediction_results['next_failure_time'], 2) if prediction_results['next_failure_time'] else 'N/A',
                                     mae=round(accuracy_metrics['mae'], 2),
                                     mse=round(accuracy_metrics['mse'], 2),
                                     rmse=round(accuracy_metrics['rmse'], 2),
                                     r2_score=round(accuracy_metrics['r2_score'], 4),
                                     accuracy=round(accuracy_metrics['accuracy'], 2))
            
            except Exception as e:
                return render_template('bp_model.html', 
                                     success=False,
                                     message=f'预测过程中发生错误: {str(e)}')
        
        # GET请求，显示页面
        return render_template('bp_model.html')
    return redirect(url_for('login'))

# BP模型API端点 - 训练
@app.route('/api/bp/train', methods=['POST'])
def api_bp_train():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401

    try:
        # 尝试解析JSON数据
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data is None:
            raw = None
            try:
                raw = request.get_data(as_text=True)
            except Exception:
                raw = None

            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400

            try:
                import json
                data = json.loads(raw)
            except Exception:
                return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
        
        train_data = data.get('train_data')
        look_back = data.get('look_back', 5)
        hidden_size = data.get('hidden_size', 10)
        lr = data.get('lr', 0.05)
        epochs = data.get('epochs', 1500)

        if not train_data or not isinstance(train_data, list) or len(train_data) < 6:
            return jsonify({'success': False, 'error': f'请提供至少{look_back + 1}个失效时间点作为训练数据'}), 400

        try:
            look_back = int(look_back)
            hidden_size = int(hidden_size)
            lr = float(lr)
            epochs = int(epochs)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400

        if look_back < 1 or look_back > 20:
            return jsonify({'success': False, 'error': '滑动窗口大小必须在1-20之间'}), 400
        if hidden_size < 1 or hidden_size > 50:
            return jsonify({'success': False, 'error': '隐含层神经元数必须在1-50之间'}), 400
        if lr <= 0 or lr > 1:
            return jsonify({'success': False, 'error': '学习率必须在0-1之间'}), 400
        if epochs < 1 or epochs > 10000:
            return jsonify({'success': False, 'error': '训练轮数必须在1-10000之间'}), 400

        model, min_val, max_val, train_losses = bp_train_model(
            train_data, look_back=look_back, hidden_size=hidden_size,
            lr=lr, epochs=epochs, verbose=False
        )
        
        resp = {
            'success': True,
            'look_back': look_back,
            'hidden_size': hidden_size,
            'lr': lr,
            'epochs': epochs,
            'final_loss': float(train_losses[-1]) if len(train_losses) > 0 else 0.0
        }
        try:
            resp['used_train_count'] = len(train_data)
            resp['used_train_sum'] = float(np.sum(train_data))
            resp['used_train_preview'] = train_data[:10]
        except Exception:
            pass
        return jsonify(resp)

    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

# BP模型API端点 - 预测
@app.route('/api/bp/predict', methods=['POST'])
def api_bp_predict():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        # 尝试解析JSON数据
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data is None:
            raw = None
            try:
                raw = request.get_data(as_text=True)
            except Exception:
                raw = None

            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400

            try:
                import json
                data = json.loads(raw)
            except Exception:
                return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
        
        # 验证输入数据
        if not data:
            return jsonify({'success': False, 'error': '无效的JSON数据'}), 400
        
        # 获取并验证参数
        data_type = data.get('data_type', 'id')
        prediction_step = data.get('prediction_step', 5)
        look_back = data.get('look_back', 5)
        hidden_size = data.get('hidden_size', 10)
        lr = data.get('lr', 0.05)
        epochs = data.get('epochs', 1500)
        
        # 参数验证
        try:
            prediction_step = int(prediction_step)
            look_back = int(look_back)
            hidden_size = int(hidden_size)
            lr = float(lr)
            epochs = int(epochs)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400
        
        if prediction_step < 1 or prediction_step > 100:
            return jsonify({'success': False, 'error': '预测步长必须在1-100之间'}), 400
        if look_back < 1 or look_back > 20:
            return jsonify({'success': False, 'error': '滑动窗口大小必须在1-20之间'}), 400
        if hidden_size < 1 or hidden_size > 50:
            return jsonify({'success': False, 'error': '隐含层神经元数必须在1-50之间'}), 400
        if lr <= 0 or lr > 1:
            return jsonify({'success': False, 'error': '学习率必须在0-1之间'}), 400
        if epochs < 1 or epochs > 10000:
            return jsonify({'success': False, 'error': '训练轮数必须在1-10000之间'}), 400
        
        # 优先使用请求中提供的 train_data
        train_data = data.get('train_data')
        test_data = data.get('test_data')
        if train_data and isinstance(train_data, list) and len(train_data) >= look_back + 1:
            failure_data = train_data
        else:
            if data_type == 'id':
                failure_data = SAMPLE_FAILURE_DATA
            else:
                failure_data = SAMPLE_FAILURE_DATA
            # 如果前端没有显式拆分测试集，则让后端自动划分验证集
            if not test_data:
                test_data = None
        
        # 执行BP模型预测
        try:
            # 训练模型
            model, min_val, max_val, train_losses = bp_train_model(
                failure_data, look_back=look_back, hidden_size=hidden_size,
                lr=lr, epochs=epochs, verbose=False
            )
            
            # 预测未来失效
            prediction_results = bp_predict_future_failures(
                model, failure_data, prediction_step=prediction_step, look_back=look_back
            )
            
            # 计算模型准确率
            # 如果前端没有提供足够长的 test_data，则传入 None，由函数内部自动划分验证集
            effective_test_data = None
            if isinstance(test_data, list) and len(test_data) >= look_back + 1:
                effective_test_data = test_data
            accuracy_metrics = bp_calculate_model_accuracy(
                model, failure_data, test_data=effective_test_data, look_back=look_back
            )
            
            # 准备响应数据
            response_data = {
                'success': True,
                'predicted_times': [round(float(t), 2) for t in prediction_results['predicted_times']],
                'predicted_intervals': [round(float(interval), 2) for interval in prediction_results['predicted_intervals']],
                'cumulative_times': [round(float(time), 2) for time in prediction_results['cumulative_times']],
                'next_failure_time': round(float(prediction_results['next_failure_time']), 2) if prediction_results['next_failure_time'] else None,
                'mae': round(float(accuracy_metrics['mae']), 2),
                'mse': round(float(accuracy_metrics['mse']), 2),
                'rmse': round(float(accuracy_metrics['rmse']), 2),
                'r2_score': round(float(accuracy_metrics['r2_score']), 4),
                'accuracy': round(float(accuracy_metrics['accuracy']), 2),
                'final_loss': float(train_losses[-1]) if len(train_losses) > 0 else 0.0
            }
            
            try:
                response_data['used_train_count'] = len(failure_data)
                response_data['used_train_sum'] = float(np.sum(failure_data)) if len(failure_data) > 0 else 0.0
                response_data['used_train_preview'] = failure_data[:10]
            except Exception:
                pass
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'模型预测错误: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500


# NHPP 模型 API 端点 - 训练参数
@app.route('/api/nhpp/train', methods=['POST'])
def api_nhpp_train():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401

    try:
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data is None:
            raw = None
            try:
                raw = request.get_data(as_text=True)
            except Exception:
                raw = None

            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400

            try:
                import json
                data = json.loads(raw)
            except Exception:
                return jsonify({'success': False, 'error': '无效的JSON数据'}), 400

        failure_data = data.get('train_data')
        model_type = data.get('model_type', 'exponential')

        if not failure_data or not isinstance(failure_data, list) or len(failure_data) < 2:
            return jsonify({'success': False, 'error': '请提供至少2个失效时间点作为训练数据'}), 400

        try:
            failure_data = [float(x) for x in failure_data]
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '训练数据必须为数值类型'}), 400

        params, used_model_type = nhpp_model_parameter_estimation(failure_data, model_type=model_type)

        resp = {
            'success': True,
            'params': [float(p) for p in params],
            'model_type': used_model_type,
            'used_train_count': len(failure_data),
            'used_train_sum': float(np.sum(failure_data)),
            'used_train_preview': failure_data[:10]
        }
        return jsonify(resp)

    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500


# NHPP 模型 API 端点 - 预测
@app.route('/api/nhpp/predict', methods=['POST'])
def api_nhpp_predict():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401

    try:
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data is None:
            raw = None
            try:
                raw = request.get_data(as_text=True)
            except Exception:
                raw = None

            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400

            try:
                import json
                data = json.loads(raw)
            except Exception:
                return jsonify({'success': False, 'error': '无效的JSON数据'}), 400

        if not data:
            return jsonify({'success': False, 'error': '无效的JSON数据'}), 400

        data_type = data.get('data_type', 'id')
        prediction_step = data.get('prediction_step', 5)
        model_type = data.get('model_type', 'exponential')
        failure_data = data.get('train_data')

        try:
            prediction_step = int(prediction_step)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '预测步长必须为整数'}), 400

        if prediction_step < 1 or prediction_step > 100:
            return jsonify({'success': False, 'error': '预测步长必须在1-100之间'}), 400

        # 获取失效数据：优先使用前端提供的数据，否则使用示例数据
        if failure_data and isinstance(failure_data, list) and len(failure_data) >= 2:
            try:
                failure_data = [float(x) for x in failure_data]
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': '失效数据必须为数值类型'}), 400
        else:
            if data_type == 'id':
                failure_data = SAMPLE_FAILURE_DATA
            else:
                failure_data = SAMPLE_FAILURE_DATA

        try:
            params, used_model_type = nhpp_model_parameter_estimation(failure_data, model_type=model_type)

            prediction_results = nhpp_predict_future_failures(
                params, used_model_type, failure_data, prediction_step=prediction_step
            )

            accuracy_metrics = calculate_nhpp_model_accuracy(
                params, used_model_type, failure_data
            )

            time_points, reliability_values = prediction_results['reliability_curve']
            reliability_curve = [
                {'time': float(t), 'reliability': float(r)}
                for t, r in zip(time_points, reliability_values)
            ]

            response_data = {
                'success': True,
                'params': [float(p) for p in params],
                'model_type': used_model_type,
                'remaining_faults': float(prediction_results['remaining_faults']),
                'total_faults': float(prediction_results['total_faults']),
                'next_failure_time': float(prediction_results['next_failure_time']) if prediction_results['next_failure_time'] else None,
                'predicted_intervals': [float(x) for x in prediction_results['predicted_intervals']],
                'cumulative_times': [float(x) for x in prediction_results['cumulative_times']],
                'reliability_curve': reliability_curve,
                'mae': float(accuracy_metrics['mae']),
                'mse': float(accuracy_metrics['mse']),
                'rmse': float(accuracy_metrics['rmse']),
                'r2_score': float(accuracy_metrics['r2_score']),
                'accuracy': float(accuracy_metrics['accuracy'])
            }

            if prediction_results.get('warning'):
                response_data['warning'] = prediction_results['warning']

            response_data['used_train_count'] = len(failure_data)
            response_data['used_train_sum'] = float(np.sum(failure_data))
            response_data['used_train_preview'] = list(failure_data[:10])

            return jsonify(response_data)

        except Exception as e:
            return jsonify({'success': False, 'error': f'模型预测错误: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

# ARIMA模型路由
@app.route('/model/arima', methods=['GET', 'POST'])
def arima_model():
    if 'username' in session:
        return render_template('arima_model.html')
    return redirect(url_for('login'))

# ARIMA模型API端点 - 训练
@app.route('/api/arima/train', methods=['POST'])
def api_arima_train():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None
        
        if data is None:
            raw = request.get_data(as_text=True)
            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400
            import json
            data = json.loads(raw)
        
        train_data = data.get('train_data')
        order = data.get('order', [1, 1, 1])
        
        if not train_data or not isinstance(train_data, list) or len(train_data) < 5:
            return jsonify({'success': False, 'error': 'ARIMA模型至少需要5个数据点'}), 400
        
        try:
            train_data = [float(x) for x in train_data]
            order = tuple([int(x) for x in order]) if isinstance(order, list) else order
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '数据格式错误'}), 400
        
        model, metrics = arima_train_model(train_data, order=order)
        
        return jsonify({
            'success': True,
            'aic': metrics.get('aic', 0.0),
            'bic': metrics.get('bic', 0.0),
            'order': order,
            'used_train_count': len(train_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

# ARIMA模型API端点 - 预测
@app.route('/api/arima/predict', methods=['POST'])
def api_arima_predict():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None
        
        if data is None:
            raw = request.get_data(as_text=True)
            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400
            import json
            data = json.loads(raw)
        
        train_data = data.get('train_data')
        prediction_step = data.get('prediction_step', 5)
        order = data.get('order', [1, 1, 1])
        
        if not train_data or len(train_data) < 5:
            train_data = SAMPLE_FAILURE_DATA
        
        try:
            train_data = [float(x) for x in train_data]
            prediction_step = int(prediction_step)
            order = tuple([int(x) for x in order]) if isinstance(order, list) else order
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400
        
        if prediction_step < 1 or prediction_step > 100:
            return jsonify({'success': False, 'error': '预测步长必须在1-100之间'}), 400
        
        model, _ = arima_train_model(train_data, order=order)
        prediction_results = arima_predict_future_failures(model, train_data, prediction_step)
        accuracy_metrics = calculate_arima_accuracy(model, train_data)
        
        return jsonify({
            'success': True,
            'predicted_times': [float(x) for x in prediction_results['predicted_times']],
            'predicted_intervals': [float(x) for x in prediction_results['predicted_intervals']],
            'cumulative_times': [float(x) for x in prediction_results['cumulative_times']],
            'next_failure_time': float(prediction_results['next_failure_time']) if prediction_results['next_failure_time'] else None,
            'mae': float(accuracy_metrics['mae']),
            'mse': float(accuracy_metrics['mse']),
            'rmse': float(accuracy_metrics['rmse']),
            'r2_score': float(accuracy_metrics['r2_score']),
            'accuracy': float(accuracy_metrics['accuracy'])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

# GM11模型路由
@app.route('/model/gm11', methods=['GET', 'POST'])
def gm11_model():
    if 'username' in session:
        return render_template('gm11_model.html')
    return redirect(url_for('login'))

# GM11模型API端点 - 训练
@app.route('/api/gm11/train', methods=['POST'])
def api_gm11_train():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None
        
        if data is None:
            raw = request.get_data(as_text=True)
            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400
            import json
            data = json.loads(raw)
        
        train_data = data.get('train_data')
        
        if not train_data or not isinstance(train_data, list) or len(train_data) < 4:
            return jsonify({'success': False, 'error': 'GM11模型至少需要4个数据点'}), 400
        
        try:
            train_data = [float(x) for x in train_data]
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '数据格式错误'}), 400
        
        params = gm11_train_model(train_data)
        
        return jsonify({
            'success': True,
            'a': params['a'],
            'b': params['b'],
            'x0_1': params['x0_1'],
            'used_train_count': len(train_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

# GM11模型API端点 - 预测
@app.route('/api/gm11/predict', methods=['POST'])
def api_gm11_predict():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None
        
        if data is None:
            raw = request.get_data(as_text=True)
            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400
            import json
            data = json.loads(raw)
        
        train_data = data.get('train_data')
        prediction_step = data.get('prediction_step', 5)
        
        if not train_data or len(train_data) < 4:
            train_data = SAMPLE_FAILURE_DATA
        
        try:
            train_data = [float(x) for x in train_data]
            prediction_step = int(prediction_step)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400
        
        if prediction_step < 1 or prediction_step > 100:
            return jsonify({'success': False, 'error': '预测步长必须在1-100之间'}), 400
        
        params = gm11_train_model(train_data)
        prediction_results = gm11_predict_future_failures(params, train_data, prediction_step)
        accuracy_metrics = calculate_gm11_accuracy(params, train_data)
        
        return jsonify({
            'success': True,
            'a': float(params['a']),
            'b': float(params['b']),
            'predicted_times': [float(x) for x in prediction_results['predicted_times']],
            'predicted_intervals': [float(x) for x in prediction_results['predicted_intervals']],
            'cumulative_times': [float(x) for x in prediction_results['cumulative_times']],
            'next_failure_time': float(prediction_results['next_failure_time']) if prediction_results['next_failure_time'] else None,
            'mae': float(accuracy_metrics['mae']),
            'mse': float(accuracy_metrics['mse']),
            'rmse': float(accuracy_metrics['rmse']),
            'r2_score': float(accuracy_metrics['r2_score']),
            'accuracy': float(accuracy_metrics['accuracy'])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

# SVR模型路由
@app.route('/model/svr', methods=['GET', 'POST'])
def svr_model():
    if 'username' in session:
        return render_template('svr_model.html')
    return redirect(url_for('login'))

# SVR模型API端点 - 训练
@app.route('/api/svr/train', methods=['POST'])
def api_svr_train():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None
        
        if data is None:
            raw = request.get_data(as_text=True)
            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400
            import json
            data = json.loads(raw)
        
        train_data = data.get('train_data')
        look_back = data.get('look_back', 5)
        kernel = data.get('kernel', 'rbf')
        C = data.get('C', 100.0)
        gamma = data.get('gamma', 'scale')
        epsilon = data.get('epsilon', 0.1)
        
        if not train_data or not isinstance(train_data, list) or len(train_data) < look_back + 1:
            return jsonify({'success': False, 'error': f'SVR模型至少需要{look_back + 1}个数据点'}), 400
        
        try:
            train_data = [float(x) for x in train_data]
            look_back = int(look_back)
            C = float(C)
            epsilon = float(epsilon)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400
        
        model, scaler, train_metrics = svr_train_model(
            train_data,
            look_back=look_back,
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon
        )
        
        return jsonify({
            'success': True,
            'train_mae': train_metrics['mae'],
            'train_mse': train_metrics['mse'],
            'train_rmse': train_metrics['rmse'],
            'train_r2_score': train_metrics['r2_score'],
            'train_accuracy': train_metrics['accuracy'],
            'used_train_count': len(train_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

# SVR模型API端点 - 预测
@app.route('/api/svr/predict', methods=['POST'])
def api_svr_predict():
    if 'username' not in session:
        return jsonify({'success': False, 'error': '未登录'}), 401
    
    try:
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None
        
        if data is None:
            raw = request.get_data(as_text=True)
            if not raw:
                return jsonify({'success': False, 'error': '请求必须包含JSON数据'}), 400
            import json
            data = json.loads(raw)
        
        train_data = data.get('train_data')
        prediction_step = data.get('prediction_step', 5)
        look_back = data.get('look_back', 5)
        kernel = data.get('kernel', 'rbf')
        C = data.get('C', 100.0)
        gamma = data.get('gamma', 'scale')
        epsilon = data.get('epsilon', 0.1)
        
        if not train_data or len(train_data) < look_back + 1:
            train_data = SAMPLE_FAILURE_DATA
        
        try:
            train_data = [float(x) for x in train_data]
            prediction_step = int(prediction_step)
            look_back = int(look_back)
            C = float(C)
            epsilon = float(epsilon)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': '参数类型错误'}), 400
        
        if prediction_step < 1 or prediction_step > 100:
            return jsonify({'success': False, 'error': '预测步长必须在1-100之间'}), 400
        
        model, scaler, _ = svr_train_model(
            train_data,
            look_back=look_back,
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon
        )
        
        prediction_results = svr_predict_future_failures(
            model, scaler, train_data, prediction_step, look_back
        )
        accuracy_metrics = calculate_svr_accuracy(model, scaler, train_data, look_back)
        
        return jsonify({
            'success': True,
            'predicted_times': [float(x) for x in prediction_results['predicted_times']],
            'predicted_intervals': [float(x) for x in prediction_results['predicted_intervals']],
            'cumulative_times': [float(x) for x in prediction_results['cumulative_times']],
            'next_failure_time': float(prediction_results['next_failure_time']) if prediction_results['next_failure_time'] else None,
            'mae': float(accuracy_metrics['mae']),
            'mse': float(accuracy_metrics['mse']),
            'rmse': float(accuracy_metrics['rmse']),
            'r2_score': float(accuracy_metrics['r2_score']),
            'accuracy': float(accuracy_metrics['accuracy'])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500

if __name__ == '__main__':
    # 确保static目录存在用于保存图表
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
