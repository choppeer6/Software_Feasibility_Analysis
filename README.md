# 可靠性分析工具

[TOC]

## 登录

账号：admin
密码：password123

## 运行环境

- Python 3.8+（已在项目中用 Python 3.13 测试）
- 依赖库：numpy, matplotlib, flask（如果运行 web 服务）

## 如何启动后端服务

在项目根目录下执行：

```powershell
python app.py
```

服务默认在 `http://127.0.0.1:5000` 启动。

## API 快速调试示例

- 预测接口：`POST /api/jm/predict`
  - JSON 请求示例：

```json
{
  "data_type": "custom",
  "prediction_step": 5,
  "ex": 0.001,
  "ey": 0.001,
  "train_data": [9,12,11,4,7,2,5,8]
}
```

## 界面模型展示

### 登录界面

>http://127.0.0.1:5000/login
>
<<<<<<< Updated upstream


=======
>>>>>>> Stashed changes

### 菜单

> http://127.0.0.1:5000/dashboard
>
<<<<<<< Updated upstream
> 
=======
>>>>>>> Stashed changes

###  模型

> http://127.0.0.1:5000/models
>


#### JM模型
> http://127.0.0.1:5000/model/jm
<<<<<<< Updated upstream
>
=======



# 默认数据添加训练比例方法

这是一个非常好的改进想法。让所有模型都支持“默认数据集”的比例划分，可以更方便地验证模型的泛化能力。

基于您提供的文件，我将分别针对 **BP 神经网络模型 (`bp_model.html`)** 和 **NHPP 模型 (`nhpp_model.html`)** 提供修改方案。修改逻辑与您之前修改 JM 模型完全一致：

1. **HTML层**：将滑块移出“自定义”区域，使其全局可见。
2. **JS层**：修改加载逻辑，无论选择什么数据集，都读取滑块值进行切割。

------

### 1. 修改 BP 神经网络 (`bp_model.html`)

BP 模型最需要划分训练集和测试集来验证效果。

#### 第一步：修改 HTML (移动滑块)

找到 `id="dataset-content"` 区域：

1. 删除 `#custom-dataset-group` 内部的 `slider-container` div。
2. 将其移动到 `#custom-dataset-group` 的**下方**（即同级），并添加 `onchange` 事件。

HTML

```
<div id="custom-dataset-group" style="display: none;">
    <div class="form-group">
        <label for="train-data-input">输入失效时间数据...</label>
        <textarea id="train-data-input" ...></textarea>
    </div>
    <div class="form-group">
        <button id="load-dataset-btn">加载数据集</button>
    </div>
</div>

<div class="slider-container" style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 15px;">
    <div class="slider-label" style="display: flex; justify-content: space-between; margin-bottom: 5px;">
        <span>训练数据比例: <strong id="train-ratio-value" style="color: var(--primary-color);">70%</strong></span>
    </div>
    <input type="range" id="train-ratio-slider" min="10" max="90" value="70" style="width: 100%;" onchange="reloadCurrentDatasetSplit()">
</div>
```

#### 第二步：修改 JavaScript

在 `<script>` 标签内，添加/替换以下两个函数：

JavaScript

```
    // 1. 【新增】辅助函数：滑块拖动时重新划分
    function reloadCurrentDatasetSplit() {
        const select = document.getElementById('dataset-select');
        if (select.value !== 'custom') {
            loadDataset(select.value);
        } else if (document.getElementById('train-data-input').value.trim() !== "") {
            loadDataset('custom');
        }
        document.getElementById('train-ratio-value').textContent = document.getElementById('train-ratio-slider').value + '%';
    }

    // 2. 【替换】loadDataset 函数 (核心修改)
    function loadDataset(datasetName) {
        try {
            let dataset;
            // 获取原始数据
            if (datasetName === 'custom') {
                dataset = parseCustomDataset();
                if (!dataset) return false;
            } else if (datasetName in DATASETS) {
                dataset = DATASETS[datasetName];
            } else {
                showMessage(`未知的数据集: ${datasetName}`, 'error');
                return false;
            }
            
            currentDataset = dataset;
            
            // 【核心修改逻辑】：无论 datasetName 是什么，都读取滑块进行划分
            const ratio = parseInt(document.getElementById('train-ratio-slider').value);
            const splitResult = splitTrainTestData(dataset, ratio);
            
            trainData = splitResult.train;
            testData = splitResult.test;
            
            // 更新提示信息
            const typeName = datasetName === 'default' ? '默认数据集' : (datasetName === 'custom' ? '自定义数据集' : datasetName);
            showMessage(`${typeName} 加载成功 (训练集:${trainData.length}, 测试集:${testData.length})`, 'success');
            
            updateDatasetPreview(dataset);
            
            // 在预览区显示划分详情
            const preview = document.getElementById('dataset-preview');
            preview.innerHTML += `<div style="margin-top:5px; font-size:0.9em; color:green;">当前划分: 前 ${trainData.length} 个训练，后 ${testData.length} 个验证</div>`;
            
            modelTrained = false;
            updateCharts(trainData, []);
            return true;
        } catch (error) { showMessage(`加载数据集失败: ${error.message}`, 'error'); return false; }
    }
```

------

### 2. 修改 NHPP 模型 (`nhpp_model.html`)

NHPP 的代码比较紧凑，我们需要展开修改。

#### 第一步：修改 HTML (移动滑块)

操作同上，将滑块移出 `custom-dataset-group`。

HTML

```
<div id="custom-dataset-group" style="display: none;">
    <div class="form-group"><label>输入数据:</label><textarea id="train-data-input" rows="5"></textarea></div>
    <button id="load-dataset-btn">加载数据</button>
</div>

<div class="slider-container" style="margin-top: 10px; padding-top:10px; border-top:1px solid #eee;">
    <label>训练比例 <span id="train-ratio-value" style="color:#1890ff; font-weight:bold;">70%</span></label>
    <input type="range" id="train-ratio-slider" min="10" max="90" value="70" style="width:100%;" onchange="reloadCurrentDatasetSplit()">
</div>
```

#### 第二步：修改 JavaScript

NHPP 的 JS 代码写得很简略，我们需要替换 `loadDS` 函数并添加 `reloadCurrentDatasetSplit`。

JavaScript

```
    // 1. 【新增】重新划分函数
    function reloadCurrentDatasetSplit() {
        const val = document.getElementById('dataset-select').value;
        loadDS(val);
        document.getElementById('train-ratio-value').innerText = document.getElementById('train-ratio-slider').value + '%';
    }

    // 2. 【替换】loadDS 函数
    function loadDS(n) {
        let d;
        // 获取数据源
        if (n === 'custom') {
            const input = document.getElementById('train-data-input').value;
            if (!input.trim()) return false;
            d = input.split(',').map(Number).filter(x => x > 0);
        } else {
            d = DATASETS[n];
        }

        if (!d || d.length < 4) {
            msg("无效数据 (至少需要4个点)", 'error');
            return false;
        }

        // 获取滑块比例
        const ratio = parseInt(document.getElementById('train-ratio-slider').value) / 100;
        const splitIdx = Math.floor(d.length * ratio);
        
        // 确保训练集至少有3个点
        const safeIdx = Math.max(3, splitIdx);

        // 执行划分
        trainData = d.slice(0, safeIdx);
        testData = d.slice(safeIdx);

        document.getElementById('dataset-preview').innerHTML = 
            `总数据: ${d.length}点<br><span style="color:green">训练: ${trainData.length}, 验证: ${testData.length}</span>`;
        
        upCharts([], trainData, []);
        msg(`加载成功 (训练比 ${Math.round(ratio*100)}%)`, 'success');
        return true;
    }
    
    // 3. 【补充】确保滑块拖动时数字变化 (可选，增强体验)
    document.getElementById('train-ratio-slider').addEventListener('input', function() {
        document.getElementById('train-ratio-value').innerText = this.value + '%';
    });
```

------

### 3. 关于 GO 模型

虽然您没有上传 `go_model.html`，但修改逻辑与 JM 模型完全一样：

1. **HTML**: 把滑块从 `display:none` 的 div 里拿出来。
2. **JS**: 复制 JM 模型的 `loadDataset` 和 `reloadCurrentDatasetSplit` 逻辑覆盖过去即可。

### 总结

完成上述修改后，您的所有模型页面将具备统一的交互体验：

- **默认数据集**也可以拖动滑块调整 70%/30% 或 80%/20%。
- **后端 API** 会自动接收切割后的 `trainData` 进行训练，并使用 `testData` 计算验证集精度（MSE/MAE等）。
>>>>>>> Stashed changes
