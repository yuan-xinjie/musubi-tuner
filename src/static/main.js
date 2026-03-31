const state = {
    currentPage: 'home',
    tasks: [],
    templates: {},
    currentTask: null,
    isEditMode: false,
    oldName: '',
    last_rendered_template: null,
    backendRunning: false,
    activeLogLine: null // Terminus Model 核心状态：当前活动行
};

// 默认值映射
const DEFAULTS_QWEN = {
    'learning_rate': '1e-4',
    'max_train_epochs': 40,
    'save_every_n_epochs': 2,
    'sample_every_n_epochs': 2,
    'timestep_sampling': 'qwen_shift',
    'loraplus_lr_ratio': 3,
    'network_dim': 32,
    'network_alpha': 16,
    'blocks_to_swap': 20,
    'network_weights': ''
};

const DEFAULTS_WAN = {
    'learning_rate': '1e-4',
    'max_train_epochs': 10,
    'save_every_n_epochs': 2,
    'timestep_sampling': 'shift',
    'loraplus_lr_ratio': 4,
    'network_dim': 32,
    'network_alpha': 16,
    'blocks_to_swap': 16,
    'network_weights': '',
    'max_data_loader_n_workers': 8,
    'task': 't2v-A14B'
};

// 内部隐藏键，不渲染为常规表单项
const HIDDEN_KEYS = ['datasets', 'samples', 'general', 'model_version', 'dim_from_weights', 'caption_extension', 'enable_bucket', 'bucket_no_upscale'];

// 翻译映射表 (Localization Map)
const LABEL_MAP = {
    'output_name': 'Lora名',
    'output_dir': '输出文件夹',
    'max_train_epochs': '训练轮数',
    'save_every_n_epochs': '每*次保存',
    'sample_every_n_epochs': '每*次采样',
    'learning_rate': '学习率',
    'timestep_sampling': '时间采样方法',
    'loraplus_lr_ratio': '学习率倍数',
    'blocks_to_swap': '交换块',
    'network_dim': 'Network Dim',
    'network_alpha': 'Network Alpha',
    'dit': '底模路径',
    'vae': 'VAE路径',
    'text_encoder': '文本编码器路径',
    'network_weights': '从已有的权重继续训练',
    'resolution': '分辨率',
    'dataset_config': '数据集配置',
    'batch_size': '批次大小',
    'num_repeats': '重复次数',
    'qwen_image_edit_control_resolution': '控制图分辨率',
    'image_directory': '目标图路径',
    'cache_directory': '缓存路径',
    'control_directory': '控制图路径',
    'width': '宽',
    'height': '高',
    'sample_steps': '采样步数',
    'guidance_scale': 'CFG',
    'seed': '种子',
    'discrete_flow_shift': '偏移',
    'control_image_path': '控制图路径',
    'prompt': '提示词',
    'frame_count': '帧数',
    'P1': '控制图1',
    'P2': '控制图2',
    'P3': '控制图3',
    'video_directory': '视频路径',
    'max_frames': '最大帧数',
    'target_frames': '目标帧数',
    'frame_sample': '帧采样',
    'frame_extraction': '帧提取方式',
    'task': '任务类型(task)',
    'clip': 'CLIP路径',
    't5': 'T5路径',
    'min_timestep': '最小时间步',
    'max_timestep': '最大时间步',
    'max_data_loader_n_workers': '数据加载线程数'
};

const getLabel = (key) => LABEL_MAP[key] || key;

// 工具函数
const $ = (id) => document.getElementById(id);
const showToast = (msg) => {
    const toast = $('toast');
    toast.innerText = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 5000);
};

// 需要文件/文件夹选择按钮的字段配置
const PATH_PICKER_CONFIG = {
    // 文件选择器（底模、VAE、文本编码器）
    'dit': { mode: 'file', title: '选择底模文件', filetypes: ['.safetensors', '.pt', '.ckpt'] },
    'vae': { mode: 'file', title: '选择VAE文件', filetypes: ['.safetensors', '.pt'] },
    'text_encoder': { mode: 'file', title: '选择文本编码器', filetypes: ['.safetensors', '.pt'] },
    't5': { mode: 'file', title: '选择T5文件', filetypes: ['.safetensors', '.pt', '.pth'] },
    'clip': { mode: 'file', title: '选择CLIP文件', filetypes: ['.safetensors', '.pt'] },
    // 文件夹选择器
    'image_directory': { mode: 'folder', title: '选择目标图文件夹' },
    'video_directory': { mode: 'folder', title: '选择视频文件夹' },
    'cache_directory': { mode: 'folder', title: '选择缓存文件夹' },
    'control_directory': { mode: 'folder', title: '选择控制图文件夹' },
    'control_image_path': { mode: 'file', title: '选择控制图', filetypes: ['.png', '.jpg', '.jpeg', '.webp', '.bmp'] }
};

// 弹出文件/文件夹选择对话框
window.pickPath = async function (inputId, config) {
    try {
        const res = await fetch('/api/pick_path', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (data.path) {
            const input = document.getElementById(inputId) || document.querySelector(`input[name="${inputId}"]`);
            if (input) {
                input.value = data.path;
                input.dispatchEvent(new Event('input'));
                input.dispatchEvent(new Event('change'));
            }
        }
    } catch (e) {
        showToast('选择路径失败');
    }
}

window.updateCacheDir = (val) => {
    const cacheInput = document.getElementById('input_tmpl_datasets_0_cache_directory') || document.querySelector(`input[name="tmpl_datasets_0_cache_directory"]`);
    if (cacheInput && val) {
        const sep = val.includes('\\') ? '\\' : '/';
        const stripped = val.endsWith(sep) ? val.slice(0, -1) : val;
        cacheInput.value = stripped + sep + 'cache';
    }
};

// 生成带选择按钮的路径输入框 HTML
function renderPathInput(name, label, value, pickerKey) {
    const config = PATH_PICKER_CONFIG[pickerKey];
    const configJson = JSON.stringify(config).replace(/"/g, '&quot;');
    let extraAttr = '';
    if (name === 'tmpl_datasets_0_video_directory' || name === 'tmpl_datasets_0_image_directory') {
        extraAttr = `oninput="updateCacheDir(this.value)" onchange="updateCacheDir(this.value)"`;
    }
    return `
        <div class="form-group flex-grow">
            <label>${label}</label>
            <div class="path-input-wrapper">
                <input type="text" id="input_${name}" name="${name}" value="${value || ''}" class="path-input" spellcheck="false" ${extraAttr}>
                <button type="button" class="btn-pick" onclick="pickPath('input_${name}', JSON.parse('${configJson}'))" title="浏览...">
                    <i class="fas fa-folder-open"></i>
                </button>
            </div>
        </div>
    `;
}

// 初始化
async function init() {
    await fetchTemplates();
    await loadTasks();
    setupNavigation();
    setupEventListeners();
    initLogSocket();

    $('btn-stop-task').onclick = stopTask;
    $('btn-close-log').onclick = () => showLogPanel(false);

    // 基础连接检查
    setInterval(() => {
        if (!logSocket || logSocket.readyState !== WebSocket.OPEN) {
            initLogSocket();
        }
    }, 5000);
}

// 分辨率处理函数
function formatRes(val) {
    if (Array.isArray(val)) return `${val[0]} x ${val[1]}`;
    return val;
}

function parseRes(str) {
    if (!str || typeof str !== 'string') return [1024, 1024];
    const parts = str.split(/[xX]/).map(p => parseInt(p.trim()));
    return parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1]) ? parts : [1024, 1024];
}

// 获取模板
async function fetchTemplates() {
    try {
        const res = await fetch('/api/templates');
        state.templates = await res.json();
        const selectOptions = Object.keys(state.templates).map(t => ({ value: t, label: t }));
        renderCustomSelect('train-type-select-wrapper', 'train_type', selectOptions, selectOptions[0].value);
    } catch (err) {
        showToast('获取模板失败');
    }
}

// WebSocket 日志处理
let logSocket = null;
function initLogSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/logs`;

    if (logSocket) {
        try { logSocket.close(); } catch (e) { }
    }

    logSocket = new WebSocket(wsUrl);

    // 心跳保活
    if (logSocket.heartbeat) clearInterval(logSocket.heartbeat);
    logSocket.heartbeat = setInterval(() => {
        if (logSocket.readyState === WebSocket.OPEN) {
            logSocket.send('ping');
        }
    }, 30000);

    logSocket.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            const content = $('log-content');

            if (msg.type === 'init') {
                updateRunningState(msg.is_running);
                if (msg.history) {
                    clearLogs(content); // 彻底重连/初始化时清空并重置状态
                    appendLogs(content, msg.history);
                }
                return;
            }

            if (msg.type === 'state') {
                updateRunningState(msg.is_running);
                return;
            }

            if (msg.type === 'log') {
                const isScrolledToBottom = content.scrollHeight - content.clientHeight <= content.scrollTop + 50;
                appendLogs(content, msg.content);
                if (isScrolledToBottom) {
                    content.scrollTop = content.scrollHeight;
                }
            }
        } catch (e) {
            console.error('WebSocket message error:', e);
        }
    };

    logSocket.onclose = (e) => {
        if (logSocket.heartbeat) clearInterval(logSocket.heartbeat);
        if (!e.wasClean) {
            setTimeout(initLogSocket, 3000);
        }
    };
}

// 基础连接检查
setInterval(() => {
    // 仅在完全断开 (CLOSED) 或未初始化时重连
    // 避免干扰 CONNECTING (0) 或 OPEN (1) 状态
    if (!logSocket || logSocket.readyState === WebSocket.CLOSED) {
        initLogSocket();
    }
}, 5000);

function updateRunningState(isRunning) {
    state.backendRunning = isRunning;
    const dot = $('running-indicator');
    if (dot) {
        isRunning ? dot.classList.add('active') : dot.classList.remove('active');
    }
}

/**
 * 终极适配版日志渲染引擎
 * 针对 Windows 环境 (\r\n) 与 tqdm (\r) 的复杂嵌套进行了专项优化。
 */
function appendLogs(container, text) {
    if (!text) return;

    // 1. 预处理：将 Windows 的 \r\n 归一化为 \n，防止误跳覆盖逻辑
    const normalizedText = text.replace(/\r\n/g, '\n');

    // 2. 按物理换行拆分
    const parts = normalizedText.split('\n');

    parts.forEach((line, idx) => {
        // 如果当前没有活动行，或者遇到了 \n (idx > 0)
        if (!state.activeLogLine || idx > 0) {
            state.activeLogLine = document.createElement('div');
            container.appendChild(state.activeLogLine);
        }

        // 3. 处理行内 \r (tqdm 原地更新)
        // 核心规则：只有当行内确实含 \r 时，才提取最后一个 \r 后的内容并“覆盖”当前 DIV
        if (line.includes('\r')) {
            const rParts = line.split('\r');
            state.activeLogLine.textContent = rParts[rParts.length - 1];
        } else {
            // 普通文本：直接追加到当前物理 DIV
            state.activeLogLine.textContent += line;
        }
    });

    // 4. 内存自动回收 (1000行)
    while (container.childNodes.length > 1000) {
        container.removeChild(container.firstChild);
    }
}

// 统一清屏与状态重置
function clearLogs(container) {
    if (!container) container = $('log-content');
    container.innerHTML = '';
    state.activeLogLine = null;
}

function showLogPanel(show = true) {
    const navConsole = $('nav-console');
    const navHome = document.querySelector('.nav-links li[data-page="home"]');

    if (show) {
        document.body.classList.add('executing-mode');
        navConsole.classList.add('active');
        navHome.classList.remove('active');
    } else {
        document.body.classList.remove('executing-mode');
        navConsole.classList.remove('active');
        navHome.classList.add('active');
    }
}

async function stopTask() {
    try {
        const res = await fetch('/api/stop', { method: 'POST' });
        if (res.ok) {
            showToast('已发送停止指令');
        }
    } catch (e) {
        showToast('停止请求失败');
    }
}

// 模拟原生 Select 的辅助函数
function renderCustomSelect(container, name, options, selectedValue, onSelect) {
    const containerEl = typeof container === 'string' ? $(container) : container;
    if (!containerEl) return;
    const currentLabel = options.find(o => o.value === selectedValue)?.label || selectedValue;

    containerEl.innerHTML = `
        <div class="custom-select-container" id="container_${name}">
            <div class="custom-select-trigger" onclick="toggleDropdown('${name}')">
                <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${currentLabel}</span>
                <i class="fas fa-chevron-down" style="margin-left: 8px;"></i>
            </div>
            <div class="dropdown-menu">
                ${options.map(opt => `
                    <div class="dropdown-item ${opt.value === selectedValue ? 'selected' : ''}" 
                         onclick="selectDropdownOption('${name}', '${opt.value}', '${opt.label}')">
                        ${opt.label}
                    </div>
                `).join('')}
            </div>
            <input type="hidden" name="${name}" value="${selectedValue}" id="input_${name}">
        </div>
    `;
}

window.toggleDropdown = (name) => {
    const el = $(`container_${name}`);
    const isActive = el.classList.contains('active');
    document.querySelectorAll('.custom-select-container').forEach(c => c.classList.remove('active'));
    if (!isActive) el.classList.add('active');
};

window.toggleNoiseSwitch = (target) => {
    const low = document.getElementById('switch_noise_low');
    const high = document.getElementById('switch_noise_high');
    if (!low || !high) return;

    if (target === 'low') {
        if (!low.checked) low.checked = true; // prevent unchecking both
        high.checked = false;
    } else {
        if (!high.checked) high.checked = true;
        low.checked = false;
    }
    applyNoisePresets();
};

window.applyNoisePresets = () => {
    const taskInput = document.getElementById('input_fixed_task');
    const highInput = document.getElementById('switch_noise_high');
    const shiftInput = document.getElementById('input_fixed_discrete_flow_shift');
    const minInput = document.getElementById('input_fixed_min_timestep');
    const maxInput = document.getElementById('input_fixed_max_timestep');

    if (!taskInput || !highInput || !shiftInput || !minInput || !maxInput) return;

    const task = taskInput.value || '';
    const isHighNoise = highInput.checked;

    if (task.includes('i2v')) {
        shiftInput.value = 5;
        if (isHighNoise) {
            minInput.value = 900;
            maxInput.value = 1000;
        } else {
            minInput.value = 0;
            maxInput.value = 900;
        }
    } else if (task.includes('t2v')) {
        shiftInput.value = 12;
        if (isHighNoise) {
            minInput.value = 875;
            maxInput.value = 1000;
        } else {
            minInput.value = 0;
            maxInput.value = 875;
        }
    }
    // Only show toast if user actively clicked the switch or selected dropdown
    if (window.event && window.event.type !== 'DOMContentLoaded') {
        if (typeof showToast === 'function') showToast(`已应用 ${task} ${isHighNoise ? '高噪' : '低噪'} 默认参数`);
    }
};

window.selectDropdownOption = (name, value, label) => {
    const container = $(`container_${name}`);
    const input = $(`input_${name}`);
    const triggerSpan = container.querySelector('.custom-select-trigger span');

    input.value = value;
    triggerSpan.innerText = label;

    container.querySelectorAll('.dropdown-item').forEach(item => {
        item.classList.remove('selected');
        if (item.innerText === label) item.classList.add('selected');
    });

    container.classList.remove('active');

    if (name === 'train_type') {
        const currentData = getCurrentFormData();
        const isWan = value === 'Wan2.2';
        const activeDefaults = isWan ? DEFAULTS_WAN : DEFAULTS_QWEN;
        const oldDefaults = isWan ? DEFAULTS_QWEN : DEFAULTS_WAN;

        for (let k in activeDefaults) {
            // If the field matches the old default, or if it was entirely empty/unused by the previous form block
            if (currentData.fixed[k] === oldDefaults[k] || currentData.fixed[k] === '' || currentData.fixed[k] === undefined) {
                currentData.fixed[k] = activeDefaults[k];
            }
        }
        
        const oldType = $('input_train_type').dataset.oldType;
        if (oldType && oldType !== value) {
            currentData.template = JSON.parse(JSON.stringify(state.templates[value] || state.templates['Qwen-Image']));
            $('input_train_type').dataset.oldType = value;
        }

        renderFormFields(value, currentData.fixed, currentData.template);
        if (isWan) {
            setTimeout(() => { applyNoisePresets(); }, 50);
        }
    } else if (name === 'fixed_task') {
        applyNoisePresets();
    }
};

document.addEventListener('click', (e) => {
    if (!e.target.closest('.custom-select-container')) {
        document.querySelectorAll('.custom-select-container').forEach(c => c.classList.remove('active'));
    }
});

function getCurrentFormData() {
    const form = $('config-form');
    if (!form) return { fixed: {}, template: null };

    const formData = new FormData(form);
    const fixed = {};
    const rawTemplate = state.last_rendered_template ? JSON.parse(JSON.stringify(state.last_rendered_template)) : null;

    if (rawTemplate) {
        const ALL_KEYS = [
            'output_name', 'output_dir', 'network_weights',
            'dit', 'vae', 'text_encoder', 't5', 'clip', 'task',
            'max_train_epochs', 'save_every_n_epochs', 'sample_every_n_epochs',
            'learning_rate', 'timestep_sampling', 'loraplus_lr_ratio',
            'network_dim', 'network_alpha', 'blocks_to_swap',
            'min_timestep', 'max_timestep', 'discrete_flow_shift', 'max_data_loader_n_workers'
        ];
        for (let key of ALL_KEYS) {
            let val = formData.get(`fixed_${key}`);
            if (val === null) {
                const checkbox = document.querySelector(`input[name="fixed_${key}"][type="checkbox"]`);
                if (checkbox) val = 'false';
            }
            if (val !== null) fixed[key] = castType(key, val);
        }
        updateObjectFromForm(rawTemplate.general, 'tmpl_general', formData);
        if (rawTemplate.datasets && rawTemplate.datasets.length > 0) {
            updateObjectFromForm(rawTemplate.datasets[0], 'tmpl_datasets_0', formData);
        }
        rawTemplate.samples = (rawTemplate.samples || []).map((_, idx) => {
            const sample = rawTemplate.samples[idx];
            updateObjectFromForm(sample, `tmpl_samples_${idx}`, formData);
            return sample;
        });
    }
    return { fixed, template: rawTemplate };
}

async function loadTasks() {
    try {
        const res = await fetch('/api/tasks');
        state.tasks = await res.json();
        renderTaskList();
    } catch (err) {
        showToast('加载任务失败');
    }
}

async function fetchThumbnailData(outputName) {
    try {
        const res = await fetch(`/api/config/${outputName}`);
        if (!res.ok) return null;
        const data = await res.json();
        const imgDir = data.template_params.datasets[0].image_directory || data.template_params.datasets[0].video_directory;
        if (imgDir) {
            const metaRes = await fetch(`/api/thumbnail_meta?path=${encodeURIComponent(imgDir)}`);
            if (metaRes.ok) return await metaRes.json();
        }
    } catch (e) { }
    return null;
}

async function renderTaskList() {
    const list = $('task-list');
    if (state.tasks.length === 0) {
        list.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 5rem 0;">暂无任务，点击右上方新建</div>';
        return;
    }

    const cardsHtml = await Promise.all(state.tasks.map(async task => {
        const thumbData = await fetchThumbnailData(task.output_name);
        
        let mediaHtml = `<img src="https://placehold.co/200x200?text=No+Image" alt="">`;
        if (thumbData && thumbData.url) {
            if (thumbData.is_video) {
                mediaHtml = `<video src="${thumbData.url}" loop muted playsinline preload="metadata" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px 8px 0 0;" onmouseover="this.play()" onmouseout="this.pause(); this.currentTime=0;"></video>`;
            } else {
                mediaHtml = `<img src="${thumbData.url}" alt="" onerror="this.src='https://placehold.co/200x200?text=No+Image';this.onerror=null;">`;
            }
        }

        return `
            <div class="task-card">
                <div class="task-card-header">
                    <h3>${task.output_name}</h3>
                </div>
                <div class="task-card-body">
                    <div class="thumbnail-wrapper">
                        ${mediaHtml}
                    </div>
                </div>
                <div class="task-card-footer">
                    <div class="btn-row">
                        <button class="btn btn-ghost btn-xs grow" onclick="editTask('${task.output_name}')"><i class="fas fa-edit"></i> 编辑</button>
                        <button class="btn btn-ghost btn-xs grow" onclick="cloneTask('${task.output_name}')"><i class="fas fa-copy"></i> 克隆</button>
                        <button class="btn btn-danger-ghost btn-xs grow" onclick="deleteTask('${task.output_name}')"><i class="fas fa-trash"></i> 删除</button>
                    </div>
                    <div class="btn-row">
                        <button class="btn btn-ghost btn-xs grow" onclick="cacheTask('${task.output_name}')"><i class="fas fa-database"></i> 缓存</button>
                        <button class="btn btn-primary btn-xs grow" onclick="trainTask('${task.output_name}')"><i class="fas fa-play"></i> 训练</button>
                    </div>
                </div>
            </div>
        `;
    }));
    list.innerHTML = cardsHtml.join('');
}

async function cacheTask(name) {
    showLogPanel(true);
    const content = $('log-content');
    clearLogs(content);
    content.innerHTML = `[INFO] 正在为任务 "${name}" 开启缓存流程...\n------------------------------------------------\n`;
    state.activeLogLine = null; // 确保下一条日志是全新的
    try {
        const res = await fetch(`/api/cache/${name}`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            showToast(`启动失败: ${err.detail}`);
            $('log-content').innerHTML += `\n[ERROR] ${err.detail}\n`;
        }
    } catch (e) {
        showToast('启动命令失败');
    }
}

async function trainTask(name) {
    showLogPanel(true);
    const content = $('log-content');
    clearLogs(content);
    content.innerHTML = `[INFO] 正在为任务 "${name}" 开启训练流程...\n------------------------------------------------\n`;
    state.activeLogLine = null;
    try {
        const res = await fetch(`/api/train/${name}`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            showToast(`启动失败: ${err.detail}`);
            $('log-content').innerHTML += `\n[ERROR] ${err.detail}\n`;
        }
    } catch (e) {
        showToast('启动命令失败');
    }
}

async function cloneTask(name) {
    try {
        const res = await fetch(`/api/config/${name}`);
        if (!res.ok) throw new Error();
        const data = await res.json();

        const newName = `${name}_clone`;
        const fixed = JSON.parse(JSON.stringify(data.fixed_params));
        const template = JSON.parse(JSON.stringify(data.template_params));

        fixed.output_name = newName;
        fixed.output_dir = `./output/${newName}`;
        fixed.dataset_config = `./src/${newName}.toml`;
        fixed.sample_prompts = `./src/${newName}.toml`;

        let cloneType = 'Qwen-Image';
        if (template.datasets && template.datasets[0] && template.datasets[0].control_directory) {
            cloneType = 'Qwen-Image-Edit-2511';
        } else if (fixed.train_type === 'Wan2.2' || (fixed.model_version && fixed.model_version.toLowerCase().includes('wan')) || fixed.task) {
            cloneType = 'Wan2.2';
        }

        const saveRes = await fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                is_new: true,
                fixed_params: fixed,
                template_params: template,
                train_type: cloneType
            })
        });

        if (saveRes.ok) {
            showToast(`已克隆为 ${newName}`);
            loadTasks();
        } else {
            const err = await saveRes.json();
            showToast(err.detail || '克隆失败');
        }
    } catch (e) {
        showToast('克隆失败');
    }
}

function setupNavigation() {
    document.querySelectorAll('.nav-links li').forEach(li => {
        li.addEventListener('click', () => {
            const page = li.getAttribute('data-page');
            if (page === 'console') {
                showLogPanel(true);
                return;
            }
            if (page === 'config' && state.currentPage !== 'config') prepareNewTask();
            if (page === 'advanced') fetchRawConfig();
            switchPage(page);
        });
    });
}

function switchPage(page) {
    state.currentPage = page;
    if (page !== 'home') {
        document.body.classList.remove('executing-mode');
    }
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    const target = $(`page-${page}`);
    if (target) target.classList.add('active');
    document.querySelectorAll('.nav-links li').forEach(li => {
        li.classList.toggle('active', li.getAttribute('data-page') === page);
    });
}

function setupEventListeners() {
    $('btn-new-task').addEventListener('click', () => { prepareNewTask(); switchPage('config'); });
    $('btn-save-config').addEventListener('click', saveConfig);
    $('btn-save-raw-config').addEventListener('click', saveRawConfig);
}

async function fetchRawConfig() {
    try {
        const res = await fetch('/api/raw_config');
        if (!res.ok) throw new Error();
        const data = await res.json();
        $('raw-config-editor').value = data.content;
    } catch (e) {
        showToast('读取配置文件失败');
    }
}

async function saveRawConfig() {
    const content = $('raw-config-editor').value;
    try {
        const res = await fetch('/api/raw_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });
        if (res.ok) {
            showToast('已保存配置文件');
        } else {
            showToast('保存失败');
        }
    } catch (e) {
        showToast('保存请求失败');
    }
}

function prepareNewTask() {
    state.isEditMode = false;
    state.oldName = '';
    $('config-title').innerText = '新建任务';

    const defaultType = Object.keys(state.templates)[0];
    const selectOptions = Object.keys(state.templates).map(t => ({ value: t, label: t }));
    renderCustomSelect('train-type-select-wrapper', 'train_type', selectOptions, defaultType);
    
    $('input_train_type').dataset.oldType = defaultType;

    const activeDefaults = defaultType === 'Wan2.2' ? DEFAULTS_WAN : DEFAULTS_QWEN;
    const fixedDefaults = JSON.parse(JSON.stringify(activeDefaults));
    renderFormFields(defaultType, fixedDefaults);
}

async function editTask(name) {
    try {
        const res = await fetch(`/api/config/${name}`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        state.isEditMode = true;
        state.oldName = name;
        $('config-title').innerText = `编辑: ${name}`;
        let type = 'Qwen-Image';
        if (data.template_params.datasets && data.template_params.datasets[0].control_directory) {
            type = 'Qwen-Image-Edit-2511';
        } else if (data.fixed_params.train_type === 'Wan2.2' || (data.fixed_params.model_version && data.fixed_params.model_version.toLowerCase().includes('wan')) || data.fixed_params.task) {
            type = 'Wan2.2';
        }

        const selectOptions = Object.keys(state.templates).map(t => ({ value: t, label: t }));
        renderCustomSelect('train-type-select-wrapper', 'train_type', selectOptions, type);
        
        $('input_train_type').dataset.oldType = type;

        renderFormFields(type, data.fixed_params, data.template_params);
        switchPage('config');
    } catch (err) {
        showToast('加载失败');
    }
}

async function deleteTask(name) {
    if (!confirm(`确定删除 ${name}？`)) return;
    try {
        const res = await fetch(`/api/tasks/${name}`, { method: 'DELETE' });
        if (res.ok) { showToast('已删除'); loadTasks(); }
    } catch (err) { showToast('删除失败'); }
}

const FIXED_BLOCKS_QWEN = [
    { keys: ['output_name', 'output_dir', 'network_weights'], class: 'block-medium' },
    { keys: ['max_train_epochs', 'save_every_n_epochs', 'sample_every_n_epochs'], class: 'block-short' },
    { keys: ['learning_rate', 'timestep_sampling', 'loraplus_lr_ratio'], class: 'block-medium' },
    { keys: ['network_dim', 'network_alpha', 'blocks_to_swap'], class: 'block-short' },
    { keys: ['dit', 'vae', 'text_encoder'], class: 'block-wide' }
];

const FIXED_BLOCKS_WAN = [
    { keys: ['task', 'ui_noise_level', 'output_name', 'output_dir'], class: 'block-medium' },
    { keys: ['max_train_epochs', 'save_every_n_epochs', 'network_dim', 'network_alpha'], class: 'block-short' },
    { keys: ['learning_rate', 'timestep_sampling', 'blocks_to_swap', 'loraplus_lr_ratio'], class: 'block-medium' },
    { keys: ['min_timestep', 'max_timestep', 'discrete_flow_shift', 'max_data_loader_n_workers'], class: 'block-short' },
    { keys: ['dit', 'vae', 't5', 'clip'], class: 'block-wide' }
];

function renderFormFields(type, fixedData = {}, templateData = null) {
    const baseTemplate = state.templates[type];
    let template = templateData ? mergeTemplateWithNewType(templateData, baseTemplate) : JSON.parse(JSON.stringify(baseTemplate));
    state.last_rendered_template = template;

    const isWan = type === 'Wan2.2';
    const activeBlocks = isWan ? FIXED_BLOCKS_WAN : FIXED_BLOCKS_QWEN;
    const activeDefaults = isWan ? DEFAULTS_WAN : DEFAULTS_QWEN;

    const fixedContainer = $('fixed-params-grid');
    fixedContainer.innerHTML = activeBlocks.map(block => {
        return `<div class="params-block ${block.class}">${block.keys.map(key => {
            let value = (fixedData[key] !== undefined && fixedData[key] !== '') ? fixedData[key] : (activeDefaults[key] !== undefined ? activeDefaults[key] : '');
            if (key === 'timestep_sampling') {
                const options = [
                    { value: 'qwen_shift', label: 'qwen_shift' },
                    { value: 'flux_shift', label: 'flux_shift' },
                    { value: 'qinglong_qwen', label: 'qinglong_qwen' },
                    { value: 'shift', label: 'shift' }
                ];
                setTimeout(() => {
                    renderCustomSelect(`dropdown_${key}`, `fixed_${key}`, options, value);
                }, 0);
                return `<div class="form-group dropdown-group"><label>${getLabel(key)}</label><div id="dropdown_${key}"></div></div>`;
            }
            if (key === 'task') {
                const options = [
                    { value: 'i2v-A14B', label: 'i2v-A14B' },
                    { value: 't2v-A14B', label: 't2v-A14B' },
                    { value: 'i2v-5B', label: 'i2v-5B' },
                    { value: 't2v-5B', label: 't2v-5B' }
                ];
                setTimeout(() => {
                    renderCustomSelect(`dropdown_${key}`, `fixed_${key}`, options, value || 't2v-A14B');
                }, 0);
                return `<div class="form-group dropdown-group"><label>${getLabel(key)}</label><div id="dropdown_${key}"></div></div>`;
            }
            if (key === 'ui_noise_level') {
                return `<div class="form-group row-group">
                            <label style="position:relative;top:5px;display:block;">高低噪选择</label>
                            <div style="display: flex; align-items: center; gap: 15px; margin-top: 10px;">
                                <div style="display: flex; align-items: center; gap: 5px;">
                                    <label class="switch">
                                        <input type="checkbox" id="switch_noise_low" checked onchange="toggleNoiseSwitch('low')">
                                        <span class="slider"></span>
                                    </label>
                                    <span style="font-size: 0.85rem; color: #666;">低噪</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 5px;">
                                    <label class="switch">
                                        <input type="checkbox" id="switch_noise_high" onchange="toggleNoiseSwitch('high')">
                                        <span class="slider"></span>
                                    </label>
                                    <span style="font-size: 0.85rem; color: #666;">高噪</span>
                                </div>
                            </div>
                        </div>`;
            }
            if (typeof value === 'boolean') return renderToggleField(`fixed_${key}`, getLabel(key), value);

            // 检查是否需要路径选择器
            if (PATH_PICKER_CONFIG[key]) {
                return renderPathInput(`fixed_${key}`, getLabel(key), value, key);
            }

            const isInt = ['max_train_epochs', 'save_every_n_epochs', 'sample_every_n_epochs', 'network_dim', 'blocks_to_swap', 'loraplus_lr_ratio'].includes(key);
            let onInputAttr = isInt ? `oninput="this.value=this.value.replace(/[^0-9]/g,'')"` : '';
            if (key === 'output_name') onInputAttr += ` onkeyup="updateOutputDir(this.value)"`;

            return `<div class="form-group ${isInt ? 'input-mini-container' : ''}"><label>${getLabel(key)}</label><input type="text" id="input_fixed_${key}" name="fixed_${key}" value="${value}" ${onInputAttr} class="${isInt ? 'input-mini' : ''}"></div>`;
        }).join('')}</div>`;
    }).join('');

    const gdGroup = $('general-dataset-group');
    gdGroup.innerHTML = '';
    const datasets = template.datasets || [];
    const rowSection = document.createElement('div');
    rowSection.className = 'form-section';
    rowSection.innerHTML = `<div class="section-header"><h3><i class="fas fa-layer-group"></i> ${getLabel('dataset_config')}</h3></div><div class="params-grid" id="dataset-row-params"></div>`;
    gdGroup.appendChild(rowSection);

    const dsRow = $('dataset-row-params');
    const flatGeneral = filterHidden(template.general);
    const flatDataset = datasets.length > 0 ? filterHidden(datasets[0]) : {};
    renderDatasetRow(dsRow, flatGeneral, flatDataset, type);

    const sSection = $('page-config-samples');
    if (template.samples) {
        sSection.style.display = 'block';
        sSection.innerHTML = `<div class="section-header"><h3><i class="fas fa-vial"></i> 采样测试</h3><button type="button" class="btn btn-primary btn-sm" id="btn-add-sample"><i class="fas fa-plus"></i> 添加采样</button></div><div id="samples-list"></div>`;
        renderSamplesList(template.samples);
        $('btn-add-sample').onclick = () => {
            const current = getCurrentFormData();
            state.last_rendered_template = current.template;
            const defaultSample = state.templates[type].samples ? JSON.parse(JSON.stringify(state.templates[type].samples[0])) : {};
            if (!state.last_rendered_template.samples) state.last_rendered_template.samples = [];
            state.last_rendered_template.samples.push(defaultSample);
            renderSamplesList(state.last_rendered_template.samples);
        };
    } else {
        sSection.style.display = 'none';
        sSection.innerHTML = '';
    }
}

window.updateOutputDir = (val) => {
    const dirInput = $('input_fixed_output_dir');
    if (dirInput) dirInput.value = `./output/${val}`;
};

function renderDatasetRow(container, general, dataset, type) {
    const all = { ...general, ...dataset };
    let keys = Object.keys(all).filter(k => !HIDDEN_KEYS.includes(k));

    const widthKey = keys.find(k => k === 'width');
    const heightKey = keys.find(k => k === 'height');
    if (widthKey && heightKey) {
        keys = keys.filter(k => k !== widthKey && k !== heightKey);
        keys.unshift(heightKey);
        keys.unshift(widthKey);
    }

    if (type === 'Qwen-Image-Edit-2511' && keys.includes('qwen_image_edit_control_resolution')) {
        const ctrlRes = 'qwen_image_edit_control_resolution';
        keys = keys.filter(k => k !== ctrlRes);
        keys.unshift(ctrlRes);
    }

    container.innerHTML = keys.map(key => {
        const val = all[key];
        const prefix = general[key] !== undefined ? 'tmpl_general' : 'tmpl_datasets_0';
        const fullKey = `${prefix}_${key}`;

        if (key.includes('resolution')) {
            let mainLabel = '分辨率';
            if (key === 'qwen_image_edit_control_resolution') mainLabel = '控制图分辨率';

            return `
                <div class="form-group input-mini-container">
                    <label>${mainLabel}</label>
                    <div class="input-res-container">
                        <input type="text" name="${fullKey}_0" value="${val[0]}" placeholder="宽">
                        <span class="res-x">x</span>
                        <input type="text" name="${fullKey}_1" value="${val[1]}" placeholder="高">
                    </div>
                </div>
            `;
        }

        const isNumeric = typeof val === 'number' || key.includes('seed') || key.includes('scale') || key.includes('steps');
        const isPath = key.includes('directory') || key.includes('path') || key === 'prompt';

        if (typeof val === 'boolean') return `<div class="form-group row-group"><label>${getLabel(key)}</label><label class="switch"><input type="checkbox" name="${fullKey}" ${val ? 'checked' : ''} value="true"><span class="slider"></span></label></div>`;

        if (key === 'frame_extraction') {
            const options = [
                { value: 'head', label: 'head' },
                { value: 'chunk', label: 'chunk' },
                { value: 'uniform', label: 'uniform' },
                { value: 'full', label: 'full' }
            ];
            setTimeout(() => {
                renderCustomSelect(`dropdown_${fullKey}`, fullKey, options, val || 'uniform');
            }, 0);
            return `<div class="form-group dropdown-group"><label>${getLabel(key)}</label><div id="dropdown_${fullKey}"></div></div>`;
        }

        // 检查是否需要路径选择器（目录字段）
        if (PATH_PICKER_CONFIG[key]) {
            return renderPathInput(fullKey, getLabel(key), val, key);
        }

        return `<div class="form-group ${isPath ? 'flex-grow' : (isNumeric ? 'input-mini-container' : '')}"><label>${getLabel(key)}</label><input type="text" name="${fullKey}" value="${val}" class="${isNumeric ? 'input-mini' : ''}"></div>`;
    }).join('');
}

function renderToggleField(name, label, checked) {
    return `<div class="form-group row-group"><label>${label}</label><label class="switch"><input type="checkbox" name="${name}" ${checked ? 'checked' : ''} value="true"><span class="slider"></span></label></div>`;
}

function renderSamplesList(samples) {
    const list = $('samples-list');
    list.innerHTML = samples.map((sample, idx) => `
        <div class="sample-card">
            <div class="sample-card-header">
                <h4>采样测试 #${idx + 1}</h4>
                <div class="header-actions">
                    <button type="button" class="btn btn-danger-ghost btn-xs" onclick="removeSample(${idx})" title="删除此采样"><i class="fas fa-trash"></i></button>
                </div>
            </div>
            <div class="sample-prompt-row">
                <div class="form-group"><label>${getLabel('prompt')}</label><textarea name="tmpl_samples_${idx}_prompt">${sample.prompt || ''}</textarea></div>
            </div>
            <div class="sample-bottom-row">
                <div class="form-group input-mini-container">
                    <label>分辨率</label>
                    <div class="input-res-container">
                        <input type="text" name="tmpl_samples_${idx}_width" value="${sample.width}" placeholder="宽">
                        <span class="res-x">x</span>
                        <input type="text" name="tmpl_samples_${idx}_height" value="${sample.height}" placeholder="高">
                    </div>
                </div>
                ${Object.keys(sample).filter(k => k !== 'prompt' && k !== 'width' && k !== 'height' && !HIDDEN_KEYS.includes(k)).map(key => {
        const val = sample[key];
        const fullKey = `tmpl_samples_${idx}_${key}`;
        if (key === 'control_image_path') {
            const ps = Array.isArray(val) ? val : [val];
            return renderPathInput(`${fullKey}_0`, 'P1', ps[0] || '', 'control_image_path') +
                renderPathInput(`${fullKey}_1`, 'P2', ps[1] || '', 'control_image_path') +
                renderPathInput(`${fullKey}_2`, 'P3', ps[2] || '', 'control_image_path');
        }
        if (typeof val === 'boolean') return `<div class="form-group row-group"><label>${getLabel(key)}</label><label class="switch"><input type="checkbox" name="${fullKey}" ${val ? 'checked' : ''} value="true"><span class="slider"></span></label></div>`;

        // 调整：用户指定 "采样步数，CFG，种子，偏移" 需要固定短宽度 (input-mini)
        // P1/P2/P3 (control_image_path) 在上方已处理为 flex-grow
        const narrowKeys = ['seed', 'scale', 'steps', 'shift', 'frame_count'];
        const isMini = narrowKeys.some(k => key.includes(k)) || typeof val === 'number';

        return `<div class="form-group ${isMini ? 'input-mini-container' : 'flex-grow'}"><label>${getLabel(key)}</label><input type="text" name="${fullKey}" value="${val}" class="${isMini ? 'input-mini' : ''}"></div>`;
    }).join('')}
            </div>
        </div>
    `).join('');
}

function filterHidden(obj) {
    const newObj = {};
    for (let k in obj) { if (!HIDDEN_KEYS.includes(k)) newObj[k] = obj[k]; }
    return newObj;
}

function mergeTemplateWithNewType(oldTmpl, newTemplateBase) {
    const result = JSON.parse(JSON.stringify(newTemplateBase));
    if (oldTmpl.general) for (let k in result.general) { if (oldTmpl.general[k] !== undefined) result.general[k] = oldTmpl.general[k]; }
    if (oldTmpl.datasets && oldTmpl.datasets[0] && result.datasets && result.datasets[0]) {
        for (let k in result.datasets[0]) { if (oldTmpl.datasets[0][k] !== undefined) result.datasets[0][k] = oldTmpl.datasets[0][k]; }
    }
    if (oldTmpl.samples && oldTmpl.samples.length > 0 && result.samples !== undefined) {
        result.samples = oldTmpl.samples.map(oldS => {
            const newS = JSON.parse(JSON.stringify((result.samples && result.samples[0]) ? result.samples[0] : {}));
            for (let k in newS) { if (oldS[k] !== undefined) newS[k] = oldS[k]; }
            return newS;
        });
    } else if (result.samples === undefined) {
        delete result.samples;
    }
    return result;
}

window.removeSample = (idx) => {
    const current = getCurrentFormData();
    state.last_rendered_template = current.template;
    state.last_rendered_template.samples.splice(idx, 1);
    renderSamplesList(state.last_rendered_template.samples);
};

async function saveConfig() {
    const formData = new FormData($('config-form'));
    const name = formData.get('fixed_output_name') ? formData.get('fixed_output_name').trim() : '';
    if (!name) { showToast('output_name (Lora名) 不能为空'); return; }
    const fixed = {};
    const type = $('input_train_type').value;
    const isWan = type === 'Wan2.2';
    const activeBlocks = isWan ? FIXED_BLOCKS_WAN : FIXED_BLOCKS_QWEN;
    const template = JSON.parse(JSON.stringify(state.last_rendered_template));

    for (let block of activeBlocks) {
        for (let key of block.keys) {
            let val = formData.get(`fixed_${key}`);
            if (val === null) {
                const checkbox = document.querySelector(`input[name="fixed_${key}"][type="checkbox"]`);
                if (checkbox) val = 'false';
            }
            if (val !== null) fixed[key] = castType(key, val);
        }
    }

    updateObjectFromForm(template.general, 'tmpl_general', formData);
    if (template.datasets && template.datasets.length > 0) {
        updateObjectFromForm(template.datasets[0], 'tmpl_datasets_0', formData);
    }
    
    if (isWan) {
        delete template.samples;
        fixed.i2v = !!(fixed.task && fixed.task.includes('i2v'));
    } else {
        template.samples = (template.samples || []).map((_, idx) => {
            const sample = template.samples[idx];
            updateObjectFromForm(sample, `tmpl_samples_${idx}`, formData);
            return sample;
        });
    }

    // 1. 静默持久化核心：补全隐藏参数
    if (!fixed.dataset_config) {
        fixed.dataset_config = `./src/${name}.toml`;
    }
    
    // Qwen的专属参数
    if (!isWan) {
        if (!fixed.sample_prompts) fixed.sample_prompts = `./src/${name}.toml`;
        fixed.model_version = type === 'Qwen-Image-Edit-2511' ? 'edit-2511' : 'original';
    } else {
        delete fixed.sample_prompts;
        delete fixed.model_version;
        
        // Wan2.2 专有的静默推断及填充参数
        fixed.log_tracker_name = fixed.output_name;
        fixed.rank = fixed.network_dim;
        fixed.weight_decay = 0.01;
        fixed.update_proj_gap = 50;
        fixed.scale = 1;
        fixed.projection_type = 'std';
    }

    try {
        const res = await fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_new: !state.isEditMode, old_name: state.oldName, fixed_params: fixed, template_params: template, train_type: type })
        });
        if (res.ok) {
            showToast('保存成功');
            loadTasks();
            switchPage('home');
        } else {
            const err = await res.json();
            showToast(err.detail || '保存失败');
        }
    } catch (e) { showToast('保存失败'); }
}

function updateObjectFromForm(obj, prefix, formData) {
    for (let key in obj) {
        const fullKey = `${prefix}_${key}`;
        if (key.includes('resolution')) {
            const w = formData.get(`${fullKey}_0`);
            const h = formData.get(`${fullKey}_1`);
            if (w !== null && h !== null) obj[key] = [parseInt(w), parseInt(h)];
            continue;
        }
        if (key === 'control_image_path') {
            const p0 = formData.get(`${fullKey}_0`);
            const p1 = formData.get(`${fullKey}_1`);
            const p2 = formData.get(`${fullKey}_2`);
            // 过滤空路径
            obj[key] = [p0, p1, p2].filter(p => p && p.trim() !== "");
            continue;
        }
        if (key === 'target_frames') {
            const val = formData.get(fullKey);
            if (val) {
                obj[key] = val.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            }
            continue;
        }
        let val = formData.get(fullKey);
        if (val === null) {
            const checkbox = document.querySelector(`input[name="${fullKey}"][type="checkbox"]`);
            if (checkbox) val = 'false';
        }
        if (val !== null) obj[key] = castType(key, val);
    }
}

function castType(key, val) {
    if (val === 'true') return true;
    if (val === 'false') return false;
    if (['save_every_n_epochs', 'sample_every_n_epochs', 'max_train_epochs', 'batch_size', 'num_repeats', 'network_dim', 'network_alpha', 'blocks_to_swap', 'loraplus_lr_ratio', 'seed', 'discrete_flow_shift', 'sample_steps', 'width', 'height', 'guidance_scale', 'frame_count', 'max_frames', 'frame_sample', 'min_timestep', 'max_timestep', 'max_data_loader_n_workers'].includes(key)) {
        const n = parseFloat(val);
        return isNaN(n) ? val : n;
    }
    return val;
}

document.addEventListener('DOMContentLoaded', init);
