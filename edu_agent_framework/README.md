# 教育智能体速成框架

> 让高校教师3分钟创建具有个人风格的AI教学助手

## 核心价值

解决AI教育产品的关键痛点：**通用性有余，个性化不足**

| 问题 | 解决方案 |
|------|----------|
| AI缺乏教师个人教学风格 | **教学DNA指纹** - 量化语言/逻辑/互动/节奏 |
| 无法体现教师思维逻辑 | 可定义讲解顺序和推理模式 |
| 缺少教师私域知识 | 支持FAQ、案例、类比等注入 |
| 使用门槛高 | Web界面 + 一行代码创建 |

## 快速启动

### 方式1：Web界面（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置API（使用DeepSeek）
export LLM_API_KEY="your-deepseek-api-key"
export LLM_BASE_URL="https://api.deepseek.com/v1"

# 3. 启动服务
python run_web.py
```

访问 http://localhost:8001 开始使用

### 方式2：代码调用

```python
from edu_agent_framework import EduAgentFactory

# 极简创建
factory = EduAgentFactory()
agent = factory.create_quick("王老师", "高等数学")
response = agent.chat("什么是导数？")
```

## 个性化核心：教学DNA指纹

让AI"听起来像老师本人"的关键技术

```
教学DNA = 语言指纹 + 逻辑指纹 + 互动指纹 + 节奏指纹
           ↓           ↓           ↓           ↓
        口头禅     推理方式     反馈风格     难度曲线
        句式习惯   概念关联     提问模式     详略分配
        正式度     因果倾向     鼓励用语     举例时机
```

### 预置DNA模板

| 模板 | 特点 |
|------|------|
| 严谨学者型 | 定义→定理→证明，注重因果解释 |
| 亲和启发型 | 多用提问引导，先举例后理论 |
| 实战派教练 | 场景→方案→代码，直接指出问题 |

```python
from edu_agent_framework import DNA_TEMPLATES

# 使用预置模板
agent = factory.create_from_template("严谨学者型", "李教授", "数据结构")
```

## 环境配置

### 使用DeepSeek API（推荐）

```bash
export LLM_API_KEY="sk-xxx"
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"
```

### 使用其他代理

```bash
export LLM_API_KEY="your-key"
export LLM_BASE_URL="https://your-proxy.com/v1"
export LLM_MODEL="gpt-4o"
```

### 配置文件方式

复制 `.env.example` 为 `.env` 并填写配置。

## 目录结构

```
edu_agent_framework/
├── core/                     # 核心模块
│   ├── teacher_profile.py    # 教师画像配置
│   ├── teaching_dna.py       # 教学DNA指纹系统
│   ├── agent_factory.py      # 智能体工厂
│   └── base_edu_agent.py     # 智能体基类
├── generators/               # Prompt生成器
├── agents/                   # 智能体实现
├── knowledge/                # 知识处理
├── web/                      # Web界面
│   └── app.py               # Streamlit应用
├── templates/                # 配置模板
├── demo/                     # 演示代码
├── docs/                     # 技术文档
├── run_web.py               # Web启动脚本
├── config.py                # 配置管理
├── requirements.txt         # 依赖清单
└── .env.example             # 环境变量模板
```

## 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    Web界面 (8001端口)                │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  教师配置                                            │
│  (姓名/课程/风格/口头禅/FAQ/类比...)                 │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  教学DNA提取                                         │
│  语言指纹 | 逻辑指纹 | 互动指纹 | 节奏指纹           │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  Prompt生成器                                        │
│  角色定位 + DNA特征 + 教学风格 + 私域知识            │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  LLM调用 (DeepSeek / OpenAI兼容)                     │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  个性化AI助教回复                                    │
└─────────────────────────────────────────────────────┘
```

## 教师需要提供什么？

### 最简输入（3秒创建）
- 姓名
- 课程名

### 推荐输入（让AI更像你）
```yaml
# 教学风格
primary_style: "苏格拉底式"
interaction_mode: "鼓励支持型"

# 特色表达
signature_phrases:
  - "这个问题很好..."
  - "换个角度想..."

# 私域知识（最重要！）
faq:
  - question: "学生常问的问题"
    answer: "你的标准回答"

common_mistakes:
  - mistake: "学生常犯的错误"
    correction: "正确理解"

analogies:
  - concept: "抽象概念"
    analogy: "你常用的类比"
```

## 详细文档

- [技术设计文档](docs/TECHNICAL_DESIGN.md)
- [配置模板](templates/teacher_config_template.yaml)
- [环境变量说明](.env.example)
