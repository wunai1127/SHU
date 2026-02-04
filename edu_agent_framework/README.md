# 教育智能体速成框架

> 让高校教师3分钟创建具有个人风格的AI教学助手

## 核心价值

解决AI教育产品的关键痛点：**通用性有余，个性化不足**

| 问题 | 解决方案 |
|------|----------|
| AI缺乏教师个人教学风格 | 多维度教学风格配置 |
| 无法体现教师思维逻辑 | 可定义讲解顺序和深度策略 |
| 缺少教师私域知识 | 支持FAQ、案例、类比等注入 |
| 使用门槛高 | 一行代码创建智能体 |

## 快速开始

### 方式1：极简创建（3行代码）

```python
from edu_agent_framework import EduAgentFactory

factory = EduAgentFactory()
agent = factory.create_quick("王老师", "高等数学")
response = agent.chat("什么是导数？")
```

### 方式2：使用预置模板

```python
agent = factory.create_from_template(
    "理工科严谨型",  # 或 "文科启发型" / "实践案例型"
    "李老师",
    "数据结构"
)
```

### 方式3：配置文件定制

```python
agent = factory.create_from_config("my_teacher.yaml")
```

## 教师需要提供什么？

### 必填项
- 姓名
- 课程名

### 推荐填写（让AI更像你）
```yaml
# 教学风格
primary_style: "苏格拉底式"  # 启发式提问
interaction_mode: "鼓励支持型"

# 教学逻辑
explanation_order: "问题→讨论→概念→应用"
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

## 技术架构

```
教师配置 (YAML)
     ↓
┌─────────────────┐
│  Prompt生成器   │  ← 多层Prompt组装
├─────────────────┤
│  智能体工厂     │  ← 一键生成
├─────────────────┤
│  知识索引器     │  ← RAG检索增强
└─────────────────┘
     ↓
个性化AI助教
```

## 目录结构

```
edu_agent_framework/
├── core/                 # 核心模块
│   ├── teacher_profile.py    # 教师画像配置
│   ├── agent_factory.py      # 智能体工厂
│   └── base_edu_agent.py     # 智能体基类
├── generators/           # Prompt生成器
├── agents/               # 智能体实现
├── knowledge/            # 知识处理
├── templates/            # 配置模板
├── demo/                 # 演示代码
└── docs/                 # 技术文档
```

## 运行Demo

```bash
# 快速演示
python -m edu_agent_framework.demo.demo_quick_start

# 交互式对话（需配置OPENAI_API_KEY）
export OPENAI_API_KEY="your-key"
python -m edu_agent_framework.demo.demo_interactive
```

## 详细文档

- [技术设计文档](docs/TECHNICAL_DESIGN.md)
- [配置模板](templates/teacher_config_template.yaml)
