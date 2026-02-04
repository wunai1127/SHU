"""
教师画像配置模块
================
定义教师个性化智能体所需的所有输入参数

核心理念：
1. 教学风格 - 如何讲解知识
2. 知识体系 - 讲什么内容
3. 私域素材 - 独特的教学资源
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json
import yaml
from pathlib import Path


class TeachingStyle(Enum):
    """教学风格枚举"""
    SOCRATIC = "苏格拉底式"        # 启发式提问
    LECTURE = "讲授式"             # 系统讲解
    CASE_BASED = "案例式"          # 以案例为核心
    PROBLEM_BASED = "问题导向式"    # PBL教学
    DISCUSSION = "讨论式"          # 互动讨论
    FLIPPED = "翻转课堂式"         # 先学后教
    SCAFFOLDING = "支架式"         # 逐步引导
    STORYTELLING = "故事式"        # 以故事串联


class InteractionMode(Enum):
    """互动模式"""
    PATIENT = "耐心细致型"
    CHALLENGING = "挑战激励型"
    SUPPORTIVE = "鼓励支持型"
    RIGOROUS = "严谨学术型"
    HUMOROUS = "幽默风趣型"


class AssessmentStyle(Enum):
    """评价风格"""
    FORMATIVE = "过程性评价"
    SUMMATIVE = "总结性评价"
    PEER = "同伴互评"
    SELF = "自我反思"


@dataclass
class KnowledgeDomain:
    """知识领域配置"""
    name: str                                    # 学科/课程名称
    level: str                                   # 教学层次：本科/硕士/博士/通识
    core_concepts: List[str] = field(default_factory=list)  # 核心概念
    prerequisite: List[str] = field(default_factory=list)   # 先修知识
    learning_objectives: List[str] = field(default_factory=list)  # 学习目标
    knowledge_map: Dict[str, List[str]] = field(default_factory=dict)  # 知识图谱结构

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "core_concepts": self.core_concepts,
            "prerequisite": self.prerequisite,
            "learning_objectives": self.learning_objectives,
            "knowledge_map": self.knowledge_map
        }


@dataclass
class PrivateKnowledge:
    """私域知识配置"""
    # 教学文档
    lecture_notes: List[str] = field(default_factory=list)      # PPT、讲义路径
    textbooks: List[str] = field(default_factory=list)          # 教材路径
    reading_materials: List[str] = field(default_factory=list)  # 拓展阅读

    # 教学素材
    case_studies: List[Dict[str, Any]] = field(default_factory=list)  # 案例库
    examples: List[Dict[str, Any]] = field(default_factory=list)      # 例题库
    analogies: List[Dict[str, str]] = field(default_factory=list)     # 类比/比喻库

    # 历史积累
    faq: List[Dict[str, str]] = field(default_factory=list)           # 常见问题
    common_mistakes: List[Dict[str, str]] = field(default_factory=list)  # 易错点
    teaching_tips: List[str] = field(default_factory=list)            # 教学心得

    def to_dict(self) -> dict:
        return {
            "lecture_notes": self.lecture_notes,
            "textbooks": self.textbooks,
            "reading_materials": self.reading_materials,
            "case_studies": self.case_studies,
            "examples": self.examples,
            "analogies": self.analogies,
            "faq": self.faq,
            "common_mistakes": self.common_mistakes,
            "teaching_tips": self.teaching_tips
        }


@dataclass
class TeachingLogic:
    """教学逻辑配置"""
    # 讲解顺序偏好
    explanation_order: str = "概念->原理->应用->总结"  # 讲解逻辑链
    depth_preference: str = "先广后深"                 # 深度策略：先深后广/先广后深/平衡

    # 讲解习惯
    use_analogy: bool = True                          # 是否常用类比
    use_examples_first: bool = False                  # 是否先举例后讲理论
    emphasize_application: bool = True                # 是否强调应用
    connect_to_frontier: bool = False                 # 是否连接前沿研究

    # 特色短语/口头禅
    signature_phrases: List[str] = field(default_factory=list)

    # 引用偏好
    preferred_references: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "explanation_order": self.explanation_order,
            "depth_preference": self.depth_preference,
            "use_analogy": self.use_analogy,
            "use_examples_first": self.use_examples_first,
            "emphasize_application": self.emphasize_application,
            "connect_to_frontier": self.connect_to_frontier,
            "signature_phrases": self.signature_phrases,
            "preferred_references": self.preferred_references
        }


@dataclass
class TeacherProfile:
    """
    教师画像完整配置
    =================
    这是教师需要填写的核心配置，用于生成个性化智能体
    """

    # === 基本信息 ===
    teacher_id: str                              # 教师ID
    name: str                                    # 教师姓名
    title: str = ""                              # 职称
    institution: str = ""                        # 所属院校
    department: str = ""                         # 所属院系

    # === 教学风格 ===
    primary_style: TeachingStyle = TeachingStyle.LECTURE
    secondary_styles: List[TeachingStyle] = field(default_factory=list)
    interaction_mode: InteractionMode = InteractionMode.SUPPORTIVE
    assessment_style: AssessmentStyle = AssessmentStyle.FORMATIVE

    # === 知识领域 ===
    domains: List[KnowledgeDomain] = field(default_factory=list)

    # === 教学逻辑 ===
    teaching_logic: TeachingLogic = field(default_factory=TeachingLogic)

    # === 私域知识 ===
    private_knowledge: PrivateKnowledge = field(default_factory=PrivateKnowledge)

    # === 个性化配置 ===
    language_style: str = "学术但易懂"           # 语言风格
    response_length: str = "适中"                # 回复长度偏好：简洁/适中/详细
    use_emoji: bool = False                      # 是否使用emoji
    formality_level: int = 3                     # 正式程度 1-5

    # === 智能体行为配置 ===
    auto_quiz: bool = False                      # 是否自动出题检测
    provide_hints: bool = True                   # 是否提供提示
    encourage_questions: bool = True             # 是否鼓励提问
    track_progress: bool = True                  # 是否跟踪学习进度

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "teacher_id": self.teacher_id,
            "name": self.name,
            "title": self.title,
            "institution": self.institution,
            "department": self.department,
            "primary_style": self.primary_style.value,
            "secondary_styles": [s.value for s in self.secondary_styles],
            "interaction_mode": self.interaction_mode.value,
            "assessment_style": self.assessment_style.value,
            "domains": [d.to_dict() for d in self.domains],
            "teaching_logic": self.teaching_logic.to_dict(),
            "private_knowledge": self.private_knowledge.to_dict(),
            "language_style": self.language_style,
            "response_length": self.response_length,
            "use_emoji": self.use_emoji,
            "formality_level": self.formality_level,
            "auto_quiz": self.auto_quiz,
            "provide_hints": self.provide_hints,
            "encourage_questions": self.encourage_questions,
            "track_progress": self.track_progress
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """导出为JSON"""
        data = self.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            Path(path).write_text(json_str, encoding='utf-8')
        return json_str

    def to_yaml(self, path: Optional[str] = None) -> str:
        """导出为YAML（更易读）"""
        data = self.to_dict()
        yaml_str = yaml.dump(data, allow_unicode=True, default_flow_style=False)
        if path:
            Path(path).write_text(yaml_str, encoding='utf-8')
        return yaml_str

    @classmethod
    def from_dict(cls, data: dict) -> 'TeacherProfile':
        """从字典创建"""
        # 处理枚举类型
        primary_style = TeachingStyle(data.get('primary_style', '讲授式'))
        secondary_styles = [TeachingStyle(s) for s in data.get('secondary_styles', [])]
        interaction_mode = InteractionMode(data.get('interaction_mode', '鼓励支持型'))
        assessment_style = AssessmentStyle(data.get('assessment_style', '过程性评价'))

        # 处理知识领域
        domains = [KnowledgeDomain(**d) for d in data.get('domains', [])]

        # 处理教学逻辑
        teaching_logic = TeachingLogic(**data.get('teaching_logic', {}))

        # 处理私域知识
        private_knowledge = PrivateKnowledge(**data.get('private_knowledge', {}))

        return cls(
            teacher_id=data['teacher_id'],
            name=data['name'],
            title=data.get('title', ''),
            institution=data.get('institution', ''),
            department=data.get('department', ''),
            primary_style=primary_style,
            secondary_styles=secondary_styles,
            interaction_mode=interaction_mode,
            assessment_style=assessment_style,
            domains=domains,
            teaching_logic=teaching_logic,
            private_knowledge=private_knowledge,
            language_style=data.get('language_style', '学术但易懂'),
            response_length=data.get('response_length', '适中'),
            use_emoji=data.get('use_emoji', False),
            formality_level=data.get('formality_level', 3),
            auto_quiz=data.get('auto_quiz', False),
            provide_hints=data.get('provide_hints', True),
            encourage_questions=data.get('encourage_questions', True),
            track_progress=data.get('track_progress', True)
        )

    @classmethod
    def from_json(cls, path: str) -> 'TeacherProfile':
        """从JSON文件加载"""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str) -> 'TeacherProfile':
        """从YAML文件加载"""
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        return cls.from_dict(data)


# === 快速创建模板 ===

def create_quick_profile(
    teacher_id: str,
    name: str,
    course_name: str,
    style: str = "讲授式",
    **kwargs
) -> TeacherProfile:
    """
    快速创建教师画像（简化版）
    适合快速上手，只需填写最基本信息
    """
    style_map = {s.value: s for s in TeachingStyle}

    return TeacherProfile(
        teacher_id=teacher_id,
        name=name,
        primary_style=style_map.get(style, TeachingStyle.LECTURE),
        domains=[KnowledgeDomain(name=course_name, level="本科")],
        **kwargs
    )


# === 预置模板 ===

PROFILE_TEMPLATES = {
    "理工科严谨型": {
        "primary_style": "讲授式",
        "interaction_mode": "严谨学术型",
        "teaching_logic": {
            "explanation_order": "定义->定理->证明->例题->应用",
            "depth_preference": "先深后广",
            "use_analogy": False,
            "use_examples_first": False,
            "emphasize_application": True
        },
        "language_style": "严谨学术",
        "formality_level": 4
    },
    "文科启发型": {
        "primary_style": "苏格拉底式",
        "interaction_mode": "鼓励支持型",
        "teaching_logic": {
            "explanation_order": "问题->讨论->观点->总结",
            "depth_preference": "先广后深",
            "use_analogy": True,
            "use_examples_first": True,
            "emphasize_application": False
        },
        "language_style": "生动易懂",
        "formality_level": 2
    },
    "实践案例型": {
        "primary_style": "案例式",
        "interaction_mode": "挑战激励型",
        "teaching_logic": {
            "explanation_order": "案例->分析->原理->迁移",
            "depth_preference": "平衡",
            "use_analogy": True,
            "use_examples_first": True,
            "emphasize_application": True
        },
        "language_style": "实用接地气",
        "formality_level": 2
    }
}
