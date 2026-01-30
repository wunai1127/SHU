"""
HTTG Multi-Agent System
========================
六层Agent架构：Monitor → Diagnosis → Strategy → Knowledge → Communication → Coordinator

每个Agent配备专用工具(Tools)，通过EventBus通信，Coordinator编排流程。
"""

from .base import (
    BaseAgent, EventBus, AgentMessage, PatientState,
    AgentTool, ToolResult, AlertEvent, DiagnosisResult
)
from .monitor_agent import MonitorAgent
from .diagnosis_agent import DiagnosisAgent
from .strategy_agent import StrategyAgent
from .knowledge_agent import KnowledgeAgent
from .communication_agent import CommunicationAgent
from .coordinator import CoordinatorAgent

__all__ = [
    'BaseAgent', 'EventBus', 'AgentMessage', 'PatientState',
    'AgentTool', 'ToolResult', 'AlertEvent', 'DiagnosisResult',
    'MonitorAgent', 'DiagnosisAgent', 'StrategyAgent',
    'KnowledgeAgent', 'CommunicationAgent', 'CoordinatorAgent',
]
